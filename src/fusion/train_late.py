from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.common.logging_utils import get_logger
from src.common.seed import set_seed
from src.fusion.calibrate_temperature import fit_temperature
from src.fusion.dataset import FusionWindowDataset, load_fusion_dataframe
from src.fusion.model_late_mlp import LateFusionMLP

LOGGER = get_logger(__name__)


def effective_num_weights(labels: np.ndarray, beta: float = 0.999, num_classes: int = 7) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def eval_model(model: LateFusionMLP, loader: DataLoader, device: torch.device) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    y_true = []
    y_pred = []
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["label_id"].to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            logits_list.append(logits)
            labels_list.append(y)

    macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    all_logits = torch.cat(logits_list, dim=0) if logits_list else torch.empty(0, 7, device=device)
    all_labels = torch.cat(labels_list, dim=0) if labels_list else torch.empty(0, dtype=torch.long, device=device)
    return float(np.mean(losses) if losses else 0.0), macro_f1, all_logits, all_labels


def train_late_fusion(regime: str, fold: int, max_epochs: int = 40, batch_size: int = 256) -> tuple[str, float]:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_df = load_fusion_dataframe(regime, fold)
    train_df = full_df[(full_df["split"] == "train_or_val") & (full_df["fold_id"] != fold)].copy()
    val_df = full_df[(full_df["split"] == "train_or_val") & (full_df["fold_id"] == fold)].copy()

    train_ds = FusionWindowDataset(train_df)
    val_ds = FusionWindowDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LateFusionMLP().to(device)
    class_weights = effective_num_weights(train_df["label_id"].to_numpy()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    best_f1 = -1.0
    wait = 0
    patience = 7

    out_path = Path(f"experiments/artifacts/fusion/late_{regime}_fold_{fold}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_logits = torch.empty(0, 7, device=device)
    best_val_labels = torch.empty(0, dtype=torch.long, device=device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["label_id"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss, val_macro_f1, val_logits, val_labels = eval_model(model, val_loader, device)
        LOGGER.info("fusion-late | regime=%s fold=%s epoch=%s val_loss=%.4f val_macro_f1=%.4f", regime, fold, epoch, val_loss, val_macro_f1)

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            wait = 0
            best_val_logits = val_logits
            best_val_labels = val_labels
            torch.save(model.state_dict(), out_path)
        else:
            wait += 1
            if wait >= patience:
                LOGGER.info("Early stopping at epoch %s", epoch)
                break

    temperature = 1.0
    if best_val_logits.numel() > 0:
        temperature = fit_temperature(best_val_logits, best_val_labels)

    temp_path = Path(f"experiments/artifacts/fusion/late_{regime}_fold_{fold}_temperature.txt")
    temp_path.write_text(f"{temperature:.8f}\n", encoding="utf-8")
    return str(out_path), temperature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train late-fusion MLP.")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt, temperature = train_late_fusion(args.regime, args.fold, max_epochs=args.max_epochs, batch_size=args.batch_size)
    LOGGER.info("Saved late fusion checkpoint: %s", ckpt)
    LOGGER.info("Saved calibrated temperature: %.6f", temperature)


if __name__ == "__main__":
    main()
