from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.common.logging_utils import get_logger
from src.common.seed import set_seed
from src.text.dataset import TextWindowDataset
from src.text.model_bert_context import BertContextEmotionModel

LOGGER = get_logger(__name__)


def effective_num_weights(labels: np.ndarray, beta: float = 0.999, num_classes: int = 7) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def eval_model(model: BertContextEmotionModel, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label_id"].to(device)
            _, logits = model(ids, mask)
            loss = criterion(logits, y)
            losses.append(loss.item())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    macro_f1 = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    return float(np.mean(losses) if losses else 0.0), macro_f1


def train_text_branch(regime: str, fold: int, max_epochs: int = 20, batch_size: int = 32) -> str:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(f"data/processed/{regime}/window_manifest.parquet")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Window manifest not found: {manifest_path}")

    df = pd.read_parquet(manifest_path)
    train_df = df[(df["split"] == "train_or_val") & (df["fold_id"] != fold)].copy()
    val_df = df[(df["split"] == "train_or_val") & (df["fold_id"] == fold)].copy()

    train_ds = TextWindowDataset(train_df)
    val_ds = TextWindowDataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BertContextEmotionModel().to(device)
    model.freeze_embeddings_and_encoder_0_7()

    class_weights = effective_num_weights(train_df["label_id"].to_numpy())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)

    best_f1 = -1.0
    patience = 4
    wait = 0

    out_path = Path(f"experiments/artifacts/text/{regime}_fold_{fold}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        if epoch == 3:
            model.unfreeze_encoder_8_11()
            optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
        if epoch == 6:
            model.unfreeze_all()
            optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label_id"].to(device)

            optimizer.zero_grad()
            _, logits = model(ids, mask)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss, val_macro_f1 = eval_model(model, val_loader, device)
        LOGGER.info("text | regime=%s fold=%s epoch=%s val_loss=%.4f val_macro_f1=%.4f", regime, fold, epoch, val_loss, val_macro_f1)

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1
            wait = 0
            torch.save(model.state_dict(), out_path)
        else:
            wait += 1
            if wait >= patience:
                LOGGER.info("Early stopping at epoch %s", epoch)
                break

    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text branch.")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = train_text_branch(args.regime, args.fold, max_epochs=args.max_epochs, batch_size=args.batch_size)
    LOGGER.info("Saved text checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
