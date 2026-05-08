from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.text.dataset import TextWindowDataset
from src.text.model_bert_context import BertContextEmotionModel


def infer_text_embeddings(regime: str, fold: int) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path(f"data/processed/{regime}/window_manifest.parquet")
    ckpt_path = Path(f"experiments/artifacts/text/{regime}_fold_{fold}.pt")

    df = pd.read_parquet(manifest_path)
    ds = TextWindowDataset(df)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = BertContextEmotionModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    records = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            window_ids = batch["window_id"]
            emb, logits = model(ids, mask)
            probs = torch.softmax(logits, dim=1)
            for i in range(ids.shape[0]):
                records.append(
                    {
                        "window_id": window_ids[i],
                        "text_embedding": emb[i].cpu().numpy().tolist(),
                        "text_probs": probs[i].cpu().numpy().tolist(),
                    }
                )

    out_path = Path(f"data/features/text/{regime}_fold{fold}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer text embeddings and probabilities.")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = infer_text_embeddings(args.regime, args.fold)
    print(output)


if __name__ == "__main__":
    main()
