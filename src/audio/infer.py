from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.audio.dataset import AudioWindowDataset
from src.audio.model_wav2vec2 import Wav2Vec2EmotionModel


def infer_audio_embeddings(regime: str, fold: int) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path(f"data/processed/{regime}/window_manifest.parquet")
    ckpt_path = Path(f"experiments/artifacts/audio/fold_{fold}.pt")

    df = pd.read_parquet(manifest_path)
    ds = AudioWindowDataset(df)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = Wav2Vec2EmotionModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    records = []
    with torch.no_grad():
        row_index = 0
        for batch in loader:
            x = batch["input_values"].to(device)
            window_ids = batch["window_id"]
            emb, logits = model(x)
            probs = torch.softmax(logits, dim=1)
            for i in range(x.shape[0]):
                records.append(
                    {
                        "window_id": window_ids[i],
                        "audio_embedding": emb[i].cpu().numpy().tolist(),
                        "audio_probs": probs[i].cpu().numpy().tolist(),
                    }
                )
                row_index += 1

    out_path = Path(f"data/features/audio/{regime}_fold{fold}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(out_path, index=False)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer audio embeddings and probabilities.")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = infer_audio_embeddings(args.regime, args.fold)
    print(output)


if __name__ == "__main__":
    main()
