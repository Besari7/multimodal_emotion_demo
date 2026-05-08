from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _safe_vec(value, dim: int = 256) -> np.ndarray:
    if isinstance(value, list) and len(value) == dim:
        return np.array(value, dtype=np.float32)
    if isinstance(value, np.ndarray) and value.shape[0] == dim:
        return value.astype(np.float32)
    return np.zeros(dim, dtype=np.float32)


def load_fusion_dataframe(regime: str, fold: int) -> pd.DataFrame:
    base = pd.read_parquet(f"data/processed/{regime}/window_manifest.parquet")
    audio = pd.read_parquet(f"data/features/audio/{regime}_fold{fold}.parquet")
    video = pd.read_parquet(f"data/features/video/{regime}_fold{fold}.parquet")
    text = pd.read_parquet(f"data/features/text/{regime}_fold{fold}.parquet")

    merged = base.merge(audio, on="window_id", how="left")
    merged = merged.merge(video, on="window_id", how="left")
    merged = merged.merge(text, on="window_id", how="left")
    return merged


class FusionWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        audio_emb = _safe_vec(row.get("audio_embedding", None), 256)
        video_emb = _safe_vec(row.get("video_embedding", None), 256)
        text_emb = _safe_vec(row.get("text_embedding", None), 256)

        asr_conf = float(row.get("asr_conf_mean", 1.0))
        face_conf = float(row.get("face_det_conf_mean", 1.0))
        missing_audio = float(bool(row.get("missing_audio", False)))
        missing_video = float(bool(row.get("missing_video", False)))
        missing_text = float(bool(row.get("missing_text", False)))

        feature = np.concatenate(
            [
                audio_emb,
                video_emb,
                text_emb,
                np.array([asr_conf, face_conf, missing_audio, missing_video, missing_text], dtype=np.float32),
            ]
        )

        return {
            "window_id": row["window_id"],
            "x": torch.tensor(feature, dtype=torch.float32),
            "label_id": int(row.get("label_id", -1)),
        }
