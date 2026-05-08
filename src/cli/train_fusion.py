from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger
from src.fusion.dataset import load_fusion_dataframe
from src.fusion.train_late import train_late_fusion
from src.fusion.train_stacking import train_stacking_meta_learner

LOGGER = get_logger(__name__)


def materialize_fusion_features(regime: str, fold: int) -> str:
    df = load_fusion_dataframe(regime, fold)

    def row_to_feature(row: pd.Series) -> list[float]:
        audio = row.get("audio_embedding", [0.0] * 256)
        video = row.get("video_embedding", [0.0] * 256)
        text = row.get("text_embedding", [0.0] * 256)
        extra = [
            float(row.get("asr_conf_mean", 1.0)),
            float(row.get("face_det_conf_mean", 1.0)),
            float(bool(row.get("missing_audio", False))),
            float(bool(row.get("missing_video", False))),
            float(bool(row.get("missing_text", False))),
        ]

        audio = audio if isinstance(audio, list) and len(audio) == 256 else [0.0] * 256
        video = video if isinstance(video, list) and len(video) == 256 else [0.0] * 256
        text = text if isinstance(text, list) and len(text) == 256 else [0.0] * 256

        vec = np.array(audio + video + text + extra, dtype=np.float32)
        return vec.tolist()

    out_df = pd.DataFrame({
        "window_id": df["window_id"],
        "fusion_input": df.apply(row_to_feature, axis=1),
    })

    out_path = Path(f"data/features/fusion/{regime}_fold{fold}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fusion model.")
    parser.add_argument("--mode", choices=["late", "stacking"], required=True)
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fusion_feature_path = materialize_fusion_features(args.regime, args.fold)
    LOGGER.info("Saved fusion features: %s", fusion_feature_path)

    if args.mode == "late":
        ckpt, temp = train_late_fusion(args.regime, args.fold)
        LOGGER.info("Late fusion checkpoint: %s", ckpt)
        LOGGER.info("Temperature: %.6f", temp)
    else:
        out = train_stacking_meta_learner(args.regime, args.fold)
        LOGGER.info("Stacking model: %s", out)


if __name__ == "__main__":
    main()
