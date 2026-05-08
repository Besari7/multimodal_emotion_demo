from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _extract_probs_column(df: pd.DataFrame, column: str) -> np.ndarray:
    values = df[column].to_list()
    if not values:
        return np.empty((0, 7), dtype=np.float32)
    return np.array(values, dtype=np.float32)


def train_stacking_meta_learner(regime: str, fold: int) -> str:
    audio = pd.read_parquet(f"data/features/audio/{regime}_fold{fold}.parquet")
    video = pd.read_parquet(f"data/features/video/{regime}_fold{fold}.parquet")
    text = pd.read_parquet(f"data/features/text/{regime}_fold{fold}.parquet")
    base = pd.read_parquet(f"data/processed/{regime}/window_manifest.parquet")

    df = base.merge(audio[["window_id", "audio_probs"]], on="window_id", how="left")
    df = df.merge(video[["window_id", "video_probs"]], on="window_id", how="left")
    df = df.merge(text[["window_id", "text_probs"]], on="window_id", how="left")

    train_df = df[(df["split"] == "train_or_val") & (df["fold_id"] != fold)].copy()
    val_df = df[(df["split"] == "train_or_val") & (df["fold_id"] == fold)].copy()

    x_train = np.concatenate(
        [
            _extract_probs_column(train_df, "audio_probs"),
            _extract_probs_column(train_df, "video_probs"),
            _extract_probs_column(train_df, "text_probs"),
        ],
        axis=1,
    )
    x_val = np.concatenate(
        [
            _extract_probs_column(val_df, "audio_probs"),
            _extract_probs_column(val_df, "video_probs"),
            _extract_probs_column(val_df, "text_probs"),
        ],
        axis=1,
    )

    y_train = train_df["label_id"].to_numpy()
    y_val = val_df["label_id"].to_numpy()

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
    )
    clf.fit(x_train, y_train)

    val_pred = clf.predict(x_val)
    macro_f1 = f1_score(y_val, val_pred, average="macro") if len(y_val) else 0.0
    LOGGER.info("stacking | regime=%s fold=%s val_macro_f1=%.4f", regime, fold, macro_f1)

    out_dir = Path("experiments/artifacts/fusion")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stacking_{regime}_fold_{fold}.npz"
    np.savez(
        out_path,
        coef_=clf.coef_,
        intercept_=clf.intercept_,
        classes_=clf.classes_,
        val_macro_f1=np.array([macro_f1], dtype=np.float32),
    )
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stacking meta-learner from unimodal predictions.")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = train_stacking_meta_learner(args.regime, args.fold)
    LOGGER.info("Saved stacking model: %s", output)


if __name__ == "__main__":
    main()
