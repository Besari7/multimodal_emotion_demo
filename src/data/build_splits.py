from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


REQUIRED_COLUMNS = [
    "record_id",
    "dataset",
    "label_id",
    "dialogue_id",
    "session_id",
]


def _assign_dataset_folds(df: pd.DataFrame, group_col: str, n_splits: int, seed: int) -> pd.DataFrame:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df = df.copy()
    df["fold_id"] = -1

    y = df["label_id"].values
    groups = df[group_col].fillna("missing_group").astype(str).values

    for fold_id, (_, val_idx) in enumerate(splitter.split(df, y=y, groups=groups)):
        df.iloc[val_idx, df.columns.get_loc("fold_id")] = fold_id

    return df


def build_splits(utterance_manifest_path: str, output_path: str, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = pd.read_parquet(utterance_manifest_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    meld_df = df[df["dataset"].str.lower() == "meld"].copy()
    iemocap_df = df[df["dataset"].str.lower() == "iemocap"].copy()
    other_df = df[~df["dataset"].str.lower().isin(["meld", "iemocap"])].copy()

    if not meld_df.empty:
        meld_df = _assign_dataset_folds(meld_df, group_col="dialogue_id", n_splits=n_splits, seed=seed)
        meld_df["split"] = "train_or_val"

    if not iemocap_df.empty:
        iemocap_df = _assign_dataset_folds(iemocap_df, group_col="session_id", n_splits=n_splits, seed=seed)
        iemocap_df["split"] = "train_or_val"

    if not other_df.empty:
        other_df["fold_id"] = -1
        other_df["split"] = "mosei_test"

    merged = pd.concat([meld_df, iemocap_df, other_df], axis=0).sort_values("record_id").reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    LOGGER.info("Saved fold assignments: %s", output_path)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe grouped CV splits.")
    parser.add_argument("--utterance-manifest", default="data/processed/asr/utterance_manifest.parquet")
    parser.add_argument("--output", default="data/splits/meld_iemocap_folds.parquet")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_splits(
        utterance_manifest_path=args.utterance_manifest,
        output_path=args.output,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
