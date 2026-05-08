from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.asr.whisper_transcribe import transcribe_manifest
from src.common.logging_utils import get_logger
from src.data.build_manifests import build_manifests
from src.data.build_splits import build_splits
from src.data.windowing import WindowConfig, build_window_manifest

LOGGER = get_logger(__name__)


def apply_split_info(utterance_path: str, window_path: str, split_path: str, regime: str) -> None:
    utterance_df = pd.read_parquet(utterance_path)
    split_df = pd.read_parquet(split_path)[["record_id", "fold_id", "split"]]

    utterance_df = utterance_df.drop(columns=[c for c in ["fold_id", "split"] if c in utterance_df.columns], errors="ignore")
    utterance_df = utterance_df.merge(split_df, on="record_id", how="left")
    utterance_df.to_parquet(utterance_path, index=False)

    window_df = build_window_manifest(utterance_df, regime=regime, config=WindowConfig())
    window_df.to_parquet(window_path, index=False)


def prepare_data_pipeline(regime: str, input_table: str, run_asr: bool) -> None:
    out_dir = f"data/processed/{regime}"
    utterance_path, window_path = build_manifests(input_table_path=input_table, out_dir=out_dir, regime=regime)

    if regime == "asr" and run_asr:
        transcribe_manifest(utterance_path, utterance_path)

    split_path = "data/splits/meld_iemocap_folds.parquet"
    build_splits(utterance_manifest_path=utterance_path, output_path=split_path, n_splits=5, seed=42)
    apply_split_info(utterance_path, window_path, split_path, regime)

    LOGGER.info("Prepared %s manifests", regime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare processed manifests and grouped CV splits.")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--input-table", default="data/interim/normalized_utterances.parquet")
    parser.add_argument("--run-asr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.input_table).exists():
        raise FileNotFoundError(f"Input table not found: {args.input_table}")
    prepare_data_pipeline(args.regime, args.input_table, run_asr=args.run_asr)


if __name__ == "__main__":
    main()
