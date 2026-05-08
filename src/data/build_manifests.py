from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.constants import CLASS_TO_ID
from src.common.io_utils import write_parquet
from src.common.logging_utils import get_logger
from src.data.label_mapping import map_iemocap_to_meld
from src.data.windowing import WindowConfig, build_window_manifest

LOGGER = get_logger(__name__)


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def normalize_row(row: pd.Series) -> str:
        dataset = str(row.get("dataset", "")).lower()
        raw = str(row.get("raw_label", "")).strip().lower()
        if dataset == "iemocap":
            return map_iemocap_to_meld(raw)
        return raw

    df["label_7"] = df.apply(normalize_row, axis=1)
    df["label_id"] = df["label_7"].map(CLASS_TO_ID)
    if df["label_id"].isna().any():
        bad = df[df["label_id"].isna()][["dataset", "raw_label"]].head(10)
        raise ValueError(f"Unknown labels after normalization. Sample:\n{bad}")
    df["label_id"] = df["label_id"].astype("int8")
    return df


def _add_group_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group_id"] = df.apply(
        lambda r: r["dialogue_id"] if str(r.get("dataset", "")).lower() == "meld" else r["session_id"],
        axis=1,
    )
    return df


def _validate_english(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "whisper_language" not in df.columns:
        df["whisper_language"] = "en"
    df["non_english_flag"] = df["whisper_language"].fillna("en").str.lower() != "en"
    return df


def build_manifests(input_table_path: str, out_dir: str, regime: str) -> tuple[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_table_path)

    defaults = {
        "speaker_id": "unknown",
        "utterance_index": 0,
        "asr_conf_mean": 1.0,
        "asr_conf_min": 1.0,
        "asr_low_conf_flag": False,
        "face_det_conf_mean": 1.0,
        "face_det_conf_min": 1.0,
        "fold_id": -1,
        "split": "train_or_val",
        "transcript_prev_1": "",
        "transcript_prev_2": "",
    }

    for key, value in defaults.items():
        if key not in df.columns:
            df[key] = value

    df = _normalize_labels(df)
    df = _add_group_id(df)
    df = _validate_english(df)
    df["transcript_source"] = regime

    utterance_path = str(out / "utterance_manifest.parquet")
    write_parquet(df, utterance_path)

    window_df = build_window_manifest(df, regime=regime, config=WindowConfig())
    window_path = str(out / "window_manifest.parquet")
    write_parquet(window_df, window_path)

    LOGGER.info("Saved utterance manifest: %s", utterance_path)
    LOGGER.info("Saved window manifest: %s", window_path)
    return utterance_path, window_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build utterance and window manifests.")
    parser.add_argument("--input-table", default="data/interim/normalized_utterances.parquet")
    parser.add_argument("--out-dir", default="data/processed/asr")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifests(input_table_path=args.input_table, out_dir=args.out_dir, regime=args.regime)


if __name__ == "__main__":
    main()
