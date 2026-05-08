from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from jiwer import wer

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


BUCKETS = [0.0, 0.4, 0.6, 0.8, 1.01]
BUCKET_NAMES = ["very_low", "low", "medium", "high"]


def run_whisper_hallucination_impact(asr_manifest: str, output_path: str = "experiments/reports/whisper_hallucination_impact.csv") -> str:
    df = pd.read_parquet(asr_manifest)
    if "oracle_text" not in df.columns:
        df["oracle_text"] = ""

    df["asr_conf_bucket"] = pd.cut(df["asr_conf_mean"], bins=BUCKETS, labels=BUCKET_NAMES, right=False)

    rows = []
    for bucket, bucket_df in df.groupby("asr_conf_bucket", dropna=False):
        if len(bucket_df) == 0:
            continue
        refs = bucket_df["oracle_text"].fillna("").astype(str).tolist()
        hyps = bucket_df["transcript_text"].fillna("").astype(str).tolist()
        bucket_wer = wer(refs, hyps) if any(refs) else None

        rows.append(
            {
                "bucket": str(bucket),
                "num_samples": int(len(bucket_df)),
                "mean_asr_conf": float(bucket_df["asr_conf_mean"].mean()),
                "wer_vs_oracle": bucket_wer,
                "macro_f1": None,
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    LOGGER.info("Saved hallucination impact report: %s", out)
    return str(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Whisper hallucination impact analysis.")
    parser.add_argument("--asr-manifest", default="data/processed/asr/window_manifest.parquet")
    parser.add_argument("--output", default="experiments/reports/whisper_hallucination_impact.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_whisper_hallucination_impact(args.asr_manifest, output_path=args.output)
    LOGGER.info(output)


if __name__ == "__main__":
    main()
