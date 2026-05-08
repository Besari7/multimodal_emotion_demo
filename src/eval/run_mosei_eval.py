from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


def run_mosei_eval(input_manifest: str, output_path: str = "experiments/reports/mosei_generalization.csv") -> str:
    df = pd.read_parquet(input_manifest)
    mosei_df = df[df["dataset"].str.lower() == "cmu_mosei"].copy()

    report = pd.DataFrame(
        [
            {
                "dataset": "CMU-MOSEI",
                "usage": "test_only",
                "num_samples": int(len(mosei_df)),
                "macro_f1": None,
                "weighted_f1": None,
                "status": "template",
            }
        ]
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    LOGGER.info("Saved MOSEI generalization template: %s", out)
    return str(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CMU-MOSEI generalization report template.")
    parser.add_argument("--input-manifest", default="data/processed/asr/window_manifest.parquet")
    parser.add_argument("--output", default="experiments/reports/mosei_generalization.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_mosei_eval(args.input_manifest, output_path=args.output)
    LOGGER.info(output)


if __name__ == "__main__":
    main()
