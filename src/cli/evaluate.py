from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.logging_utils import get_logger
from src.eval.run_ablation_suite import run_ablation_suite
from src.eval.run_mosei_eval import run_mosei_eval
from src.eval.whisper_hallucination_impact import run_whisper_hallucination_impact

LOGGER = get_logger(__name__)


def build_cv_summary(regime: str, output_path: str = "experiments/reports/cv_summary.csv") -> str:
    rows = []
    for fold in range(5):
        rows.append(
            {
                "regime": regime,
                "fold": fold,
                "macro_f1": None,
                "weighted_f1": None,
                "status": "pending_metric_ingestion",
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return str(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation suite templates and reports.")
    parser.add_argument("--suite", choices=["full", "quick"], default="full")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cv_path = build_cv_summary(args.regime)
    ablation_path = run_ablation_suite(args.regime)
    mosei_path = run_mosei_eval(f"data/processed/{args.regime}/window_manifest.parquet")

    if args.regime == "asr":
        whisper_path = run_whisper_hallucination_impact(f"data/processed/{args.regime}/window_manifest.parquet")
        LOGGER.info("Whisper report: %s", whisper_path)

    LOGGER.info("CV summary: %s", cv_path)
    LOGGER.info("Ablation summary: %s", ablation_path)
    LOGGER.info("MOSEI report: %s", mosei_path)


if __name__ == "__main__":
    main()
