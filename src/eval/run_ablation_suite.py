from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


REQUIRED_ABLATIONS = [
    "A01_audio_only",
    "A02_video_only",
    "A03_text_only_oracle",
    "A04_text_only_asr",
    "A05_fusion_all_oracle",
    "A06_fusion_all_asr",
    "A07_drop_audio",
    "A08_drop_video",
    "A09_drop_text",
    "A10_audio_video",
    "A11_audio_text",
    "A12_video_text",
    "A13_oracle_vs_asr_gap",
    "A14_whisper_hallucination_impact",
    "A15_cmu_mosei_generalization",
]


def run_ablation_suite(regime: str, output_path: str = "experiments/reports/ablation_summary.csv") -> str:
    rows = []
    for exp_id in REQUIRED_ABLATIONS:
        rows.append(
            {
                "experiment_id": exp_id,
                "regime": regime,
                "status": "planned",
                "macro_f1": None,
                "notes": "Use src.cli.evaluate to execute and fill results.",
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    LOGGER.info("Saved ablation suite template: %s", out)
    return str(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ablation suite tracker CSV.")
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--output", default="experiments/reports/ablation_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_ablation_suite(args.regime, output_path=args.output)
    LOGGER.info(output)


if __name__ == "__main__":
    main()
