from __future__ import annotations

import argparse

from src.common.logging_utils import get_logger
from src.serving.onnx_infer import run_fusion_onnx
from src.serving.schema_validate import DISTRIBUTION_SCHEMA, TIMELINE_SCHEMA
from src.common.io_utils import write_json
from jsonschema import validate

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference with ONNX fusion model.")
    parser.add_argument("--window-manifest", default="data/processed/asr/window_manifest.parquet")
    parser.add_argument("--fusion-features", default="data/features/fusion/asr_fold0.parquet")
    parser.add_argument("--fusion-onnx", default="experiments/artifacts/onnx/fusion_mlp.onnx")
    parser.add_argument("--timeline-out", default="experiments/reports/timeline_output.json")
    parser.add_argument("--distribution-out", default="experiments/reports/distribution_output.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline, distribution = run_fusion_onnx(args.window_manifest, args.fusion_features, args.fusion_onnx)
    validate(instance=timeline, schema=TIMELINE_SCHEMA)
    validate(instance=distribution, schema=DISTRIBUTION_SCHEMA)
    write_json(timeline, args.timeline_out)
    write_json(distribution, args.distribution_out)
    LOGGER.info("Saved timeline JSON: %s", args.timeline_out)
    LOGGER.info("Saved distribution JSON: %s", args.distribution_out)


if __name__ == "__main__":
    main()
