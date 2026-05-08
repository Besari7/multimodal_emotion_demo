from __future__ import annotations

import argparse
import json
from pathlib import Path

from jsonschema import validate

TIMELINE_SCHEMA = {
    "type": "object",
    "required": [
        "video_id",
        "model_version",
        "transcript_regime",
        "window_size_sec",
        "window_stride_sec",
        "class_order",
        "timeline",
    ],
    "properties": {
        "video_id": {"type": "string"},
        "model_version": {"type": "string"},
        "transcript_regime": {"type": "string", "enum": ["oracle", "asr"]},
        "window_size_sec": {"type": "number"},
        "window_stride_sec": {"type": "number"},
        "class_order": {"type": "array", "items": {"type": "string"}, "minItems": 7, "maxItems": 7},
        "timeline": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "window_index",
                    "start_time_sec",
                    "end_time_sec",
                    "effective_duration_sec",
                    "is_padded",
                    "pad_right_sec",
                    "predicted_label",
                    "probability_vector",
                    "confidence",
                    "missing_modality_flags",
                ],
            },
        },
    },
}

DISTRIBUTION_SCHEMA = {
    "type": "object",
    "required": ["video_id", "aggregation", "total_effective_duration_sec", "distribution", "dominant_label"],
    "properties": {
        "video_id": {"type": "string"},
        "aggregation": {"type": "string", "enum": ["duration_weighted"]},
        "total_effective_duration_sec": {"type": "number"},
        "distribution": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label", "duration_sec", "percentage", "soft_percentage"],
            },
        },
        "dominant_label": {"type": ["string", "null"]},
    },
}


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate timeline and distribution output schemas.")
    parser.add_argument("--timeline", default="experiments/reports/timeline_output.json")
    parser.add_argument("--distribution", default="experiments/reports/distribution_output.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline = load_json(args.timeline)
    distribution = load_json(args.distribution)
    validate(instance=timeline, schema=TIMELINE_SCHEMA)
    validate(instance=distribution, schema=DISTRIBUTION_SCHEMA)
    print("Schema validation passed")


if __name__ == "__main__":
    main()
