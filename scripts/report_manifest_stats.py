from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report useful statistics for a JSONL manifest.")
    parser.add_argument("--manifest", required=True, help="Path to a JSONL manifest.")
    parser.add_argument("--output", default=None, help="Optional path to write stats as JSON.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skipping invalid JSON on line {line_number}: {exc}", file=sys.stderr)
                continue
            if not isinstance(value, dict):
                print(f"warning: skipping non-object JSON on line {line_number}", file=sys.stderr)
                continue
            rows.append(value)
    return rows


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _feature_missing(row: dict[str, Any], feature_key: str, metadata_key: str) -> bool:
    metadata = _metadata(row)
    return not bool(row.get(feature_key)) or bool(metadata.get(metadata_key))


def collect_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_ids = [str(row.get("sample_id", "")) for row in rows if row.get("sample_id")]
    sample_id_counts = Counter(sample_ids)
    duplicate_ids = sorted(sample_id for sample_id, count in sample_id_counts.items() if count > 1)

    stats = {
        "num_samples": len(rows),
        "label_counts": dict(sorted(Counter(str(row.get("label", "<missing>")) for row in rows).items())),
        "source_counts": dict(
            sorted(Counter(str(_metadata(row).get("source", "<missing>")) for row in rows).items())
        ),
        "split_counts": dict(
            sorted(Counter(str(_metadata(row).get("split", row.get("split", "<missing>"))) for row in rows).items())
        ),
        "empty_text_count": sum(1 for row in rows if not str(row.get("text", "")).strip()),
        "missing_audio_path": sum(
            1 for row in rows if not bool(row.get("audio_path")) or bool(_metadata(row).get("missing_audio"))
        ),
        "missing_video_path": sum(
            1 for row in rows if not bool(row.get("video_path")) or bool(_metadata(row).get("missing_video"))
        ),
        "missing_audio_features": sum(
            1 for row in rows if _feature_missing(row, "audio_feature_path", "missing_audio_features")
        ),
        "missing_video_features": sum(
            1 for row in rows if _feature_missing(row, "video_feature_path", "missing_video_features")
        ),
        "duplicate_sample_ids": duplicate_ids,
    }
    return stats


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    stats = collect_stats(_read_jsonl(manifest_path))
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote stats to {output_path}")


if __name__ == "__main__":
    main()
