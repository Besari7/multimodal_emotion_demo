from __future__ import annotations

import argparse
import json

from multimodal_emotion.data.manifest import load_manifest, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a JSONL multimodal manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_manifest(load_manifest(args.manifest), args.labels)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
