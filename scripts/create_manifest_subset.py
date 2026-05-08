from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic balanced subset from a JSONL manifest.")
    parser.add_argument("--input", required=True, help="Input JSONL manifest.")
    parser.add_argument("--output", required=True, help="Output JSONL subset manifest.")
    parser.add_argument(
        "--max-per-label",
        required=True,
        type=int,
        help="Maximum number of samples to keep for each label.",
    )
    parser.add_argument("--seed", default=42, type=int, help="Random seed for deterministic shuffling.")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(value)
    return rows


def create_subset(rows: list[dict[str, Any]], max_per_label: int, seed: int) -> list[dict[str, Any]]:
    if max_per_label < 1:
        raise ValueError("--max-per-label must be at least 1")

    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("label", "<missing>"))].append(row)

    subset: list[dict[str, Any]] = []
    for label in sorted(by_label):
        label_rows = by_label[label]
        rng.shuffle(label_rows)
        subset.extend(label_rows[:max_per_label])

    rng.shuffle(subset)
    return subset


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_path}")

    subset = create_subset(_read_jsonl(input_path), args.max_per_label, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in subset:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = dict(sorted(Counter(str(row.get("label", "<missing>")) for row in subset).items()))
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"wrote {len(subset)} samples to {output_path}")


if __name__ == "__main__":
    main()
