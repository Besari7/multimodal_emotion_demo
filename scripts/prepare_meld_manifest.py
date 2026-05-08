from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


TARGET_LABELS = {"neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"}
REQUIRED_COLUMNS = {
    "dialogue_id": ("Dialogue_ID", "dialogue_id"),
    "utterance_id": ("Utterance_ID", "utterance_id"),
    "speaker": ("Speaker", "speaker"),
    "utterance": ("Utterance", "utterance"),
    "emotion": ("Emotion", "emotion"),
}
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a MELD-only JSONL manifest from a MELD *_sent_emo.csv file."
    )
    parser.add_argument("--csv", required=True, help="Path to a MELD CSV file, e.g. train_sent_emo.csv.")
    parser.add_argument(
        "--video-root",
        required=True,
        help="Directory containing MELD video clips named dia{Dialogue_ID}_utt{Utterance_ID}.mp4.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--split", required=True, help="Split name to write into metadata, e.g. train/val/test.")
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Optional directory containing audio clips matching dia{Dialogue_ID}_utt{Utterance_ID}.*.",
    )
    return parser.parse_args()


def _resolve_columns(fieldnames: list[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("CSV has no header row.")

    lower_to_original = {name.lower(): name for name in fieldnames}
    resolved: dict[str, str] = {}
    missing: list[str] = []

    for canonical, aliases in REQUIRED_COLUMNS.items():
        for alias in aliases:
            if alias in fieldnames:
                resolved[canonical] = alias
                break
            if alias.lower() in lower_to_original:
                resolved[canonical] = lower_to_original[alias.lower()]
                break
        else:
            missing.append("/".join(aliases))

    if missing:
        raise ValueError(f"Missing required MELD CSV columns: {', '.join(missing)}")
    return resolved


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _find_audio_path(audio_root: Path | None, stem: str) -> tuple[str, bool]:
    if audio_root is None:
        return "", False

    for extension in AUDIO_EXTENSIONS:
        candidate = audio_root / f"{stem}{extension}"
        if candidate.exists():
            return candidate.as_posix(), False
    return "", True


def build_manifest(csv_path: Path, video_root: Path, output_path: Path, split: str, audio_root: Path | None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not video_root.exists():
        print(f"warning: video root does not exist: {video_root}", file=sys.stderr)
    if audio_root is not None and not audio_root.exists():
        print(f"warning: audio root does not exist: {audio_root}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    missing_video_count = 0
    missing_audio_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        columns = _resolve_columns(reader.fieldnames)

        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row_number, row in enumerate(reader, start=2):
                dialogue_id = _clean(row.get(columns["dialogue_id"]))
                utterance_id = _clean(row.get(columns["utterance_id"]))
                speaker = _clean(row.get(columns["speaker"]))
                text = _clean(row.get(columns["utterance"]))
                original_label = _clean(row.get(columns["emotion"]))
                mapped_label = original_label.lower()

                if mapped_label not in TARGET_LABELS:
                    skipped += 1
                    print(
                        f"warning: skipping row {row_number} with unsupported label {original_label!r}",
                        file=sys.stderr,
                    )
                    continue

                clip_stem = f"dia{dialogue_id}_utt{utterance_id}"
                video_path = video_root / f"{clip_stem}.mp4"
                missing_video = not video_path.exists()
                if missing_video:
                    missing_video_count += 1

                audio_path, missing_audio = _find_audio_path(audio_root, clip_stem)
                if missing_audio:
                    missing_audio_count += 1

                metadata: dict[str, Any] = {
                    "source": "MELD",
                    "split": split,
                    "dialogue_id": dialogue_id,
                    "utterance_id": utterance_id,
                    "original_label": original_label,
                    "mapped_label": mapped_label,
                    "mapping_confidence": "high",
                    "modality": "text_audio_video" if audio_path else "text_video",
                }
                if missing_video:
                    metadata["missing_video"] = True
                if missing_audio:
                    metadata["missing_audio"] = True

                record = {
                    "sample_id": f"meld_{split}_dia{dialogue_id}_utt{utterance_id}",
                    "label": mapped_label,
                    "text": text,
                    "group_id": f"dia{dialogue_id}",
                    "speaker_id": speaker,
                    "audio_path": audio_path,
                    "video_path": video_path.as_posix(),
                    "metadata": metadata,
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

    print(f"wrote {written} samples to {output_path}")
    if skipped:
        print(f"skipped {skipped} rows with labels outside the MELD target set")
    if missing_video_count:
        print(f"warning: {missing_video_count} samples are missing video files", file=sys.stderr)
    if audio_root is not None and missing_audio_count:
        print(f"warning: {missing_audio_count} samples are missing audio files", file=sys.stderr)


def main() -> None:
    args = parse_args()
    build_manifest(
        csv_path=Path(args.csv),
        video_root=Path(args.video_root),
        output_path=Path(args.output),
        split=args.split,
        audio_root=Path(args.audio_root) if args.audio_root else None,
    )


if __name__ == "__main__":
    main()
