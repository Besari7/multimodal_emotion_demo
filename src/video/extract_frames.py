from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


def extract_frames(video_path: str, out_dir: str, fps: int = 3) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_pattern = str(Path(out_dir) / "frame_%06d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        out_pattern,
    ]
    subprocess.run(cmd, check=True)
    LOGGER.info("Extracted frames to %s", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames at fixed FPS.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_frames(args.video, args.out_dir, fps=args.fps)


if __name__ == "__main__":
    main()
