from __future__ import annotations

import argparse

from src.audio.infer import infer_audio_embeddings
from src.audio.train import train_audio_branch
from src.common.logging_utils import get_logger
from src.text.infer import infer_text_embeddings
from src.text.train import train_text_branch
from src.video.infer import infer_video_embeddings
from src.video.train import train_video_branch

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a unimodal branch and materialize features.")
    parser.add_argument("--branch", choices=["audio", "video", "text"], required=True)
    parser.add_argument("--regime", choices=["oracle", "asr"], required=True)
    parser.add_argument("--fold", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.branch == "audio":
        train_audio_branch(args.regime, args.fold)
        out = infer_audio_embeddings(args.regime, args.fold)
    elif args.branch == "video":
        train_video_branch(args.regime, args.fold)
        out = infer_video_embeddings(args.regime, args.fold)
    else:
        train_text_branch(args.regime, args.fold)
        out = infer_text_embeddings(args.regime, args.fold)

    LOGGER.info("Saved %s features: %s", args.branch, out)


if __name__ == "__main__":
    main()
