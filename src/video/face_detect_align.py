from __future__ import annotations

import argparse
from pathlib import Path

from facenet_pytorch import MTCNN
from PIL import Image
import torch

from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)


def detect_and_align_faces(input_dir: str, output_dir: str, image_size: int = 224) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mtcnn = MTCNN(image_size=image_size, margin=10, post_process=True, device="cuda" if torch.cuda.is_available() else "cpu")

    for image_path in sorted(in_dir.glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        face, prob = mtcnn(image, return_prob=True)
        if face is None:
            continue
        out_path = out_dir / image_path.name
        aligned = face.permute(1, 2, 0).cpu().numpy()
        Image.fromarray((aligned * 255).astype("uint8")).save(out_path)
        LOGGER.debug("Saved face crop %s conf=%.4f", out_path, float(prob))

    LOGGER.info("Face alignment done for %s", input_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and align faces from extracted frames.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detect_and_align_faces(args.input_dir, args.output_dir, image_size=args.image_size)


if __name__ == "__main__":
    main()
