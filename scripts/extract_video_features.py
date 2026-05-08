from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cached video embeddings for the multimodal project.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def sample_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(video_path)
    frames: list[np.ndarray] = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        capture.release()
        return frames

    frame_indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()
        if success:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    capture.release()
    return frames


def main() -> None:
    from multimodal_emotion.training.runtime import ensure_training_dependencies

    ensure_training_dependencies("cv2", "PIL")
    import cv2
    import torch
    from PIL import Image
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    from multimodal_emotion.data.manifest import ManifestSample, load_manifest, write_manifest
    from multimodal_emotion.training.engine import resolve_device

    args = parse_args()
    device = resolve_device(args.device)

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights).to(device).eval()
    feature_extractor = torch.nn.Sequential(model.features, model.avgpool)
    preprocess = weights.transforms()

    samples = load_manifest(args.input_manifest)
    feature_dir = Path(args.feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    updated_samples: list[ManifestSample] = []
    with torch.no_grad():
        for sample in samples:
            if not sample.video_path:
                updated_samples.append(sample)
                continue

            frames = sample_frames(sample.video_path, args.num_frames)
            if not frames:
                updated_samples.append(sample)
                continue

            batch = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames]).to(device)
            features = feature_extractor(batch).flatten(1)
            embedding = features.mean(dim=0).cpu().numpy().astype(np.float32)
            output_path = feature_dir / f"{sample.sample_id}.npy"
            np.save(output_path, embedding)
            sample.video_features_path = str(output_path)
            updated_samples.append(sample)

    write_manifest(updated_samples, args.output_manifest)


if __name__ == "__main__":
    main()
