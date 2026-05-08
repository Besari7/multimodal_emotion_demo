from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract cached audio embeddings for the multimodal project.")
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--model-name", default="facebook/wav2vec2-base")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    from multimodal_emotion.training.runtime import ensure_training_dependencies

    ensure_training_dependencies("librosa")
    import librosa
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    from multimodal_emotion.data.manifest import ManifestSample, load_manifest, write_manifest
    from multimodal_emotion.training.engine import resolve_device

    args = parse_args()
    device = resolve_device(args.device)
    extractor = AutoFeatureExtractor.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    samples = load_manifest(args.input_manifest)

    feature_dir = Path(args.feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    updated_samples: list[ManifestSample] = []
    with torch.no_grad():
        for sample in samples:
            if not sample.audio_path:
                updated_samples.append(sample)
                continue

            waveform, _ = librosa.load(sample.audio_path, sr=args.sample_rate, mono=True)
            inputs = extractor(
                waveform,
                sampling_rate=args.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            outputs = model(
                input_values=inputs["input_values"].to(device),
                attention_mask=inputs.get("attention_mask", None).to(device)
                if inputs.get("attention_mask", None) is not None
                else None,
            )
            hidden = outputs.last_hidden_state
            embedding = hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            output_path = feature_dir / f"{sample.sample_id}.npy"
            np.save(output_path, embedding)
            sample.audio_features_path = str(output_path)
            updated_samples.append(sample)

    write_manifest(updated_samples, args.output_manifest)


if __name__ == "__main__":
    main()
