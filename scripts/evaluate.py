from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_emotion.training.runtime import ensure_training_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained multimodal emotion model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    ensure_training_dependencies()
    import torch
    from transformers import AutoTokenizer

    from multimodal_emotion.config import load_config
    from multimodal_emotion.data.dataset import build_dataloader
    from multimodal_emotion.models import MultimodalEmotionModel
    from multimodal_emotion.training.engine import evaluate_model, resolve_device

    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(config.training.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    label_to_id = config.label_to_id()
    dataloader = build_dataloader(
        manifest_path=args.manifest,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        audio_dim=config.model.audio_feature_dim,
        video_dim=config.model.video_feature_dim,
        max_text_length=config.model.max_text_length,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    model = MultimodalEmotionModel(config.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics, predictions = evaluate_model(model, dataloader, device, config.model.labels)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_root / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
