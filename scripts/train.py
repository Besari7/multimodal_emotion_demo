from __future__ import annotations

import argparse

from multimodal_emotion.training.runtime import ensure_training_dependencies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal emotion model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    ensure_training_dependencies()
    from transformers import AutoTokenizer

    from multimodal_emotion.config import load_config, save_config_snapshot
    from multimodal_emotion.data.dataset import build_dataloader
    from multimodal_emotion.models import MultimodalEmotionModel
    from multimodal_emotion.training.engine import set_seed, train_model

    args = parse_args()
    config = load_config(args.config)
    set_seed(config.training.seed)
    save_config_snapshot(config, f"{args.output_dir}/config_snapshot.json")

    tokenizer = AutoTokenizer.from_pretrained(config.model.text_model_name)
    label_to_id = config.label_to_id()

    train_loader = build_dataloader(
        manifest_path=args.train_manifest,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        audio_dim=config.model.audio_feature_dim,
        video_dim=config.model.video_feature_dim,
        max_text_length=config.model.max_text_length,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = build_dataloader(
        manifest_path=args.val_manifest,
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
    summary = train_model(model, train_loader, val_loader, config, args.output_dir)
    print(summary)


if __name__ == "__main__":
    main()
