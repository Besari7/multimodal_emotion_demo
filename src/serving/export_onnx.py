from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.audio.model_wav2vec2 import Wav2Vec2EmotionModel
from src.fusion.model_late_mlp import LateFusionMLP
from src.text.model_bert_context import BertContextEmotionModel
from src.video.model_efficientnet_attnpool import VideoEmotionModel


def _export_audio(output_path: Path) -> None:
    model = Wav2Vec2EmotionModel().eval()
    dummy = torch.randn(1, 80000)
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["input_values"],
        output_names=["embedding", "logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "samples"}, "embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def _export_video(output_path: Path) -> None:
    model = VideoEmotionModel().eval()
    dummy = torch.randn(1, 16, 3, 224, 224)
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["frames"],
        output_names=["embedding", "logits"],
        dynamic_axes={"frames": {0: "batch", 1: "time"}, "embedding": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def _export_text(output_path: Path) -> None:
    model = BertContextEmotionModel().eval()
    input_ids = torch.ones(1, 192, dtype=torch.long)
    attn = torch.ones(1, 192, dtype=torch.long)
    torch.onnx.export(
        model,
        (input_ids, attn),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding", "logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


def _export_fusion(output_path: Path) -> None:
    model = LateFusionMLP().eval()
    dummy = torch.randn(1, 773)
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["fusion_input"],
        output_names=["logits"],
        dynamic_axes={"fusion_input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def export_onnx(component: str | None = None) -> None:
    root = Path("experiments/artifacts/onnx")
    root.mkdir(parents=True, exist_ok=True)

    if component in (None, "audio"):
        _export_audio(root / "audio_branch.onnx")
    if component in (None, "video"):
        _export_video(root / "video_branch.onnx")
    if component in (None, "text"):
        _export_text(root / "text_branch.onnx")
    if component in (None, "fusion"):
        _export_fusion(root / "fusion_mlp.onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export branch and fusion models to ONNX.")
    parser.add_argument("--component", choices=["audio", "video", "text", "fusion", "all"], default="all")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    component = None if args.all or args.component == "all" else args.component
    export_onnx(component)


if __name__ == "__main__":
    main()
