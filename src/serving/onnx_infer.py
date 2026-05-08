from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from src.common.constants import CLASS_ORDER
from src.common.io_utils import write_json


def _build_duration_weights(window_df: pd.DataFrame) -> np.ndarray:
    if window_df.empty:
        return np.array([], dtype=np.float32)

    centers = ((window_df["window_start_sec"].to_numpy() + window_df["window_end_sec"].to_numpy()) / 2.0).astype(np.float32)
    boundaries = np.zeros(len(centers) + 1, dtype=np.float32)
    boundaries[0] = float(window_df["window_start_sec"].iloc[0])
    boundaries[-1] = float(window_df["window_end_sec"].max())
    for i in range(1, len(centers)):
        boundaries[i] = 0.5 * (centers[i - 1] + centers[i])
    return np.maximum(boundaries[1:] - boundaries[:-1], 1e-6)


def run_fusion_onnx(window_manifest: str, fusion_features: str, model_path: str) -> tuple[dict, dict]:
    win_df = pd.read_parquet(window_manifest)
    feat_df = pd.read_parquet(fusion_features)

    merged = win_df.merge(feat_df, on="window_id", how="left")
    x = np.stack(merged["fusion_input"].to_list(), axis=0).astype(np.float32)

    sess = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    logits = sess.run(["logits"], {"fusion_input": x})[0]
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-8, None)
    pred_idx = probs.argmax(axis=1)

    timeline_items = []
    for i, row in merged.iterrows():
        timeline_items.append(
            {
                "window_index": int(row["window_index"]),
                "start_time_sec": float(row["window_start_sec"]),
                "end_time_sec": float(row["window_end_sec"]),
                "effective_duration_sec": float(row["effective_duration_sec"]),
                "is_padded": bool(row["is_padded"]),
                "pad_right_sec": float(row["pad_right_sec"]),
                "predicted_label": CLASS_ORDER[int(pred_idx[i])],
                "probability_vector": probs[i].astype(float).tolist(),
                "confidence": float(probs[i].max()),
                "missing_modality_flags": {
                    "audio": bool(row.get("missing_audio", False)),
                    "video": bool(row.get("missing_video", False)),
                    "text": bool(row.get("missing_text", False)),
                },
                "quality_signals": {
                    "asr_conf_mean": float(row.get("asr_conf_mean", 1.0)),
                    "face_det_conf_mean": float(row.get("face_det_conf_mean", 1.0)),
                },
                "fallback_strategy": "none",
            }
        )

    weights = _build_duration_weights(merged)
    total = float(weights.sum()) if len(weights) else 1.0

    distribution = []
    for class_idx, label in enumerate(CLASS_ORDER):
        hard_duration = float(weights[pred_idx == class_idx].sum()) if len(weights) else 0.0
        soft_duration = float((weights * probs[:, class_idx]).sum()) if len(weights) else 0.0
        distribution.append(
            {
                "label": label,
                "duration_sec": hard_duration,
                "percentage": 100.0 * hard_duration / total,
                "soft_percentage": 100.0 * soft_duration / total,
            }
        )

    distribution = sorted(distribution, key=lambda x: x["percentage"], reverse=True)

    timeline_json = {
        "video_id": "unknown",
        "model_version": "0.1.0",
        "transcript_regime": str(merged["regime"].iloc[0]) if not merged.empty else "asr",
        "window_size_sec": 5.0,
        "window_stride_sec": 2.5,
        "class_order": CLASS_ORDER,
        "timeline": timeline_items,
    }

    distribution_json = {
        "video_id": "unknown",
        "aggregation": "duration_weighted",
        "total_effective_duration_sec": total,
        "distribution": distribution,
        "dominant_label": distribution[0]["label"] if distribution else None,
    }

    return timeline_json, distribution_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX fusion inference and produce timeline/distribution JSON outputs.")
    parser.add_argument("--window-manifest", default="data/processed/asr/window_manifest.parquet")
    parser.add_argument("--fusion-features", default="data/features/fusion/asr_fold0.parquet")
    parser.add_argument("--fusion-onnx", default="experiments/artifacts/onnx/fusion_mlp.onnx")
    parser.add_argument("--timeline-out", default="experiments/reports/timeline_output.json")
    parser.add_argument("--distribution-out", default="experiments/reports/distribution_output.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeline, distribution = run_fusion_onnx(args.window_manifest, args.fusion_features, args.fusion_onnx)
    write_json(timeline, args.timeline_out)
    write_json(distribution, args.distribution_out)


if __name__ == "__main__":
    main()
