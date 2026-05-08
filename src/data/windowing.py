from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from src.common.constants import DEFAULT_WINDOW_OVERLAP, DEFAULT_WINDOW_SEC, DEFAULT_WINDOW_STRIDE_SEC


@dataclass
class WindowConfig:
    length_sec: float = DEFAULT_WINDOW_SEC
    overlap: float = DEFAULT_WINDOW_OVERLAP
    stride_sec: float = DEFAULT_WINDOW_STRIDE_SEC


@dataclass
class WindowRecord:
    window_index: int
    window_start_sec: float
    window_end_sec: float
    effective_duration_sec: float
    is_padded: bool
    pad_right_sec: float


def iter_windows(duration_sec: float, config: WindowConfig) -> Iterable[WindowRecord]:
    if duration_sec <= 0:
        return
    step = config.stride_sec if config.stride_sec > 0 else config.length_sec * (1.0 - config.overlap)
    step = max(step, 1e-6)

    start = 0.0
    index = 0
    while start < duration_sec:
        end = min(start + config.length_sec, duration_sec)
        effective = end - start
        is_padded = effective < config.length_sec
        pad = max(config.length_sec - effective, 0.0)
        yield WindowRecord(
            window_index=index,
            window_start_sec=float(round(start, 4)),
            window_end_sec=float(round(start + config.length_sec, 4)),
            effective_duration_sec=float(round(effective, 4)),
            is_padded=bool(is_padded),
            pad_right_sec=float(round(pad, 4)),
        )
        start += step
        index += 1


def build_window_manifest(utterance_df: pd.DataFrame, regime: str, config: WindowConfig | None = None) -> pd.DataFrame:
    config = config or WindowConfig()
    rows: List[dict] = []

    for _, row in utterance_df.iterrows():
        duration = float(row.get("duration_sec", 0.0))
        for win in iter_windows(duration, config):
            rows.append(
                {
                    "window_id": f"{row['record_id']}_w{win.window_index:04d}",
                    "record_id": row["record_id"],
                    "dataset": row["dataset"],
                    "regime": regime,
                    "window_index": win.window_index,
                    "window_start_sec": win.window_start_sec,
                    "window_end_sec": win.window_end_sec,
                    "effective_duration_sec": win.effective_duration_sec,
                    "is_padded": win.is_padded,
                    "pad_right_sec": win.pad_right_sec,
                    "audio_chunk_path": row.get("audio_path_16khz", ""),
                    "frame_paths": [],
                    "text_current": row.get("transcript_text", ""),
                    "text_ctx_prev_1": row.get("transcript_prev_1", ""),
                    "text_ctx_prev_2": row.get("transcript_prev_2", ""),
                    "asr_conf_mean": row.get("asr_conf_mean", 1.0),
                    "face_det_conf_mean": row.get("face_det_conf_mean", 1.0),
                    "missing_audio": False,
                    "missing_video": False,
                    "missing_text": row.get("transcript_text", "") == "",
                    "label_id": row.get("label_id", -1),
                    "label_7": row.get("label_7", "neutral"),
                    "group_id": row.get("group_id", ""),
                    "fold_id": row.get("fold_id", -1),
                    "split": row.get("split", "train_or_val"),
                }
            )

    return pd.DataFrame(rows)
