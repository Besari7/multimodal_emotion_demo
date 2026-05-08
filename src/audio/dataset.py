from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sample_rate: int = 16000, window_sec: float = 5.0) -> None:
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_sec)

    def __len__(self) -> int:
        return len(self.df)

    def _load_waveform(self, audio_path: str) -> torch.Tensor:
        if not audio_path or not Path(audio_path).exists():
            return torch.zeros(self.window_samples)

        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)

        if wav.numel() < self.window_samples:
            pad = self.window_samples - wav.numel()
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[: self.window_samples]

        return wav

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        waveform = self._load_waveform(str(row.get("audio_chunk_path", row.get("audio_path_16khz", ""))))
        label_id = int(row.get("label_id", -1))
        return {
            "window_id": row.get("window_id", f"row_{idx}"),
            "input_values": waveform,
            "label_id": label_id,
        }
