from __future__ import annotations

import ast
from pathlib import Path
from typing import List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VideoWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_frames: int = 16) -> None:
        self.df = df.reset_index(drop=True)
        self.max_frames = max_frames
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def _parse_paths(self, frame_paths) -> List[str]:
        if isinstance(frame_paths, list):
            return [str(p) for p in frame_paths]
        if isinstance(frame_paths, str):
            frame_paths = frame_paths.strip()
            if frame_paths.startswith("["):
                try:
                    parsed = ast.literal_eval(frame_paths)
                    if isinstance(parsed, list):
                        return [str(p) for p in parsed]
                except (ValueError, SyntaxError):
                    return []
            if frame_paths:
                return [frame_paths]
        return []

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        paths = self._parse_paths(row.get("frame_paths", []))[: self.max_frames]

        frames = []
        for path in paths:
            if Path(path).exists():
                image = Image.open(path).convert("RGB")
                frames.append(self.transform(image))

        if not frames:
            frames = [torch.zeros(3, 224, 224)]

        while len(frames) < self.max_frames:
            frames.append(frames[-1].clone())

        tensor = torch.stack(frames[: self.max_frames], dim=0)

        return {
            "window_id": row.get("window_id", f"row_{idx}"),
            "frames": tensor,
            "label_id": int(row.get("label_id", -1)),
        }
