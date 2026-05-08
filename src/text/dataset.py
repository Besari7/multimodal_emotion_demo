from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, checkpoint: str = "bert-base-uncased", max_length: int = 192) -> None:
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def __len__(self) -> int:
        return len(self.df)

    def _build_text(self, row: pd.Series) -> str:
        prev2 = str(row.get("text_ctx_prev_2", "") or "")
        prev1 = str(row.get("text_ctx_prev_1", "") or "")
        current = str(row.get("text_current", row.get("transcript_text", "")) or "")
        return " [SEP] ".join([prev2, prev1, current]).strip()

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = self._build_text(row)
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "window_id": row.get("window_id", f"row_{idx}"),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_id": int(row.get("label_id", -1)),
        }
