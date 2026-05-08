from __future__ import annotations

from typing import Dict, List

CLASS_ORDER: List[str] = [
    "neutral",
    "surprise",
    "fear",
    "sadness",
    "joy",
    "disgust",
    "anger",
]

CLASS_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(CLASS_ORDER)}
ID_TO_CLASS: Dict[int, str] = {idx: label for label, idx in CLASS_TO_ID.items()}

IEMOCAP_TO_MELD: Dict[str, str] = {
    "happy": "joy",
    "excited": "joy",
    "sad": "sadness",
    "angry": "anger",
    "frustrated": "anger",
    "neutral": "neutral",
    "disgust": "disgust",
    "fear": "fear",
    "surprised": "surprise",
}

DEFAULT_WINDOW_SEC: float = 5.0
DEFAULT_WINDOW_OVERLAP: float = 0.5
DEFAULT_WINDOW_STRIDE_SEC: float = 2.5
