from __future__ import annotations

from typing import Iterable, List

from src.common.constants import CLASS_TO_ID, IEMOCAP_TO_MELD


def map_iemocap_to_meld(label: str) -> str:
    key = label.strip().lower()
    if key not in IEMOCAP_TO_MELD:
        raise ValueError(f"Unsupported IEMOCAP label: {label}")
    return IEMOCAP_TO_MELD[key]


def to_label_id(label_7: str) -> int:
    key = label_7.strip().lower()
    if key not in CLASS_TO_ID:
        raise ValueError(f"Unknown MELD label: {label_7}")
    return CLASS_TO_ID[key]


def map_iemocap_series(labels: Iterable[str]) -> List[str]:
    return [map_iemocap_to_meld(label) for label in labels]
