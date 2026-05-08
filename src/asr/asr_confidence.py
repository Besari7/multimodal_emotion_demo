from __future__ import annotations

from typing import Iterable

import numpy as np


def confidence_from_token_logprobs(token_logprobs: Iterable[float]) -> tuple[float, float]:
    values = np.array(list(token_logprobs), dtype=np.float32)
    if values.size == 0:
        return 0.0, 0.0
    probs = np.exp(np.clip(values, -20.0, 0.0))
    return float(probs.mean()), float(probs.min())


def low_confidence_flag(conf_mean: float, threshold: float = 0.6) -> bool:
    return conf_mean < threshold
