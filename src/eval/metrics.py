from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_recall_fscore_support


LABEL_IDS = [0, 1, 2, 3, 4, 5, 6]


@dataclass
class EvalReport:
    macro_f1: float
    weighted_f1: float
    per_class_precision: List[float]
    per_class_recall: List[float]
    per_class_f1: List[float]
    confusion_counts: List[List[int]]
    confusion_row_normalized: List[List[float]]
    brier_macro_ovr: float


def macro_brier_ovr(y_true: np.ndarray, y_prob: np.ndarray, label_ids: List[int] | None = None) -> float:
    labels = label_ids or LABEL_IDS
    scores: List[float] = []
    for c in labels:
        target = (y_true == c).astype(int)
        scores.append(brier_score_loss(target, y_prob[:, c], pos_label=1))
    return float(np.mean(scores))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> EvalReport:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS, normalize=None)
    cm_norm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS, normalize="true")

    return EvalReport(
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted")),
        per_class_precision=p.astype(float).tolist(),
        per_class_recall=r.astype(float).tolist(),
        per_class_f1=f.astype(float).tolist(),
        confusion_counts=cm.astype(int).tolist(),
        confusion_row_normalized=np.nan_to_num(cm_norm).astype(float).tolist(),
        brier_macro_ovr=macro_brier_ovr(y_true, y_prob),
    )


def report_to_dict(report: EvalReport) -> Dict[str, object]:
    return {
        "macro_f1": report.macro_f1,
        "weighted_f1": report.weighted_f1,
        "per_class_precision": report.per_class_precision,
        "per_class_recall": report.per_class_recall,
        "per_class_f1": report.per_class_f1,
        "confusion_counts": report.confusion_counts,
        "confusion_row_normalized": report.confusion_row_normalized,
        "brier_macro_ovr": report.brier_macro_ovr,
    }
