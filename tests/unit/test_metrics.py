import numpy as np

from src.eval.metrics import compute_metrics


def test_metrics_output_shapes() -> None:
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6])
    y_prob = np.eye(7)

    report = compute_metrics(y_true, y_pred, y_prob)
    assert report.macro_f1 == 1.0
    assert len(report.per_class_precision) == 7
    assert len(report.confusion_counts) == 7
