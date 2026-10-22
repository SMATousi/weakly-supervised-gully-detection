"""Metrics, bootstrap CIs, McNemar."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y: np.ndarray, p1: np.ndarray, thr: float = 0.5) -> dict:
    y, p1 = np.asarray(y).astype(int), np.asarray(p1, dtype=float)
    pred = (p1 >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    both = len(np.unique(y)) == 2
    return {
        "n": len(y), "acc": (tp + tn) / len(y), "precision": prec, "recall": rec,
        "f1": div(2 * prec * rec, prec + rec) if tp else 0.0, "npv": div(tn, tn + fn),
        "roc_auc": float(roc_auc_score(y, p1)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, p1)) if both else float("nan"),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def bootstrap_ci(y, p1, metric: str = "acc", n_boot: int = 2000, seed: int = 0, thr: float = 0.5):
    """95% percentile CI of `metric` over resampled locations."""
    y, p1 = np.asarray(y), np.asarray(p1)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(binary_metrics(y[i], p1[i], thr)[metric])
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def mcnemar(y, pred_a, pred_b) -> float:
    """Exact two-sided McNemar p-value on paired hard predictions."""
    a, b = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)
