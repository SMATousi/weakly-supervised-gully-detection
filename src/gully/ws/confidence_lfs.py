"""Confidence-valued labeling functions (fixes audit B10: 3 binary LFs give only ~8 distinct posteriors).

A VLM labeler gives P(yes) per location. Two ways to use it, both with thresholds chosen on DEV only:
  * `abstain_band`: vote 1 above `hi`, 0 below `lo`, abstain in between (binary LF with abstention).
  * `soft_vote`: keep P(yes) as a continuous vote (logit), for the soft aggregator below.
`SoftLogitAggregator` combines continuous votes with per-LF weights fit on dev (logistic regression on the
dev labels), so it needs dev labels for the weights but never test labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def abstain_band(p_yes: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Votes in {-1, 0, 1}: 1 if p >= hi, 0 if p <= lo, else -1 (abstain). NaN -> abstain."""
    p = np.asarray(p_yes, dtype=float)
    v = np.full(p.shape, -1, dtype=int)
    v[p >= hi] = 1
    v[p <= lo] = 0
    v[np.isnan(p)] = -1
    return v


def tune_band(p_dev: np.ndarray, y_dev: np.ndarray, min_coverage: float = 0.5, grid: int = 21):
    """Pick (lo, hi) on dev maximizing accuracy-on-covered subject to coverage >= min_coverage."""
    qs = np.linspace(0.02, 0.98, grid)
    cand = np.unique(np.quantile(p_dev[~np.isnan(p_dev)], qs))
    best, best_acc = (float(np.median(cand)),) * 2, -1.0
    for lo in cand:
        for hi in cand[cand >= lo]:
            v = abstain_band(p_dev, lo, hi)
            cov = v != -1
            if cov.mean() < min_coverage or cov.sum() == 0:
                continue
            acc = float((v[cov] == y_dev[cov]).mean())
            if acc > best_acc + 1e-12:
                best, best_acc = (float(lo), float(hi)), acc
    return best, best_acc


class SoftLogitAggregator:
    """Weighted sum of LF logits; weights and bias fit on dev labels (L2-regularized logistic regression)."""

    name = "softlogit"

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, P_dev: np.ndarray, y_dev: np.ndarray):
        self.fill_ = np.nanmean(logit(np.where(np.isnan(P_dev), 0.5, P_dev)), 0) * 0.0  # abstain -> 0 logit
        self.lr = LogisticRegression(C=self.C).fit(self._x(P_dev), y_dev)
        return self

    @staticmethod
    def _x(P: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(P), 0.0, logit(np.where(np.isnan(P), 0.5, P)))

    def predict_proba(self, P: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(self._x(P))


def fold_ids(n: int, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Deterministic fold assignment 0..n_folds-1 for n development locations."""
    return np.random.default_rng(seed).permutation(n) % n_folds


def crossfit_band(p_dev: np.ndarray, y_dev: np.ndarray, min_coverage: float = 0.5, n_folds: int = 5,
                  seed: int = 0) -> np.ndarray:
    """Abstention-band votes for the development locations, each from a band tuned WITHOUT that location's fold.

    Tuning a band on the development labels and scoring the votes on the same labels is optimistic; with many
    tuned labelers it is badly so. This gives honest development-split votes.
    """
    p_dev, y_dev = np.asarray(p_dev, dtype=float), np.asarray(y_dev)
    folds = fold_ids(len(p_dev), n_folds, seed)
    votes = np.full(len(p_dev), -1, dtype=int)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        (lo, hi), _ = tune_band(p_dev[tr], y_dev[tr], min_coverage)
        votes[te] = abstain_band(p_dev[te], lo, hi)
    return votes
