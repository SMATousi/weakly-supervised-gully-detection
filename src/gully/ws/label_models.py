"""Label models behind one interface: fit(L, class_balance) -> predict_proba(L) -> (n, 2) array.

`class_balance` is P(y=1) as a length-2 vector [P(0), P(1)] or None. It must come from dev labels
or be unspecified, never from test labels (audit B4).
"""
from __future__ import annotations

import numpy as np


class MajorityVote:
    name = "mv"

    def fit(self, L: np.ndarray, class_balance=None, seed: int = 0):
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        pos, neg = (L == 1).sum(1), (L == 0).sum(1)
        p1 = np.where(pos + neg == 0, 0.5, pos / np.maximum(pos + neg, 1))
        return np.stack([1 - p1, p1], 1)


class SnorkelLM:
    name = "snorkel"

    def __init__(self, n_epochs: int = 100, lr: float = 0.01):
        self.n_epochs, self.lr = n_epochs, lr

    def fit(self, L: np.ndarray, class_balance=None, seed: int = 12):
        from snorkel.labeling.model import LabelModel

        self.lm = LabelModel(cardinality=2, verbose=False)
        kw = {} if class_balance is None else {"class_balance": list(class_balance)}
        self.lm.fit(L, n_epochs=self.n_epochs, lr=self.lr, seed=seed, log_freq=10**6, **kw)
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        return self.lm.predict_proba(L)


def class_balance_from(y: np.ndarray) -> list[float]:
    """[P(0), P(1)] from binary labels."""
    p1 = float(np.mean(y))
    return [1 - p1, p1]
