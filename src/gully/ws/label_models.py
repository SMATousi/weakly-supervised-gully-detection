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


class DawidSkene:
    """Binary Dawid-Skene (EM) over hard votes in {-1 abstain, 0, 1}; per-LF 2x2 confusion matrices."""

    name = "ds"

    def __init__(self, n_iter: int = 50, smooth: float = 1.0):
        self.n_iter, self.smooth = n_iter, smooth

    def fit(self, L: np.ndarray, class_balance=None, seed: int = 0):
        m = L.shape[1]
        prior = np.array(class_balance if class_balance is not None else [0.5, 0.5], dtype=float)
        # init posteriors from majority vote
        q = MajorityVote().predict_proba(L)[:, 1]
        for _ in range(self.n_iter):
            conf = np.zeros((m, 2, 2))  # conf[j, y, vote]
            for j in range(m):
                for v in (0, 1):
                    mask = L[:, j] == v
                    conf[j, 1, v] = (q[mask].sum() + self.smooth)
                    conf[j, 0, v] = ((1 - q[mask]).sum() + self.smooth)
                conf[j] /= conf[j].sum(1, keepdims=True)
            self.conf_, self.prior_ = conf, prior
            q = self._posterior(L)
            if class_balance is None:
                prior = np.array([1 - q.mean(), q.mean()])
        self.conf_, self.prior_ = conf, prior
        return self

    def _posterior(self, L: np.ndarray) -> np.ndarray:
        logp = np.tile(np.log(self.prior_), (len(L), 1))
        for j in range(L.shape[1]):
            for v in (0, 1):
                mask = L[:, j] == v
                logp[mask, 0] += np.log(self.conf_[j, 0, v])
                logp[mask, 1] += np.log(self.conf_[j, 1, v])
        logp -= logp.max(1, keepdims=True)
        p = np.exp(logp)
        return (p / p.sum(1, keepdims=True))[:, 1]

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        p1 = self._posterior(L)
        return np.stack([1 - p1, p1], 1)


class TripletLM:
    """Triplet-method label model (our reimplementation of the estimator at the core of FlyingSquid).

    Votes map to {-1, +1} (0 = abstain). With conditionally independent LFs, E[v_i v_j] = a_i a_j where
    a_i = E[v_i y] on non-abstained points, so |a_i| = sqrt(E[v_i v_j] E[v_i v_k] / E[v_j v_k]) over triplets
    (averaged), assuming LFs are better than random (a_i > 0). Posterior = prior + sum_j v_j * log-odds_j.
    The official FlyingSquid package is not used because it needs an outdated pgmpy API.
    """

    name = "triplet"

    def fit(self, L: np.ndarray, class_balance=None, seed: int = 0):
        V = np.where(L == 1, 1.0, np.where(L == 0, -1.0, 0.0))
        m = V.shape[1]
        fired = V != 0

        def moment(i, j):
            both = fired[:, i] & fired[:, j]
            return (V[both, i] * V[both, j]).mean() if both.any() else 0.0

        M = np.array([[moment(i, j) if i != j else np.nan for j in range(m)] for i in range(m)])
        acc = np.zeros(m)
        for i in range(m):
            ests = []
            for j in range(m):
                for k in range(j + 1, m):
                    if i in (j, k):
                        continue
                    num, den = M[i, j] * M[i, k], M[j, k]
                    if den > 1e-6 and num > 0:
                        ests.append(np.sqrt(num / den))
            acc[i] = np.clip(np.mean(ests) if ests else 0.1, 0.01, 0.99)
        self.a_ = acc
        self.w_ = np.log((1 + acc) / (1 - acc))
        pb = np.array(class_balance if class_balance is not None else [0.5, 0.5], dtype=float)
        self.b_ = float(np.log(pb[1] / pb[0]))
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        V = np.where(L == 1, 1.0, np.where(L == 0, -1.0, 0.0))
        z = self.b_ + V @ self.w_
        p1 = 1 / (1 + np.exp(-z))
        return np.stack([1 - p1, p1], 1)


MODELS = {"mv": MajorityVote, "snorkel": SnorkelLM, "ds": DawidSkene, "triplet": TripletLM}
