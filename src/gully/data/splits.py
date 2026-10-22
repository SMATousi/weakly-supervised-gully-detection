"""Spatially blocked dev/test split of the expert-labeled locations + pool exclusion buffer.

Labeled locations are grouped into square blocks (metres, projected CRS); each block goes whole to
dev or test. The assignment is a seeded random search for a dev set of the target size whose
positive rate matches the overall one. Pool locations within `buffer_m` of ANY labeled location
(including the ones without a defined label) become `pool_excluded`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def assign_blocks(df: pd.DataFrame, block_m: float, dev_target: int, seed: int, n_trials: int = 5000):
    """df: rows with x, y, y (0/1). Returns {block_id: 'dev'|'test'}."""
    blk = (np.floor(df["x"] / block_m).astype(int).astype(str) + "_"
           + np.floor(df["y"] / block_m).astype(int).astype(str))
    g = df.assign(block=blk).groupby("block")["label"].agg(n="size", pos="sum")
    p_all = df["label"].mean()
    rng = np.random.default_rng(seed)
    best, best_cost = None, np.inf
    ids = g.index.to_numpy()
    for _ in range(n_trials):
        order = rng.permutation(len(ids))
        n = pos = 0
        chosen = []
        for i in order:
            if n + g.n.iloc[i] > dev_target * 1.1:
                continue
            chosen.append(i)
            n += g.n.iloc[i]
            pos += g.pos.iloc[i]
            if n >= dev_target:
                break
        cost = abs(n - dev_target) / dev_target + 2 * abs(pos / max(n, 1) - p_all)
        if cost < best_cost:
            best, best_cost = chosen, cost
    dev_blocks = set(ids[best])
    return {b: ("dev" if b in dev_blocks else "test") for b in ids}, blk


def min_distance(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    """For each row of a, distance to the nearest row of b."""
    return cKDTree(b_xy).query(a_xy)[0]
