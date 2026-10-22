import numpy as np
import pandas as pd

from gully.data.splits import assign_blocks, min_distance


def _toy(n=400, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"x": r.uniform(0, 20000, n), "y": r.uniform(0, 20000, n),
                         "label": r.integers(0, 2, n)})


def test_blocks_are_whole_and_deterministic():
    df = _toy()
    a, blk = assign_blocks(df, 1000, 120, seed=1, n_trials=200)
    b, _ = assign_blocks(df, 1000, 120, seed=1, n_trials=200)
    assert a == b
    sp = blk.map(a)
    assert not set(blk[sp == "dev"]) & set(blk[sp == "test"])
    assert abs((sp == "dev").sum() - 120) <= 40


def test_min_distance():
    d = min_distance(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0], [10.0, 0.0]]))
    assert d[0] == 5.0
