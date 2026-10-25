import numpy as np
import pandas as pd

from gully.labelers.dem import apply_lf, build_records, fit_lf


def _data(n=400, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    return y, (y - 0.5) * 2 + r.normal(0, 0.8, n)  # informative, positive direction


def test_fit_direction_band_and_scores():
    y, x = _data()
    fit = fit_lf(x, y)
    assert fit["sign"] == 1.0 and fit["dev_auc_oriented"] > 0.8
    fit_neg = fit_lf(-x, y)  # reversed attribute: the sign must flip
    assert fit_neg["sign"] == -1.0 and abs(fit_neg["dev_auc_oriented"] - fit["dev_auc_oriented"]) < 1e-9
    votes, p = apply_lf(fit, x)
    cov = votes != -1
    assert 0.45 <= cov.mean() <= 1.0 and (votes[cov] == y[cov]).mean() > 0.85
    assert ((p >= 0) & (p <= 1)).all() and np.corrcoef(p, x)[0, 1] > 0.9  # monotone score
    assert apply_lf(fit, np.array([np.nan]))[0][0] == -1


def test_records_format():
    y, x = _data(50)
    df = pd.DataFrame({"loc_id": [f"L-{i:06d}" for i in range(50)], "twi_max": x})
    recs = build_records(df, "twi_max", fit_lf(x, y))
    assert len(recs) == 50 and recs[0]["labeler"] == "dem_twi_max" and set(recs[0]) >= {"loc_id", "label", "p_yes", "abstain"}
