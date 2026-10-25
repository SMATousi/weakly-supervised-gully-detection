"""DEM-based labelers: one labeling function per terrain attribute.

For each attribute (a curated, hydrology-motivated list fixed in advance, not picked by score) we use the DEVELOPMENT split only to
(1) choose the direction (the sign of AUC - 0.5), (2) tune an abstention band (tau_lo, tau_hi) with the same rule as the VLM
labelers (best accuracy on the covered locations subject to coverage >= 50%), and (3) fit the monotone map from the oriented
attribute to a score in [0, 1] (the dev empirical CDF), used as `p_yes`. No test labels are read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gully.ws.confidence_lfs import abstain_band, tune_band

DEM_LF_FEATURES = [
    "twi_max", "twi_p90", "spi_p99", "acc_max", "frac_acc_500", "frac_acc_100",
    "plan_curv_min", "prof_curv_min", "lap_p98", "frac_concave", "sink_depth_max",
    "tpi5_min", "tpi15_min", "tpi15_frac_hollow", "resid_p01", "resid_std",
    "slope_p90", "relief", "tri_mean", "aspect_circ_var", "channel_plan_curv_min",
]


def fit_lf(x_dev: np.ndarray, y_dev: np.ndarray, min_coverage: float = 0.5) -> dict:
    """Fit direction, band and score map of one attribute on the development locations."""
    auc = float(roc_auc_score(y_dev, x_dev))
    sign = 1.0 if auc >= 0.5 else -1.0
    z = sign * x_dev
    (lo, hi), cov_acc = tune_band(z, y_dev, min_coverage)
    return {"sign": sign, "dev_auc_raw": auc, "dev_auc_oriented": max(auc, 1 - auc), "lo": lo, "hi": hi,
            "dev_acc_covered": cov_acc, "z_dev_sorted": np.sort(z).tolist()}


def apply_lf(fit: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(votes in {-1, 0, 1}, p_yes in [0, 1]) for attribute values x (NaN -> abstain, p = 0.5)."""
    z = fit["sign"] * np.asarray(x, dtype=float)
    votes = abstain_band(np.where(np.isnan(z), np.nan, z), fit["lo"], fit["hi"])
    zs = np.asarray(fit["z_dev_sorted"])
    p = np.interp(z, zs, np.linspace(0.0, 1.0, len(zs)))
    return votes, np.where(np.isnan(z), 0.5, p)


def crossfit_dev(x_dev: np.ndarray, y_dev: np.ndarray, n_folds: int = 5, seed: int = 0):
    """Out-of-fold (votes, p_yes) for the development locations: each from a labeler fitted without its own fold."""
    from gully.ws.confidence_lfs import fold_ids

    folds = fold_ids(len(x_dev), n_folds, seed)
    votes, p = np.full(len(x_dev), -1, dtype=int), np.full(len(x_dev), 0.5)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        v, pp = apply_lf(fit_lf(x_dev[tr], y_dev[tr]), x_dev[te])
        votes[te], p[te] = v, pp
    return votes, p


def build_records(df: pd.DataFrame, feature: str, fit: dict, override: dict | None = None) -> list[dict]:
    """Records for every row of df; `override` maps loc_id -> (vote, p_yes) (the out-of-fold values for dev rows)."""
    votes, p = apply_lf(fit, df[feature].to_numpy(dtype=float))
    if override:
        for i, loc in enumerate(df.loc_id):
            if loc in override:
                votes[i], p[i] = override[loc]
    return [{"loc_id": loc, "labeler": f"dem_{feature}", "model": "DEM", "revision": None, "prompt_id": "terrain",
             "image_mode": "dem", "raw": "", "label": int(v), "p_yes": float(pp), "abstain": bool(v == -1),
             "latency_s": None, "ts": None} for loc, v, pp in zip(df.loc_id, votes, p, strict=True)]
