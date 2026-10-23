import numpy as np

from gully.ws.confidence_lfs import SoftLogitAggregator, abstain_band, tune_band
from gully.ws.label_models import MODELS, DawidSkene, MajorityVote


def _synthetic(n=3000, accs=(0.85, 0.7, 0.6, 0.6), seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    L = np.stack([np.where(r.random(n) < a, y, 1 - y) for a in accs], 1)
    L[r.random(L.shape) < 0.1] = -1
    return y, L


def test_ds_beats_or_matches_mv_and_recovers_accuracies():
    y, L = _synthetic()
    mv = ((MajorityVote().predict_proba(L)[:, 1] >= 0.5) == y).mean()
    ds = DawidSkene().fit(L)
    acc = ((ds.predict_proba(L)[:, 1] >= 0.5) == y).mean()
    assert acc >= mv - 0.01
    est = [(ds.conf_[j, 0, 0] * ds.prior_[0] + ds.conf_[j, 1, 1] * ds.prior_[1]) for j in range(4)]
    assert est[0] > est[2]  # ranks the good LF above a weak one


def test_models_registry_runs():
    y, L = _synthetic()
    for name in ("mv", "ds", "triplet", "snorkel"):
        p = MODELS[name]().fit(L, class_balance=[0.5, 0.5]).predict_proba(L)
        assert p.shape == (len(L), 2) and np.isfinite(p).all()
        assert ((p[:, 1] >= 0.5) == y).mean() > 0.7, name


def test_abstain_band_and_tuning():
    r = np.random.default_rng(1)
    y = r.integers(0, 2, 1000)
    p = np.clip(0.5 + (y - 0.5) * 0.5 + r.normal(0, 0.25, 1000), 0, 1)
    (lo, hi), acc = tune_band(p, y, min_coverage=0.5)
    v = abstain_band(p, lo, hi)
    assert 0.5 <= (v != -1).mean() <= 1 and acc > 0.8 and lo <= hi
    assert abstain_band(np.array([np.nan]), 0.3, 0.7)[0] == -1


def test_soft_logit_uses_confidence():
    r = np.random.default_rng(2)
    y = r.integers(0, 2, 600)
    good = np.clip(0.5 + (y - 0.5) * 0.8 + r.normal(0, 0.15, 600), 0.01, 0.99)
    bad = r.random(600)
    P = np.stack([good, bad], 1)
    agg = SoftLogitAggregator().fit(P[:300], y[:300])
    assert ((agg.predict_proba(P[300:])[:, 1] >= 0.5) == y[300:]).mean() > 0.9
    assert len(np.unique(agg.predict_proba(P)[:, 1].round(6))) > 100  # many distinct posteriors (B10)


def test_triplet_recovers_accuracy_order():
    from gully.ws.label_models import TripletLM

    _, L = _synthetic(n=20000)
    lm = TripletLM().fit(L)
    true_a = 2 * np.array([0.85, 0.7, 0.6, 0.6]) - 1
    assert np.abs(lm.a_ - true_a).max() < 0.06
