import numpy as np
import pytest
import torch

from gully.student.robust import (
    MODES,
    confident_mask,
    n_steps,
    per_sample_loss,
    sharpen,
    train_robust,
)


def test_confident_mask_is_class_balanced():
    r = np.random.default_rng(0)
    p1 = np.concatenate([r.uniform(0.0, 0.2, 900), r.uniform(0.8, 1.0, 100)])  # 90% negatives
    m = confident_mask(p1, 0.5)
    assert abs(m[:900].mean() - 0.5) < 0.05 and abs(m[900:].mean() - 0.5) < 0.1  # both classes keep ~half
    assert m[:900].sum() > 0 and m[900:].sum() > 0
    # the kept ones are the most confident of their class
    assert np.abs(p1[:900][m[:900]] - 0.5).min() >= np.abs(p1[:900][~m[:900]] - 0.5).max() - 1e-9


def test_losses_and_sharpen():
    logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    t1 = torch.tensor([0.0, 1.0])
    for kind in ("kl", "gce"):
        good = per_sample_loss(logits, t1, kind)
        bad = per_sample_loss(logits, 1 - t1, kind)
        assert (good < bad).all()
    # robust: the GCE loss of a confidently wrong prediction is bounded (1/q), the cross entropy is not
    assert per_sample_loss(torch.tensor([[20.0, -20.0]]), torch.tensor([1.0]), "gce").item() < 1 / 0.7 + 1e-6
    assert per_sample_loss(torch.tensor([[20.0, -20.0]]), torch.tensor([1.0]), "kl").item() > 30
    assert abs(sharpen(torch.tensor([0.5]), 0.5).item() - 0.5) < 1e-6 and sharpen(torch.tensor([0.7]), 0.5).item() > 0.7


def test_step_counts():
    assert n_steps("all_kl", 1000, 500, 10, 4, 100) == 100
    assert n_steps("drop_gce", 1000, 500, 10, 4, 100) == 50
    assert n_steps("sil_late", 1000, 500, 10, 4, 100) == 4 * 5 + 6 * 10
    assert n_steps("sil_early", 1000, 500, 10, 4, 100) == 100


def _toy(n=600, d=16, seed=0):
    r = np.random.default_rng(seed)
    y = r.integers(0, 2, n)
    X = r.normal(size=(n, 8, 65, d)).astype(np.float16)
    X[:, :, 0, 0] += (2 * y - 1)[:, None] * 3
    flip = r.random(n) < 0.2  # 20% label noise on the pseudo-labels
    base = np.where(flip, 1 - y, y).astype(float)
    p1 = 0.5 + (base - 0.5) * r.uniform(0.1, 1.0, n)  # soft labels with varying confidence
    return X, np.ones((n, 8), dtype=bool), p1, y


@pytest.mark.parametrize("mode", MODES)
def test_modes_run_and_use_the_right_samples(mode):
    X, V, p1, y = _toy()
    _, hist, p_dev, _ep = train_robust(X[:450], V[:450], p1[:450], X[450:], V[450:], y[450:], head="mlp", mode=mode,
                                      epochs=6, warm_epochs=3, ramp_epochs=2, bs=64, device="cpu",
                                      head_kwargs={"dim": 16, "hidden": 32})
    assert len(hist) == 6 and np.isfinite(p_dev).all()
    n_cert = int(confident_mask(p1[:450], 0.5).sum())
    if mode.startswith("drop"):
        assert all(h["n_active"] == n_cert and h["n_student_targets"] == 0 for h in hist)
    if mode.startswith("all"):
        assert all(h["n_active"] == 450 and h["n_student_targets"] == 0 for h in hist)
    if mode == "sil_late":  # certain only during warm-up, student targets afterwards
        assert all(h["n_active"] == n_cert and h["n_student_targets"] == 0 for h in hist[:3])
        assert all(h["n_active"] == 450 and h["n_student_targets"] > 0 for h in hist[3:])
    if mode == "sil_early":
        assert all(h["n_active"] == 450 and h["n_student_targets"] > 0 for h in hist)


def test_hard_labels_are_rejected_for_confidence_modes():
    X, V, p1, y = _toy()
    hard = (p1 >= 0.5).astype(float)
    with pytest.raises(ValueError):
        train_robust(X[:450], V[:450], hard[:450], X[450:], V[450:], y[450:], head="mlp", mode="drop_gce", epochs=2,
                     device="cpu", head_kwargs={"dim": 16, "hidden": 32})
