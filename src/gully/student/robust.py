"""Noise-robust and student-in-the-loop training on cached features (experiment E13).

One training loop, six modes, so that they differ only in what the question asks:
  all_kl     plain noise-aware KL on every sample (reproduces the earlier students; reference)
  all_gce    robust loss on every sample (control: robust loss only)
  drop_kl    keep only the confident pseudo-labels, plain KL (control: dropping only)
  drop_gce   V1: keep only the confident pseudo-labels and train with a robust loss
  sil_late   V2: warm up on the confident samples; then the uncertain samples are added with TARGETS FROM THE
             STUDENT (an EMA copy of itself, sharpened) instead of the labeler's uncertain labels
  sil_early  V3: the same student targets for the uncertain samples, but from epoch 0

Confidence of a pseudo-label p1 is |p1 - 0.5|. The confident set is the `keep_frac` most confident samples of EACH
pseudo-class (so dropping cannot remove a class). No test labels are read; epoch selection uses dev PR-AUC.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from gully.eval.metrics import binary_metrics
from gully.seed import seed_everything
from gully.student.heads import HEADS
from gully.student.train import predict

MODES = ("all_kl", "all_gce", "drop_kl", "drop_gce", "sil_late", "sil_early")


def confident_mask(p1: np.ndarray, keep_frac: float) -> np.ndarray:
    """Class-balanced: the keep_frac most confident samples of each pseudo-class (ties at the cut are kept)."""
    p1 = np.asarray(p1, dtype=float)
    conf = np.abs(p1 - 0.5)
    mask = np.zeros(len(p1), dtype=bool)
    for cls in (p1 >= 0.5, p1 < 0.5):
        idx = np.where(cls)[0]
        if len(idx) == 0:
            continue
        cut = np.quantile(conf[idx], 1.0 - keep_frac)
        mask[idx[conf[idx] >= cut]] = True
    return mask


def per_sample_loss(logits: torch.Tensor, t1: torch.Tensor, kind: str, q: float = 0.7) -> torch.Tensor:
    """Loss of every sample against the soft target distribution [1 - t1, t1]."""
    t = torch.stack([1 - t1, t1], 1)
    if kind == "kl":
        return F.kl_div(F.log_softmax(logits, -1), t, reduction="none").sum(-1)
    if kind == "gce":  # generalized cross entropy (Zhang & Sabuncu 2018), expectation over the soft target
        p = F.softmax(logits, -1).clamp_min(1e-6)
        return (t * (1 - p**q) / q).sum(-1)
    raise ValueError(kind)


def sharpen(p1: torch.Tensor, temp: float) -> torch.Tensor:
    a, b = (1 - p1).clamp_min(1e-6) ** (1 / temp), p1.clamp_min(1e-6) ** (1 / temp)
    return b / (a + b)


def n_steps(mode: str, n: int, n_certain: int, epochs: int, warm_epochs: int, bs: int) -> int:
    """Exact number of optimizer steps, so the one-cycle schedule ends fully annealed in every mode."""
    steps = 0
    for ep in range(epochs):
        full = mode.startswith("all") or mode == "sil_early" or (mode == "sil_late" and ep >= warm_epochs)
        size = n if full else n_certain
        steps += int(np.ceil(size / bs))
    return steps


def train_robust(X_tr, V_tr, p1_tr, X_dev, V_dev, y_dev, head: str = "temporal", seed: int = 0, mode: str = "drop_gce",
                 epochs: int = 30, lr: float = 1e-3, wd: float = 1e-2, bs: int = 128, keep_frac: float = 0.5,
                 q: float = 0.7, warm_epochs: int = 10, ramp_epochs: int = 5, ema_decay: float = 0.995,
                 temp: float = 0.5, device: str = "cuda", select: str = "pr_auc", head_kwargs: dict | None = None):
    """Returns (best_model, history, dev_probs, best_epoch). history entries carry n_active / n_student_targets."""
    assert mode in MODES, mode
    assert head in ("mlp", "temporal"), "the EMA copy does not track batch-norm statistics (linear head)"
    seed_everything(seed)
    kind = "gce" if mode.endswith("gce") else "kl"
    model = HEADS[head](**(head_kwargs or {})).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(p1_tr)
    cert_np = confident_mask(p1_tr, keep_frac)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=n_steps(mode, n, int(cert_np.sum()), epochs, warm_epochs, bs) + 1)
    p1 = torch.as_tensor(np.asarray(p1_tr), dtype=torch.float32, device=device)
    certain = torch.as_tensor(cert_np, device=device)
    if mode != "all_kl" and mode != "all_gce" and cert_np.all():
        raise ValueError("every pseudo-label is 'confident' (hard 0/1 labels or ties): this mode needs soft labels")
    Xg = X_tr if torch.is_tensor(X_tr) else torch.as_tensor(np.asarray(X_tr)).to(device)
    Vg = torch.as_tensor(V_tr).to(device)
    rng = np.random.default_rng(seed)
    best, best_score, hist = None, -np.inf, []
    for ep in range(epochs):
        use_student = mode in ("sil_late", "sil_early") and (mode == "sil_early" or ep >= warm_epochs)
        start = 0 if mode == "sil_early" else warm_epochs
        w_unc = min(1.0, (ep - start + 1) / ramp_epochs) if use_student else 0.0
        if mode.startswith("all") or use_student:
            active = np.arange(n)
        else:
            active = np.where(certain.cpu().numpy())[0]
        model.train()
        order = active[rng.permutation(len(active))]
        tot, n_stu = 0.0, 0
        for i in range(0, len(active), bs):
            idx = torch.as_tensor(order[i:i + bs], device=device)
            if len(idx) < 2:
                continue
            x, v = Xg[idx], Vg[idx]
            t1, w = p1[idx], torch.ones(len(idx), device=device)
            if use_student:
                unc = ~certain[idx]
                if unc.any():
                    with torch.no_grad():
                        t1 = torch.where(unc, sharpen(F.softmax(ema(x, v), -1)[:, 1], temp), t1)
                    w = torch.where(unc, torch.full_like(w, w_unc), w)
                    n_stu += int(unc.sum())
            loss = (per_sample_loss(model(x, v), t1, kind, q) * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                    pe.mul_(ema_decay).add_(pm.detach(), alpha=1 - ema_decay)
            tot += loss.item() * len(idx)
        p_dev = predict(model, X_dev, V_dev, device)
        m = binary_metrics(y_dev, p_dev)
        hist.append({"epoch": ep, "train_loss": tot / max(len(active), 1), "n_active": len(active),
                     "n_student_targets": n_stu, "w_uncertain": w_unc,
                     **{k: m[k] for k in ("acc", "f1", "roc_auc", "pr_auc")}})
        if m[select] > best_score:
            best_score, best = m[select], (copy.deepcopy(model.state_dict()), ep, p_dev)
    model.load_state_dict(best[0])
    return model, hist, best[2], best[1]
