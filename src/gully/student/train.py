"""Train a head on frozen features with (soft) pseudo-labels; select the epoch on the expert-labeled dev split.

One optimizer step per iteration (audit B2). Test labels are never read here; evaluation of the
final configuration is scripts/evaluate.py (audit B3/B6).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from gully.eval.metrics import binary_metrics
from gully.seed import seed_everything
from gully.student.heads import HEADS


def load_features(feat_dir: str, device: str):
    tok = np.load(f"{feat_dir}/tokens.npy", mmap_mode="r")
    return tok, np.load(f"{feat_dir}/valid.npy"), np.load(f"{feat_dir}/ids.npy")


def _batches(n: int, bs: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


@torch.no_grad()
def predict(model, X, V, device: str, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.as_tensor(np.asarray(X[i:i + bs])).to(device)
        v = torch.as_tensor(V[i:i + bs]).to(device)
        out.append(F.softmax(model(x, v), -1)[:, 1].cpu())
    return torch.cat(out).numpy()


def train_student(X_tr, V_tr, p1_tr, X_dev, V_dev, y_dev, head: str = "temporal", seed: int = 0,
                  epochs: int = 30, lr: float = 1e-3, wd: float = 1e-2, bs: int = 128, hard: bool = False,
                  device: str = "cuda", select: str = "pr_auc", head_kwargs: dict | None = None):
    """p1_tr: pseudo-label P(y=1) per training row. Returns (best_model, history, dev_probs)."""
    seed_everything(seed)
    model = HEADS[head](**(head_kwargs or {})).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(X_tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    target = torch.as_tensor(p1_tr, dtype=torch.float32)
    if hard:
        target = (target >= 0.5).float()
    Xg = torch.as_tensor(np.asarray(X_tr)).to(device)
    Vg, Tg = torch.as_tensor(V_tr).to(device), target.to(device)
    rng = np.random.default_rng(seed)
    best, best_score, hist = None, -np.inf, []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for idx in _batches(len(Xg), bs, rng):
            if len(idx) < 2:  # BatchNorm needs >1
                continue
            idx_t = torch.as_tensor(idx, device=device)
            logp = F.log_softmax(model(Xg[idx_t], Vg[idx_t]), -1)
            t1 = Tg[idx_t]
            loss = F.kl_div(logp, torch.stack([1 - t1, t1], 1), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        p_dev = predict(model, X_dev, V_dev, device)
        m = binary_metrics(y_dev, p_dev)
        hist.append({"epoch": ep, "train_loss": tot / len(Xg), **{k: m[k] for k in ("acc", "f1", "roc_auc", "pr_auc")}})
        if m[select] > best_score:
            best_score, best = m[select], (copy.deepcopy(model.state_dict()), ep, p_dev)
    model.load_state_dict(best[0])
    return model, hist, best[2], best[1]
