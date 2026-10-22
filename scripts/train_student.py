"""Train one student (head x pseudo-label set x seed) on cached features; select the epoch on dev.

Never reads test labels. Writes results/<exp>/<run>.json (+ dev probabilities and the checkpoint under
artifacts/students/) so scripts/evaluate.py can score the final configuration once on test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gully.config import git_sha
from gully.student.train import load_features, train_student


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="dir with tokens.npy / valid.npy / ids.npy")
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--pseudo", required=True, help="parquet with loc_id, p1 (pool pseudo-labels)")
    ap.add_argument("--head", default="temporal", choices=["linear", "mlp", "temporal"])
    ap.add_argument("--hard", action="store_true", help="train on hard labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exp", default="E_dev")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt-dir", default="artifacts/students")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok, valid, ids = load_features(a.features, a.device)
    pos = {k: i for i, k in enumerate(ids)}
    split = pd.read_parquet(a.index).set_index("loc_id")["split"]
    gt = pd.read_parquet(a.gt).set_index("loc_id")["y_main"].dropna()
    ps = pd.read_parquet(a.pseudo).set_index("loc_id")["p1"]

    tr_ids = [i for i in ps.index if split.get(i) == "pool" and i in pos]
    dev_ids = [i for i in gt.index if split.get(i) == "dev" and i in pos]
    tr_rows, dev_rows = [pos[i] for i in tr_ids], [pos[i] for i in dev_ids]
    y_dev = gt.loc[dev_ids].to_numpy().astype(int)
    model, hist, p_dev, best_ep = train_student(
        tok[tr_rows], valid[tr_rows], ps.loc[tr_ids].to_numpy(), tok[dev_rows], valid[dev_rows], y_dev,
        head=a.head, seed=a.seed, epochs=a.epochs, lr=a.lr, hard=a.hard, device=a.device)

    out = Path(a.out) / a.exp
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(a.ckpt_dir)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ck / f"{a.exp}__{a.run}.pt")
    np.save(ck / f"{a.exp}__{a.run}__dev_p1.npy", p_dev)
    sha = git_sha()
    res = {"run": a.run, "args": vars(a), "git": sha, "n_train": len(tr_ids), "n_dev": len(dev_ids),
           "best_epoch": best_ep, "dev": hist[best_ep], "history": hist}
    (out / f"{a.run}.json").write_text(json.dumps(res, indent=2))
    print(f"{a.run}: best epoch {best_ep} dev {hist[best_ep]}")


if __name__ == "__main__":
    main()
