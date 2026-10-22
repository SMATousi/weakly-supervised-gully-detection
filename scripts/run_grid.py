"""Train many students (pseudo-label set x head x seed) from one GPU-resident copy of the features.

Writes results/<exp>/<pseudo>__<head>__s<seed>.json (dev metrics, selected epoch, history) and the dev
probabilities / checkpoints under artifacts/students/. Reads NO test labels.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gully.student.train import load_features, to_device, train_student


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--ws-dir", default="artifacts/ws")
    ap.add_argument("--pseudo", nargs="+", required=True, help="names of pseudo_<name>.parquet; suffix :hard trains on hard labels")
    ap.add_argument("--heads", nargs="+", default=["linear", "mlp", "temporal"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exp", default="E_dev")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt-dir", default="artifacts/students")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    tok, valid, ids = load_features(a.features, a.device)
    pos = {k: i for i, k in enumerate(ids)}
    split = pd.read_parquet(a.index).set_index("loc_id")["split"]
    gt = pd.read_parquet(a.gt).set_index("loc_id")["y_main"].dropna()
    pool_ids = [i for i in ids if split.get(i) == "pool"]
    dev_ids = [i for i in gt.index if split.get(i) == "dev" and i in pos]
    y_dev = gt.loc[dev_ids].to_numpy().astype(int)
    X_pool = to_device(tok, [pos[i] for i in pool_ids], a.device)
    V_pool = valid[[pos[i] for i in pool_ids]]
    X_dev = to_device(tok, [pos[i] for i in dev_ids], a.device)
    V_dev = valid[[pos[i] for i in dev_ids]]
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True,
                         text=True).stdout.strip()
    out, ck = Path(a.out) / a.exp, Path(a.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)

    for spec in a.pseudo:
        name, _, mode = spec.partition(":")
        ps = pd.read_parquet(Path(a.ws_dir) / f"pseudo_{name}.parquet").set_index("loc_id")["p1"].loc[pool_ids]
        for head in a.heads:
            for seed in a.seeds:
                run = f"{name}{'-hard' if mode == 'hard' else ''}__{head}__s{seed}"
                model, hist, p_dev, best = train_student(
                    X_pool, V_pool, ps.to_numpy(), X_dev, V_dev, y_dev, head=head, seed=seed,
                    epochs=a.epochs, lr=a.lr, hard=(mode == "hard"), device=a.device)
                torch.save(model.state_dict(), ck / f"{a.exp}__{run}.pt")
                np.save(ck / f"{a.exp}__{run}__dev_p1.npy", p_dev)
                res = {"run": run, "pseudo": name, "mode": mode or "soft", "head": head, "seed": seed,
                       "git": sha, "n_train": len(pool_ids), "n_dev": len(dev_ids), "best_epoch": best,
                       "dev": hist[best], "history": hist, "epochs": a.epochs, "lr": a.lr}
                (out / f"{run}.json").write_text(json.dumps(res, indent=2))
                d = hist[best]
                print(f"{run}: ep{best} dev acc={d['acc']:.3f} f1={d['f1']:.3f} pr_auc={d['pr_auc']:.3f}",
                      flush=True)
                del model
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
