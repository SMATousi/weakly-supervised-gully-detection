"""E13: noise-robust / student-in-the-loop students (modes in gully.student.robust) from one GPU-resident copy of the
cached features. Writes results/<exp>/<pseudo>__<mode>__<head>__s<seed>.json. Reads NO test labels; epoch selection and
every reported metric use the development split."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from gully.config import git_sha
from gully.student.robust import MODES, train_robust
from gully.student.train import load_features, to_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--ws-dir", default="artifacts/ws")
    ap.add_argument("--pseudo", required=True, help="name of pseudo_<name>.parquet (soft labels)")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    ap.add_argument("--heads", nargs="+", default=["mlp", "temporal"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--q", type=float, default=0.7)
    ap.add_argument("--warm-epochs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--exp", default="E13_robust_sil")
    ap.add_argument("--tag", default="", help="suffix for the run name (e.g. a keep-frac sweep)")
    ap.add_argument("--out", default="results")
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
    p1 = pd.read_parquet(Path(a.ws_dir) / f"pseudo_{a.pseudo}.parquet").set_index("loc_id")["p1"].loc[pool_ids].to_numpy()
    sha = git_sha()
    out = Path(a.out) / a.exp
    out.mkdir(parents=True, exist_ok=True)
    for mode in a.modes:
        for head in a.heads:
            for seed in a.seeds:
                run = f"{a.pseudo}__{mode}{a.tag}__{head}__s{seed}"
                if (out / f"{run}.json").exists():  # resumable
                    continue
                _, hist, _, best = train_robust(
                    X_pool, V_pool, p1, X_dev, V_dev, y_dev, head=head, seed=seed, mode=mode, epochs=a.epochs,
                    keep_frac=a.keep_frac, q=a.q, warm_epochs=a.warm_epochs, device=a.device)
                res = {"run": run, "pseudo": a.pseudo, "mode": mode, "head": head, "seed": seed, "git": sha,
                       "keep_frac": a.keep_frac, "q": a.q, "warm_epochs": a.warm_epochs, "epochs": a.epochs,
                       "n_train": len(pool_ids), "n_dev": len(dev_ids), "best_epoch": best, "dev": hist[best],
                       "history": hist}
                (out / f"{run}.json").write_text(json.dumps(res, indent=2, default=float))
                d = hist[best]
                print(f"{run}: ep{best} dev acc={d['acc']:.3f} f1={d['f1']:.3f} pr_auc={d['pr_auc']:.3f}", flush=True)
                torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
