"""Fit a label model on the pool label matrix and write pseudo-labels: artifacts/ws/pseudo_<model>_<lfset>.parquet.

The class balance comes from DEV labels only (or is left unset): never from test (audit B4).
"""
import argparse
from pathlib import Path

import pandas as pd

from gully.ws.label_matrix import load_labels
from gully.ws.label_models import MajorityVote, SnorkelLM, class_balance_from


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfs", nargs="+", required=True)
    ap.add_argument("--lfset", required=True, help="short name for the output file")
    ap.add_argument("--model", default="snorkel", choices=["snorkel", "mv"])
    ap.add_argument("--prior", default="dev", choices=["dev", "none"])
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--labels", default="artifacts/labels")
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--out", default="artifacts/ws")
    a = ap.parse_args()

    idx = pd.read_parquet(a.index).set_index("loc_id")
    gt = pd.read_parquet(a.gt).set_index("loc_id")["y_main"].dropna()
    L = load_labels(a.labels, a.lfs)
    pool = idx.index[idx.index.str[0] == "U"]
    cb = None
    if a.prior == "dev":
        cb = class_balance_from(gt[[i for i in gt.index if idx.split.get(i) == "dev"]].to_numpy().astype(int))
    lm = (MajorityVote() if a.model == "mv" else SnorkelLM()).fit(L.loc[pool].to_numpy(), class_balance=cb,
                                                                  seed=a.seed)
    p1 = lm.predict_proba(L.loc[pool].to_numpy())[:, 1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"loc_id": pool, "p1": p1, "y_hat": (p1 >= 0.5).astype(int)})
    df.to_parquet(out / f"pseudo_{a.model}_{a.lfset}.parquet")
    print(f"{a.model}/{a.lfset}: class_balance={cb} pool_pos_rate={df.y_hat.mean():.3f} "
          f"distinct_p1={df.p1.round(6).nunique()}")


if __name__ == "__main__":
    main()
