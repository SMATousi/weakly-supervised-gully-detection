"""E3 dev screening: score labeler JSONL outputs on DEV only (never test), plus pairwise error correlation.

For each labeler: coverage, positive rate, accuracy / F1 at the labeler's own hard answer, PR-AUC and
ROC-AUC of P(yes) (when available), and the dev-optimal P(yes) threshold. The error-correlation matrix
(1 = same mistakes) shows which labelers add *different* errors, which is what weak supervision needs.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gully.eval.metrics import binary_metrics


def load(path: str) -> pd.Series:
    df = pd.read_json(path, lines=True).drop_duplicates("loc_id", keep="last")
    return df.set_index("loc_id")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", required=True, help="glob(s) of *.jsonl")
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--out", default="results/E3/screening.json")
    a = ap.parse_args()

    split = pd.read_parquet(a.index).set_index("loc_id")["split"]
    gt = pd.read_parquet(a.gt).set_index("loc_id")["y_main"].dropna()
    dev = [i for i in gt.index if split.get(i) == "dev"]
    y = gt.loc[dev].to_numpy().astype(int)
    rows, errs = {}, {}
    for pat in a.labels:
        for f in sorted(glob.glob(pat)):
            d = load(f)
            name = d["labeler"].iloc[0]
            d = d.reindex(dev)
            lab = d["label"].fillna(-1).astype(int).to_numpy()
            cov = lab != -1
            hard = np.where(cov, lab, 0)  # abstain counted as "no"
            r = {"file": Path(f).name, "n_dev": len(dev), "coverage": float(cov.mean()),
                 "pos_rate": float((lab == 1).sum() / max(cov.sum(), 1)),
                 "acc_covered": float((lab[cov] == y[cov]).mean()) if cov.any() else float("nan")}
            m = binary_metrics(y, hard.astype(float))
            r.update({k: m[k] for k in ("acc", "f1", "precision", "recall")})
            if "p_yes" in d and d["p_yes"].notna().all():
                p = d["p_yes"].to_numpy(dtype=float)
                pm = binary_metrics(y, p)
                r.update({"pr_auc": pm["pr_auc"], "roc_auc": pm["roc_auc"]})
                ths = np.unique(p)
                accs = [((p >= t).astype(int) == y).mean() for t in ths]
                r["dev_best_thr"], r["acc_at_dev_best_thr"] = float(ths[int(np.argmax(accs))]), float(max(accs))
            rows[name] = r
            errs[name] = (hard != y).astype(float)
    E = pd.DataFrame(errs)
    corr = E.corr().round(2) if E.shape[1] > 1 else pd.DataFrame()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"labelers": rows, "error_corr": corr.to_dict()}, indent=2))
    print(pd.DataFrame(rows).T.round(3).drop(columns=["file", "n_dev"]).to_string())
    if not corr.empty:
        print("\nerror correlation\n", corr.to_string())


if __name__ == "__main__":
    main()
