"""The ONLY script that reads test labels. Run once per final configuration.

Scores a saved student (or a labeler / label-model score file) on the test split with a fixed threshold
chosen on dev, bootstrap 95% CIs, and an optional paired McNemar test against a baseline. Single process,
exact test set, n asserted (fixes audit B5).

  --scores   parquet/csv with loc_id, p1 for the test locations (e.g. produced by predict_student.py)
  --baseline optional second score file for McNemar
  --thr      threshold selected on dev (never tuned here)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gully.eval.metrics import binary_metrics, bootstrap_ci, mcnemar

METRICS = ("acc", "precision", "recall", "f1", "npv", "roc_auc", "pr_auc")


def _read(p: str) -> pd.DataFrame:
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="results/final")
    ap.add_argument("--expected-n", type=int, default=334)
    a = ap.parse_args()

    split = pd.read_parquet(a.index).set_index("loc_id")["split"]
    gt = pd.read_parquet(a.gt).set_index("loc_id")["y_main"].dropna()
    test_ids = [i for i in gt.index if split.get(i) == "test"]
    assert len(test_ids) == a.expected_n, (len(test_ids), a.expected_n)

    def aligned(path: str):
        s = _read(path).set_index("loc_id")["p1"]
        missing = set(test_ids) - set(s.index)
        assert not missing, f"{path}: {len(missing)} test locations without a score"
        return gt.loc[test_ids].to_numpy().astype(int), s.loc[test_ids].to_numpy()

    y, p = aligned(a.scores)
    res = {"name": a.name, "threshold": a.thr, "n": len(y), "metrics": binary_metrics(y, p, a.thr)}
    res["ci95"] = {m: bootstrap_ci(y, p, m, thr=a.thr) for m in METRICS}
    if a.baseline:
        _, pb = aligned(a.baseline)
        res["mcnemar_p_vs_baseline"] = mcnemar(y, (p >= a.thr).astype(int), (pb >= a.thr).astype(int))
        res["baseline"] = a.baseline
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    m = res["metrics"]
    print(f"{a.name}: n={m['n']} acc={m['acc']:.3f} f1={m['f1']:.3f} pr_auc={m['pr_auc']:.3f}")
    assert np.isfinite(m["acc"])


if __name__ == "__main__":
    main()
