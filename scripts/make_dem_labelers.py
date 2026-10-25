"""Fit the DEM labelers on the development split and write artifacts/labels/dem_<feature>.jsonl for every location.

Also writes results/E5/dem_labelers.json (per attribute: direction, dev AUC, band, covered accuracy, coverage, positive rates).
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from gully.labelers.dem import DEM_LF_FEATURES, apply_lf, build_records, crossfit_dev, fit_lf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/dem/dem_features.parquet")
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--out", default="artifacts/labels")
    ap.add_argument("--report", default="results/E5/dem_labelers.json")
    a = ap.parse_args()
    feats = pd.read_parquet(a.features)
    idx = pd.read_parquet(a.index).set_index("loc_id")
    gt = pd.read_parquet(a.gt).set_index("loc_id")["y_main"].dropna()
    dev = [i for i in gt.index if idx.split.get(i) == "dev"]
    f = feats.set_index("loc_id")
    ok_dev = [i for i in dev if i in f.index and bool(f.loc[i, "dem_ok"])]
    y = gt.loc[ok_dev].to_numpy().astype(int)
    pool = feats[feats.loc_id.str.startswith("U-") & feats.dem_ok]
    lab = feats[feats.loc_id.str.startswith("L-") & feats.dem_ok]
    report = {"n_dev": len(ok_dev), "features": {}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in DEM_LF_FEATURES:
        x_dev = f.loc[ok_dev, name].to_numpy(dtype=float)
        fit = fit_lf(x_dev, y)
        v_oof, p_oof = crossfit_dev(x_dev, y)  # honest development-split votes (5-fold)
        override = {loc: (int(v), float(p)) for loc, v, p in zip(ok_dev, v_oof, p_oof, strict=True)}
        recs = build_records(feats.assign(**{name: feats[name].where(feats.dem_ok)}), name, fit, override)
        (out / f"dem_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        v_pool, _ = apply_lf(fit, pool[name].to_numpy(dtype=float))
        v_lab, _ = apply_lf(fit, lab[name].to_numpy(dtype=float))
        v_dev = v_oof
        cov = v_dev != -1
        report["features"][name] = {
            "sign": fit["sign"], "dev_auc_in_sample": round(fit["dev_auc_oriented"], 3),
            "dev_auc_out_of_fold": round(float(roc_auc_score(y, p_oof)), 3), "band": [fit["lo"], fit["hi"]],
            "dev_acc_covered": round(float((v_dev[cov] == y[cov]).mean()), 3), "dev_coverage": round(float(cov.mean()), 3),
            "pool_coverage": round(float((v_pool != -1).mean()), 3),
            "pool_pos_rate": round(float((v_pool == 1).sum() / max((v_pool != -1).sum(), 1)), 3),
            "labeled_pos_rate": round(float((v_lab == 1).sum() / max((v_lab != -1).sum(), 1)), 3)}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2, default=float))
    t = pd.DataFrame(report["features"]).T[["sign", "dev_auc_in_sample", "dev_auc_out_of_fold", "dev_acc_covered", "dev_coverage", "pool_pos_rate", "labeled_pos_rate"]]
    print(t.sort_values("dev_auc_out_of_fold", ascending=False).to_string())


if __name__ == "__main__":
    main()
