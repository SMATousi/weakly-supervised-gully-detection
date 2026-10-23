"""E6: label models on the real label matrix; binary LFs vs confidence-valued LFs. Scored on DEV only.

Binary:      each labeler -> hard vote in {-1, 0, 1}; models MV, Dawid-Skene, Snorkel, triplet.
Confidence:  labelers with P(yes) become abstain-band votes (band tuned on dev) and a soft logit aggregator
             (weights fit on dev). Writes results/E6/e6.json and pseudo-labels for the chosen configuration.
Class prior for the models comes from DEV labels only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gully.eval.metrics import binary_metrics
from gully.ws.confidence_lfs import SoftLogitAggregator, abstain_band, tune_band
from gully.ws.label_models import MODELS, class_balance_from


def load_matrix(labels_dir: str, names: list[str]):
    """Hard votes (rows loc_id, -1 abstain) and P(yes) (NaN where absent) per labeler."""
    hard, pyes = {}, {}
    for n in names:
        d = pd.read_json(Path(labels_dir) / f"{n}.jsonl", lines=True).drop_duplicates("loc_id", keep="last")
        d = d.set_index("loc_id")
        hard[n] = d["label"]
        pyes[n] = d["p_yes"] if "p_yes" in d and d["p_yes"].notna().any() else pd.Series(dtype=float)
    return pd.DataFrame(hard).fillna(-1).astype(int), pd.DataFrame(pyes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfs", nargs="+", required=True)
    ap.add_argument("--lfset", required=True)
    ap.add_argument("--labels", default="artifacts/labels")
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--out", default="results/E6")
    ap.add_argument("--ws-out", default="artifacts/ws")
    a = ap.parse_args()

    H, P = load_matrix(a.labels, a.lfs)
    idx = pd.read_parquet(a.index).set_index("loc_id")
    gt = pd.read_parquet(a.gt).set_index("loc_id")["y_main"].dropna()
    pool = [i for i in H.index if i.startswith("U-")]
    dev = [i for i in gt.index if idx.split.get(i) == "dev" and i in H.index]
    y_dev = gt.loc[dev].to_numpy().astype(int)
    cb = class_balance_from(y_dev)
    res = {"lfs": a.lfs, "n_pool": len(pool), "n_dev": len(dev), "class_balance_dev": cb, "binary": {}, "confidence": {}}

    for name in ("mv", "ds", "triplet", "snorkel"):
        lm = MODELS[name]().fit(H.loc[pool].to_numpy(), class_balance=cb, seed=12)
        p_dev = lm.predict_proba(H.loc[dev].to_numpy())[:, 1]
        p_pool = lm.predict_proba(H.loc[pool].to_numpy())[:, 1]
        res["binary"][name] = {**binary_metrics(y_dev, p_dev), "pool_pos_rate": float((p_pool >= 0.5).mean()),
                               "distinct_pool_posteriors": int(pd.Series(p_pool).round(6).nunique())}
        pd.DataFrame({"loc_id": pool, "p1": p_pool}).to_parquet(Path(a.ws_out) / f"pseudo_{name}_{a.lfset}.parquet")

    scored = [n for n in a.lfs if len(P[n].dropna()) > 0] if not P.empty else []
    if scored:
        Hc = H.copy()
        bands = {}
        for n in scored:
            (lo, hi), acc = tune_band(P.loc[dev, n].to_numpy(dtype=float), y_dev, a.min_coverage)
            Hc[n] = np.where(P[n].reindex(Hc.index).isna(), Hc[n], abstain_band(P[n].reindex(Hc.index).to_numpy(), lo, hi))
            bands[n] = {"lo": lo, "hi": hi, "dev_acc_on_covered": acc}
        res["confidence"]["bands"] = bands
        for name in ("mv", "ds", "triplet", "snorkel"):
            lm = MODELS[name]().fit(Hc.loc[pool].to_numpy(), class_balance=cb, seed=12)
            p_dev = lm.predict_proba(Hc.loc[dev].to_numpy())[:, 1]
            p_pool = lm.predict_proba(Hc.loc[pool].to_numpy())[:, 1]
            res["confidence"][f"band_{name}"] = {**binary_metrics(y_dev, p_dev),
                                                "pool_pos_rate": float((p_pool >= 0.5).mean()),
                                                "distinct_pool_posteriors": int(pd.Series(p_pool).round(6).nunique())}
            pd.DataFrame({"loc_id": pool, "p1": p_pool}).to_parquet(Path(a.ws_out) / f"pseudo_band_{name}_{a.lfset}.parquet")
        # soft logit aggregator: weights fit on dev (cross-fitted 5-fold on dev to avoid scoring on its own fit)
        Pm = P[scored].reindex(H.index).to_numpy(dtype=float)
        pos = {k: i for i, k in enumerate(H.index)}
        dv = np.array([pos[i] for i in dev])
        folds = np.arange(len(dv)) % 5
        oof = np.zeros(len(dv))
        for k in range(5):
            tr, te = folds != k, folds == k
            oof[te] = SoftLogitAggregator().fit(Pm[dv[tr]], y_dev[tr]).predict_proba(Pm[dv[te]])[:, 1]
        agg = SoftLogitAggregator().fit(Pm[dv], y_dev)
        pp = np.array([pos[i] for i in pool])
        p_pool = agg.predict_proba(Pm[pp])[:, 1]
        res["confidence"]["softlogit_cv_dev"] = {**binary_metrics(y_dev, oof), "pool_pos_rate": float((p_pool >= 0.5).mean()),
                                                "distinct_pool_posteriors": int(pd.Series(p_pool).round(6).nunique())}
        pd.DataFrame({"loc_id": pool, "p1": p_pool}).to_parquet(Path(a.ws_out) / f"pseudo_softlogit_{a.lfset}.parquet")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"e6_{a.lfset}.json").write_text(json.dumps(res, indent=2, default=float))
    for grp in ("binary", "confidence"):
        for k, v in res[grp].items():
            if isinstance(v, dict) and "acc" in v:
                print(f"{grp:10s} {k:16s} dev acc={v['acc']:.3f} f1={v['f1']:.3f} pr_auc={v['pr_auc']:.3f} "
                      f"pool_pos={v['pool_pos_rate']:.3f} distinct={v['distinct_pool_posteriors']}")


if __name__ == "__main__":
    main()
