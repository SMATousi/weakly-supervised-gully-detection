"""E0: reproduce the old label-model step from the legacy labeler outputs, then remove the test leakage.

Variants (same 3 LFs and seed as the old notebook):
  legacy_leaky : class balance from ALL 484 expert labels (what `fit(L_train, Y_dev=test GT)` did, B4)
  dev_prior    : class balance from the dev split only
  no_prior     : Snorkel's default (uniform) class balance
Scoring is on dev only. The leaky variant is also compared with the saved old pseudo-labels, which
checks that the old step is reproduced without touching test labels for scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gully.eval.metrics import binary_metrics
from gully.ws.label_matrix import coverage, load_labels
from gully.ws.label_models import MajorityVote, SnorkelLM, class_balance_from

LFS = ["legacy_llama3.2-v-90b_mq", "legacy_llama3.2-v-90b_sq", "legacy_qwen2.5-vl-72b_sq"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="artifacts/labels")
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--old", default="/root/Gully_Detection/weak-supervision/large_dataset/results/"
                                      "train_lable_model_results.json")
    ap.add_argument("--out", default="results/E0")
    args = ap.parse_args()

    L = load_labels(args.labels, LFS)
    idx = pd.read_parquet(args.index)[["loc_id", "legacy_id", "split"]]
    gt = pd.read_parquet(args.gt)[["loc_id", "y_main"]].dropna()
    pool_ids = idx.loc[idx.loc_id.str[0] == "U", "loc_id"]
    split = idx.set_index("loc_id")["split"]
    dev_gt = gt[gt.loc_id.map(split) == "dev"]
    all_gt = gt[gt.loc_id.isin(idx.loc_id)]
    L_pool, L_dev = L.loc[pool_ids].to_numpy(), L.loc[dev_gt.loc_id].to_numpy()
    y_dev = dev_gt.y_main.to_numpy().astype(int)

    res = {"lfs": LFS, "coverage_pool": coverage(L.loc[pool_ids]).to_dict(),
           "coverage_dev": coverage(L.loc[dev_gt.loc_id]).to_dict(), "n_dev": len(y_dev), "variants": {}}
    with open(args.old) as f:
        old = json.load(f)
    old_p1 = np.array([old[str(i)]["proba"][1] for i in idx.loc[idx.loc_id.str[0] == "U", "legacy_id"]])

    variants = {
        "legacy_leaky": class_balance_from(all_gt.y_main.to_numpy().astype(int)),
        "dev_prior": class_balance_from(y_dev),
        "no_prior": None,
    }
    probs = {}
    for name, cb in variants.items():
        lm = SnorkelLM().fit(L_pool, class_balance=cb, seed=12)
        p_pool, p_dev = lm.predict_proba(L_pool)[:, 1], lm.predict_proba(L_dev)[:, 1]
        probs[name] = p_pool
        r = {"class_balance": cb, "dev": binary_metrics(y_dev, p_dev),
             "pool_pos_rate": float((p_pool >= 0.5).mean()),
             "distinct_pool_posteriors": len(np.unique(p_pool.round(6)))}
        if name == "legacy_leaky":
            r["agreement_with_saved_pseudolabels"] = float(((p_pool >= 0.5) == (old_p1 >= 0.5)).mean())
            r["max_abs_proba_diff_vs_saved"] = float(np.abs(p_pool - old_p1).max())
        res["variants"][name] = r
    res["max_abs_posterior_diff_vs_legacy_leaky"] = {
        k: float(np.abs(v - probs["legacy_leaky"]).max()) for k, v in probs.items() if k != "legacy_leaky"}
    mv = MajorityVote().fit(L_pool)
    res["variants"]["majority_vote"] = {"dev": binary_metrics(y_dev, mv.predict_proba(L_dev)[:, 1]),
                                        "pool_pos_rate": float((mv.predict_proba(L_pool)[:, 1] >= 0.5).mean())}
    for j, n in enumerate(LFS):  # single LFs on dev (abstain -> counted wrong)
        ok = L_dev[:, j] != -1
        res.setdefault("single_lf_dev", {})[n] = {
            "acc_on_covered": float((L_dev[ok, j] == y_dev[ok]).mean()), "coverage": float(ok.mean())}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "e0_label_model.json").write_text(json.dumps(res, indent=2))
    for k, v in res["variants"].items():
        d = v["dev"]
        print(f"{k:14s} dev acc={d['acc']:.3f} f1={d['f1']:.3f} pr_auc={d['pr_auc']:.3f} "
              f"pool_pos={v['pool_pos_rate']:.3f} " + (f"agree_saved={v.get('agreement_with_saved_pseudolabels')}"
                                                       if 'agreement_with_saved_pseudolabels' in v else ""))


if __name__ == "__main__":
    main()
