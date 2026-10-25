"""LaTeX tables and number macros for the paper. Every number comes from artifacts/ and results/ (no hand-typing).

All model-quality numbers here are on the DEV split (n = 150) until the final configuration is frozen and
scripts/evaluate.py has been run once on test; test numbers never appear in these tables.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from gully.data.gt import ANNOTATORS, SCHEMES, kappa_for_scheme
from gully.eval.metrics import binary_metrics

LABELERS = [  # (file stem, display name, image input)
    ("legacy_llama3.2-v-90b_sq", "Llama-3.2-Vision-90B, single question", "collage"),
    ("legacy_llama3.2-v-90b_mq", "Llama-3.2-Vision-90B, multi-question", "collage"),
    ("legacy_qwen2.5-vl-72b_sq", "Qwen2.5-VL-72B, single question", "collage"),
    ("qwen3-vl-8b_sq_multiimg__sq_v4", "Qwen3-VL-8B", "8 images"),
    ("qwen3-vl-30b-a3b-fp8_sq_multiimg__sq_v4", "Qwen3-VL-30B-A3B (FP8)", "8 images"),
    ("qwen3-vl-32b-fp8_sq_multiimg__sq_v4", "Qwen3-VL-32B (FP8)", "8 images"),
    ("qwen3-vl-32b-fp8_sq_collage__sq_v4", "Qwen3-VL-32B (FP8)", "collage"),
]
LABEL_SETS = {  # E7 pseudo-label names -> display
    "single_q32hard": "32B labeler alone, hard",
    "single_q32soft": "32B labeler alone, $P(\\mathrm{yes})$",
    "snorkel_legacy3_q32": "Snorkel, binary labelers",
    "band_snorkel_legacy3_q32": "Snorkel, confidence-valued 32B",
    "ds_legacy3_q32": "Dawid--Skene, binary labelers",
    "band_ds_legacy3_q32": "Dawid--Skene, confidence-valued 32B",
    "mv_legacy3_q32-hard": "Majority vote, hard",
}
HEADS = {"mlp": "MLP", "temporal": "Temporal attention"}


def f(x: float, d: int = 3) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _table(caption: str, label: str, colspec: str, header: str, rows: list[str], star: bool = False) -> str:
    env = "table*" if star else "table"
    return "\n".join([f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
                      f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule",
                      *rows, "\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""])


def _dev(root: Path):
    idx = pd.read_parquet(root / "artifacts/index/locations.parquet").set_index("loc_id")
    gt = pd.read_parquet(root / "artifacts/gt/gt.parquet").set_index("loc_id")["y_main"].dropna()
    dev = [i for i in gt.index if idx.split.get(i) == "dev"]
    return idx, gt, dev


def gt_table(root: Path) -> str:
    gt = pd.read_parquet(root / "artifacts/gt/gt.parquet")
    ref = pd.read_csv(root / "data/reece_initial_labels.csv")
    m = gt.merge(ref, on="loc_id")
    names = [("strict_pos", "Strict pos."), ("lenient_pos", "Lenient pos."),
             ("lenient_neg", "Lenient neg."), ("strict_neg", "Strict neg. (main)")]
    rows = []
    for key, name in names:
        d = m.dropna(subset=[f"y_{key}"])
        y, p = d[f"y_{key}"].astype(int), d["reece_label"]
        rows.append(" & ".join([name, f"{int(y.sum())} / {int((1 - y).sum())}", f(kappa_for_scheme(gt, key), 2),
                                f(accuracy_score(y, p)), f(precision_score(y, p, average="macro")),
                                f(recall_score(y, p, average="macro")), f(f1_score(y, p, average="macro"))]) + " \\\\")
    cap = ("Expert ground-truth schemes: positive/negative counts among locations with a defined label, Fleiss' "
           "$\\kappa$ over the three annotators (locations where all three give a vote), and agreement with the "
           "initial labels of the polygon database~\\cite{reece2023using}. Precision, recall and F1 are "
           "macro-averaged over the two classes (so F1 is not the harmonic mean of the P and R shown); "
           "locations without a defined label are excluded from the agreement columns.")
    return _table(cap, "tab:gt_metrics", "lcccccc",
                  "Scheme & Pos / Neg & $\\kappa$ & Acc & P & R & F1 \\\\", rows)


def _labeler_metrics(root: Path, stem: str) -> dict:
    _, gt, dev = _dev(root)
    y = gt.loc[dev].to_numpy().astype(int)
    fp = root / f"artifacts/labels/{stem}.jsonl"
    if not fp.exists():
        fp = root / f"artifacts/labels/{stem}.labeled.jsonl"
    d = pd.read_json(fp, lines=True).drop_duplicates("loc_id", keep="last").set_index("loc_id").reindex(dev)
    lab = d["label"].fillna(-1).astype(int).to_numpy()
    m = binary_metrics(y, (lab == 1).astype(float))
    pm = binary_metrics(y, d["p_yes"].to_numpy(dtype=float)) if d["p_yes"].notna().all() else {}
    return {**m, "pos": float((lab == 1).mean()), "roc_auc": pm.get("roc_auc"), "pr_auc": pm.get("pr_auc")}


def _pool_pos_rate(root: Path, stem: str) -> float:
    d = pd.read_json(root / f"artifacts/labels/{stem}.jsonl", lines=True)
    d = d[d.loc_id.str.startswith("U-")]
    return float((d.label == 1).mean())


def _student_means(root: Path) -> pd.DataFrame:
    runs = []
    for fp in glob.glob(str(root / "results/E7_new_labelers/*.json")):
        d = json.loads(Path(fp).read_text())
        runs.append({"labels": d["pseudo"] + ("-hard" if d["mode"] == "hard" else ""), "head": d["head"],
                     **{k: d["dev"][k] for k in ("acc", "f1", "roc_auc", "pr_auc")}})
    return pd.DataFrame(runs).groupby(["labels", "head"]).mean()


def labeler_table(root: Path) -> str:
    _, gt, dev = _dev(root)
    y = gt.loc[dev].to_numpy().astype(int)
    rows = []
    for stem, name, inp in LABELERS:
        fp = root / f"artifacts/labels/{stem}.jsonl"
        if not fp.exists():  # labeled-set-only labelers have no merged file
            fp = root / f"artifacts/labels/{stem}.labeled.jsonl"
        d = pd.read_json(fp, lines=True).drop_duplicates("loc_id", keep="last")
        d = d.set_index("loc_id").reindex(dev)
        lab = d["label"].fillna(-1).astype(int).to_numpy()
        cov = float((lab != -1).mean())
        hard = (lab == 1).astype(float)  # abstain counted as "no"
        m = binary_metrics(y, hard)
        has_p = "p_yes" in d and d["p_yes"].notna().all()
        pm = binary_metrics(y, d["p_yes"].to_numpy(dtype=float)) if has_p else {"roc_auc": None, "pr_auc": None}
        rows.append(" & ".join([name, inp, f(float((lab == 1).mean()), 2), f(cov, 2), f(m["acc"]), f(m["f1"]),
                                f(m["precision"]), f(m["recall"]), f(pm["roc_auc"]), f(pm["pr_auc"])]) + " \\\\")
    cap = ("Individual open-weight labelers on the development split ($n=150$). \\emph{Pos.}: fraction of locations "
           "answered ``yes''; metrics use the labeler's own answer (abstentions count as ``no''); ROC-AUC and PR-AUC "
           "use $P(\\mathrm{yes})$ when the model provides it. The first three rows use the collage input and the "
           "prompt of the earlier zero-shot study; the Qwen3-VL rows use prompt \\texttt{sq\\_v4}.")
    return _table(cap, "tab:labelers", "llccccccrr",
                  "Labeler & Input & Pos. & Cov. & Acc & F1 & P & R & ROC & PR \\\\", rows, star=True)


def labelmodel_table(root: Path) -> str:
    e6 = json.loads((root / "results/E6/e6_legacy3_q32.json").read_text())
    e0 = json.loads((root / "results/E0/e0_label_model.json").read_text())
    rows = []

    def row(name, m):
        rows.append(" & ".join([name, f(m["acc"]), f(m["f1"]), f(m["pr_auc"]),
                                str(m.get("distinct_pool_posteriors", "--"))]) + " \\\\")

    def block(title):
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{title}}}}} \\\\")
    block("3 earlier labelers, binary")
    for key, name in (("majority_vote", "Majority vote"), ("dev_prior", "Snorkel")):
        v = e0["variants"][key]
        row(name, {**v["dev"], "distinct_pool_posteriors": v.get("distinct_pool_posteriors", "--")})
    block("+ Qwen3-VL-32B, binary")
    for key, name in (("mv", "Majority vote"), ("ds", "Dawid--Skene"), ("triplet", "Triplet method"),
                      ("snorkel", "Snorkel")):
        row(name, e6["binary"][key])
    block("+ Qwen3-VL-32B, confidence-valued")
    for key, name in (("mv", "Majority vote"), ("ds", "Dawid--Skene"), ("triplet", "Triplet method"),
                      ("snorkel", "Snorkel")):
        row(name, e6["confidence"][f"band_{key}"])
    cap = ("Label models on the development split ($n=150$). The first block uses the three earlier labelers; the "
           "other blocks add the Qwen3-VL-32B labeler, whose $P(\\mathrm{yes})$ is a binary vote (second block) or a "
           "vote with an abstention band tuned on the development split (third block). The class prior of every "
           "model comes from the development labels. \\emph{Post.}: distinct pseudo-label values over the pool.")
    return _table(cap, "tab:labelmodels", "lcccc", "Label model & Acc & F1 & PR-AUC & Post. \\\\", rows)


def student_table(root: Path) -> str:
    runs = []
    for fp in glob.glob(str(root / "results/E7_new_labelers/*.json")):
        d = json.loads(Path(fp).read_text())
        runs.append({"labels": d["pseudo"] + ("-hard" if d["mode"] == "hard" else ""), "head": d["head"],
                     **{k: d["dev"][k] for k in ("acc", "f1", "roc_auc", "pr_auc")}})
    df = pd.DataFrame(runs)
    rows = []
    for key, name in LABEL_SETS.items():
        for head, hname in HEADS.items():
            s = df[(df.labels == key) & (df["head"] == head)]
            if s.empty:
                continue
            ms = {c: (s[c].mean(), s[c].std(ddof=1)) for c in ("acc", "f1", "roc_auc", "pr_auc")}
            rows.append(" & ".join([name, hname, f"{ms['acc'][0]:.3f}$\\pm${ms['acc'][1]:.3f}",
                                    f"{ms['f1'][0]:.3f}$\\pm${ms['f1'][1]:.3f}", f(ms["roc_auc"][0]),
                                    f"{ms['pr_auc'][0]:.3f}$\\pm${ms['pr_auc'][1]:.3f}"]) + " \\\\")
    n = int(df.groupby(["labels", "head"]).size().iloc[0])
    cap = (f"Student models on the development split ($n=150$; mean $\\pm$ standard deviation over {n} seeds). "
           "The student is a head on frozen FlexiViT-S features of the eight images; rows differ in the "
           "pseudo-labels it is trained on. Threshold 0.5; the epoch is chosen on the development split.")
    return _table(cap, "tab:students", "llcccc", "Training labels & Head & Acc & F1 & ROC & PR-AUC \\\\", rows,
                  star=True)


E13_MODES = {
    "all_kl": "All samples, KL (reference)",
    "all_gce": "All samples, robust loss",
    "drop_kl": "Confident only, KL",
    "drop_gce": "V1: confident only, robust loss",
    "sil_late": "V2: student labels the uncertain, after warm-up",
    "sil_early": "V3: student labels the uncertain, from the start",
}
E13_SOURCES = {"single_q32soft": "32B labeler $P(\\mathrm{yes})$", "band_snorkel_legacy3_q32": "Snorkel, confidence-valued 32B"}


def robust_means(root: Path) -> pd.DataFrame:
    runs = []
    for fp in glob.glob(str(root / "results/E13_robust_sil/*.json")):
        d = json.loads(Path(fp).read_text())
        if d.get("keep_frac") != 0.5 or d["run"].count("__") != 3:  # main grid only (sweeps carry a tag)
            continue
        runs.append({"pseudo": d["pseudo"], "mode": d["mode"], "head": d["head"],
                     **{k: d["dev"][k] for k in ("acc", "f1", "roc_auc", "pr_auc")}})
    if not runs:
        return pd.DataFrame()
    g = pd.DataFrame(runs).groupby(["pseudo", "mode", "head"])
    return g.agg(["mean", "std"])


def robust_table(root: Path) -> str:
    m = robust_means(root)
    if m.empty:
        return "\\TODO{robust and student-in-the-loop results pending}\n"
    rows = []
    for src, sname in E13_SOURCES.items():
        if src not in m.index.get_level_values(0):
            continue
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{Pseudo-labels: {sname}}}}} \\\\")
        for mode, mname in E13_MODES.items():
            if not any((src, mode, h) in m.index for h in ("mlp", "temporal")):
                continue  # controls that were not run
            cells = []
            for head in ("mlp", "temporal"):
                if (src, mode, head) in m.index:
                    r = m.loc[(src, mode, head)]
                    cells += [f"{r[('acc', 'mean')]:.3f}$\\pm${r[('acc', 'std')]:.3f}",
                              f"{r[('pr_auc', 'mean')]:.3f}$\\pm${r[('pr_auc', 'std')]:.3f}"]
                else:
                    cells += ["--", "--"]
            rows.append(" & ".join([mname, *cells]) + " \\\\")
    cap = ("Noise-robust and student-in-the-loop training on the development split ($n=150$; mean $\\pm$ standard "
           "deviation over 3 seeds). ``Confident'' keeps the 50\\% most confident pseudo-labels of each class; the "
           "robust loss is generalized cross-entropy ($q=0.7$); in V2 and V3 the uncertain locations are labeled by "
           "an exponential moving average of the student itself, starting after 10 epochs (V2) or at epoch 0 (V3).")
    return _table(cap, "tab:robust", "lcccc", "Training & \\multicolumn{2}{c}{MLP} & \\multicolumn{2}{c}{Temporal} \\\\\n"
                  " & Acc & PR-AUC & Acc & PR-AUC \\\\", rows, star=True)


def numbers(root: Path) -> str:
    idx = pd.read_parquet(root / "artifacts/index/locations.parquet")
    gt = pd.read_parquet(root / "artifacts/gt/gt.parquet")
    e1 = json.loads((root / "results/E1/split_stats.json").read_text())
    sc = pd.read_parquet(root / "artifacts/gt/scores.parquet")
    main = gt["y_main"].dropna()
    vals = {
        "nPool": f"{int((idx.loc_id.str[0] == 'U').sum()):,}".replace(",", "{,}"),
        "nPoolImages": f"{int((idx.loc_id.str[0] == 'U').sum()) * 8:,}".replace(",", "{,}"),
        "nLabeledLoc": int((idx.loc_id.str[0] == "L").sum()), "nScored": int(sc.legacy_id.nunique()),
        "nDefined": len(main), "nPos": int(main.sum()), "nNeg": int((1 - main).sum()),
        "nDev": int(e1["counts"]["dev"]), "nTest": int(e1["counts"]["test"]),
        "nUndefined": int(e1["counts"]["labeled_null"]),
        "minPoolDistKm": f"{e1['labeled_to_nearest_pool_m']['0'] / 1000:.1f}",
        "medPoolDistKm": f"{e1['labeled_to_nearest_pool_m']['0.5'] / 1000:.1f}",
        "bufferM": int(e1["buffer_m"]), "blockKm": f"{e1['block_size_m'] / 1000:.0f}",
        "kappaMain": f(kappa_for_scheme(gt, "strict_neg"), 2),
        "devPosRate": f"{e1['pos_rate']['dev']:.2f}",
        "nAnnotators": len(ANNOTATORS), "nSchemes": len(SCHEMES),
    }
    lc = _labeler_metrics(root, "qwen3-vl-32b-fp8_sq_collage__sq_v4")
    lm = _labeler_metrics(root, "qwen3-vl-32b-fp8_sq_multiimg__sq_v4")
    l8 = _labeler_metrics(root, "qwen3-vl-8b_sq_multiimg__sq_v4")
    l30 = _labeler_metrics(root, "qwen3-vl-30b-a3b-fp8_sq_multiimg__sq_v4")
    lq = _labeler_metrics(root, "legacy_qwen2.5-vl-72b_sq")
    e6 = json.loads((root / "results/E6/e6_legacy3_q32.json").read_text())
    e0 = json.loads((root / "results/E0/e0_label_model.json").read_text())
    sm = _student_means(root)
    rob = robust_means(root)
    ws = sm[~sm.index.get_level_values(0).str.startswith("single")]
    vals.update({
        "lcAcc": f(lc["acc"]), "lcROC": f(lc["roc_auc"], 2), "lcPR": f(lc["pr_auc"], 2), "lcPos": f(lc["pos"], 2), "lcPrec": f(lc["precision"], 2), "lcRec": f(lc["recall"], 2),
        "lmAcc": f(lm["acc"]), "lmROC": f(lm["roc_auc"], 2), "lmPR": f(lm["pr_auc"], 2), "lmPos": f(lm["pos"], 2),
        "lsmallROC": f(l8["roc_auc"], 2), "lsmallPos": f(l8["pos"], 2), "lmoeROC": f(l30["roc_auc"], 2),
        "lmoePos": f(l30["pos"], 2), "lqAcc": f(lq["acc"]),
        "lmOldMVPR": f(e0["variants"]["majority_vote"]["dev"]["pr_auc"], 2),
        "lmOldSnorkelPR": f(e0["variants"]["dev_prior"]["dev"]["pr_auc"], 2),
        "lmNewMVPR": f(e6["binary"]["mv"]["pr_auc"], 2), "lmNewSnorkelPR": f(e6["binary"]["snorkel"]["pr_auc"], 2),
        "lmBandSnorkelPR": f(e6["confidence"]["band_snorkel"]["pr_auc"], 2),
        "lmBandDSPR": f(e6["confidence"]["band_ds"]["pr_auc"], 2),
        "lmPosPoolLow": f(min(e6["binary"][k]["pool_pos_rate"] for k in ("mv", "ds", "triplet", "snorkel")), 2),
        "lmPosPoolHigh": f(max(e6["binary"][k]["pool_pos_rate"] for k in ("mv", "ds", "triplet", "snorkel")), 2),
        "pPoolLlama": f(_pool_pos_rate(root, "legacy_llama3.2-v-90b_sq"), 2),
        "pPoolQwen": f(_pool_pos_rate(root, "legacy_qwen2.5-vl-72b_sq"), 2),
        "pPoolThirtyTwo": f(_pool_pos_rate(root, "qwen3-vl-32b-fp8_sq_multiimg__sq_v4"), 2),
        "pDevLlama": f(_labeler_metrics(root, "legacy_llama3.2-v-90b_sq")["pos"], 2),
        "stDistPR": f(sm.loc[("single_q32hard", "temporal"), "pr_auc"], 2),
        "stDistAcc": f(sm.loc[("single_q32hard", "temporal"), "acc"], 2),
        "stWSBestPR": f(ws["pr_auc"].max(), 2), "stWSBestAcc": f(ws["acc"].max(), 2),
        "stWSWorstPR": f(ws["pr_auc"].min(), 2),
        "stMLPPR": f(sm.xs("mlp", level=1)["pr_auc"].mean(), 2),
        "stTempPR": f(sm.xs("temporal", level=1)["pr_auc"].mean(), 2),
    })
    if not rob.empty:
        for head, hk in (("temporal", "T"), ("mlp", "M")):
            for mode, mk in (("all_kl", "Ref"), ("drop_gce", "VOne"), ("sil_late", "VTwo"), ("sil_early", "VThree")):
                r = rob.loc[("single_q32soft", mode, head)]
                vals[f"rob{hk}{mk}PR"] = f(r[("pr_auc", "mean")], 2)
                vals[f"rob{hk}{mk}Acc"] = f(r[("acc", "mean")], 2)
    summ = root / "results/E13_robust_sil/summary.json"
    if summ.exists():
        ph = json.loads(summ.read_text())["sil_late_phase"]
        vals["robVTwoWins"] = int(sum(r["after_wins"] for r in ph))
        vals["robVTwoRuns"] = len(ph)
    return "% generated by scripts/paper_tables.py; do not edit\n" + "\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in vals.items()) + "\n"
