"""Summarize E13 (robust / student-in-the-loop training) on the development split; writes results/E13_robust_sil/summary.json.

Adds a phase analysis for sil_late: does the second phase (the student labels the uncertain samples) ever beat the best
epoch of the warm-up phase (confident samples only)?
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results/E13_robust_sil")
rows, phase = [], []
for f in sorted(glob.glob(str(R / "*.json"))):
    d = json.loads(Path(f).read_text())
    if d.get("keep_frac") != 0.5 or d["run"].count("__") != 3:
        continue
    h = d["history"]
    rows.append({"pseudo": d["pseudo"], "mode": d["mode"], "head": d["head"], "seed": d["seed"], "best_epoch": d["best_epoch"],
                 **{k: d["dev"][k] for k in ("acc", "f1", "roc_auc", "pr_auc")}})
    if d["mode"] == "sil_late":
        w = d["warm_epochs"]
        pr = np.array([e["pr_auc"] for e in h])
        phase.append({"head": d["head"], "seed": d["seed"], "best_warmup": float(pr[:w].max()), "best_after": float(pr[w:].max()),
                      "after_wins": bool(pr[w:].max() > pr[:w].max()), "pr_last_epoch": float(pr[-1])})
df, ph = pd.DataFrame(rows), pd.DataFrame(phase)
g = df.groupby(["pseudo", "head", "mode"])[["acc", "f1", "roc_auc", "pr_auc"]].agg(["mean", "std"]).round(3)
print(g.to_string())
if len(ph):
    print("\nsil_late phase analysis (PR-AUC):")
    print(ph.round(3).to_string(index=False))
    print("runs where phase 2 beat the warm-up best:", int(ph.after_wins.sum()), "of", len(ph))
out = {"means": json.loads(g.to_json()), "sil_late_phase": json.loads(ph.to_json(orient="records")) if len(ph) else []}
(R / "summary.json").write_text(json.dumps(out, indent=2))
