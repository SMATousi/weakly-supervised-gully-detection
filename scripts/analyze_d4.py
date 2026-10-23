"""D-4: pool prevalence and realistic precision from the annotators' labels on the 100 random pool locations.

Inputs: the three annotators' JSONs for the D-4 bundle (same format as data/gt_raw, tile ids = blind ids),
results/D4/sample.csv (blind -> loc_id), a pseudo-label parquet. Outputs results/D4/summary.json:
prevalence of the main GT scheme (+ Wilson 95% CI), and for the pseudo-labels / any score file: precision,
recall, accuracy against the new labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gully.data.gt import build_gt
from gully.eval.metrics import binary_metrics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir with annotator_{1,2,3}.json for the D-4 tiles")
    ap.add_argument("--sample", default="results/D4/sample.csv")
    ap.add_argument("--scores", nargs="*", default=[], help="parquet files with loc_id, p1 to score")
    ap.add_argument("--out", default="results/D4/summary.json")
    a = ap.parse_args()
    from gully.data.gt import load_scores

    gt = build_gt(load_scores(a.raw))
    sample = pd.read_csv(a.sample)
    gt["legacy_id"] = gt["legacy_id"].astype(int)
    df = sample.merge(gt, left_on="blind_id", right_on="legacy_id", suffixes=("", "_gt"))
    out = {"n_sampled": len(sample), "n_labeled": len(df)}
    for scheme in ("strict_neg", "lenient_neg", "lenient_pos", "strict_pos"):
        y = df[f"y_{scheme}"].dropna()
        k, n = int(y.sum()), len(y)
        out[scheme] = {"n": n, "positives": k, "prevalence": k / n, "wilson95": wilson(k, n)}
    d = df.dropna(subset=["y_main"])
    out["scores"] = {}
    for f in a.scores:
        s = pd.read_parquet(f).set_index("loc_id")["p1"]
        m = d[d.loc_id.isin(s.index)]
        out["scores"][Path(f).stem] = binary_metrics(m.y_main.to_numpy().astype(int), s.loc[m.loc_id].to_numpy())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "scores"}, indent=2, default=float))


if __name__ == "__main__":
    main()
