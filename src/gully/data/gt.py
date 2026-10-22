"""Expert ground truth: 3 annotators score each of the 8 images of a location (0-4).

Each annotator's scores are turned into a binary location label by one of four schemes; the
final label is the majority over annotators. Rules were recovered from the legacy files
(data/gt_legacy) and match them exactly; see tests/test_gt.py.

Scores: 0 = no gully ... 4 = clear gully; 2 = unsure. Locations whose images are all 2 are null.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATORS = ("annotator_1", "annotator_2", "annotator_3")


def _strict_neg(s: list[int]) -> int | None:
    s = [x for x in s if x != 2]
    return None if not s else int(any(x > 0 for x in s))  # negative only if every other image is 0


def _lenient_neg(s: list[int]) -> int | None:
    return None if all(x == 2 for x in s) else int(any(x >= 3 for x in s))


def _lenient_pos(s: list[int]) -> int | None:
    if all(x == 2 for x in s):
        return None
    return int(any(x == 4 for x in s) and sum(x >= 1 for x in s) >= 2)


def _strict_pos(s: list[int]) -> int | None:
    if all(x == 2 for x in s):
        return None
    return int(any(x == 4 for x in s) and sum(x >= 3 for x in s) >= 2)


# scheme name -> (rule, legacy file stem). `y_main` = strict_neg.
SCHEMES = {
    "strict_neg": (_strict_neg, "neg_strict"),
    "lenient_neg": (_lenient_neg, "neg_len_1or0"),
    "lenient_pos": (_lenient_pos, "pos_len_4-1"),
    "strict_pos": (_strict_pos, "pos_strict"),
}


def majority(votes: list[int | None]) -> int | None:
    """Majority of non-null votes; null if fewer than 2 votes or a tie."""
    v = [x for x in votes if x is not None]
    if len(v) < 2:
        return None
    ones = sum(v)
    if 2 * ones == len(v):
        return None
    return int(2 * ones > len(v))


def load_scores(raw_dir: str | Path) -> pd.DataFrame:
    """Long table: legacy_id, annotator, image_role, score."""
    rows = []
    for a in ANNOTATORS:
        with open(Path(raw_dir) / f"{a}.json") as f:
            d = json.load(f)
        for tid, t in d.items():
            for role, sc in t["images"].items():
                rows.append((tid, a, role, int(sc)))
    return pd.DataFrame(rows, columns=["legacy_id", "annotator", "image_role", "score"])


def build_gt(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per location: per-annotator and majority labels for every scheme."""
    per = {}
    for (tid, a), g in scores.groupby(["legacy_id", "annotator"]):
        per[(tid, a)] = g["score"].tolist()
    out = []
    for tid in sorted({k[0] for k in per}, key=int):
        row = {"legacy_id": tid, "loc_id": f"L-{int(tid):06d}"}
        for name, (rule, _) in SCHEMES.items():
            votes = [rule(per[(tid, a)]) if (tid, a) in per else None for a in ANNOTATORS]
            for a, v in zip(ANNOTATORS, votes):
                row[f"{name}__{a}"] = v
            row[f"y_{name}"] = majority(votes)
        out.append(row)
    df = pd.DataFrame(out)
    df["y_main"] = df["y_strict_neg"]
    return df


def fleiss_kappa(votes: np.ndarray) -> float:
    """Fleiss' kappa for binary votes; `votes` is (n_items, n_raters) with complete rows only."""
    n, m = votes.shape
    n1 = votes.sum(1)
    counts = np.stack([m - n1, n1], 1)
    p_i = ((counts**2).sum(1) - m) / (m * (m - 1))
    p_j = counts.sum(0) / (n * m)
    pe = (p_j**2).sum()
    return float((p_i.mean() - pe) / (1 - pe))


def kappa_for_scheme(df: pd.DataFrame, scheme: str) -> float:
    cols = [f"{scheme}__{a}" for a in ANNOTATORS]
    sub = df[cols].dropna()
    return fleiss_kappa(sub.to_numpy(dtype=int))
