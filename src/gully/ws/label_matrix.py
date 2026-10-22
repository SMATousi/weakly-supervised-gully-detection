"""Assemble the label matrix L (rows = loc_id, columns = labelers, values in {-1, 0, 1})."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_labels(labels_dir: str | Path, names: list[str]) -> pd.DataFrame:
    """Wide table of hard labels over all locations in the given JSONL files (-1 = abstain / missing)."""
    cols = {}
    for n in names:
        df = pd.read_json(Path(labels_dir) / f"{n}.jsonl", lines=True)
        cols[n] = df.set_index("loc_id")["label"]
    return pd.DataFrame(cols).fillna(-1).astype(int)


def coverage(L: pd.DataFrame) -> pd.Series:
    return (L != -1).mean()
