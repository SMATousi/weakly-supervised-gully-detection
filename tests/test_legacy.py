from pathlib import Path

import pandas as pd
import pytest

from gully.config import REPO_ROOT

LABELS = REPO_ROOT / "artifacts/labels"
GT = REPO_ROOT / "artifacts/gt/gt.parquet"
pytestmark = pytest.mark.skipif(not (LABELS / "legacy_qwen2.5-vl-72b_sq.jsonl").exists() or not GT.exists(),
                                reason="needs scripts/ingest_legacy.py and build_gt.py outputs")


def _acc(name: str):
    df = pd.read_json(Path(LABELS) / f"{name}.jsonl", lines=True)
    df = df[df.loc_id.str[0] == "L"]
    gt = pd.read_parquet(GT)[["loc_id", "y_main"]].dropna()
    m = gt.merge(df, on="loc_id", how="left")
    cov = (~m.abstain.fillna(True)).mean()
    voted = m[~m.abstain.fillna(True)]
    return (voted.label == voted.y_main).mean(), cov


@pytest.mark.parametrize("name,acc,cov", [
    ("legacy_qwen2.5-vl-72b_sq", 0.653, 1.0),
    ("legacy_llama3.2-v-90b_sq", 0.607, 1.0),
    ("legacy_llama3.2-v-90b_mq", 0.568, 0.876),
])
def test_reproduces_audit_numbers(name, acc, cov):
    a, c = _acc(name)
    assert abs(a - acc) < 0.002 and abs(c - cov) < 0.002
