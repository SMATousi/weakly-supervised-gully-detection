import json

import pandas as pd
import pytest

from gully.config import REPO_ROOT
from gully.data.gt import SCHEMES, build_gt, load_scores, majority


@pytest.fixture(scope="module")
def gt():
    return build_gt(load_scores(REPO_ROOT / "data/gt_raw"))


def test_majority_rules():
    assert majority([1, 1, 0]) == 1
    assert majority([1, 0, None]) is None  # tie
    assert majority([1, None, None]) is None
    assert majority([None, 0, 0]) == 0


def test_main_gt_counts(gt):
    vc = gt["y_main"].value_counts()
    assert (vc[1], vc[0]) == (257, 227)


@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_matches_legacy_majority(gt, scheme):
    stem = SCHEMES[scheme][1]
    with open(REPO_ROOT / f"data/gt_legacy/{stem}_labels_majority.json") as f:
        legacy = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    mine = dict(zip(gt["legacy_id"], gt[f"y_{scheme}"]))
    for k, v in legacy.items():
        got = mine.get(k)
        got = None if pd.isna(got) else got
        assert got == v, (scheme, k, got, v)
