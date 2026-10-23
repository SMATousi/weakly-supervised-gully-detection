import numpy as np

from gully.labelers.clip_rs import location_scores


def test_location_scores_aggregation():
    logits = np.zeros((2, 8, 6))
    logits[0, 3, :3] = 5.0  # one date strongly "gully"
    s_max = location_scores(logits, 3, "max")
    s_mean = location_scores(logits, 3, "mean")
    assert s_max[0] > 0.99 and abs(s_max[1] - 0.5) < 1e-9
    assert s_mean[0] < s_max[0]
    assert location_scores(logits, 3, "top2")[0] < s_max[0]
