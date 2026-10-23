import json

from gully.eval.report import fmt, grid_table, main_table


def test_fmt():
    assert fmt(0.6789) == "67.9"
    assert "[60.0, 70.0]" in fmt(0.65, (0.6, 0.7))


def test_main_table_marks_significance(tmp_path):
    base = {"metrics": {k: 0.7 for k in ("acc", "f1", "precision", "recall", "npv", "roc_auc", "pr_auc")},
            "ci95": {k: [0.6, 0.8] for k in ("acc", "f1", "precision", "recall", "npv", "roc_auc", "pr_auc")}}
    (tmp_path / "a.json").write_text(json.dumps({**base, "mcnemar_p_vs_baseline": 0.01}))
    (tmp_path / "b.json").write_text(json.dumps({**base, "mcnemar_p_vs_baseline": 0.5}))
    t = main_table(tmp_path, [("a", "Ours"), ("b", "Base")], "cap", "tab:x")
    assert "Ours$^*$" in t and "Base$^*$" not in t and t.count("\\\\") >= 3


def test_grid_table(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps({"pseudo": "snorkel_x", "mode": "soft", "head": "mlp",
                                                 "dev": {"acc": 0.7, "f1": 0.7, "pr_auc": 0.7}}))
    assert "snorkel\\_x & mlp" in grid_table(tmp_path, "c", "l")
