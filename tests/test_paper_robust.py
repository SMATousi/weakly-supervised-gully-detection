import json

from gully.eval import paper


def test_robust_table_from_fake_results(tmp_path):
    d = tmp_path / "results/E13_robust_sil"
    d.mkdir(parents=True)
    for mode in paper.E13_MODES:
        for head in ("mlp", "temporal"):
            for seed in range(3):
                run = f"single_q32soft__{mode}__{head}__s{seed}"
                (d / f"{run}.json").write_text(json.dumps({
                    "run": run, "pseudo": "single_q32soft", "mode": mode, "head": head, "seed": seed, "keep_frac": 0.5,
                    "dev": {"acc": 0.6 + 0.01 * seed, "f1": 0.6, "roc_auc": 0.7, "pr_auc": 0.7}}))
    # a sweep run (tagged) must be ignored by the main table
    (d / "single_q32soft__drop_gce_k30__mlp__s0.json").write_text(json.dumps({
        "run": "single_q32soft__drop_gce_k30__mlp__s0", "pseudo": "single_q32soft", "mode": "drop_gce", "head": "mlp",
        "seed": 0, "keep_frac": 0.3, "dev": {"acc": 0.1, "f1": 0.1, "roc_auc": 0.1, "pr_auc": 0.1}}))
    t = paper.robust_table(tmp_path)
    assert "V3: student labels the uncertain, from the start" in t and "0.610$\\pm$0.010" in t
    assert "0.100" not in t
    assert "pending" in paper.robust_table(tmp_path / "nothing")
