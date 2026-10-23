"""results/*.json -> LaTeX tables. Every number in the paper comes from here (no notebooks)."""
from __future__ import annotations

import json
from pathlib import Path

COLS = [("acc", "Acc"), ("f1", "F1"), ("precision", "P"), ("recall", "R"), ("npv", "NPV"),
        ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]


def fmt(v: float, ci: tuple[float, float] | None = None) -> str:
    s = f"{100 * v:.1f}"
    return s if ci is None else f"{s} \\scriptsize{{[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]}}"


def main_table(final_dir: str | Path, rows: list[tuple[str, str]], caption: str, label: str,
               baseline_name: str | None = None) -> str:
    """rows = [(file stem in final_dir, display name)]. Test metrics with 95% bootstrap CIs; marks McNemar p<0.05 vs
    the baseline stored in each file with an asterisk."""
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{l" + "c" * len(COLS) + "}", "\\toprule",
             "Method & " + " & ".join(n for _, n in COLS) + " \\\\", "\\midrule"]
    for stem, name in rows:
        d = json.loads((Path(final_dir) / f"{stem}.json").read_text())
        star = "$^*$" if d.get("mcnemar_p_vs_baseline", 1.0) < 0.05 else ""
        cells = [fmt(d["metrics"][k], tuple(d["ci95"][k])) for k, _ in COLS]
        lines.append(f"{name}{star} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def grid_table(run_dir: str | Path, caption: str, label: str) -> str:
    """Dev-set grid (rows = pseudo-label set x head; columns Acc / F1 / PR-AUC); means over seeds if several."""
    import collections

    import numpy as np

    acc = collections.defaultdict(list)
    for f in sorted(Path(run_dir).glob("*.json")):
        d = json.loads(f.read_text())
        acc[(d["pseudo"] + ("-hard" if d.get("mode") == "hard" else ""), d["head"])].append(
            [d["dev"]["acc"], d["dev"]["f1"], d["dev"]["pr_auc"]])
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{llccc}", "\\toprule", "Labels & Head & Acc & F1 & PR-AUC \\\\", "\\midrule"]
    for (lab, head), v in sorted(acc.items()):
        m = np.mean(v, 0)
        esc = lab.replace("_", "\\_")
        lines.append(f"{esc} & {head} & " + " & ".join(f"{100 * x:.1f}" for x in m) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)
