"""Regenerate every LaTeX table from results/. One command: python scripts/make_tables.py"""
import argparse
from pathlib import Path

from gully.eval.report import grid_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = Path(a.results)
    if (r / "E_early_legacy").exists():
        (out / "early_legacy_grid.tex").write_text(
            grid_table(r / "E_early_legacy", "Dev-set students on the legacy labels (one seed).", "tab:early_legacy"))
    # main test table: add once results/final/*.json exist (scripts/evaluate.py), see eval.report.main_table
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
