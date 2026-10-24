"""Write every paper table and the number macros to results/tables/ (copy them into the paper repo)."""
import argparse
from pathlib import Path

from gully.config import REPO_ROOT
from gully.eval import paper


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results/tables"))
    ap.add_argument("--paper-dir", help="also copy the files into <paper-dir>/tables/")
    a = ap.parse_args()
    root, out = REPO_ROOT, Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {"tab_gt.tex": paper.gt_table(root), "tab_labelers.tex": paper.labeler_table(root),
             "tab_labelmodels.tex": paper.labelmodel_table(root), "tab_students.tex": paper.student_table(root),
             "numbers.tex": paper.numbers(root)}
    for name, text in files.items():
        (out / name).write_text(text)
        if a.paper_dir:
            (Path(a.paper_dir) / "tables").mkdir(parents=True, exist_ok=True)
            (Path(a.paper_dir) / "tables" / name).write_text(text)
    print("wrote", ", ".join(files))


if __name__ == "__main__":
    main()
