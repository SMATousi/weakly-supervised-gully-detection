"""Build artifacts/gt/gt.parquet (+ long scores) from the raw annotator scores in data/gt_raw."""
import argparse
from pathlib import Path

from gully.config import REPO_ROOT
from gully.data.gt import SCHEMES, build_gt, kappa_for_scheme, load_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO_ROOT / "data/gt_raw"))
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts/gt"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.raw)
    df = build_gt(scores)
    scores.to_parquet(out / "scores.parquet")
    df.to_parquet(out / "gt.parquet")
    print(f"{len(df)} locations")
    for s in SCHEMES:
        vc = df[f"y_{s}"].value_counts(dropna=False).to_dict()
        print(f"{s}: {vc}  fleiss_kappa={kappa_for_scheme(df, s):.3f}")


if __name__ == "__main__":
    main()
