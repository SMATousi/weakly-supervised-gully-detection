"""Merge a labeler's labeled-set file and pool shards into one artifacts/labels/<labeler>.jsonl.

  python scripts/merge_labels.py --name qwen3-vl-32b-fp8_sq_multiimg__sq_v4 --src artifacts/labels \
      --expect-pool 18179 --expect-labeled 548

Checks that every location appears exactly once and that the expected counts are present; duplicates keep the
last record (resumed jobs can re-emit a location).
"""
import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--src", default="artifacts/labels")
    ap.add_argument("--expect-pool", type=int)
    ap.add_argument("--expect-labeled", type=int)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.src}/{a.name}.*.jsonl"))
    if not files:
        raise SystemExit(f"no files for {a.name} in {a.src}")
    recs: dict[str, str] = {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated tail of a killed job
                recs[r["loc_id"]] = line if line.endswith("\n") else line + "\n"
    n_u = sum(k.startswith("U-") for k in recs)
    n_l = sum(k.startswith("L-") for k in recs)
    print(f"{a.name}: {len(recs)} locations ({n_u} pool, {n_l} labeled) from {len(files)} files")
    if a.expect_pool is not None and n_u != a.expect_pool:
        raise SystemExit(f"expected {a.expect_pool} pool locations, got {n_u}")
    if a.expect_labeled is not None and n_l != a.expect_labeled:
        raise SystemExit(f"expected {a.expect_labeled} labeled locations, got {n_l}")
    out = Path(a.src) / f"{a.name}.jsonl"
    with open(out, "w") as fh:
        fh.writelines(recs[k] for k in sorted(recs))
    print("wrote", out)


if __name__ == "__main__":
    main()
