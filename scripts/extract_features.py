"""Extract frozen features for every location in the index (run as a GPU job on the cluster)."""
import argparse

import pandas as pd

from gully.student.features import extract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--pvc-root", action="append", required=True, help="PVC=/mount/path (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="flexivit_small.1200ep_in1k")
    ap.add_argument("--tag", default="flexivit_s")
    ap.add_argument("--highres-size", type=int, default=512)
    ap.add_argument("--highres-patch", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    roots = dict(r.split("=", 1) for r in a.pvc_root)
    extract(pd.read_parquet(a.index), roots, f"{a.out}/{a.tag}", a.backbone, a.highres_size,
            a.highres_patch, a.batch_size, a.workers, limit=a.limit)


if __name__ == "__main__":
    main()
