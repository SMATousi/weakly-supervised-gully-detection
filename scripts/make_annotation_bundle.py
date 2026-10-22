"""Copy the D-4 sample into a folder the old labeling tool can read: rgb_{0..6,11}_tile_{blind_id}.tif.

Run where the PVCs are mounted (cluster pod). The pool high-res file `rgb_highres_*` becomes `rgb_11_*`.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--pvc-root", action="append", required=True, help="PVC=/mount (repeatable)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    roots = dict(r.split("=", 1) for r in a.pvc_root)
    idx = pd.read_parquet(a.index).set_index("loc_id")
    out = Path(a.out) / "all_tiles_25"
    out.mkdir(parents=True, exist_ok=True)
    for _, s in pd.read_csv(a.sample).iterrows():
        r = idx.loc[s.loc_id]
        src = Path(roots[r["pvc"]]) / r["img_dir"]
        for k in range(7):
            shutil.copy(src / r[f"img_naip_{k}"], out / f"rgb_{k}_tile_{s.blind_id}.tif")
        shutil.copy(src / r["img_highres"], out / f"rgb_11_tile_{s.blind_id}.tif")
    print(f"wrote {len(list(out.iterdir()))} files to {out}")


if __name__ == "__main__":
    main()
