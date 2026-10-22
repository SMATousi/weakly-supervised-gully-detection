"""Build artifacts/index/locations.parquet for the pool (U-) and the expert-labeled test tiles (L-).

Paths in the index are relative to image_dir_pool / image_dir_test (columns img_dir, dem_dir hold the
dir relative to the PVC root), so the same index works wherever the PVCs are mounted.
"""
import argparse
from pathlib import Path

import pandas as pd

from gully.config import load_config
from gully.data.index import build_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-pvc", required=True, help="mount point of the PVC with the pool")
    ap.add_argument("--test-pvc", required=True, help="mount point of the PVC with the test tiles")
    ap.add_argument("--out", default="artifacts/index")
    args = ap.parse_args()
    cfg = load_config("data")

    pool_dir = cfg["pool"]["rgb_images"]
    test_dir = cfg["test"]["rgb_images"]
    pool = build_index(Path(args.pool_pvc) / pool_dir, "U", "highres", Path(args.pool_pvc) / cfg["pool"]["dem"])
    pool["pvc"], pool["img_dir"], pool["dem_dir"] = cfg["pvc_pool"], pool_dir, cfg["pool"]["dem"]
    test = build_index(Path(args.test_pvc) / test_dir, "L", "11")
    test["pvc"], test["img_dir"] = cfg["pvc_test"], test_dir
    df = pd.concat([pool, test], ignore_index=True)
    df["split"] = df["loc_id"].str[0].map({"U": "pool", "L": "unassigned_labeled"})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "locations.parquet")
    print(df.groupby("split").size())
    print(df.groupby(df.loc_id.str[0])[["tile_width_m", "naip_px", "highres_px", "highres_bands"]].describe().T)
    print("missing dem:", int(pool["dem"].isna().sum()))


if __name__ == "__main__":
    main()
