"""Terrain attributes for every location (pool: pool/dem; labeled: pos/dem or neg/dem) -> one parquet (loc_id + 35 features)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from gully.data.dem import terrain_features


def _one(args):
    loc_id, path = args
    try:
        with rasterio.open(path) as src:
            z = src.read(1).astype(float)
            res = float(src.res[0])
        z[(z < -500) | (z > 3000)] = np.nan  # nodata sentinels
        if np.isnan(z).mean() > 0.5:
            return {"loc_id": loc_id, "dem_ok": False}
        return {"loc_id": loc_id, "dem_ok": True, **terrain_features(z, res)}
    except Exception as e:  # noqa: BLE001  (record and continue: a few tiles may be unreadable)
        return {"loc_id": loc_id, "dem_ok": False, "error": str(e)[:80]}


def dem_paths(index: pd.DataFrame, pool_dem: str, test_root: str, reece: pd.DataFrame) -> list[tuple[str, str]]:
    side = dict(zip(reece.loc_id, reece.reece_label, strict=True))
    out = []
    for loc, tid in zip(index.loc_id, index.legacy_id, strict=True):
        if loc.startswith("U-"):
            out.append((loc, f"{pool_dem}/dem_tile_{tid}.tif"))
        else:
            out.append((loc, f"{test_root}/{'pos' if side[loc] == 1 else 'neg'}/dem/dem_tile_{tid}.tif"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--reece", default="data/reece_initial_labels.csv")
    ap.add_argument("--pool-dem", required=True)
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--out", default="artifacts/dem/dem_features.parquet")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    idx = pd.read_parquet(a.index)
    if a.limit:
        idx = pd.concat([idx[idx.loc_id.str[0] == "U"].head(a.limit), idx[idx.loc_id.str[0] == "L"].head(a.limit)])
    jobs = dem_paths(idx, a.pool_dem, a.test_root, pd.read_csv(a.reece))
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:  # file reads dominate: many workers, few CPUs
        for k, r in enumerate(ex.map(_one, jobs, chunksize=16)):
            rows.append(r)
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out)
    print(f"{len(df)} tiles, ok={int(df.dem_ok.sum())}, failed={int((~df.dem_ok).sum())}, features={df.shape[1] - 2}")
    if (~df.dem_ok).any():
        print(df[~df.dem_ok].head())


if __name__ == "__main__":
    main()
