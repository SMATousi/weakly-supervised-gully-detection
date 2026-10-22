"""Location index: one row per location, image paths by role (never by sort position; audit B1)."""
from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import rasterio

N_NAIP = 7
_TILE_RE = re.compile(r"^rgb_(\d+|highres)_tile_(\d+)\.tif$")


def scan_tiles(rgb_dir: str | Path) -> dict[str, dict[str, str]]:
    """{legacy_id: {role_token: filename}} where role_token is '0'..'6', '11' or 'highres'."""
    out: dict[str, dict[str, str]] = {}
    for name in sorted(Path(rgb_dir).iterdir()):
        m = _TILE_RE.match(name.name)
        if m:
            out.setdefault(m.group(2), {})[m.group(1)] = name.name
    return out


def _probe(args: tuple) -> dict:
    """Read one location's headers (top-level so it can run in a process pool)."""
    rgb_dir, tid, files, prefix, highres_token, dem_dir = args
    rgb_dir = Path(rgb_dir)
    with rasterio.open(rgb_dir / files["0"]) as src:
        b = src.bounds
        crs, res, size = str(src.crs), src.res[0], src.width
    with rasterio.open(rgb_dir / files[highres_token]) as src:
        hr_size, hr_res, hr_bands = src.width, src.res[0], src.count
    row = {
        "loc_id": f"{prefix}-{int(tid):06d}",
        "legacy_id": tid,
        "x": (b.left + b.right) / 2,
        "y": (b.bottom + b.top) / 2,
        "crs": crs,
        "tile_width_m": b.right - b.left,
        "naip_px": size,
        "naip_res_m": res,
        "highres_px": hr_size,
        "highres_res_m": hr_res,
        "highres_bands": hr_bands,
        "img_highres": files[highres_token],
    }
    for i in range(N_NAIP):
        row[f"img_naip_{i}"] = files[str(i)]
    if dem_dir is not None:
        dem = Path(dem_dir) / f"dem_tile_{tid}.tif"
        row["dem"] = dem.name if dem.exists() else None
    return row


def build_index(rgb_dir, prefix: str, highres_token: str, dem_dir=None, workers: int = 32):
    """Index of one image directory. Locations missing any of the 8 images raise (checked, not skipped)."""
    tiles = scan_tiles(rgb_dir)
    expected = {str(i) for i in range(N_NAIP)} | {highres_token}
    for tid, files in tiles.items():
        if set(files) != expected:
            raise ValueError(f"location {tid}: roles {sorted(files)} != expected {sorted(expected)}")
    jobs = [(str(rgb_dir), t, tiles[t], prefix, highres_token, dem_dir and str(dem_dir))
            for t in sorted(tiles, key=int)]
    with ProcessPoolExecutor(workers) as ex:  # network volume: latency bound, many processes
        rows = list(ex.map(_probe, jobs, chunksize=16))
    return pd.DataFrame(rows)
