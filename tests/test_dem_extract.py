import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from gully.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("extract_dem", REPO_ROOT / "scripts/extract_dem_features.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _tif(path: Path, z: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
                       crs="EPSG:6344", transform=from_origin(0, 0, 0.914, 0.914)) as d:
        d.write(z.astype("float32"), 1)


def test_paths_and_extraction(tmp_path):
    idx = pd.DataFrame({"loc_id": ["U-000001", "L-000002", "L-000003"], "legacy_id": ["1", "2", "3"]})
    reece = pd.DataFrame({"loc_id": ["L-000002", "L-000003"], "reece_label": [1, 0]})
    paths = dict(mod.dem_paths(idx, str(tmp_path / "pool"), str(tmp_path / "t"), reece))
    assert paths["L-000002"].endswith("/pos/dem/dem_tile_2.tif") and paths["L-000003"].endswith("/neg/dem/dem_tile_3.tif")
    z = 0.05 * np.mgrid[0:64, 0:64][0].astype(float)[::-1]
    for p in paths.values():
        _tif(Path(p), z)
    rows = [mod._one(x) for x in paths.items()]
    assert all(r["dem_ok"] for r in rows) and len(rows[0]) > 35
    bad = tmp_path / "bad.tif"
    _tif(bad, np.full((64, 64), -9999.0))
    assert mod._one(("U-9", str(bad)))["dem_ok"] is False
    assert mod._one(("U-8", str(tmp_path / "missing.tif")))["dem_ok"] is False
