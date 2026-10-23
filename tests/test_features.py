import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from gully.data.images import read_rgb
from gully.student.features import LocationImages


def _tif(path, size, bands):
    data = np.random.default_rng(0).integers(0, 255, (bands, size, size), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands, dtype="uint8",
                       crs="EPSG:6344", transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(data)


def test_read_rgb_drops_alpha(tmp_path):
    _tif(tmp_path / "a.tif", 20, 4)
    assert read_rgb(tmp_path / "a.tif").shape == (20, 20, 3)


def test_dataset_roles_and_masking(tmp_path):
    row = {"loc_id": "U-1", "pvc": "p", "img_dir": ".", "highres_px": 700, "img_highres": "hr.tif"}
    for k in range(7):
        row[f"img_naip_{k}"] = f"n{k}.tif"
        if k != 3:  # one unreadable image
            _tif(tmp_path / f"n{k}.tif", 128, 3)
    _tif(tmp_path / "hr.tif", 700, 4)
    ds = LocationImages(pd.DataFrame([row]), {"p": str(tmp_path)}, highres_size=64)
    naip, hr, valid = ds[0]
    assert naip.shape == (7, 3, 128, 128) and hr.shape == (3, 64, 64)
    assert valid.tolist() == [True, True, True, False, True, True, True, True]
    row["highres_px"] = 15  # truncated tile
    assert not LocationImages(pd.DataFrame([row]), {"p": str(tmp_path)}, 64)[0][2][7]
