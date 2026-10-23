"""Light-weight GeoTIFF readers (no torch): shared by feature extraction and the VLM labelers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import rasterio.io


def decode_rgb(data: bytes) -> np.ndarray:
    """uint8 (H, W, 3) from GeoTIFF bytes; alpha band dropped (high-res tiles are RGBA)."""
    with rasterio.io.MemoryFile(data) as mf, mf.open() as src:
        arr = src.read([1, 2, 3])
    return np.moveaxis(arr, 0, -1)


def read_bytes(path: str | Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def read_rgb(path: str | Path) -> np.ndarray:
    return decode_rgb(read_bytes(path))
