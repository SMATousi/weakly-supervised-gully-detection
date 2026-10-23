"""Frozen-backbone feature extraction, once per (backbone, location).

Each location has 8 images in a fixed ROLE ORDER: naip_0..naip_6 then highres (never sort order;
audit B1). Output layout under <out>/<backbone>/:
  tokens.npy  float16 (N, 8, 1+G*G, D)   CLS + patch tokens per image
  valid.npy   bool    (N, 8)             False for truncated / unreadable images
  ids.npy     str     (N,)               loc_id, in index-row order
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gully.data.images import decode_rgb, read_bytes, read_rgb  # noqa: F401  (read_rgb re-exported)

ROLES = [f"naip_{i}" for i in range(7)] + ["highres"]


class LocationImages(Dataset):
    """Yields (naip (7,3,128,128) float, highres (3,S,S) float, valid (8,) bool), normalized."""

    def __init__(self, index: pd.DataFrame, roots: dict[str, str], highres_size: int,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), min_highres_px: int = 600):
        self.rows = index.reset_index(drop=True)
        self.roots, self.S = roots, highres_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.min_highres_px = min_highres_px

    def __len__(self) -> int:
        return len(self.rows)

    def _norm(self, a: np.ndarray, size: int | None) -> torch.Tensor:
        t = torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
        if size is not None and t.shape[-1] != size:
            t = F.interpolate(t[None], size=(size, size), mode="bilinear", antialias=True,
                              align_corners=False)[0]
        return (t - self.mean) / self.std

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        d = Path(self.roots[r["pvc"]]) / r["img_dir"]
        valid = np.ones(8, dtype=bool)
        naip = torch.zeros(7, 3, 128, 128)
        paths = [d / r[f"img_naip_{k}"] for k in range(7)] + [d / r["img_highres"]]
        with ThreadPoolExecutor(8) as ex:  # network volume: latency bound, so read the 8 files at once
            blobs = list(ex.map(self._try_read, paths))
        for k in range(7):
            if blobs[k] is None:  # unreadable -> masked, zeros
                valid[k] = False
            else:
                naip[k] = self._norm(decode_rgb(blobs[k]), 128)
        if blobs[7] is None:
            hr, valid[7] = torch.zeros(3, self.S, self.S), False
        else:
            hr = self._norm(decode_rgb(blobs[7]), self.S)
            valid[7] = r["highres_px"] >= self.min_highres_px  # truncated tiles at coverage edges
        return naip, hr, torch.from_numpy(valid)

    @staticmethod
    def _try_read(path) -> bytes | None:
        try:
            return read_bytes(path)
        except OSError:
            return None


def load_backbone(name: str, img_size: int, patch_size: int, device: str):
    import timm

    m = timm.create_model(name, pretrained=True, img_size=img_size, patch_size=patch_size, num_classes=0)
    return m.eval().to(device).requires_grad_(False)


@torch.no_grad()
def extract(index: pd.DataFrame, roots: dict[str, str], out_dir: str | Path, backbone: str,
            highres_size: int = 512, highres_patch: int = 64, batch_size: int = 32,
            workers: int = 16, device: str = "cuda", limit: int | None = None) -> None:
    if limit:
        index = index.iloc[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m_naip = load_backbone(backbone, 128, 16, device)
    m_hr = load_backbone(backbone, highres_size, highres_patch, device)
    n, g2 = len(index), 65
    tok = np.lib.format.open_memmap(out / "tokens.npy", mode="w+", dtype=np.float16, shape=(n, 8, g2, 384))
    valid_all = np.zeros((n, 8), dtype=bool)
    dl = DataLoader(LocationImages(index, roots, highres_size), batch_size=batch_size, num_workers=workers)
    pos = 0
    for naip, hr, valid in dl:
        b = naip.shape[0]
        f_naip = m_naip.forward_features(naip.flatten(0, 1).to(device)).view(b, 7, g2, -1)
        f_hr = m_hr.forward_features(hr.to(device)).view(b, 1, g2, -1)
        tok[pos:pos + b] = torch.cat([f_naip, f_hr], 1).half().cpu().numpy()
        valid_all[pos:pos + b] = valid.numpy()
        pos += b
        if (pos // batch_size) % 10 == 0:
            print(f"{pos}/{n}", flush=True)
    tok.flush()
    np.save(out / "valid.npy", valid_all)
    np.save(out / "ids.npy", index["loc_id"].to_numpy().astype(str))
