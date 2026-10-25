"""Terrain attributes from a DEM tile (pure numpy/scipy; the tiles are 128 x 128 at ~0.91 m).

Everything is computed within the tile, so flow accumulation counts only the area inside the tile and edge cells drain out
of it. Slopes in radians, curvatures in 1/m, flow accumulation in cells. No absolute elevation is returned (it would only
encode where a tile lies).
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_EPS = 1e-4


def fill_depressions(z: np.ndarray) -> np.ndarray:
    """Priority-flood fill with a tiny gradient on flats (Barnes et al. 2014); the tile border is the outlet."""
    h, w = z.shape
    out = np.full_like(z, np.inf, dtype=float)
    done = np.zeros((h, w), dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for i in range(h):
        for j in range(w):
            if i in (0, h - 1) or j in (0, w - 1):
                out[i, j] = z[i, j]
                done[i, j] = True
                heapq.heappush(heap, (z[i, j], i, j))
    while heap:
        zc, i, j = heapq.heappop(heap)
        for di, dj in _NB:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w and not done[a, b]:
                done[a, b] = True
                out[a, b] = max(z[a, b], zc + _EPS)
                heapq.heappush(heap, (out[a, b], a, b))
    return out


def d8_flow_accumulation(zf: np.ndarray) -> np.ndarray:
    """Number of cells draining through each cell (including itself), D8 routing on the filled DEM."""
    h, w = zf.shape
    acc = np.ones((h, w))
    order = np.argsort(-zf, axis=None)  # highest first
    for flat in order:
        i, j = divmod(int(flat), w)
        best, bi, bj = 0.0, -1, -1
        for di, dj in _NB:
            a, b = i + di, j + dj
            if 0 <= a < h and 0 <= b < w:
                drop = (zf[i, j] - zf[a, b]) / (1.4142 if di and dj else 1.0)
                if drop > best:
                    best, bi, bj = drop, a, b
        if bi >= 0:
            acc[bi, bj] += acc[i, j]
    return acc


def slope_aspect(z: np.ndarray, res: float):
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)
    return np.arctan(np.hypot(dzdx, dzdy)), np.arctan2(dzdy, -dzdx)


def curvatures(z: np.ndarray, res: float):
    """Zevenbergen-Thorne profile and plan curvature, and the Laplacian (positive = concave / hollow)."""
    zp = np.pad(z, 1, mode="edge")
    z1, z2, z3 = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    z4, z5, z6 = zp[1:-1, :-2], zp[1:-1, 1:-1], zp[1:-1, 2:]
    z7, z8, z9 = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    L = res
    D, E = ((z4 + z6) / 2 - z5) / L**2, ((z2 + z8) / 2 - z5) / L**2
    F = (-z1 + z3 + z7 - z9) / (4 * L**2)
    G, H = (-z4 + z6) / (2 * L), (z2 - z8) / (2 * L)
    den = G**2 + H**2
    safe = den > 1e-12
    prof = np.where(safe, -2 * (D * G**2 + E * H**2 + F * G * H) / np.where(safe, den, 1), 0.0)
    plan = np.where(safe, 2 * (D * H**2 + E * G**2 - F * G * H) / np.where(safe, den, 1), 0.0)
    lap = (z2 + z8 + z4 + z6 - 4 * z5) / L**2
    return prof, plan, lap


def terrain_features(z: np.ndarray, res: float = 0.914) -> dict[str, float]:
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).all():  # fill the few nodata cells with the tile median
        z = np.where(np.isfinite(z), z, np.nanmedian(z))
    f: dict[str, float] = {}
    slope, aspect = slope_aspect(z, res)
    prof, plan, lap = curvatures(z, res)
    f.update(slope_mean=slope.mean(), slope_p90=np.quantile(slope, 0.9), slope_max=slope.max(), slope_std=slope.std(),
             relief=np.ptp(z))
    f.update(plan_curv_min=np.quantile(plan, 0.02), plan_curv_p98=np.quantile(plan, 0.98),
             prof_curv_min=np.quantile(prof, 0.02), prof_curv_p98=np.quantile(prof, 0.98),
             lap_p98=np.quantile(lap, 0.98), frac_concave=float((lap > 0.05).mean()))
    zf = fill_depressions(z)
    depth = zf - z
    f.update(sink_depth_max=depth.max(), sink_frac=float((depth > 0.05).mean()), sink_volume=float(depth.sum() * res**2))
    acc = d8_flow_accumulation(zf)
    a_spec = acc * res  # specific catchment area per unit contour width (m)
    tanb = np.tan(np.clip(slope, 0.001, 1.5))
    twi, spi = np.log(a_spec / tanb), a_spec * tanb
    f.update(acc_max=acc.max(), frac_acc_100=float((acc >= 100).mean()), frac_acc_500=float((acc >= 500).mean()),
             frac_acc_2000=float((acc >= 2000).mean()), twi_mean=twi.mean(), twi_p90=np.quantile(twi, 0.9),
             twi_max=twi.max(), spi_p99=np.quantile(spi, 0.99), spi_max=spi.max(),
             spi_log_mean=np.log1p(spi).mean())
    ch = acc >= 100  # concentrated-flow cells
    f.update(channel_slope_mean=float(slope[ch].mean()) if ch.any() else 0.0,
             channel_plan_curv_min=float(plan[ch].min()) if ch.any() else 0.0)
    for r in (5, 15):  # topographic position index: negative = hollow
        tpi = z - ndimage.uniform_filter(z, size=2 * r + 1, mode="nearest")
        f[f"tpi{r}_min"] = float(np.quantile(tpi, 0.02))
        f[f"tpi{r}_frac_hollow"] = float((tpi < -0.15).mean())
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    A = np.c_[xx.ravel() * res, yy.ravel() * res, np.ones(z.size)]
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    resid = z.ravel() - A @ coef
    f.update(resid_std=resid.std(), resid_p01=np.quantile(resid, 0.01), resid_p99=np.quantile(resid, 0.99))
    zp = np.pad(z, 1, mode="edge")
    tri = np.mean([np.abs(zp[1 + di:1 + di + z.shape[0], 1 + dj:1 + dj + z.shape[1]] - z) for di, dj in _NB], axis=0)
    f["tri_mean"] = tri.mean()
    m = slope > np.deg2rad(2)
    f["aspect_circ_var"] = float(1 - np.abs(np.exp(1j * aspect[m]).mean())) if m.sum() > 10 else 0.0
    return {k: float(v) for k, v in f.items()}
