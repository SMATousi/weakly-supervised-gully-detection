import numpy as np

from gully.data.dem import d8_flow_accumulation, fill_depressions, slope_aspect, terrain_features

RES = 1.0


def _plane(slope_deg=5.0, n=64):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.tan(np.deg2rad(slope_deg)) * xx * RES + 0 * yy


def test_slope_of_a_plane():
    s, _ = slope_aspect(_plane(5.0), RES)
    assert abs(np.degrees(s[10:-10, 10:-10]).mean() - 5.0) < 0.05


def test_fill_removes_a_pit_and_flat_gets_a_gradient():
    z = _plane(2.0)
    z[30, 30] -= 3.0
    zf = fill_depressions(z)
    assert (zf - z)[30, 30] > 2.0 and (zf >= z - 1e-9).all()


def test_flow_accumulates_along_a_carved_channel():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    z = 0.05 * (n - yy) * RES  # slopes down towards the bottom row
    z = z - 1.0 * np.exp(-((xx - 32) ** 2) / 8.0)  # a swale in the middle column
    acc = d8_flow_accumulation(fill_depressions(z))
    assert acc[-2, 32] > 5 * np.median(acc) and acc[-2, 32] > acc[-2, 5] * 3


def test_channel_tile_scores_higher_than_a_plane_on_flow_and_hollow_features():
    n = 64
    yy, xx = np.mgrid[0:n, 0:n]
    base = 0.05 * (n - yy) * RES
    plane = terrain_features(base, RES)
    swale = terrain_features(base - 1.0 * np.exp(-((xx - 32) ** 2) / 8.0), RES)
    for k in ("acc_max", "twi_max", "frac_acc_100", "spi_max", "tpi15_frac_hollow", "resid_std"):
        assert swale[k] > plane[k], k
    assert swale["tpi5_min"] < plane["tpi5_min"] and swale["resid_p01"] < plane["resid_p01"]
    assert swale["plan_curv_min"] < plane["plan_curv_min"]


def test_sink_features_and_nodata_handling():
    z = _plane(2.0)
    z[20:23, 20:23] -= 1.5
    f = terrain_features(z, RES)
    assert f["sink_depth_max"] > 1.0 and f["sink_volume"] > 5
    z2 = z.copy()
    z2[5, 5] = np.nan
    assert all(np.isfinite(v) for v in terrain_features(z2, RES).values())
    assert len(f) >= 35
