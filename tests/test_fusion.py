import numpy as np
from scipy.stats import rankdata

from factorylens.fusion import ReliabilityConfig, ReliabilityFusion, background_level

RNG = np.random.default_rng(0)
SIZE = 16
# the object is everything except a 3x3 corner where the fake defects go, so a
# defect never moves an image's background level
FG = np.ones((SIZE, SIZE), bool)
FG[2:5, 2:5] = False


def maps(n, level=1.0):
    # images differ in overall level, as real validation images do
    return (
        level + 0.1 * RNG.standard_normal((n, 1, 1)) + 0.05 * RNG.standard_normal((n, SIZE, SIZE))
    )


def with_defects(m, idx):
    m = m.copy()
    m[list(idx), 2:5, 2:5] += 3.0
    return m


def scored(m):
    return m.max(axis=(1, 2)), m


def fg(n):
    return np.broadcast_to(FG, (n, SIZE, SIZE))


VAL_RGB, VAL_GEO = maps(30), maps(30)
POINTS = RNG.normal(1000, 20, 30)


def fitted(config=None):
    return ReliabilityFusion(config).fit(scored(VAL_RGB), scored(VAL_GEO), fg(30), POINTS)


def test_background_level_ignores_a_small_defect():
    m = with_defects(maps(2), [1])
    level = background_level(m, np.ones((2, SIZE, SIZE), bool), 0.25)
    assert abs(level[1] - level[0]) < 0.3
    assert m[1].max() > m[0].max() + 2


def test_clean_inputs_rank_exactly_like_plain_late_fusion():
    f = fitted()
    rgb, geo = with_defects(VAL_RGB, range(10)), with_defects(VAL_GEO, range(5, 15))
    (fused, _), w = f.fuse(scored(rgb), scored(geo), fg(30), POINTS)
    assert (w.rgb == 1).all() and (w.geo == 1).all()
    zr = (rgb.max(axis=(1, 2)) - f.rgb_val[0].mean()) / f.rgb_val[0].std()
    zg = (geo.max(axis=(1, 2)) - f.geo_val[0].mean()) / f.geo_val[0].std()
    np.testing.assert_array_equal(rankdata(fused), rankdata(zr + zg))


def test_a_noisy_sensor_is_switched_off():
    f = fitted()
    labels = np.arange(30) < 15
    geo = with_defects(VAL_GEO, range(15))
    noisy = maps(30, level=3.0) + RNG.uniform(0, 2, (30, 1, 1))  # nothing to do with defects
    (fused, _), w = f.fuse(scored(noisy), scored(geo), fg(30), POINTS)
    assert (w.rgb == 0).all() and (w.geo == 1).all()
    assert fused[labels].min() > fused[~labels].max()


def test_missing_points_switch_off_the_3d_sensor():
    f = fitted()
    _, w = f.fuse(scored(VAL_RGB), scored(VAL_GEO), fg(30), POINTS / 3)
    assert (w.geo == 0).all() and (w.rgb == 1).all()
    assert (w.u_geo > 0).all() and (w.u_rgb <= 0).all()


def test_both_sensors_off_falls_back_to_both():
    f = fitted()
    (fused, m), w = f.fuse(scored(maps(5, level=5)), scored(maps(5, level=5)), fg(5), POINTS[:5])
    assert (w.rgb == 1).all() and (w.geo == 1).all()
    assert np.isfinite(fused).all() and np.isfinite(m).all()


def test_weight_ramps_down_over_the_configured_width():
    f = fitted(ReliabilityConfig(ramp=2.0))
    worst = f.ref_rgb.worst
    u = np.array([worst - 1, worst, worst + 1, worst + 2, worst + 5])
    np.testing.assert_allclose(f._weight(u, worst), [1, 1, 0.5, 0, 0], atol=1e-12)


def test_objects_without_foreground_use_the_whole_map():
    m = maps(1)
    np.testing.assert_allclose(
        background_level(m, np.zeros((1, SIZE, SIZE), bool), 0.25), [np.quantile(m[0], 0.25)]
    )
