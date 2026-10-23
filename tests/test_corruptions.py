import numpy as np
import pytest

from factorylens import corruptions as C
from factorylens.data.cache import Prepared
from factorylens.data.mvtec3d import valid_points


def prepared(n=2, size=64):
    """A dome of points in the middle of the image on a textured background."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(40, 200, (n, size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:size, :size]
    r2 = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (size / 3) ** 2
    inside = r2 < 1
    xyz = np.zeros((n, size, size, 3), np.float32)
    xyz[:, inside, 0] = (xx[inside] - size / 2) * 2e-4
    xyz[:, inside, 1] = (yy[inside] - size / 2) * 2e-4
    xyz[:, inside, 2] = 0.3 - 0.01 * (1 - r2[inside])
    fg = np.broadcast_to(inside, (n, size, size)).copy()
    return Prepared(
        rgb,
        xyz,
        fg,
        np.zeros((n, size, size), bool),
        ["good"] * n,
        [f"{i:03d}" for i in range(n)],
        {"category": "fake", "split": "test", "settings": {"size": size}},
    )


@pytest.mark.parametrize("name", list(C.CORRUPTIONS))
def test_every_corruption_is_deterministic_and_touches_one_sensor(name):
    p = prepared()
    a = C.corrupt(p, name, 1, seed=3)
    b = C.corrupt(p, name, 1, seed=3)
    np.testing.assert_array_equal(a.rgb, b.rgb)
    np.testing.assert_array_equal(a.xyz, b.xyz)
    assert a.rgb.dtype == np.uint8 and a.xyz.dtype == np.float32
    changed_rgb, changed_xyz = not np.array_equal(a.rgb, p.rgb), not np.array_equal(a.xyz, p.xyz)
    assert (changed_rgb, changed_xyz) == (
        (True, False) if C.CORRUPTIONS[name].sensor == "rgb" else (False, True)
    )
    np.testing.assert_array_equal(a.mask, p.mask)
    assert a.meta["settings"]["corruption"]["name"] == name
    assert a.meta["split"] != p.meta["split"]  # cached features must not collide


def test_randomness_follows_the_stem_not_the_order():
    p = prepared(3)
    full = C.corrupt(p, "gaussian_noise", 2)
    one = C.corrupt(
        Prepared(
            p.rgb[2:], p.xyz[2:], p.foreground[2:], p.mask[2:], p.defect[2:], p.stem[2:], p.meta
        ),
        "gaussian_noise",
        2,
    )
    np.testing.assert_array_equal(full.rgb[2], one.rgb[0])
    assert not np.array_equal(full.rgb[0], full.rgb[1])


def test_same_stem_in_two_defect_folders_gets_different_noise():
    p = prepared(2)
    p.stem[1] = p.stem[0]
    p.defect[1] = "crack"
    c = C.corrupt(p, "gaussian_noise", 1)
    assert not np.array_equal(c.rgb[0].astype(int) - p.rgb[0], c.rgb[1].astype(int) - p.rgb[1])


@pytest.mark.parametrize(
    "name, measure",
    [
        ("gaussian_noise", lambda p, c: np.abs(c.rgb.astype(int) - p.rgb).mean()),
        ("defocus_blur", lambda p, c: np.abs(c.rgb.astype(int) - p.rgb).mean()),
        ("brightness", lambda p, c: c.rgb.mean()),
        ("jpeg", lambda p, c: np.abs(c.rgb.astype(int) - p.rgb).mean()),
        ("occlusion", lambda p, c: (c.rgb == 128).all(-1).mean()),
        ("depth_noise", lambda p, c: np.abs(c.xyz[..., 2] - p.xyz[..., 2]).mean()),
        ("holes", lambda p, c: 1 - valid_points(c.xyz).sum() / valid_points(p.xyz).sum()),
        ("quantise", lambda p, c: np.abs(c.xyz[..., 2] - p.xyz[..., 2]).mean()),
        ("tilt", lambda p, c: np.abs(c.xyz - p.xyz).mean()),
        ("misregister", lambda p, c: np.abs(c.xyz - p.xyz).mean()),
    ],
)
def test_severity_makes_it_worse(name, measure):
    p = prepared()
    values = [measure(p, C.corrupt(p, name, s)) for s in range(1, 6)]
    assert values[-1] > values[0]
    assert np.all(np.diff(values) >= -1e-6 * max(values)), values


def test_holes_remove_the_target_share_and_update_the_foreground():
    p = prepared()
    c = C.corrupt(p, "holes", 3)
    share = 1 - valid_points(c.xyz).sum() / valid_points(p.xyz).sum()
    assert C.HOLES[2] <= share < C.HOLES[2] + 0.15
    np.testing.assert_array_equal(c.foreground, valid_points(c.xyz))


def test_tilt_is_rigid():
    p = prepared(1)
    c = C.corrupt(p, "tilt", 5)
    v = valid_points(p.xyz[0])
    a, b = p.xyz[0][v].astype(float), c.xyz[0][v].astype(float)
    i = np.random.default_rng(0).integers(len(a), size=(200, 2))
    d0 = np.linalg.norm(a[i[:, 0]] - a[i[:, 1]], axis=1)
    d1 = np.linalg.norm(b[i[:, 0]] - b[i[:, 1]], axis=1)
    np.testing.assert_allclose(d0, d1, atol=1e-6)
    np.testing.assert_allclose(a.mean(0), b.mean(0), atol=1e-6)


def test_quantise_keeps_every_point_on_the_grid():
    p = prepared(1)
    c = C.corrupt(p, "quantise", 2)
    z = c.xyz[0][valid_points(c.xyz[0]), 2]
    step = C.DEPTH_STEP_M[1]
    np.testing.assert_allclose(np.round(z / step) * step, z, atol=1e-6)
    assert valid_points(c.xyz).sum() == valid_points(p.xyz).sum()


def test_misregistration_is_a_pure_shift():
    p = prepared(1)
    c = C.corrupt(p, "misregister", 4)
    d = C.SHIFT_PX[3]
    shifts = [(dy, dx) for dy in range(-d, d + 1) for dx in range(-d, d + 1)]
    assert any(np.array_equal(np.roll(p.xyz[0], s, axis=(0, 1)), c.xyz[0]) for s in shifts)


def test_missing_sensors_and_bad_severities():
    p = prepared()
    assert C.corrupt(p, "rgb_missing", 1).rgb.max() == 0
    assert not valid_points(C.corrupt(p, "xyz_missing", 1).xyz).any()
    with pytest.raises(ValueError):
        C.corrupt(p, "rgb_missing", 2)
    with pytest.raises(ValueError):
        C.corrupt(p, "jpeg", 6)
