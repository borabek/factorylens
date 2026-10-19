import numpy as np
import pytest

from factorylens.data import preprocess as pp


def tilted_scene(size=60, bump_height=0.02, noise=0.0, seed=0):
    """A tilted table at ~0.55 m with a square box in the middle and a few missing points."""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:size, 0:size].astype(np.float32)
    x = (u - size / 2) * 0.001
    y = (v - size / 2) * 0.001
    z = 0.55 + 0.2 * x + 0.1 * y + rng.normal(0, noise, (size, size))
    box = (abs(u - size / 2) < 8) & (abs(v - size / 2) < 8)
    z = np.where(box, z - bump_height, z)  # closer to the camera
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    xyz[5, 5] = 0
    return xyz, box


def test_resize_xyz_never_invents_points():
    xyz, _ = tilted_scene()
    small = pp.resize_xyz(xyz, 17)
    assert small.shape == (17, 17, 3)
    original = {tuple(p) for p in xyz.reshape(-1, 3)}
    assert all(tuple(p) in original for p in small.reshape(-1, 3))


def test_resize_xyz_keeps_non_square_layout():
    xyz = np.zeros((40, 90, 3), np.float32)
    xyz[:, 45:] = 1  # right half valid
    small = pp.resize_xyz(xyz, 10)
    assert (small[:, :5] == 0).all() and (small[:, 5:] == 1).all()


def test_resize_rgb_and_mask_shapes():
    rgb = np.zeros((40, 90, 3), np.uint8)
    mask = np.zeros((40, 90), bool)
    mask[10:20, 10:20] = True
    assert pp.resize_rgb(rgb, 32).shape == (32, 32, 3)
    small = pp.resize_mask(mask, 32)
    assert small.dtype == bool and small.shape == (32, 32) and small.any()


def test_fit_plane_recovers_tilt_despite_outliers():
    xyz, box = tilted_scene(noise=0.0002)
    plane = pp.fit_plane(xyz[~box & pp.valid_points(xyz)])
    expected = np.array([0.2, 0.1, -1.0])
    expected /= np.linalg.norm(expected)
    assert abs(abs(plane.normal @ expected) - 1) < 1e-4


def test_fit_plane_needs_three_points():
    with pytest.raises(ValueError):
        pp.fit_plane(np.zeros((2, 3)))


def test_remove_background_keeps_only_the_object():
    xyz, box = tilted_scene(noise=0.0005)
    cleaned, fg, _ = pp.remove_background(xyz, threshold=0.005, border=5)
    assert fg[box].mean() > 0.99
    assert fg[~box].mean() < 0.01
    assert (cleaned[~fg] == 0).all()
    assert np.array_equal(cleaned[fg], xyz[fg])


def test_remove_background_is_deterministic():
    xyz, _ = tilted_scene(noise=0.001)
    a = pp.remove_background(xyz, border=5, seed=3)[1]
    b = pp.remove_background(xyz, border=5, seed=3)[1]
    assert np.array_equal(a, b)


def test_to_point_cloud_skips_missing_points():
    xyz, _ = tilted_scene()
    points, pixels = pp.to_point_cloud(xyz)
    assert len(points) == 60 * 60 - 1
    assert [5, 5] not in pixels.tolist()
    r, c = pixels[100]
    assert np.array_equal(points[100], xyz[r, c])


def test_drop_small_components_removes_specks_only():
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:40] = True  # object, 900 px
    mask[70:80, 70:75] = True  # second real part, 50 px
    mask[95, 95] = True  # speck
    cleaned = pp.drop_small_components(mask, min_fraction=0.002)  # 20 px
    assert cleaned[10:40, 10:40].all() and cleaned[70:80, 70:75].all()
    assert not cleaned[95, 95]


def test_keep_largest_component_drops_everything_else():
    mask = np.zeros((100, 100), bool)
    mask[10:40, 10:40] = True
    mask[70:80, 70:75] = True
    largest = pp.keep_largest_component(mask)
    assert largest.sum() == 900


def test_component_helpers_on_empty_mask():
    empty = np.zeros((5, 5), bool)
    assert not pp.drop_small_components(empty).any()
    assert not pp.keep_largest_component(empty).any()


def test_remove_background_drops_floating_specks():
    xyz, box = tilted_scene(noise=0.0005)
    xyz[2, 50, 2] -= 0.05  # a single point floating 5 cm above the table
    _, with_cleaning, _ = pp.remove_background(xyz, border=5, min_component=0.001)
    _, without, _ = pp.remove_background(xyz, border=5, min_component=0)
    assert without[2, 50] and not with_cleaning[2, 50]
    assert with_cleaning[box].mean() > 0.99
