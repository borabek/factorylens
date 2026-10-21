import numpy as np
import pytest
import torch

from factorylens.data import cache
from factorylens.models import fpfh


def scene(size=48, bump=True):
    """A flat patch of table with an optional dome in the middle, in metres."""
    v, u = np.mgrid[0:size, 0:size].astype(np.float32)
    x, y = (u - size / 2) * 0.001, (v - size / 2) * 0.001
    z = np.full_like(x, 0.55)
    if bump:
        z -= np.clip(0.008 - (x**2 + y**2) * 40, 0, None)
    xyz = np.stack([x, y, z], axis=-1)
    xyz[:3, :3] = 0  # a few missing points
    return xyz


def fake_fpfh_map(xyz, config=None):
    """Stand-in for open3d: one 'histogram' per point built from its depth."""
    out = np.repeat(xyz[..., 2:3] * 100, fpfh.N_BINS, axis=-1).astype(np.float32)
    out[~xyz.any(axis=-1)] = 0
    return out


def test_fpfh_tells_a_dome_from_a_flat_table():
    pytest.importorskip("open3d")
    cfg = fpfh.FPFHConfig(voxel_size=0.002)  # radii scaled to the 48 mm toy scene
    flat, dome = fpfh.fpfh_map(scene(bump=False), cfg), fpfh.fpfh_map(scene(), cfg)
    assert flat.shape == (48, 48, fpfh.N_BINS)
    assert not flat[:3, :3].any()  # no point, no feature
    sums = dome[5:, 5:].sum(axis=-1)
    assert np.allclose(sums, sums.flat[0], rtol=1e-4)  # fixed total per point
    centre = (slice(20, 28), slice(20, 28))
    assert np.abs(dome[centre] - flat[centre]).sum(axis=-1).mean() > 10


def test_fpfh_returns_zeros_for_an_almost_empty_scan():
    pytest.importorskip("open3d")
    xyz = np.zeros((10, 10, 3), np.float32)
    xyz[0, :5] = 0.5
    assert not fpfh.fpfh_map(xyz).any()


@pytest.mark.parametrize(("pad", "grid"), [(False, 28), (True, 28), (False, 7)])
def test_to_patches_lands_on_the_grid(pad, grid):
    maps = np.random.default_rng(0).random((2, 224, 224, fpfh.N_BINS), dtype=np.float32)
    out = fpfh.to_patches(maps, fpfh.FPFHConfig(pad=pad, grid=grid))
    assert out.shape == (2, grid, grid, fpfh.N_BINS) and out.dtype == torch.float32


def test_patches_are_cached_per_settings(root, tmp_path, monkeypatch):
    calls = []

    def counting(xyz, config=None):
        calls.append(1)
        return fake_fpfh_map(xyz, config)

    monkeypatch.setattr(fpfh, "fpfh_map", counting)
    prepared = cache.prepare(root, "bagel", "train", cache.Settings(size=16))
    cfg = fpfh.FPFHConfig(grid=4)
    first = fpfh.patches_for(prepared, cfg, tmp_path)
    assert len(calls) == 3 and first.shape == (3, 4, 4, fpfh.N_BINS)
    again = fpfh.patches_for(prepared, cfg, tmp_path)
    assert len(calls) == 3 and torch.equal(first, again)  # read back, not recomputed
    fpfh.patches_for(prepared, fpfh.FPFHConfig(grid=2), tmp_path)
    assert len(calls) == 6  # other settings, other cache entry
