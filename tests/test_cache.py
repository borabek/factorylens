import numpy as np
import pytest

from factorylens.data import cache

SMALL = cache.Settings(size=4)


def test_prepare_shapes_and_types(root):
    p = cache.prepare(root, "bagel", "test", SMALL)
    assert len(p) == 3
    assert p.rgb.shape == (3, 4, 4, 3) and p.rgb.dtype == np.uint8
    assert p.xyz.shape == (3, 4, 4, 3) and p.xyz.dtype == np.float32
    assert p.foreground.dtype == bool and p.mask.dtype == bool
    assert p.defect == ["crack", "good", "good"]
    assert p.is_anomalous.tolist() == [True, False, False]
    assert p.mask[0].any() and not p.mask[1:].any()
    # background is zeroed in xyz wherever foreground is False
    assert (p.xyz[~p.foreground] == 0).all()


def test_train_split_has_empty_masks(root):
    p = cache.prepare(root, "bagel", "train", SMALL)
    assert not p.mask.any()
    assert p.foreground.any()  # the 1 cm box survives the plane cut


def test_save_load_round_trip(root, tmp_path):
    p = cache.prepare(root, "bagel", "test", SMALL)
    path = tmp_path / "x.npz"
    cache.save(p, path)
    q = cache.load(path)
    for name in ("rgb", "xyz", "foreground", "mask"):
        np.testing.assert_array_equal(getattr(p, name), getattr(q, name))
    assert q.defect == p.defect and q.stem == p.stem
    assert q.meta["settings"]["size"] == 4
    assert not list(tmp_path.glob("*.tmp.npz"))


def test_get_uses_the_cache(root, tmp_path, monkeypatch):
    first = cache.get(root, tmp_path, "bagel", "validation", SMALL)
    assert cache.cache_path(tmp_path, "bagel", "validation", SMALL).exists()

    def fail(*args, **kwargs):
        raise AssertionError("should have been read from the cache")

    monkeypatch.setattr(cache, "prepare", fail)
    second = cache.get(root, tmp_path, "bagel", "validation", SMALL)
    np.testing.assert_array_equal(first.xyz, second.xyz)


def test_settings_change_the_cache_location(tmp_path):
    a = cache.cache_path(tmp_path, "bagel", "train", cache.Settings())
    b = cache.cache_path(tmp_path, "bagel", "train", cache.Settings(threshold=0.005))
    c = cache.cache_path(tmp_path, "bagel", "train", cache.Settings(size=256))
    assert len({a.parent, b.parent, c.parent}) == 3
    assert a.parent.name.startswith("224px-")


def test_take_selects_by_stem(root):
    p = cache.prepare(root, "bagel", "train", SMALL)
    sub = cache.take(p, ["002", "000"])
    assert sub.stem == ["000", "002"]
    np.testing.assert_array_equal(sub.rgb, p.rgb[[0, 2]])
    with pytest.raises(ValueError, match="not in this split"):
        cache.take(p, ["000", "123"])
