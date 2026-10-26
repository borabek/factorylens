import numpy as np
import pytest
from test_fpfh import fake_fpfh_map
from test_sweep import TINY

from factorylens import corruptions
from factorylens.data import cache, mvtec3d
from factorylens.fusion import ReliabilityConfig
from factorylens.inspector import Inspector, InspectorConfig
from factorylens.models import fpfh

CONFIG = InspectorConfig(sweep=TINY, reliability=ReliabilityConfig(window=3), alpha=0.3)


@pytest.fixture(autouse=True)
def no_open3d(monkeypatch):
    monkeypatch.setattr(fpfh, "fpfh_map", fake_fpfh_map)


@pytest.fixture
def splits(root):
    return {
        s: cache.prepare(root, "bagel", s, TINY.settings) for s in ("train", "validation", "test")
    }


@pytest.fixture
def inspector(splits):
    return Inspector(CONFIG, device="cpu", pretrained=False).fit(
        splits["train"], splits["validation"]
    )


def parts(p):
    return [(p.rgb[i], p.xyz[i], p.foreground[i]) for i in range(len(p))]


def test_a_clean_part_gets_full_weights_and_a_valid_p_value(inspector, splits):
    out = inspector.inspect(*parts(splits["validation"])[0])
    assert out.weight_rgb == 1 and out.weight_3d == 1
    assert 0 < out.p_value <= 1
    assert out.anomaly_map.shape == (32, 32)
    assert out.alarm == (out.score > inspector.threshold)
    d = out.as_dict()
    assert "anomaly_map" not in d and isinstance(d["alarm"], bool)
    assert len(out.as_dict(with_map=True)["anomaly_map"]) == 32


def test_saving_and_loading_gives_the_same_answers(inspector, splits, tmp_path):
    inspector.save(tmp_path / "model.pt")
    loaded = Inspector.load(tmp_path / "model.pt", device="cpu")
    assert loaded.meta["category"] == "bagel"
    for part in parts(splits["test"]):
        a, b = inspector.inspect(*part), loaded.inspect(*part)
        assert a.score == pytest.approx(b.score) and a.alarm == b.alarm
        np.testing.assert_allclose(a.anomaly_map, b.anomaly_map, rtol=1e-5)


def test_a_dead_camera_is_switched_off_after_the_window_fills(inspector, splits):
    dead = corruptions.corrupt(splits["test"], "rgb_missing", 1)
    weights = [inspector.inspect(*part).weight_rgb for part in parts(dead)]
    assert weights[-1] == 0
    inspector.reset()
    assert inspector.inspect(*parts(splits["validation"])[0]).weight_rgb == 1


def test_raw_scans_are_preprocessed_first(inspector, root):
    sample = mvtec3d.list_samples(root, "bagel", "test")[0]
    out = inspector.inspect_raw(
        mvtec3d.load_rgb(sample.rgb_path), mvtec3d.load_xyz(sample.xyz_path)
    )
    assert np.isfinite(out.score)


def test_old_model_files_are_refused(inspector, tmp_path):
    import torch

    inspector.save(tmp_path / "model.pt")
    state = torch.load(tmp_path / "model.pt", weights_only=False)
    state["format"] = 99
    torch.save(state, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="format"):
        Inspector.load(tmp_path / "old.pt")
