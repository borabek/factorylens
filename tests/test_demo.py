import numpy as np
import pytest
from test_inspector import CONFIG, no_open3d  # noqa: F401  (fixture)

from factorylens import demo
from factorylens.data import cache
from factorylens.inspector import Inspector


@pytest.fixture
def setup(root):
    splits = {
        s: cache.prepare(root, "bagel", s, CONFIG.sweep.settings)
        for s in ("train", "validation", "test")
    }
    inspector = Inspector(CONFIG, device="cpu", pretrained=False).fit(
        splits["train"], splits["validation"]
    )
    return inspector, splits["test"], splits["validation"]


def test_images_have_the_right_shape():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    xyz = np.zeros((16, 16, 3), np.float32)
    xyz[4:12, 4:12] = [0.0, 0.0, 0.3]
    assert demo.shade(xyz).shape == (16, 16, 3) and demo.shade(xyz)[0, 0, 0] == 255
    hot = demo.overlay(rgb, np.full((16, 16), 10.0))
    cold = demo.overlay(rgb, np.full((16, 16), -1.0))
    assert hot.dtype == np.uint8 and not np.array_equal(hot, rgb)
    np.testing.assert_array_equal(cold, rgb)  # nothing above normal: the image alone


def test_one_part_with_and_without_a_broken_sensor(setup):
    inspector, test, validation = setup
    clean = demo.inspect_one(inspector, test, validation, 0, demo.NONE, 3)
    assert clean.fused.shape == (32, 32, 3) and "Ground truth" in clean.text
    broken = demo.inspect_one(inspector, test, validation, 0, "rgb_missing", 1)
    assert broken.rgb.max() == 0
    assert broken.result.score_rgb != clean.result.score_rgb
    # the window was filled with blank images too, so the camera is switched off
    assert broken.result.weight_rgb == 0


def test_a_stream_with_a_fault_half_way(setup):
    inspector, test, _ = setup
    s = demo.run_stream(inspector, test, "xyz_missing", 1)
    assert len(s.rows) == len(test)
    assert sum(r["after"] for r in s.rows) == len(test) - round(0.5 * len(test))
    # broken from the first part: once the 3-part window is full of missing scans, 3D is off
    s = demo.run_stream(inspector, test, "xyz_missing", 1, onset=0.0)
    assert s.rows[-1]["weight_3d"] == 0 and s.rows[-1]["weight_rgb"] == 1


def test_a_warm_window_keeps_the_first_parts_steady(setup):
    inspector, test, validation = setup
    s = demo.run_stream(inspector, test, "xyz_missing", 1, onset=1.0, validation=validation)
    assert all(r["weight_rgb"] == 1 and r["weight_3d"] == 1 for r in s.rows)
    assert "reliability fusion" in s.text
    assert s.figure is not None


def test_the_app_builds(setup, tmp_path, root):
    pytest.importorskip("gradio")
    inspector = setup[0]
    inspector.save(tmp_path / "bagel.pt")
    app = demo.build_app(tmp_path, root, tmp_path / "cache")
    assert app is not None
    with pytest.raises(FileNotFoundError):
        demo.build_app(tmp_path / "empty", root, tmp_path / "cache")
