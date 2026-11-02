import numpy as np
import pytest
import torch
from test_fpfh import fake_fpfh_map
from test_sweep import TINY

from factorylens import conformal, corruptions
from factorylens import inspector as insp
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
    assert out.anomaly_map.shape == out.map_rgb.shape == out.map_3d.shape == (32, 32)
    # with both weights at 1 the fused map is the two sensor maps added, Stouffer-scaled
    np.testing.assert_allclose(out.anomaly_map, (out.map_rgb + out.map_3d) / np.sqrt(2), rtol=1e-5)
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
    results = [inspector.inspect(*part) for part in parts(dead)]
    assert results[-1].weight_rgb == 0
    assert results[-1].sensor_alert == "rgb" and results[-1].as_dict()["sensor_alert"] == "rgb"
    status = inspector.status()
    assert status["weight_rgb"] == 0 and status["parts_in_window"] == 3
    assert status["sensor_alert"] == "rgb"
    # its score isn't on the calibration scale, so it can't be confirmed as good
    with pytest.raises(ValueError, match="down-weighted"):
        inspector.confirm_good(results[-1].part)
    inspector.reset()
    assert inspector.status()["parts_in_window"] == 0
    clean = inspector.inspect(*parts(splits["validation"])[0])
    assert clean.weight_rgb == 1 and clean.sensor_alert is None


def test_both_sensors_off_is_an_alert_too(inspector, splits):
    both = corruptions.corrupt(
        corruptions.corrupt(splits["test"], "rgb_missing", 1), "xyz_missing", 1
    )
    results = [inspector.inspect(*part) for part in parts(both)]
    # nothing is left to decide with, so both keep full weight, but a person has to look
    assert results[-1].weight_rgb == results[-1].weight_3d == 1
    assert results[-1].sensor_alert == "rgb+3d"
    assert "both_off" not in results[-1].as_dict()
    assert inspector.status()["sensor_alert"] == "rgb+3d"


def test_confirmed_good_parts_join_the_calibration_set(inspector, splits):
    n = len(inspector.calibration)
    out = inspector.inspect(*parts(splits["validation"])[0])
    assert out.part == 1 and out.as_dict()["part"] == 1
    inspector.confirm_good(out.part)
    assert len(inspector.calibration) == n + 1 and inspector.calibration[-1] == out.score
    assert inspector.meta["confirmed"] == 1
    assert inspector.threshold == conformal.threshold(inspector.calibration, CONFIG.alpha)
    with pytest.raises(KeyError):
        inspector.confirm_good(out.part)


def test_the_alarm_rate_is_judged_after_enough_parts(inspector, splits):
    good = parts(splits["validation"]) * insp.RECENT
    for part in good[: insp.MIN_FOR_RATE - 1]:
        inspector.inspect(*part)
    assert "alarm_rate" not in inspector.status()
    inspector.threshold = -np.inf  # every part alarms from here, as if the line had drifted
    for part in good[: insp.RECENT]:
        inspector.inspect(*part)
    status = inspector.status()
    lo, hi = status["alarm_rate_expected"]
    n = len(inspector.calibration)
    assert (lo, hi) == conformal.alarm_rate_range(n, CONFIG.alpha, insp.RECENT) and hi < 1
    assert status["alarm_rate"] == 1 and status["alarm_rate_high"]


def test_raw_scans_are_preprocessed_first(inspector, root):
    sample = mvtec3d.list_samples(root, "bagel", "test")[0]
    out = inspector.inspect_raw(
        mvtec3d.load_rgb(sample.rgb_path), mvtec3d.load_xyz(sample.xyz_path)
    )
    assert np.isfinite(out.score)
    t = inspector.timings
    assert list(t) == ["preprocess", "backbone", "rgb_bank", "fpfh", "fpfh_bank", "fusion", "total"]
    assert all(v >= 0 for v in t.values())
    assert inspector.status()["ms_per_part"] == pytest.approx(1000 * t["total"])


def test_old_model_files_are_refused(inspector, tmp_path):
    inspector.save(tmp_path / "model.pt")
    state = torch.load(tmp_path / "model.pt", weights_only=False)
    state["format"] = 99
    torch.save(state, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="format"):
        Inspector.load(tmp_path / "old.pt")


def test_half_precision_banks_are_smaller_and_still_load(inspector, splits, tmp_path):
    # how close the scores stay is measured on real data (docs/inference.md); the fake
    # images here are nearly identical, so their distances sit at float16's resolution
    inspector.save(tmp_path / "full.pt")
    inspector.save(tmp_path / "half.pt", half=True)
    assert (tmp_path / "half.pt").stat().st_size < (tmp_path / "full.pt").stat().st_size
    half = Inspector.load(tmp_path / "half.pt", device="cpu")
    assert half.rgb_bank.bank.dtype == torch.float32
    assert np.isfinite(half.inspect(*parts(splits["test"])[0]).score_rgb)


@pytest.mark.parametrize("fill", [0.0, np.nan, np.inf])
def test_a_dead_3d_sensor_on_a_raw_scan_is_judged_not_a_crash(inspector, root, fill):
    sample = mvtec3d.list_samples(root, "bagel", "test")[0]
    rgb, xyz = mvtec3d.load_rgb(sample.rgb_path), mvtec3d.load_xyz(sample.xyz_path)
    for _ in range(3):
        out = inspector.inspect_raw(rgb, np.full_like(xyz, fill))
    # on the 3D side only: the fixture's random backbone can make RGB look off as well
    assert np.isfinite(out.score) and out.health_3d > 0 and "3d" in out.sensor_alert


def test_both_sensors_dead_on_raw_scans_is_an_alert(inspector, root):
    sample = mvtec3d.list_samples(root, "bagel", "test")[0]
    rgb, xyz = mvtec3d.load_rgb(sample.rgb_path), mvtec3d.load_xyz(sample.xyz_path)
    for _ in range(3):
        out = inspector.inspect_raw(np.zeros_like(rgb), np.zeros_like(xyz))
    assert np.isfinite(out.score) and "3d" in out.sensor_alert


def test_nan_marks_a_missing_point_like_zero_does(inspector, root):
    sample = mvtec3d.list_samples(root, "bagel", "test")[0]
    rgb, xyz = mvtec3d.load_rgb(sample.rgb_path), mvtec3d.load_xyz(sample.xyz_path)
    with_zeros = xyz.copy()
    with_zeros[:, ::7] = 0
    with_nan = xyz.copy()
    with_nan[:, ::7] = np.nan
    a = inspector.inspect_raw(rgb, with_zeros)
    inspector.reset()
    b = inspector.inspect_raw(rgb, with_nan)
    assert a.score == pytest.approx(b.score)


@pytest.mark.parametrize(
    "rgb, xyz, message",
    [
        (np.zeros((8, 8), np.uint8), np.zeros((8, 8, 3), np.float32), "rgb must be"),
        (np.zeros((8, 8, 3)), np.zeros((8, 8, 3), np.float32), "rgb must be"),
        (np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.float32), "xyz must be"),
        (np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.int16), "floats in metres"),
        (np.zeros((8, 8, 3), np.uint8), np.zeros((9, 8, 3), np.float32), "differ in size"),
    ],
)
def test_what_cannot_be_a_scan_is_refused(inspector, rgb, xyz, message):
    with pytest.raises(ValueError, match=message):
        inspector.inspect_raw(rgb, xyz)


def test_a_preprocessed_part_of_the_wrong_size_is_refused(inspector, splits):
    rgb, xyz, fg = parts(splits["test"])[0]
    with pytest.raises(ValueError, match="expected parts"):
        inspector.inspect(rgb[:10], xyz[:10], fg[:10])


def test_a_file_that_is_not_a_model_is_refused(tmp_path):
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="not a FactoryLens model"):
        Inspector.load(garbage)
    other = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(1)}, other)
    with pytest.raises(ValueError, match="not a FactoryLens model"):
        Inspector.load(other)
