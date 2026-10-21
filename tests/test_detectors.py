from dataclasses import replace

import numpy as np
import pytest
import torch
from test_fpfh import fake_fpfh_map

from factorylens import experiments
from factorylens.data import cache
from factorylens.models import fpfh
from factorylens.models.detectors import (
    BTFDetector,
    FPFHDetector,
    FPFHDetectorConfig,
    RGBDetector,
)
from factorylens.models.memory_bank import BankConfig
from factorylens.models.patchcore import BTF_RGB, PatchCoreConfig

SETTINGS = cache.Settings(size=64)  # layer3 must survive a 3x3 unpadded average


@pytest.fixture
def split(root):
    return {s: cache.prepare(root, "bagel", s, SETTINGS) for s in ("train", "test")}


@pytest.fixture(autouse=True)
def no_open3d(monkeypatch):
    monkeypatch.setattr(fpfh, "fpfh_map", fake_fpfh_map)


def small_fpfh():
    return FPFHDetectorConfig(fpfh.FPFHConfig(grid=4), BankConfig(blur_sigma=0))


def test_rgb_detector(split):
    det = RGBDetector(
        PatchCoreConfig(arch="resnet18", embed_dim=16), device="cpu", pretrained=False
    )
    scores, maps = det.fit(split["train"]).score(split["test"])
    assert scores.shape == (3,) and maps.shape == (3, 64, 64)
    assert det.config_dict()["model"] == "patchcore_rgb" and det.bank_size > 0


def test_fpfh_detector(split, tmp_path):
    det = FPFHDetector(small_fpfh(), feature_cache=tmp_path)
    scores, maps = det.fit(split["train"]).score(split["test"])
    assert scores.shape == (3,) and maps.shape == (3, 64, 64)
    assert det.bank_size == 3 * 4 * 4
    assert det.config_dict()["fpfh"]["grid"] == 4


def test_btf_detector_concatenates_both_feature_sets(split):
    rgb = replace(BTF_RGB, arch="resnet18", grid=4, blur_sigma=0)
    det = BTFDetector(rgb, device="cpu", pretrained=False)
    patches = det.patches(split["train"])
    assert patches.shape == (3, 4, 4, 128 + 256 + fpfh.N_BINS)
    scores, maps = det.fit(split["train"]).score(split["test"])
    assert scores.shape == (3,) and maps.shape == (3, 64, 64)


def test_btf_detector_refuses_mismatched_grids(split):
    rgb = replace(BTF_RGB, arch="resnet18", grid=4)
    det = BTFDetector(rgb, features=fpfh.FPFHConfig(grid=3), device="cpu", pretrained=False)
    with pytest.raises(ValueError, match="grids differ"):
        det.fit(split["train"])


def test_generic_run_saves_a_result(root, tmp_path):
    det = FPFHDetector(small_fpfh(), feature_cache=tmp_path / "features")
    result = experiments.run(
        det,
        root,
        tmp_path / "cache",
        "bagel",
        settings=SETTINGS,
        results_dir=tmp_path / "results",
        scores_dir=tmp_path / "scores",
    )
    assert 0 <= result["aupro"] <= 1
    [path] = (tmp_path / "results").glob("bagel_kall_s0_*.json")
    assert '"model": "fpfh"' in path.read_text()
    assert np.load(next((tmp_path / "scores").glob("*.npz")))["maps"].shape == (3, 64, 64)


def test_btf_balanced_weight_equalises_the_two_halves(split):
    from factorylens.models.detectors import spread

    rgb = replace(BTF_RGB, arch="resnet18", grid=4, blur_sigma=0)
    det = BTFDetector(rgb, device="cpu", pretrained=False, fpfh_weight="balanced")
    det.fit(split["train"])
    assert det.fitted_weight is not None and det.fitted_weight > 0
    r, g = det.parts(split["train"])
    assert spread(g * det.fitted_weight) == pytest.approx(spread(r), rel=1e-4)
    assert det.config_dict()["fpfh_weight"] == "balanced"
    with pytest.raises(ValueError):
        BTFDetector(rgb, device="cpu", pretrained=False, fpfh_weight="huge")


def test_btf_fixed_weight_scales_only_the_fpfh_half(split):
    rgb = replace(BTF_RGB, arch="resnet18", grid=4, blur_sigma=0)
    torch.manual_seed(0)  # random-init backbones: same seed, same RGB half
    full = BTFDetector(rgb, device="cpu", pretrained=False).patches(split["train"])
    torch.manual_seed(0)
    tenth = BTFDetector(rgb, device="cpu", pretrained=False, fpfh_weight=0.1)
    tenth = tenth.patches(split["train"])
    n = fpfh.N_BINS
    torch.testing.assert_close(tenth[..., :-n], full[..., :-n])
    torch.testing.assert_close(tenth[..., -n:], full[..., -n:] * 0.1)
