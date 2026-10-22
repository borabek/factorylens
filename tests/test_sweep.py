import json

import numpy as np
import pytest
from test_fpfh import fake_fpfh_map

from factorylens import sweep
from factorylens.data import cache
from factorylens.models import fpfh
from factorylens.models.patchcore import PatchCoreConfig

TINY = sweep.SweepConfig(
    rgb=PatchCoreConfig(arch="resnet18", embed_dim=16, grid=4),
    fpfh=fpfh.FPFHConfig(grid=4),
    settings=cache.Settings(size=32),
)


@pytest.fixture(autouse=True)
def no_open3d(monkeypatch):
    monkeypatch.setattr(fpfh, "fpfh_map", fake_fpfh_map)


@pytest.fixture
def kshot(tmp_path):
    path = tmp_path / "kshot.json"
    path.write_text(json.dumps({"bagel": {"0": {"1": ["001"], "2": ["000", "002"]}}}))
    return path


def run(root, tmp_path, kshot, **kwargs):
    return sweep.run_category(
        root,
        tmp_path / "cache",
        "bagel",
        config=TINY,
        kshot_path=kshot,
        results_dir=tmp_path / "results",
        scores_dir=tmp_path / "scores",
        feature_cache=None,
        device="cpu",
        pretrained=False,
        log=lambda line: None,
        **kwargs,
    )


def test_every_model_gets_a_result_per_k_and_seed(root, tmp_path, kshot):
    rows = run(root, tmp_path, kshot, ks=[1, 2, None], seeds=[0])
    assert {(r["model"], r["k"]) for r in rows} == {
        (m, k) for m in sweep.MODELS for k in (1, 2, None)
    }
    for model in sweep.MODELS:
        assert len(list((tmp_path / "results" / model).glob("*.json"))) == 3
        # scores are kept only for the full training set
        assert len(list((tmp_path / "scores" / model).glob("*.npz"))) == 1
    one = [r for r in rows if r["model"] == "rgb" and r["k"] == 1][0]
    assert one["n_train"] == 1 and one["n_test"] == 3


def test_early_fusion_records_its_weight(root, tmp_path, kshot):
    run(root, tmp_path, kshot, ks=[2], seeds=[0], models=("early",))
    [path] = (tmp_path / "results" / "early").glob("*.json")
    weight = json.loads(path.read_text())["config"]["fpfh_weight_fitted"]
    assert weight > 0


def test_augmented_rgb_only_for_small_k(root, tmp_path, kshot):
    rows = run(root, tmp_path, kshot, ks=[1, None], seeds=[0], models=("rgb", "rgb_aug"))
    aug = [r for r in rows if r["model"] == "rgb_aug"]
    assert len(aug) == 1 and aug[0]["k"] == 1 and aug[0]["bank_size"] == 8 * 4 * 4


def test_dihedral_gives_eight_distinct_views():
    img = np.arange(2 * 3 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3, 3)
    views = sweep.dihedral(img)
    assert views.shape == (16, 3, 3, 3)
    # layout is [rot_k(all), flipped rot_k(all)] for k = 0..3, so image 0 sits at even indices
    assert len({v.tobytes() for v in views[0::2]}) == 8


def test_standardise_uses_the_validation_scores():
    val = (np.array([1.0, 3.0]), np.array([[[0.0, 2.0]]]))
    test = (np.array([2.0, 5.0]), np.array([[[1.0, 3.0]]]))
    s, m = sweep.standardise(test, val)
    np.testing.assert_allclose(s, [0.0, 3.0])  # val mean 2, std 1
    np.testing.assert_allclose(m, [[[0.0, 2.0]]])
