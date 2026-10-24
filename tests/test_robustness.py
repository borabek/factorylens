import json

import numpy as np
import pytest
from test_fpfh import fake_fpfh_map
from test_sweep import TINY

from factorylens import robustness
from factorylens.models import fpfh


@pytest.fixture(autouse=True)
def no_open3d(monkeypatch):
    monkeypatch.setattr(fpfh, "fpfh_map", fake_fpfh_map)


def run(root, tmp_path, conditions):
    return robustness.run_category(
        root,
        tmp_path / "cache",
        "bagel",
        conditions=conditions,
        config=TINY,
        results_dir=tmp_path / "results",
        scores_dir=tmp_path / "scores",
        feature_cache=tmp_path / "features",
        device="cpu",
        pretrained=False,
        log=lambda line: None,
    )


def test_clean_plus_each_condition_for_every_model(root, tmp_path):
    rows = run(root, tmp_path, [("gaussian_noise", 2), ("holes", 1)])
    assert {(r["model"], r["corruption"]) for r in rows} == {
        (m, c) for m in robustness.ALL_MODELS for c in ("clean", "gaussian_noise", "holes")
    }
    for model in robustness.ALL_MODELS:
        assert len(list((tmp_path / "results" / model).glob("*.json"))) == 3
    assert len(list((tmp_path / "scores").glob("*.npz"))) == 3
    [path] = (tmp_path / "results" / "early").glob("*holes1*.json")
    config = json.loads(path.read_text())["config"]
    assert config["severity"] == 1 and config["fpfh_weight_fitted"] > 0


def test_a_corrupted_sensor_leaves_the_other_model_alone(root, tmp_path):
    rows = run(root, tmp_path, [("rgb_missing", 1), ("xyz_missing", 1)])
    auroc = {(r["model"], r["corruption"]): r["image_auroc"] for r in rows}
    assert auroc[("fpfh", "rgb_missing")] == auroc[("fpfh", "clean")]
    assert auroc[("rgb", "xyz_missing")] == auroc[("rgb", "clean")]


def test_reliable_fusion_drops_a_missing_sensor(root, tmp_path):
    rows = run(root, tmp_path, [("xyz_missing", 1)])
    auroc = {(r["model"], r["corruption"]): r["image_auroc"] for r in rows}
    # with no points the 3D weight goes to 0, so the fusion ranks like RGB alone
    assert auroc[("reliable", "xyz_missing")] == auroc[("rgb", "xyz_missing")]
    [npz] = (tmp_path / "scores").glob("*xyz_missing*.npz")
    assert (np.load(npz)["w_fpfh"] == 0).all()


def test_models_can_be_chosen(root, tmp_path):
    rows = robustness.run_category(
        root,
        tmp_path / "cache",
        "bagel",
        conditions=[],
        models=("reliable",),
        config=TINY,
        results_dir=tmp_path / "results",
        scores_dir=None,
        feature_cache=None,
        device="cpu",
        pretrained=False,
        log=lambda line: None,
    )
    assert [r["model"] for r in rows] == ["reliable"]
    assert not (tmp_path / "results" / "early").exists()


def test_all_conditions_lists_every_severity():
    conditions = robustness.all_conditions()
    assert ("rgb_missing", 1) in conditions and ("rgb_missing", 2) not in conditions
    assert len(conditions) == 10 * 5 + 2
