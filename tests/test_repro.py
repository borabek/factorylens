import json
import math
import random

import numpy as np
import pytest

from factorylens import repro


def test_seed_everything_repeats_draws():
    repro.seed_everything(7)
    a = (random.random(), np.random.rand(3))
    repro.seed_everything(7)
    b = (random.random(), np.random.rand(3))
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])


def test_config_hash_ignores_key_order():
    assert repro.config_hash({"k": 5, "backbone": "wrn50"}) == repro.config_hash(
        {"backbone": "wrn50", "k": 5}
    )


def test_config_hash_changes_with_values():
    assert repro.config_hash({"k": 5}) != repro.config_hash({"k": 10})
    assert len(repro.config_hash({"k": 5})) == 12


def test_save_and_load_round_trip(tmp_path):
    config = {"category": "bagel", "k": 10, "seed": 0}
    path = repro.save_result(tmp_path / "runs" / "bagel.json", {"image_auroc": 0.91}, config)

    record = repro.load_result(path)
    assert record["metrics"] == {"image_auroc": 0.91}
    assert record["config"] == config
    assert record["config_hash"] == repro.config_hash(config)
    assert set(record["run"]) == {"time", "git", "python", "platform", "packages"}
    assert "numpy" in record["run"]["packages"]


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_save_refuses_non_finite_metrics(tmp_path, bad):
    with pytest.raises(ValueError):
        repro.save_result(tmp_path / "r.json", {"image_auroc": bad}, {})
    assert not (tmp_path / "r.json").exists()


def test_git_state_outside_a_repo(tmp_path):
    assert repro.git_state(tmp_path) == {"commit": None, "dirty": None}


def test_git_state_inside_this_repo():
    state = repro.git_state()
    if state["commit"] is None:
        pytest.skip("not running from a git checkout")
    assert len(state["commit"]) == 40
    assert isinstance(state["dirty"], bool)


def test_result_file_is_plain_json(tmp_path):
    path = repro.save_result(tmp_path / "r.json", {"n": 3}, {"a": [1, 2]})
    assert json.loads(path.read_text())["config"]["a"] == [1, 2]


def test_run_info_captured_early_is_what_gets_saved(tmp_path):
    early = repro.run_info()
    early["git"] = {"commit": "a" * 40, "dirty": False}  # pretend the checkout changed since
    path = repro.save_result(tmp_path / "r.json", {"x": 1}, {}, run=early)
    assert repro.load_result(path)["run"]["git"]["commit"] == "a" * 40


def test_run_info_reads_git_from_the_package_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a git repo
    from_package = repro.run_info()["git"]
    if from_package["commit"] is None:
        pytest.skip("package not imported from a git checkout")
    assert len(from_package["commit"]) == 40
