import numpy as np

from factorylens import cli, experiments
from factorylens.models.patchcore import PatchCoreConfig
from factorylens.repro import load_result

TINY = PatchCoreConfig(arch="resnet18", embed_dim=32)


def run(root, tmp_path, **kwargs):
    return experiments.run_rgb(
        root,
        tmp_path / "cache",
        "bagel",
        model_config=TINY,
        results_dir=tmp_path / "results",
        scores_dir=tmp_path / "scores",
        device="cpu",
        pretrained=False,
        **kwargs,
    )


def test_run_rgb_writes_result_and_scores(root, tmp_path):
    result = run(root, tmp_path)
    assert 0 <= result["image_auroc"] <= 1 and 0 <= result["aupro"] <= 1
    assert result["image_auroc_ci_low"] <= result["image_auroc"] <= result["image_auroc_ci_high"]
    assert result["n_train"] == 3 and result["n_test"] == 3

    [json_file] = (tmp_path / "results").glob("bagel_kall_s0_*.json")
    record = load_result(json_file)
    assert record["config"]["model"] == "patchcore_rgb"
    assert record["config"]["layers"] == ["layer2", "layer3"]
    assert record["metrics"]["aupro"] == result["aupro"]

    scores = np.load(tmp_path / "scores" / json_file.with_suffix(".npz").name)
    assert scores["scores"].shape == (3,) and scores["maps"].shape == (3, 224, 224)
    assert scores["labels"].tolist() == [True, False, False]


def test_kshot_run_uses_the_subset(root, tmp_path):
    import json

    kshot = tmp_path / "kshot.json"
    kshot.write_text(json.dumps({"bagel": {"1": {"1": ["002"]}}}))
    result = run(root, tmp_path, k=1, seed=1, kshot_path=kshot)
    assert result["n_train"] == 1
    assert list((tmp_path / "results").glob("bagel_k1_s1_*.json"))


def test_results_table(root, tmp_path, capsys):
    run(root, tmp_path)
    df = experiments.collect(tmp_path / "results")
    assert df.loc[0, "layers"] == "layer2+layer3" and df.loc[0, "category"] == "bagel"

    assert cli.main(["results", "table", str(tmp_path / "results"), "--by", "layers"]) == 0
    assert "layer2+layer3" in capsys.readouterr().out
    assert cli.main(["results", "table", str(tmp_path / "empty")]) == 1
