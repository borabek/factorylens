import pytest
from fakes import FAKE_COUNTS, make_ff_frame, write_ff_csvs

from factorylens import cli
from factorylens.data import mvtec3d, subsets


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "factorylens" in capsys.readouterr().out


def test_data_check_ok(root, monkeypatch, capsys):
    monkeypatch.setattr(mvtec3d, "EXPECTED_COUNTS", FAKE_COUNTS)

    code = cli.main(["data", "check", str(root), "--category", "bagel", "--deep"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("bagel        3/1/3        ok")


def test_data_check_reports_problems(root, capsys):
    code = cli.main(["data", "check", str(root), "--category", "bagel"])

    out = capsys.readouterr().out
    assert code == 1
    assert "bagel/train: 3 samples, expected 244" in out


def test_unknown_category_is_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["data", "check", "--category", "pretzel"])
    assert exc.value.code == 2


def test_data_kshot_writes_file(root, tmp_path, capsys):
    out = tmp_path / "kshot.json"
    args = ["data", "kshot", str(root), "--out", str(out), "--category", "bagel"]
    args += ["--k", "1", "--k", "2"]

    assert cli.main(args) == 0
    written = subsets.load(out)
    assert set(written) == {"bagel"}
    assert len(written["bagel"]["4"]["2"]) == 2
    assert "k=(1, 2)" in capsys.readouterr().out


def test_ff_blocks(tmp_path, capsys):
    write_ff_csvs(tmp_path, make_ff_frame())

    assert cli.main(["ff", "blocks", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "train (day 1): 2 cycles" in out
    assert "test (day 2): 2 cycles" in out


def test_data_prepare(root, tmp_path, capsys):
    out = tmp_path / "processed"
    args = ["data", "prepare", str(root), "--cache", str(out), "--size", "4"]
    assert cli.main(args + ["--category", "bagel", "--jobs", "1"]) == 0
    assert "bagel" in capsys.readouterr().out
    assert len(list(out.glob("4px-*/bagel_*.npz"))) == 3


def test_run_fpfh_and_btf_from_the_cli(root, tmp_path, monkeypatch, capsys):
    from test_fpfh import fake_fpfh_map

    from factorylens.models import fpfh

    monkeypatch.setattr(fpfh, "fpfh_map", fake_fpfh_map)
    base = [str(root), "--cache", str(tmp_path / "c"), "--features", str(tmp_path / "f")]
    base += ["--category", "bagel", "--no-scores", "--device", "cpu"]
    assert cli.main(["run", "fpfh", *base, "--results", str(tmp_path / "r1")]) == 0
    assert "fpfh   bagel" in capsys.readouterr().out
    assert len(list((tmp_path / "r1").glob("*.json"))) == 1


def test_data_fpfh_precomputes_features(root, tmp_path, monkeypatch):
    from test_fpfh import fake_fpfh_map

    from factorylens.models import fpfh

    monkeypatch.setattr(fpfh, "fpfh_map", fake_fpfh_map)
    args = ["data", "fpfh", str(root), "--cache", str(tmp_path / "c")]
    args += ["--features", str(tmp_path / "f"), "--category", "bagel", "--jobs", "1"]
    monkeypatch.setattr(cli, "ProcessPoolExecutor", _InlinePool)
    assert cli.main(args) == 0
    assert len(list((tmp_path / "f").glob("*/bagel_*.pt"))) == 3


class _InlinePool:
    """Runs jobs in this process, so monkeypatches still apply."""

    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs):
        return map(fn, jobs)


def test_sweep_command(root, tmp_path, monkeypatch, capsys):
    from factorylens import sweep

    calls = []
    monkeypatch.setattr(sweep, "run_category", lambda *a, **kw: calls.append(kw))
    args = ["sweep", str(root), "--category", "bagel", "--k", "1", "--k", "all", "--seed", "3"]
    assert cli.main([*args, "--model", "late"]) == 0
    assert calls[0]["ks"] == [1, None] and calls[0]["seeds"] == [3]
    assert calls[0]["models"] == ("late",)


def test_robustness_command(root, monkeypatch):
    from factorylens import robustness

    calls = []
    monkeypatch.setattr(robustness, "run_category", lambda *a, **kw: calls.append((a, kw)))
    args = ["robustness", str(root), "--category", "bagel", "--corruption", "holes"]
    assert cli.main([*args, "--corruption", "rgb_missing", "--seed", "2"]) == 0
    ((a, kw),) = calls
    assert a[2] == "bagel" and kw["seed"] == 2
    assert kw["conditions"] == [("rgb_missing", 1), *[("holes", s) for s in range(1, 6)]]


def test_fit_and_inspect_commands(root, tmp_path, monkeypatch, capsys):
    from test_fpfh import fake_fpfh_map
    from test_inspector import CONFIG

    from factorylens import inspector as insp
    from factorylens.models import fpfh

    class Tiny(insp.Inspector):
        """The real thing with a small, randomly initialised backbone."""

        def __init__(self, config=None, device=None, pretrained=True):
            super().__init__(CONFIG, device="cpu", pretrained=False)

    monkeypatch.setattr(fpfh, "fpfh_map", fake_fpfh_map)
    monkeypatch.setattr(insp, "InspectorConfig", lambda **kw: CONFIG)
    monkeypatch.setattr(insp, "Inspector", Tiny)
    out = tmp_path / "bagel.pt"
    args = ["fit", str(root), "--category", "bagel", "--cache", str(tmp_path / "cache")]
    assert cli.main([*args, "--out", str(out), "--coreset-ratio", "0.5"]) == 0
    assert out.exists() and "alarm threshold" in capsys.readouterr().out

    s = mvtec3d.list_samples(root, "bagel", "test")[0]
    assert cli.main(["inspect", str(out), str(s.rgb_path), str(s.xyz_path)]) == 0
    line = capsys.readouterr().out.strip()
    assert '"alarm"' in line and str(s.rgb_path) in line
    assert cli.main(["inspect", str(out), str(s.rgb_path)]) == 2
