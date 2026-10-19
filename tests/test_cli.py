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
