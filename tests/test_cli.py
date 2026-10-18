import pytest
from fakes import FAKE_COUNTS

from factorylens import cli
from factorylens.data import mvtec3d


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
