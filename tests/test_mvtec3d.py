import json
import shutil

import numpy as np
import pytest
import tifffile
from PIL import Image

from factorylens.data import mvtec3d

H, W = 6, 8


def write_sample(split_dir, defect, idx, with_gt, mask_value=0):
    d = split_dir / defect
    for sub in ("rgb", "xyz") + (("gt",) if with_gt else ()):
        (d / sub).mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((H, W, 3), 120, np.uint8)).save(d / "rgb" / f"{idx:03d}.png")

    xyz = np.full((H, W, 3), 0.5, np.float32)
    xyz[0, 0] = 0  # one pixel the sensor missed
    tifffile.imwrite(d / "xyz" / f"{idx:03d}.tiff", xyz, photometric="rgb")

    if with_gt:
        mask = np.zeros((H, W), np.uint8)
        mask[2:4, 2:4] = mask_value
        Image.fromarray(mask).save(d / "gt" / f"{idx:03d}.png")


@pytest.fixture
def root(tmp_path):
    """A tiny bagel category: 3 train, 1 validation, 2 good + 1 crack in test."""
    cat = tmp_path / "bagel"
    for i in range(3):
        write_sample(cat / "train", "good", i, with_gt=False)
    write_sample(cat / "validation", "good", 0, with_gt=False)
    for i in range(2):
        write_sample(cat / "test", "good", i, with_gt=True)
    write_sample(cat / "test", "crack", 0, with_gt=True, mask_value=254)
    (cat / "class_ids.json").write_text(json.dumps({"254": "crack", "0": "good"}))
    return tmp_path


FAKE_COUNTS = {"bagel": (3, 1, 3)}


def test_list_samples_pairs_files(root):
    test = mvtec3d.list_samples(root, "bagel", "test")
    assert [s.defect for s in test] == ["crack", "good", "good"]
    assert all(s.xyz_path.stem == s.rgb_path.stem for s in test)
    assert all(s.gt_path is not None for s in test)
    assert [s.is_anomalous for s in test] == [True, False, False]
    assert test[0].name == "bagel/test/crack/000"


def test_train_samples_have_no_mask(root):
    train = mvtec3d.list_samples(root, "bagel", "train")
    assert len(train) == 3
    assert all(s.gt_path is None for s in train)


def test_rejects_unknown_category_and_split(root):
    with pytest.raises(ValueError):
        mvtec3d.list_samples(root, "pretzel", "train")
    with pytest.raises(ValueError):
        mvtec3d.list_samples(root, "bagel", "val")


def test_loaders_return_expected_types(root):
    s = mvtec3d.list_samples(root, "bagel", "test")[0]
    rgb = mvtec3d.load_rgb(s.rgb_path)
    xyz = mvtec3d.load_xyz(s.xyz_path)
    mask = mvtec3d.load_mask(s.gt_path)

    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    assert xyz.shape == (H, W, 3) and xyz.dtype == np.float32
    assert mask.dtype == bool and mask.sum() == 4

    valid = mvtec3d.valid_points(xyz)
    assert not valid[0, 0] and valid.sum() == H * W - 1


def test_defect_names(root):
    names = mvtec3d.defect_names(root, "bagel")
    assert names[254] == "crack"
    assert names[0] == "good"


def test_check_passes_on_consistent_data(root):
    assert mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS, deep=True) == []


def test_check_reports_wrong_counts(root):
    problems = mvtec3d.check_category(root, "bagel", expected={"bagel": (4, 1, 3)})
    assert problems == ["bagel/train: 3 samples, expected 4"]


def test_check_reports_missing_xyz(root):
    (root / "bagel" / "train" / "good" / "xyz" / "001.tiff").unlink()
    problems = mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS)
    assert problems == ["bagel/train/good/001: no xyz file"]


def test_deep_check_catches_size_mismatch(root):
    path = root / "bagel" / "validation" / "good" / "xyz" / "000.tiff"
    tifffile.imwrite(path, np.ones((H + 1, W, 3), np.float32), photometric="rgb")
    problems = mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS, deep=True)
    assert len(problems) == 1
    assert "rgb (6, 8, 3) vs xyz (7, 8, 3)" in problems[0]


def test_deep_check_catches_empty_defect_mask(root):
    path = root / "bagel" / "test" / "crack" / "gt" / "000.png"
    Image.fromarray(np.zeros((H, W), np.uint8)).save(path)
    problems = mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS, deep=True)
    assert problems == ["bagel/test/crack/000: defect sample with an empty mask"]


def test_missing_split(root):
    shutil.rmtree(root / "bagel" / "validation")
    with pytest.raises(FileNotFoundError):
        mvtec3d.list_samples(root, "bagel", "validation")
    problems = mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS)
    assert problems == ["bagel/validation: missing"]


def test_check_flags_defects_in_training_data(root):
    write_sample(root / "bagel" / "train", "crack", 0, with_gt=False)
    problems = mvtec3d.check_category(root, "bagel", expected={"bagel": (4, 1, 3)})
    assert problems == ["bagel/train: contains non-good samples"]


def test_check_reports_missing_mask(root):
    (root / "bagel" / "test" / "good" / "gt" / "001.png").unlink()
    problems = mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS)
    assert problems == ["bagel/test/good/001: no gt mask"]


def test_deep_check_catches_empty_scan(root):
    path = root / "bagel" / "train" / "good" / "xyz" / "002.tiff"
    tifffile.imwrite(path, np.zeros((H, W, 3), np.float32), photometric="rgb")
    problems = mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS, deep=True)
    assert problems == ["bagel/train/good/002: xyz has no valid points"]


def test_deep_check_catches_mask_size_mismatch(root):
    path = root / "bagel" / "test" / "good" / "gt" / "000.png"
    Image.fromarray(np.zeros((H, W + 2), np.uint8)).save(path)
    problems = mvtec3d.check_category(root, "bagel", expected=FAKE_COUNTS, deep=True)
    assert problems == ["bagel/test/good/000: rgb (6, 8) vs gt (6, 10)"]
