"""A tiny fake MVTec 3D-AD category written to a temp dir, so tests never need the real data."""

import json

import numpy as np
import tifffile
from PIL import Image

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


def make_fake_root(tmp_path):
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
