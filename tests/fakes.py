"""A tiny fake MVTec 3D-AD category written to a temp dir, so tests never need the real data."""

import json

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from factorylens.data import ff

H, W = 6, 8


def write_sample(split_dir, defect, idx, with_gt, mask_value=0):
    d = split_dir / defect
    for sub in ("rgb", "xyz") + (("gt",) if with_gt else ()):
        (d / sub).mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.full((H, W, 3), 120, np.uint8)).save(d / "rgb" / f"{idx:03d}.png")

    # a flat table 0.55 m away with a 1 cm box on it, so plane fitting has something to do
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    z = np.full((H, W), 0.55, np.float32)
    z[2:4, 2:5] -= 0.01
    xyz = np.stack([u * 0.001, v * 0.001, z], axis=-1)
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


# Future Factories


def make_ff_frame():
    """Five cycles at 1 Hz: counter 1, 2, a 3 s reset stub (0), then 1, 2 again on day 2."""
    pieces = [
        ("2023-12-11T10:00:00Z", 1, 100, ""),
        ("2023-12-11T10:01:40Z", 2, 100, "NoNose"),
        ("2023-12-11T23:59:57Z", 0, 3, ""),
        ("2023-12-12T08:00:00Z", 1, 100, ""),
        ("2023-12-12T08:01:40Z", 2, 100, "NoNose"),
    ]
    rows = []
    for start, counter, n, desc in pieces:
        for t in pd.date_range(start, periods=n, freq="1s"):
            rows.append({"time": t, "Q_Cell_CycleCount": counter, "label": desc or ff.NORMAL})
    return pd.DataFrame(rows).set_index("time")


def write_ff_csvs(tmp_path, df):
    times = df.index.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3] + "Z"
    for i, name in enumerate(ff.FILES):
        out = pd.DataFrame({"_time": times})
        if name == ff.LABEL_FILE:
            out["Q_Cell_CycleCount"] = df["Q_Cell_CycleCount"].values
            out["Description"] = df["label"].replace(ff.NORMAL, "").values
        else:
            out[f"signal_{i}"] = 68.0
            out["Description"] = ""
        out.to_csv(tmp_path / name, index=False)
