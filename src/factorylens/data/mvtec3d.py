"""Reading MVTec 3D-AD from disk.

Layout of one category:

    <category>/train/good/{rgb,xyz}/000.png, 000.tiff, ...
    <category>/validation/good/{rgb,xyz}/...
    <category>/test/<defect>/{rgb,xyz,gt}/...

The xyz maps are float32 (x, y, z) in metres, aligned pixel for pixel with the
RGB image. Pixels the sensor did not see are stored as (0, 0, 0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

CATEGORIES = (
    "bagel",
    "cable_gland",
    "carrot",
    "cookie",
    "dowel",
    "foam",
    "peach",
    "potato",
    "rope",
    "tire",
)
SPLITS = ("train", "validation", "test")

# (train, validation, test) as shipped. The totals match the paper: 2656 / 294 / 1197.
EXPECTED_COUNTS = {
    "bagel": (244, 22, 110),
    "cable_gland": (223, 23, 108),
    "carrot": (286, 29, 159),
    "cookie": (210, 22, 131),
    "dowel": (288, 34, 130),
    "foam": (236, 27, 100),
    "peach": (361, 42, 132),
    "potato": (300, 33, 114),
    "rope": (298, 33, 101),
    "tire": (210, 29, 112),
}


@dataclass(frozen=True)
class Sample:
    category: str
    split: str
    defect: str  # "good" for defect-free samples
    rgb_path: Path
    xyz_path: Path
    gt_path: Path | None  # only test samples have masks

    @property
    def is_anomalous(self) -> bool:
        return self.defect != "good"

    @property
    def name(self) -> str:
        return f"{self.category}/{self.split}/{self.defect}/{self.rgb_path.stem}"


def list_samples(root: str | Path, category: str, split: str) -> list[Sample]:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    split_dir = Path(root) / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(split_dir)

    samples = []
    for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for rgb_path in sorted((defect_dir / "rgb").glob("*.png")):
            xyz_path = defect_dir / "xyz" / f"{rgb_path.stem}.tiff"
            gt_path = defect_dir / "gt" / rgb_path.name if split == "test" else None
            samples.append(Sample(category, split, defect_dir.name, rgb_path, xyz_path, gt_path))
    return samples


def load_rgb(path: str | Path) -> np.ndarray:
    """(H, W, 3) uint8."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_xyz(path: str | Path) -> np.ndarray:
    """(H, W, 3) float32, zeros where the sensor has no measurement."""
    return tifffile.imread(path).astype(np.float32, copy=False)


def load_defect_map(path: str | Path) -> np.ndarray:
    """(H, W) uint8. 0 is good; other values map to names via class_ids.json.

    "combined" test samples contain several different values.
    """
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def load_mask(path: str | Path) -> np.ndarray:
    """(H, W) bool, True on defective pixels."""
    return load_defect_map(path) > 0


def valid_points(xyz: np.ndarray) -> np.ndarray:
    """(H, W) bool, True where the 3D sensor returned a point."""
    return np.any(xyz != 0, axis=-1)


def defect_names(root: str | Path, category: str) -> dict[int, str]:
    with open(Path(root) / category / "class_ids.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def check_category(
    root: str | Path,
    category: str,
    expected: dict[str, tuple[int, int, int]] = EXPECTED_COUNTS,
    deep: bool = False,
) -> list[str]:
    """Return a list of problems with one category on disk (empty if it looks fine).

    The cheap check looks at counts and file pairing. With deep=True every
    sample is also loaded to compare image sizes, which takes a few minutes for
    the whole dataset.
    """
    problems = []
    for split, n_expected in zip(SPLITS, expected[category], strict=True):
        try:
            samples = list_samples(root, category, split)
        except FileNotFoundError:
            problems.append(f"{category}/{split}: missing")
            continue

        if len(samples) != n_expected:
            problems.append(f"{category}/{split}: {len(samples)} samples, expected {n_expected}")
        if split != "test" and any(s.is_anomalous for s in samples):
            problems.append(f"{category}/{split}: contains non-good samples")

        for s in samples:
            if not s.xyz_path.is_file():
                problems.append(f"{s.name}: no xyz file")
                continue
            if s.gt_path is not None and not s.gt_path.is_file():
                problems.append(f"{s.name}: no gt mask")
                continue
            if deep:
                problems.extend(_check_shapes(s))
    return problems


def _check_shapes(s: Sample) -> list[str]:
    rgb = load_rgb(s.rgb_path)
    xyz = load_xyz(s.xyz_path)
    problems = []
    if xyz.shape != rgb.shape:
        problems.append(f"{s.name}: rgb {rgb.shape} vs xyz {xyz.shape}")
    if not valid_points(xyz).any():
        problems.append(f"{s.name}: xyz has no valid points")
    if s.gt_path is not None:
        mask = load_mask(s.gt_path)
        if mask.shape != rgb.shape[:2]:
            problems.append(f"{s.name}: rgb {rgb.shape[:2]} vs gt {mask.shape}")
        if s.is_anomalous and not mask.any():
            problems.append(f"{s.name}: defect sample with an empty mask")
    return problems
