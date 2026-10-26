"""Preprocessed MVTec 3D-AD at a fixed size, one compressed .npz per category and split.

Loading an 800x800 TIFF and fitting the background plane takes most of the
time in every experiment, and the result never changes, so it is done once.
Background removal runs at full resolution (more points for the plane fit),
then everything is resized: RGB bilinear, xyz and masks nearest-neighbour.

The cache directory name contains a hash of the preprocessing settings, so
changing a threshold can never silently reuse old arrays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from factorylens import __version__
from factorylens.data import mvtec3d, preprocess
from factorylens.repro import config_hash


@dataclass(frozen=True)
class Settings:
    size: int = 224
    threshold: float = preprocess.DEFAULT_THRESHOLD
    min_component: float = preprocess.MIN_COMPONENT
    # BTF's preprocessing keeps only the largest blob and blacks out the RGB
    # pixels it removed; both are off here except when reproducing BTF
    keep_largest: bool = False
    zero_background_rgb: bool = False

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "size": self.size,
            "threshold": self.threshold,
            "min_component": self.min_component,
        }
        # only written when set, so the default settings keep their old hash
        if self.keep_largest:
            d["keep_largest"] = True
        if self.zero_background_rgb:
            d["zero_background_rgb"] = True
        return d

    def dirname(self) -> str:
        return f"{self.size}px-{config_hash(self.as_dict())}"


@dataclass
class Prepared:
    rgb: np.ndarray  # (N, S, S, 3) uint8
    xyz: np.ndarray  # (N, S, S, 3) float32, background zeroed
    foreground: np.ndarray  # (N, S, S) bool
    mask: np.ndarray  # (N, S, S) bool, all False outside the test split
    defect: list[str]
    stem: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.stem)

    @property
    def is_anomalous(self) -> np.ndarray:
        return np.array([d != "good" for d in self.defect])


BTF_SETTINGS = Settings(
    threshold=preprocess.BTF_THRESHOLD, min_component=0, keep_largest=True, zero_background_rgb=True
)


def prepare_sample(
    rgb: np.ndarray, xyz: np.ndarray, settings: Settings
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One full-resolution scan -> resized RGB, background-free xyz and object mask."""
    xyz, fg, _ = preprocess.remove_background(
        xyz, threshold=settings.threshold, min_component=settings.min_component
    )
    if settings.keep_largest:
        fg = preprocess.keep_largest_component(fg)
        xyz = np.where(fg[..., None], xyz, 0).astype(np.float32)
    if settings.zero_background_rgb:
        rgb = np.where(fg[..., None], rgb, 0).astype(np.uint8)
    size = settings.size
    return (
        preprocess.resize_rgb(rgb, size),
        preprocess.resize_xyz(xyz, size),
        preprocess.resize_mask(fg, size),
    )


def prepare(root: str | Path, category: str, split: str, settings: Settings) -> Prepared:
    rgbs, xyzs, fgs, masks, defects, stems = [], [], [], [], [], []
    size = settings.size
    for s in mvtec3d.list_samples(root, category, split):
        rgb, xyz, fg = prepare_sample(
            mvtec3d.load_rgb(s.rgb_path), mvtec3d.load_xyz(s.xyz_path), settings
        )
        rgbs.append(rgb)
        xyzs.append(xyz)
        fgs.append(fg)
        if s.gt_path is not None:
            masks.append(preprocess.resize_mask(mvtec3d.load_mask(s.gt_path), size))
        else:
            masks.append(np.zeros((size, size), bool))
        defects.append(s.defect)
        stems.append(s.rgb_path.stem)
    meta = {
        "category": category,
        "split": split,
        "settings": settings.as_dict(),
        "factorylens": __version__,
    }
    return Prepared(
        np.stack(rgbs), np.stack(xyzs), np.stack(fgs), np.stack(masks), defects, stems, meta
    )


def save(prepared: Prepared, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        rgb=prepared.rgb,
        xyz=prepared.xyz,
        foreground=prepared.foreground,
        mask=prepared.mask,
        defect=np.array(prepared.defect),
        stem=np.array(prepared.stem),
        meta=np.array(json.dumps(prepared.meta)),
    )
    tmp.replace(path)  # never leave a half-written cache file behind


def load(path: str | Path) -> Prepared:
    with np.load(path) as f:
        return Prepared(
            rgb=f["rgb"],
            xyz=f["xyz"],
            foreground=f["foreground"],
            mask=f["mask"],
            defect=f["defect"].tolist(),
            stem=f["stem"].tolist(),
            meta=json.loads(str(f["meta"])),
        )


def cache_path(cache_dir: str | Path, category: str, split: str, settings: Settings) -> Path:
    return Path(cache_dir) / settings.dirname() / f"{category}_{split}.npz"


def get(
    root: str | Path,
    cache_dir: str | Path,
    category: str,
    split: str,
    settings: Settings | None = None,
) -> Prepared:
    """Load from the cache, preparing and saving first if needed."""
    settings = settings or Settings()
    path = cache_path(cache_dir, category, split, settings)
    if path.exists():
        return load(path)
    prepared = prepare(root, category, split, settings)
    save(prepared, path)
    return prepared


def take(prepared: Prepared, stems: list[str]) -> Prepared:
    """Subset by sample stem, e.g. a k-shot training set from configs/kshot.json."""
    wanted = set(stems)
    idx = [i for i, s in enumerate(prepared.stem) if s in wanted]
    if len(idx) != len(wanted):
        raise ValueError(f"{len(wanted) - len(idx)} requested stems are not in this split")
    return Prepared(
        prepared.rgb[idx],
        prepared.xyz[idx],
        prepared.foreground[idx],
        prepared.mask[idx],
        [prepared.defect[i] for i in idx],
        [prepared.stem[i] for i in idx],
        dict(prepared.meta),
    )
