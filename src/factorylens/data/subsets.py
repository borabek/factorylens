"""Fixed few-shot training subsets.

For every category and seed the training samples are shuffled once and the
first k are taken, so the 1-shot set is inside the 5-shot set and so on. That
keeps the data-efficiency curves comparable across k: going from k=5 to k=10
only adds samples, it never swaps the ones already there.

The lists are written to configs/kshot.json and committed, so every run (and
anyone reproducing the results) uses exactly the same images.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from factorylens.data import mvtec3d

KS = (1, 5, 10, 50)
SEEDS = (0, 1, 2, 3, 4)

Subsets = dict[str, dict[str, dict[str, list[str]]]]  # category -> seed -> k -> stems


def _rng(category: str, seed: int) -> np.random.Generator:
    # python's hash() is salted per process, crc32 is stable
    return np.random.default_rng([seed, zlib.crc32(category.encode())])


def nested_subsets(
    category: str,
    stems: Sequence[str],
    ks: Iterable[int] = KS,
    seeds: Iterable[int] = SEEDS,
) -> dict[str, dict[str, list[str]]]:
    stems = sorted(stems)
    ks = sorted(ks)
    if ks and ks[-1] > len(stems):
        raise ValueError(f"{category}: k={ks[-1]} but only {len(stems)} training samples")
    out = {}
    for seed in seeds:
        order = [stems[i] for i in _rng(category, seed).permutation(len(stems))]
        out[str(seed)] = {str(k): sorted(order[:k]) for k in ks}
    return out


def build(
    root: str | Path,
    categories: Iterable[str] = mvtec3d.CATEGORIES,
    ks: Iterable[int] = KS,
) -> Subsets:
    ks = tuple(ks)
    return {
        c: nested_subsets(c, [s.rgb_path.stem for s in mvtec3d.list_samples(root, c, "train")], ks)
        for c in categories
    }


def save(subsets: Subsets, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(subsets, indent=1) + "\n")


def load(path: str | Path) -> Subsets:
    subsets: Subsets = json.loads(Path(path).read_text())
    return subsets


def select(
    samples: Sequence[mvtec3d.Sample], subsets: Subsets, k: int | None, seed: int = 0
) -> list[mvtec3d.Sample]:
    """Training samples for one k-shot run; k=None means the full training set."""
    if k is None:
        return list(samples)
    if not samples:
        return []
    wanted = set(subsets[samples[0].category][str(seed)][str(k)])
    chosen = [s for s in samples if s.rgb_path.stem in wanted]
    if len(chosen) != k:
        raise ValueError(f"expected {k} samples, found {len(chosen)}; is the dataset complete?")
    return chosen
