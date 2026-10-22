"""Data-efficiency sweep: four detectors x k-shot x seeds, features computed once.

Every k and seed of a category reuses the same patch features. The RGB
backbone embeds each split once and FPFH comes from the disk cache, so only
the memory banks are refit. That is what makes 4 models x 5 k x 5 seeds x 10
categories fit in an afternoon on a laptop.

Models:
  rgb     PatchCore on the colour image
  fpfh    the same memory bank on FPFH point-cloud features
  early   RGB and FPFH patches concatenated, FPFH scaled so that both halves
          have the same spread on that run's training patches
  late    rgb and fpfh scored separately, each standardised with the mean and
          standard deviation of its scores on the validation normals, summed
  rgb_aug rgb with the k training images also flipped and rotated (k <= 10)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from factorylens.data import cache, subsets
from factorylens.data.cache import Prepared
from factorylens.experiments import evaluate, run_name
from factorylens.models import fpfh
from factorylens.models.detectors import spread
from factorylens.models.memory_bank import BankConfig, MemoryBank
from factorylens.models.patchcore import PatchCore, PatchCoreConfig
from factorylens.repro import run_info, save_result

MODELS = ("rgb", "fpfh", "early", "late")
AUGMENT_UP_TO_K = 10

Scored = tuple[np.ndarray, np.ndarray]  # image scores (N,), maps (N, H, W)


@dataclass
class SplitFeatures:
    rgb: torch.Tensor  # (N, h, w, D)
    geo: torch.Tensor  # (N, h, w, 33), same device as rgb
    stems: list[str]

    def take(self, stems: list[str]) -> SplitFeatures:
        idx = [i for i, s in enumerate(self.stems) if s in set(stems)]
        if len(idx) != len(set(stems)):
            raise ValueError("some requested stems are not in this split")
        return SplitFeatures(self.rgb[idx], self.geo[idx], [self.stems[i] for i in idx])


def split_features(
    p: Prepared, rgb: PatchCore, fpfh_config: fpfh.FPFHConfig, feature_cache: str | Path | None
) -> SplitFeatures:
    r = rgb.embed(p.rgb)
    g = fpfh.patches_for(p, fpfh_config, feature_cache).to(r.device)
    if r.shape[1:3] != g.shape[1:3]:
        raise ValueError(f"grids differ: rgb {tuple(r.shape[1:3])}, fpfh {tuple(g.shape[1:3])}")
    return SplitFeatures(r, g, p.stem)


def dihedral(images: np.ndarray) -> np.ndarray:
    """The 8 flips and 90-degree rotations of every (square) image."""
    out = []
    for k in range(4):
        rot = np.rot90(images, k, axes=(1, 2))
        out += [rot, rot[:, :, ::-1]]
    return np.ascontiguousarray(np.concatenate(out))


def fit_score(
    config: BankConfig, train: torch.Tensor, evals: list[torch.Tensor], size: tuple[int, int]
) -> tuple[list[Scored], int]:
    bank = MemoryBank(config).fit(train)
    assert bank.bank is not None
    return [bank.score(e, size) for e in evals], len(bank.bank)


def standardise(test: Scored, val: Scored) -> Scored:
    """z-scores against the validation normals, separately for images and pixels."""
    s, m = test
    vs, vm = val
    return (s - vs.mean()) / (vs.std() + 1e-12), (m - vm.mean()) / (vm.std() + 1e-12)


@dataclass(frozen=True)
class SweepConfig:
    rgb: PatchCoreConfig = PatchCoreConfig()
    fpfh: fpfh.FPFHConfig = fpfh.FPFHConfig()
    settings: cache.Settings = cache.Settings()


def run_category(
    root: str | Path,
    cache_dir: str | Path,
    category: str,
    ks: list[int | None],
    seeds: list[int],
    models: tuple[str, ...] = MODELS,
    config: SweepConfig | None = None,
    kshot_path: str | Path = "configs/kshot.json",
    results_dir: str | Path = "results/sweep",
    scores_dir: str | Path | None = "outputs/scores/sweep",
    feature_cache: str | Path | None = "outputs/features/fpfh",
    device: str | None = None,
    pretrained: bool = True,
    log: Any = print,
) -> list[dict[str, Any]]:
    run = run_info()  # once per category, before any work
    config = config or SweepConfig()
    splits = {
        s: cache.get(root, cache_dir, category, s, config.settings)
        for s in ("train", "validation", "test")
    }
    rgb_model = PatchCore(config.rgb, device=device, pretrained=pretrained)
    feats = {s: split_features(p, rgb_model, config.fpfh, feature_cache) for s, p in splits.items()}
    test, val = splits["test"], feats["validation"]
    size = (test.rgb.shape[1], test.rgb.shape[2])
    kshot = subsets.load(kshot_path)

    out = []
    for k in ks:
        for seed in seeds:
            if k is None:
                train = feats["train"]
            else:
                train = feats["train"].take(kshot[category][str(seed)][str(k)])
            bank = replace(config.rgb.bank_config(), seed=seed)
            scored: dict[str, Scored] = {}
            banks: dict[str, int] = {}
            extra: dict[str, dict[str, Any]] = {}

            if {"rgb", "late"} & set(models):
                (rgb_test, rgb_val), banks["rgb"] = fit_score(
                    bank, train.rgb, [feats["test"].rgb, val.rgb], size
                )
                scored["rgb"] = rgb_test
            if {"fpfh", "late"} & set(models):
                (geo_test, geo_val), banks["fpfh"] = fit_score(
                    bank, train.geo, [feats["test"].geo, val.geo], size
                )
                scored["fpfh"] = geo_test
            if "late" in models:
                zr, zg = standardise(rgb_test, rgb_val), standardise(geo_test, geo_val)
                scored["late"] = (zr[0] + zg[0], zr[1] + zg[1])
                banks["late"] = banks["rgb"] + banks["fpfh"]
            if "early" in models:
                w = spread(train.rgb) / spread(train.geo)
                joint = torch.cat([train.rgb, train.geo * w], dim=-1)
                test_joint = torch.cat([feats["test"].rgb, feats["test"].geo * w], dim=-1)
                (scored["early"],), banks["early"] = fit_score(bank, joint, [test_joint], size)
                extra["early"] = {"fpfh_weight_fitted": w}
            if "rgb_aug" in models and k is not None and k <= AUGMENT_UP_TO_K:
                images = splits["train"].rgb[[splits["train"].stem.index(s) for s in train.stems]]
                aug = rgb_model.embed(dihedral(images))
                (scored["rgb_aug"],), banks["rgb_aug"] = fit_score(
                    bank, aug, [feats["test"].rgb], size
                )

            for model in [m for m in models if m in scored]:
                cfg = {
                    "model": f"sweep_{model}",
                    "category": category,
                    "k": k,
                    "seed": seed,
                    "preprocess": config.settings.as_dict(),
                    "rgb": config.rgb.as_dict(),
                    "fpfh": config.fpfh.as_dict(),
                    "pretrained": pretrained,
                    **extra.get(model, {}),
                }
                scores, maps = scored[model]
                result = evaluate(test.is_anomalous, scores, test.mask, maps)
                result |= {
                    "n_train": len(train.stems),
                    "n_test": len(test),
                    "bank_size": banks[model],
                }
                name = run_name(cfg)
                save_result(Path(results_dir) / model / f"{name}.json", result, cfg, run=run)
                if scores_dir is not None and k is None:
                    path = Path(scores_dir) / model / f"{name}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        path,
                        scores=scores,
                        maps=maps.astype(np.float16),
                        labels=test.is_anomalous,
                        defect=np.array(test.defect),
                        stem=np.array(test.stem),
                    )
                out.append({"model": model, "k": k, "seed": seed, **result})
                log(
                    f"{category:12s} {model:8s} k={k or 'all':>3} seed={seed} "
                    f"image {result['image_auroc']:.3f} aupro {result['aupro']:.3f}"
                )
    return out
