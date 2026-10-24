"""The four sweep detectors on clean and FactoryLens-C test sets.

Training data is always clean: the banks are fitted once per category on the
full training set, as on a line where the sensors were fine at setup time and
one of them degraded later. Only the test split is corrupted. A corruption of
one sensor leaves the other sensor's features untouched, so only the affected
half is recomputed.

Image scores of every model are kept for every condition, together with the
validation statistics late fusion uses, so a new fusion rule can be tried on
them later without touching the features again. For the reliability-aware
fusion (`factorylens.fusion`) the per-image sensor weights are kept as well.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from factorylens import corruptions
from factorylens.data import cache
from factorylens.data.cache import Prepared
from factorylens.data.mvtec3d import valid_points
from factorylens.experiments import evaluate
from factorylens.fusion import ReliabilityConfig, ReliabilityFusion, Weights
from factorylens.models import fpfh
from factorylens.models.detectors import spread
from factorylens.models.memory_bank import MemoryBank
from factorylens.models.patchcore import PatchCore
from factorylens.repro import config_hash, run_info, save_result
from factorylens.sweep import (
    MODELS,
    Scored,
    SplitFeatures,
    SweepConfig,
    split_features,
    standardise,
)

CLEAN = ("clean", 0)
ALL_MODELS = (*MODELS, "reliable", "reliable_window")
WINDOW = 16  # parts over which reliable_window judges a sensor


def all_conditions() -> list[tuple[str, int]]:
    return [
        (name, s) for name, c in corruptions.CORRUPTIONS.items() for s in range(1, c.severities + 1)
    ]


def run_name(category: str, name: str, severity: int, config: dict[str, Any]) -> str:
    return f"{category}_{name}{severity}_{config_hash(config)}"


def points(p: Prepared) -> np.ndarray:
    return np.asarray(valid_points(p.xyz).sum(axis=(1, 2)))


class Fitted:
    """Banks for every model, fitted once on the clean training features."""

    def __init__(
        self,
        train: SplitFeatures,
        val: SplitFeatures,
        val_split: Prepared,
        config: SweepConfig,
        size: tuple[int, int],
        early: bool = True,
        reliability: ReliabilityConfig | None = None,
    ):
        bank = config.rgb.bank_config()
        self.size = size
        self.weight = spread(train.rgb) / spread(train.geo)
        self.rgb = MemoryBank(bank).fit(train.rgb)
        self.geo = MemoryBank(bank).fit(train.geo)
        self.early = (
            MemoryBank(bank).fit(torch.cat([train.rgb, train.geo * self.weight], -1))
            if early
            else None
        )
        self.rgb_val = self.rgb.score(val.rgb, size)
        self.geo_val = self.geo.score(val.geo, size)
        reliability = reliability or ReliabilityConfig()
        self.fusions = {
            name: ReliabilityFusion(cfg).fit(
                self.rgb_val, self.geo_val, val_split.foreground, points(val_split)
            )
            for name, cfg in [
                ("reliable", reliability),
                ("reliable_window", replace(reliability, window=WINDOW)),
            ]
        }
        self.weights: dict[str, Weights] = {}  # of the last call to score()

    def score(self, rgb: torch.Tensor, geo: torch.Tensor, test: Prepared) -> dict[str, Scored]:
        r = self.rgb.score(rgb, self.size)
        g = self.geo.score(geo, self.size)
        zr, zg = standardise(r, self.rgb_val), standardise(g, self.geo_val)
        out = {"rgb": r, "fpfh": g, "late": (zr[0] + zg[0], zr[1] + zg[1])}
        if self.early is not None:
            out["early"] = self.early.score(torch.cat([rgb, geo * self.weight], -1), self.size)
        for name, fusion in self.fusions.items():
            out[name], self.weights[name] = fusion.fuse(r, g, test.foreground, points(test))
        return out


def run_category(
    root: str | Path,
    cache_dir: str | Path,
    category: str,
    conditions: list[tuple[str, int]] | None = None,
    models: tuple[str, ...] = ALL_MODELS,
    config: SweepConfig | None = None,
    reliability: ReliabilityConfig | None = None,
    seed: int = 0,
    results_dir: str | Path = "results/robustness",
    scores_dir: str | Path | None = "outputs/scores/robustness",
    feature_cache: str | Path | None = "outputs/features/fpfh",
    device: str | None = None,
    pretrained: bool = True,
    log: Any = print,
) -> list[dict[str, Any]]:
    run = run_info()
    config = config or SweepConfig()
    conditions = [CLEAN, *(all_conditions() if conditions is None else conditions)]
    splits = {
        s: cache.get(root, cache_dir, category, s, config.settings)
        for s in ("train", "validation", "test")
    }
    rgb_model = PatchCore(config.rgb, device=device, pretrained=pretrained)
    feats = {s: split_features(p, rgb_model, config.fpfh, feature_cache) for s, p in splits.items()}
    test = splits["test"]
    reliability = reliability or ReliabilityConfig()
    fitted = Fitted(
        feats["train"],
        feats["validation"],
        splits["validation"],
        config,
        (test.rgb.shape[1], test.rgb.shape[2]),
        early="early" in models,
        reliability=reliability,
    )

    out = []
    for name, severity in conditions:
        rgb, geo, shown = feats["test"].rgb, feats["test"].geo, test
        if name != "clean":
            shown = corruptions.corrupt(test, name, severity, seed)
            if corruptions.CORRUPTIONS[name].sensor == "rgb":
                rgb = rgb_model.embed(shown.rgb)
            else:
                geo = fpfh.patches_for(shown, config.fpfh, feature_cache).to(rgb.device)
        scored = fitted.score(rgb, geo, shown)
        for model in models:
            cfg = {
                "model": f"robust_{model}",
                "category": category,
                "corruption": name,
                "severity": severity,
                "seed": seed,
                "k": None,
                "preprocess": config.settings.as_dict(),
                "rgb": config.rgb.as_dict(),
                "fpfh": config.fpfh.as_dict(),
                "pretrained": pretrained,
                "corruptions_version": corruptions.VERSION,
            }
            if model == "early":
                cfg["fpfh_weight_fitted"] = fitted.weight
            if model == "reliable":
                cfg["reliability"] = asdict(reliability)
            if model == "reliable_window":
                cfg["reliability"] = asdict(replace(reliability, window=WINDOW))
            scores, maps = scored[model]
            result = evaluate(test.is_anomalous, scores, test.mask, maps)
            path = Path(results_dir) / model / f"{run_name(category, name, severity, cfg)}.json"
            save_result(path, result, cfg, run=run)
            out.append({"model": model, "corruption": name, "severity": severity, **result})
        if scores_dir is not None:
            path = Path(scores_dir) / f"{category}_{name}{severity}_s{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, Any] = {m: s for m, (s, _) in scored.items()}
            for fusion, w in fitted.weights.items():
                suffix = fusion.removeprefix("reliable")
                arrays |= {
                    f"w_rgb{suffix}": w.rgb,
                    f"w_fpfh{suffix}": w.geo,
                    f"u_rgb{suffix}": w.u_rgb,
                    f"u_fpfh{suffix}": w.u_geo,
                }
            arrays |= {
                "rgb_val": fitted.rgb_val[0],
                "fpfh_val": fitted.geo_val[0],
                "labels": test.is_anomalous,
                "defect": np.array(test.defect),
                "stem": np.array(test.stem),
            }
            np.savez_compressed(path, **arrays)
        row = {r["model"]: r["image_auroc"] for r in out[-len(models) :]}
        log(
            f"{category:12s} {name:15s} {severity} " + " ".join(f"{m} {row[m]:.3f}" for m in models)
        )
    return out
