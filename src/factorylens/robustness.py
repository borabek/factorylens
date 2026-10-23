"""The four sweep detectors on clean and FactoryLens-C test sets.

Training data is always clean: the banks are fitted once per category on the
full training set, as on a line where the sensors were fine at setup time and
one of them degraded later. Only the test split is corrupted. A corruption of
one sensor leaves the other sensor's features untouched, so only the affected
half is recomputed.

Image scores of every model are kept for every condition, together with the
validation statistics late fusion uses, so a new fusion rule can be tried on
them later without touching the features again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from factorylens import corruptions
from factorylens.data import cache
from factorylens.experiments import evaluate
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


def all_conditions() -> list[tuple[str, int]]:
    return [
        (name, s) for name, c in corruptions.CORRUPTIONS.items() for s in range(1, c.severities + 1)
    ]


def run_name(category: str, name: str, severity: int, config: dict[str, Any]) -> str:
    return f"{category}_{name}{severity}_{config_hash(config)}"


class Fitted:
    """Banks for every model, fitted once on the clean training features."""

    def __init__(
        self, train: SplitFeatures, val: SplitFeatures, config: SweepConfig, size: tuple[int, int]
    ):
        bank = config.rgb.bank_config()
        self.size = size
        self.weight = spread(train.rgb) / spread(train.geo)
        self.rgb = MemoryBank(bank).fit(train.rgb)
        self.geo = MemoryBank(bank).fit(train.geo)
        self.early = MemoryBank(bank).fit(torch.cat([train.rgb, train.geo * self.weight], -1))
        self.rgb_val = self.rgb.score(val.rgb, size)
        self.geo_val = self.geo.score(val.geo, size)

    def score(self, rgb: torch.Tensor, geo: torch.Tensor) -> dict[str, Scored]:
        r = self.rgb.score(rgb, self.size)
        g = self.geo.score(geo, self.size)
        e = self.early.score(torch.cat([rgb, geo * self.weight], -1), self.size)
        zr, zg = standardise(r, self.rgb_val), standardise(g, self.geo_val)
        return {"rgb": r, "fpfh": g, "early": e, "late": (zr[0] + zg[0], zr[1] + zg[1])}


def run_category(
    root: str | Path,
    cache_dir: str | Path,
    category: str,
    conditions: list[tuple[str, int]] | None = None,
    config: SweepConfig | None = None,
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
    fitted = Fitted(
        feats["train"], feats["validation"], config, (test.rgb.shape[1], test.rgb.shape[2])
    )

    out = []
    for name, severity in conditions:
        rgb, geo = feats["test"].rgb, feats["test"].geo
        if name != "clean":
            bad = corruptions.corrupt(test, name, severity, seed)
            if corruptions.CORRUPTIONS[name].sensor == "rgb":
                rgb = rgb_model.embed(bad.rgb)
            else:
                geo = fpfh.patches_for(bad, config.fpfh, feature_cache).to(rgb.device)
        scored = fitted.score(rgb, geo)
        for model in MODELS:
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
            scores, maps = scored[model]
            result = evaluate(test.is_anomalous, scores, test.mask, maps)
            path = Path(results_dir) / model / f"{run_name(category, name, severity, cfg)}.json"
            save_result(path, result, cfg, run=run)
            out.append({"model": model, "corruption": name, "severity": severity, **result})
        if scores_dir is not None:
            path = Path(scores_dir) / f"{category}_{name}{severity}_s{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays: dict[str, Any] = {m: s for m, (s, _) in scored.items()}
            arrays |= {
                "rgb_val": fitted.rgb_val[0],
                "fpfh_val": fitted.geo_val[0],
                "labels": test.is_anomalous,
                "defect": np.array(test.defect),
                "stem": np.array(test.stem),
            }
            np.savez_compressed(path, **arrays)
        row = {r["model"]: r["image_auroc"] for r in out[-len(MODELS) :]}
        log(
            f"{category:12s} {name:15s} {severity} " + " ".join(f"{m} {row[m]:.3f}" for m in MODELS)
        )
    return out
