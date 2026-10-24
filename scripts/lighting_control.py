"""Does the clean RGB detector use the lighting difference between the test classes?

`check_acquisition_confound.py` showed that normal and defective test images
differ in global brightness in several categories. Two checks of whether the
detectors here profit from that:

1. Without rerunning anything: how strongly the clean full-data scores of
   each sweep model follow image brightness, within the normal test images
   and within the defective ones (Spearman). A detector that reacts to
   lighting ranks images by brightness inside each class too.
2. A control run: every image (train, validation, test) gets a per-channel
   gain so that its mean matches the training set's mean, like an
   auto-exposure. RGB PatchCore, early and late fusion are refitted on the
   normalised images; FPFH is unchanged. If a category's AUROC drops, part
   of it came from the lighting.

usage: OMP_NUM_THREADS=4 uv run python scripts/lighting_control.py data/raw/mvtec3d \
           --cache data/processed
Writes results/analysis/lighting_scores.csv and results/analysis/lighting_control.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from factorylens import experiments
from factorylens.data import cache
from factorylens.data.cache import Prepared
from factorylens.data.mvtec3d import CATEGORIES
from factorylens.eval.metrics import image_auroc
from factorylens.models import fpfh
from factorylens.models.patchcore import PatchCore
from factorylens.repro import run_info
from factorylens.robustness import Fitted
from factorylens.sweep import SplitFeatures, SweepConfig

MODELS = ("rgb", "fpfh", "early", "late")


def brightness(p: Prepared) -> np.ndarray:
    return p.rgb.reshape(len(p), -1).mean(axis=1)


def score_correlations(root: str, cache_dir: str, scores: Path, results: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        runs = experiments.collect(results / model)
        runs = runs[runs["k"].isna()].set_index("category")
        for category in CATEGORIES:
            test = cache.get(root, cache_dir, category, "test")
            npz = np.load(
                scores / model / Path(runs.loc[category, "file"]).with_suffix(".npz").name
            )
            assert list(npz["stem"]) == test.stem
            b, s, y = brightness(test), npz["scores"], test.is_anomalous
            rows.append(
                {
                    "model": model,
                    "category": category,
                    "auroc": image_auroc(y, s),
                    "spearman_normal": spearmanr(b[~y], s[~y]).statistic,
                    "spearman_defective": spearmanr(b[y], s[y]).statistic,
                }
            )
    return pd.DataFrame(rows)


def equalise(p: Prepared, target: np.ndarray) -> Prepared:
    """Per-image, per-channel gain so that each image's channel means equal `target`."""
    x = p.rgb.astype(np.float32)
    gain = target / np.maximum(x.mean(axis=(1, 2), keepdims=True), 1.0)
    rgb = np.clip(np.round(x * gain), 0, 255).astype(np.uint8)
    return Prepared(rgb, p.xyz, p.foreground, p.mask, p.defect, p.stem, p.meta)


def control(root: str, cache_dir: str, features: str, device: str | None) -> pd.DataFrame:
    config = SweepConfig()
    rgb_model = PatchCore(config.rgb, device=device)
    rows = []
    for category in CATEGORIES:
        splits = {
            s: cache.get(root, cache_dir, category, s, config.settings)
            for s in ("train", "validation", "test")
        }
        target = splits["train"].rgb.astype(np.float32).mean(axis=(0, 1, 2))
        feats = {}
        for s, p in splits.items():
            geo = fpfh.patches_for(p, config.fpfh, features)
            rgb = rgb_model.embed(equalise(p, target).rgb)
            feats[s] = SplitFeatures(rgb, geo.to(rgb.device), p.stem)
        test = splits["test"]
        fitted = Fitted(
            feats["train"], feats["validation"], splits["validation"], config, test.rgb.shape[1:3]
        )
        scored = fitted.score(feats["test"].rgb, feats["test"].geo, test)
        y = test.is_anomalous
        for model in MODELS:
            rows.append(
                {"model": model, "category": category, "auroc": image_auroc(y, scored[model][0])}
            )
        print(
            category,
            {m: round(r["auroc"], 3) for m, r in zip(MODELS, rows[-4:], strict=True)},
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--features", default="outputs/features/fpfh")
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/sweep"))
    parser.add_argument("--results", type=Path, default=Path("results/sweep"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    parser.add_argument("--device")
    args = parser.parse_args()

    git = run_info()["git"]
    corr = score_correlations(args.root, args.cache, args.scores, args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    corr.round(4).to_csv(args.out / "lighting_scores.csv", index=False)
    print(corr.pivot(index="category", columns="model", values="spearman_normal").round(2))

    normalised = control(args.root, args.cache, args.features, args.device)
    clean = corr.set_index(["model", "category"])["auroc"]
    normalised["clean_auroc"] = [
        clean[(m, c)] for m, c in zip(normalised["model"], normalised["category"], strict=True)
    ]
    normalised["change"] = normalised["auroc"] - normalised["clean_auroc"]
    normalised["commit"] = f"{git['commit'][:7]}{'-dirty' if git['dirty'] else ''}"
    normalised.round(4).to_csv(args.out / "lighting_control.csv", index=False)
    print(normalised.pivot(index="category", columns="model", values="change").round(3))
    print(normalised.groupby("model")[["clean_auroc", "auroc"]].mean().round(3))


if __name__ == "__main__":
    main()
