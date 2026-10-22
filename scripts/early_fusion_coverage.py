"""Why early fusion fails with few images, and whether that can be seen without labels.

For every category and k-shot draw this computes the coverage gap
(`factorylens.models.coverage`) on two sets of normal images:

  gap_val          the validation normals: no test data and no labels involved
  gap_test_normal  the normal test images, for comparison

next to early fusion's test AUROC (joint matching), the AUROC of matching each
half separately, and the late-fusion AUROC from the sweep results. If gap_val
tracks early fusion's failures, it can choose the fusion before deployment.

usage: uv run python scripts/early_fusion_coverage.py data/raw/mvtec3d --cache data/processed
Writes results/analysis/coverage_gap.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from factorylens import experiments, sweep
from factorylens.data import cache, mvtec3d, subsets
from factorylens.eval.metrics import image_auroc
from factorylens.models.coverage import coverage
from factorylens.models.detectors import spread
from factorylens.models.patchcore import PatchCore


def late_auroc(results: Path) -> pd.Series:
    runs = experiments.collect(results / "late")
    runs = runs[runs["k"].notna()]
    return runs.set_index(["category", runs["k"].astype(int), "seed"])["image_auroc"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--category", nargs="+", default=list(mvtec3d.CATEGORIES))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--features", default="outputs/features/fpfh")
    parser.add_argument("--sweep", type=Path, default=Path("results/sweep"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis/coverage_gap.csv"))
    args = parser.parse_args()

    config = sweep.SweepConfig()
    rgb = PatchCore(config.rgb)
    late = late_auroc(args.sweep)
    rows = []
    for category in args.category:
        splits = {
            s: cache.get(args.root, args.cache, category, s, config.settings)
            for s in ("train", "validation", "test")
        }
        feats = {
            s: sweep.split_features(p, rgb, config.fpfh, args.features) for s, p in splits.items()
        }
        labels = splits["test"].is_anomalous
        draws = subsets.load("configs/kshot.json")[category]
        for k in args.k:
            for seed in range(args.seeds):
                train = feats["train"].take(draws[str(seed)][str(k)])
                w = spread(train.rgb) / spread(train.geo)
                val = coverage(
                    train.rgb, train.geo, feats["validation"].rgb, feats["validation"].geo, w
                )
                test = coverage(train.rgb, train.geo, feats["test"].rgb, feats["test"].geo, w)
                rows.append(
                    {
                        "category": category,
                        "k": k,
                        "seed": seed,
                        "early_auroc": image_auroc(labels, test.joint),
                        "separate_auroc": image_auroc(labels, test.separate),
                        "late_auroc": late.get((category, k, seed), float("nan")),
                        "gap_val": val.gap.mean(),
                        "gap_test_normal": test.gap[~labels].mean(),
                        "gap_test_anomalous": test.gap[labels].mean(),
                    }
                )
                print(
                    {key: round(v, 3) if isinstance(v, float) else v for key, v in rows[-1].items()}
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
