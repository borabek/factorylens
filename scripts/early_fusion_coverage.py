"""Why early fusion breaks down with few training images.

For every k-shot draw of one category, each test patch is matched three ways:
against the concatenated (RGB, scaled FPFH) bank, as early fusion does, and
against the RGB and FPFH banks separately. If the training images contain the
combinations of appearance and geometry that the test images show, the joint
neighbour is barely farther than the two separate ones. The "excess" column is
how much farther it is, in squared distance, averaged over the patches of an
image and then over normal / anomalous images.

The "separate" score is max over patches of sqrt(d_rgb^2 + d_fpfh^2), i.e. the
same distance as early fusion but with each half matched on its own.

usage: uv run python scripts/early_fusion_coverage.py data/raw/mvtec3d --cache data/processed \
           --category bagel
Writes results/analysis/early_fusion_coverage_<category>.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from factorylens import sweep
from factorylens.data import cache, subsets
from factorylens.eval.metrics import image_auroc
from factorylens.models.detectors import spread
from factorylens.models.memory_bank import nearest_distance
from factorylens.models.patchcore import PatchCore


def match(train: sweep.SplitFeatures, test: sweep.SplitFeatures) -> dict[str, np.ndarray]:
    w = spread(train.rgb) / spread(train.geo)
    bank_rgb = train.rgb.reshape(-1, train.rgb.shape[-1])
    bank_geo = (train.geo * w).reshape(-1, train.geo.shape[-1])
    bank_joint = torch.cat([bank_rgb, bank_geo], dim=-1)
    joint, separate, excess = [], [], []
    for r, g in zip(test.rgb, test.geo * w, strict=True):
        qr, qg = r.reshape(-1, r.shape[-1]), g.reshape(-1, g.shape[-1])
        d_sep = nearest_distance(qr, bank_rgb) ** 2 + nearest_distance(qg, bank_geo) ** 2
        d_joint = nearest_distance(torch.cat([qr, qg], dim=-1), bank_joint) ** 2
        joint.append(float(d_joint.max().sqrt()))
        separate.append(float(d_sep.max().sqrt()))
        excess.append(float((d_joint - d_sep).mean() / d_sep.mean()))
    return {"joint": np.array(joint), "separate": np.array(separate), "excess": np.array(excess)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--category", default="bagel")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--features", default="outputs/features/fpfh")
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    config = sweep.SweepConfig()
    rgb = PatchCore(config.rgb)
    feats = {
        s: sweep.split_features(
            cache.get(args.root, args.cache, args.category, s, config.settings),
            rgb,
            config.fpfh,
            args.features,
        )
        for s in ("train", "test")
    }
    labels = cache.get(args.root, args.cache, args.category, "test", config.settings).is_anomalous
    draws = subsets.load("configs/kshot.json")[args.category]

    rows = []
    for k in args.k:
        for seed in range(args.seeds):
            m = match(feats["train"].take(draws[str(seed)][str(k)]), feats["test"])
            rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "joint_auroc": image_auroc(labels, m["joint"]),
                    "separate_auroc": image_auroc(labels, m["separate"]),
                    "excess_normal": m["excess"][~labels].mean(),
                    "excess_anomalous": m["excess"][labels].mean(),
                }
            )
            print(rows[-1])
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"early_fusion_coverage_{args.category}.csv"
    pd.DataFrame(rows).round(4).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
