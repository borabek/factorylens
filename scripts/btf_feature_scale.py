"""How much each half of BTF's concatenated patch features moves the distances.

BTF puts RGB (1536 dims, ImageNet features around 1) and FPFH (33 dims,
histograms in the hundreds) side by side without rescaling. For each category
this measures the typical spread of each half over the training patches, i.e.
the mean distance of a patch to the mean patch. If one half is much wider,
nearest-neighbour distances in the joint space are mostly that half.

usage: uv run python scripts/btf_feature_scale.py
Writes results/analysis/btf_feature_scale.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from factorylens.data import cache, mvtec3d
from factorylens.models.detectors import BTFDetector, spread


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/mvtec3d")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--features", default="outputs/features/fpfh")
    args = parser.parse_args()

    rows = []
    for category in mvtec3d.CATEGORIES:
        train = cache.get(args.root, args.cache, category, "train", cache.BTF_SETTINGS)
        rgb, geo = BTFDetector(feature_cache=args.features).parts(train)
        spread_rgb, spread_fpfh = spread(rgb.cpu()), spread(geo.cpu())
        rows.append(
            {
                "category": category,
                "spread_rgb": spread_rgb,
                "spread_fpfh": spread_fpfh,
                "ratio": spread_fpfh / spread_rgb,
                # share of a typical squared distance that comes from FPFH
                "fpfh_share": spread_fpfh**2 / (spread_fpfh**2 + spread_rgb**2),
            }
        )
        print(f"{category:12s} ratio {rows[-1]['ratio']:5.1f}", flush=True)
    out = Path("results/analysis/btf_feature_scale.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, float_format="%.4f")
    print("wrote", out)


if __name__ == "__main__":
    main()
