"""Can image statistics with nothing to do with the defect tell normal from defective?

For every category, the test split is scored with trivial features that a
real defect detector should not need:

  rgb_mean        mean intensity of the whole image
  background_mean mean intensity of the pixels the background removal threw
                  away; no defect is annotated there
  points          number of 3D points left after background removal
  jpeg_error      mean change when the image is saved as JPEG at quality 25

AUROC far from 0.5 in either direction means normal and defective test
images differ before any detector looks at them. The training mean is
printed next to the two test groups to show which group moved.

Found while checking the JPEG results of FactoryLens-C, which jumped between
0 and 1 for single categories.

usage: uv run python scripts/check_acquisition_confound.py data/raw/mvtec3d --cache data/processed
Writes results/analysis/acquisition_confound.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from factorylens import corruptions
from factorylens.data import cache
from factorylens.data.mvtec3d import CATEGORIES, valid_points
from factorylens.eval.metrics import image_auroc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument(
        "--out", type=Path, default=Path("results/analysis/acquisition_confound.csv")
    )
    args = parser.parse_args()

    rows = []
    for category in CATEGORIES:
        test = cache.get(args.root, args.cache, category, "test")
        train = cache.get(args.root, args.cache, category, "train")
        y = test.is_anomalous
        mean = test.rgb.mean(axis=(1, 2, 3))
        background = np.array(
            [img[~fg].mean() for img, fg in zip(test.rgb, test.foreground, strict=True)]
        )
        points = valid_points(test.xyz).sum(axis=(1, 2)).astype(float)
        jpeg = corruptions.corrupt(test, "jpeg", 1)
        jpeg_error = np.abs(jpeg.rgb.astype(float) - test.rgb).mean(axis=(1, 2, 3))
        rows.append(
            {
                "category": category,
                "rgb_mean": image_auroc(y, mean),
                "background_mean": image_auroc(y, background),
                "points": image_auroc(y, points),
                "jpeg_error": image_auroc(y, jpeg_error),
                "train_rgb_mean": train.rgb.mean(),
                "test_normal_rgb_mean": mean[~y].mean(),
                "test_defective_rgb_mean": mean[y].mean(),
            }
        )
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(args.out, index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
