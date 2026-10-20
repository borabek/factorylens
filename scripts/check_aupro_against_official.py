"""Compare our AUPRO with the official MVTec 3D-AD style computation on saved maps.

The official evaluation code (also used by BTF) samples 100 thresholds evenly
in FPR and counts 8-connected defect regions. Ours uses every threshold and
4-connected regions. This shows how much that changes the numbers.

usage: uv run python scripts/check_aupro_against_official.py --scores outputs/scores/rgb
Writes results/analysis/aupro_vs_official.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_metrics import official_aupro  # noqa: E402

from factorylens.data import cache, mvtec3d  # noqa: E402
from factorylens.eval import metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="outputs/scores/rgb")
    parser.add_argument("--root", default="data/raw/mvtec3d")
    parser.add_argument("--cache", default="data/processed")
    args = parser.parse_args()

    print(f"{'category':12s} {'ours':>6s} {'official':>8s}")
    rows = []
    for category in mvtec3d.CATEGORIES:
        [path] = sorted(Path(args.scores).glob(f"{category}_kall_s0_*.npz"))
        maps = np.load(path)["maps"].astype(np.float32)
        masks = cache.get(args.root, args.cache, category, "test").mask
        rows.append((metrics.aupro(masks, maps), official_aupro(masks, maps)))
        print(f"{category:12s} {rows[-1][0]:6.3f} {rows[-1][1]:8.3f}")
    ours, official = np.mean(rows, axis=0)
    print(f"{'mean':12s} {ours:6.3f} {official:8.3f}")
    out = Path("results/analysis/aupro_vs_official.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, index=list(mvtec3d.CATEGORIES), columns=["ours", "official"])
    table.rename_axis("category").to_csv(out, float_format="%.4f")
    print("wrote", out)


if __name__ == "__main__":
    main()
