"""How much of each defect survives background-plane removal, per category.

usage: uv run python scripts/check_background_removal.py data/raw/mvtec3d [--threshold 0.005]

A defect pixel that lies on the background plane (e.g. table showing through a
cut or crack) is removed by design, so 100% is not reachable everywhere. The
point is to see where the threshold eats into the object itself.
"""

import argparse

import numpy as np

from factorylens.data import mvtec3d, preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--threshold", type=float, default=preprocess.DEFAULT_THRESHOLD)
    args = parser.parse_args()

    print(f"threshold {args.threshold * 1000:.0f} mm")
    print(f"{'category':12s} {'fg':>5s} {'kept':>6s} {'p5':>6s} {'<50%':>8s}")
    for category in mvtec3d.CATEGORIES:
        kept, fg_frac = [], []
        for s in mvtec3d.list_samples(args.root, category, "test"):
            xyz = mvtec3d.load_xyz(s.xyz_path)
            _, fg, _ = preprocess.remove_background(xyz, threshold=args.threshold)
            fg_frac.append(fg.mean())
            if s.is_anomalous and s.gt_path is not None:
                mask = mvtec3d.load_mask(s.gt_path) & mvtec3d.valid_points(xyz)
                if mask.any():
                    kept.append(fg[mask].mean())
        k = np.array(kept)
        print(
            f"{category:12s} {np.mean(fg_frac):5.2f} {k.mean():6.3f} {np.percentile(k, 5):6.3f}"
            f" {int((k < 0.5).sum()):3d}/{len(k):<4d}"
        )


if __name__ == "__main__":
    main()
