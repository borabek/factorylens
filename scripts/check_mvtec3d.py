"""Check that a local copy of MVTec 3D-AD is complete and consistent.

usage: uv run python scripts/check_mvtec3d.py data/raw/mvtec3d [--deep]
"""

import argparse
import sys
import time

from factorylens.data import mvtec3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--deep", action="store_true", help="load every sample and compare sizes")
    args = parser.parse_args()

    all_problems = []
    for category in mvtec3d.CATEGORIES:
        start = time.perf_counter()
        problems = mvtec3d.check_category(args.root, category, deep=args.deep)
        counts = "/".join(str(n) for n in mvtec3d.EXPECTED_COUNTS[category])
        status = "ok" if not problems else f"{len(problems)} problems"
        print(f"{category:12s} {counts:12s} {status:12s} {time.perf_counter() - start:5.1f}s")
        all_problems.extend(problems)

    for p in all_problems:
        print("  " + p)
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
