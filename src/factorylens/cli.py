"""Command line entry point: `factorylens <group> <command>`."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from factorylens import __version__
from factorylens.data import mvtec3d, subsets

DEFAULT_ROOT = "data/raw/mvtec3d"
DEFAULT_KSHOT = "configs/kshot.json"


def data_check(args: argparse.Namespace) -> int:
    categories = args.category or mvtec3d.CATEGORIES
    all_problems = []
    for category in categories:
        start = time.perf_counter()
        problems = mvtec3d.check_category(
            args.root, category, expected=mvtec3d.EXPECTED_COUNTS, deep=args.deep
        )
        counts = "/".join(str(n) for n in mvtec3d.EXPECTED_COUNTS[category])
        status = "ok" if not problems else f"{len(problems)} problems"
        print(f"{category:12s} {counts:12s} {status:12s} {time.perf_counter() - start:5.1f}s")
        all_problems.extend(problems)

    for p in all_problems:
        print("  " + p)
    return 1 if all_problems else 0


def data_kshot(args: argparse.Namespace) -> int:
    ks = tuple(args.k or subsets.KS)
    built = subsets.build(args.root, args.category or mvtec3d.CATEGORIES, ks)
    subsets.save(built, args.out)
    print(f"wrote {len(built)} categories x {len(subsets.SEEDS)} seeds, k={ks} to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="factorylens")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    groups = parser.add_subparsers(dest="group", required=True)

    data = groups.add_parser("data", help="dataset utilities")
    data_cmds = data.add_subparsers(dest="command", required=True)

    check = data_cmds.add_parser("check", help="check a local MVTec 3D-AD copy")
    check.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    check.add_argument(
        "--category",
        action="append",
        choices=mvtec3d.CATEGORIES,
        help="only check this category (can be repeated)",
    )
    check.add_argument("--deep", action="store_true", help="load every sample and compare sizes")
    check.set_defaults(func=data_check)

    kshot = data_cmds.add_parser("kshot", help="write the fixed few-shot training subsets")
    kshot.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    kshot.add_argument("--out", default=DEFAULT_KSHOT)
    kshot.add_argument("--category", action="append", choices=mvtec3d.CATEGORIES)
    kshot.add_argument("--k", action="append", type=int, help=f"default: {subsets.KS}")
    kshot.set_defaults(func=data_kshot)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
