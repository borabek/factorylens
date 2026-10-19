"""Command line entry point: `factorylens <group> <command>`."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

from factorylens import __version__
from factorylens.data import cache, ff, mvtec3d, subsets

DEFAULT_ROOT = "data/raw/mvtec3d"
DEFAULT_KSHOT = "configs/kshot.json"
DEFAULT_FF_ROOT = "data/raw/ff_analog/Analog Dataset"
DEFAULT_CACHE = "data/processed"


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


def _prepare_category(job: tuple[str, str, str, cache.Settings]) -> str:
    root, cache_dir, category, settings = job
    start = time.perf_counter()
    sizes = []
    for split in mvtec3d.SPLITS:
        cache.get(root, cache_dir, category, split, settings)
        sizes.append(cache.cache_path(cache_dir, category, split, settings).stat().st_size)
    return f"{category:12s} {sum(sizes) / 1e6:7.1f} MB {time.perf_counter() - start:6.1f}s"


def data_prepare(args: argparse.Namespace) -> int:
    settings = cache.Settings(size=args.size)
    categories = args.category or mvtec3d.CATEGORIES
    jobs = [(args.root, args.cache, c, settings) for c in categories]
    print(f"writing to {args.cache}/{settings.dirname()}")
    if args.jobs == 1:
        for job in jobs:
            print(_prepare_category(job))
        return 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(_prepare_category, jobs):
            print(line, flush=True)
    return 0


def ff_blocks(args: argparse.Namespace) -> int:
    cycles = ff.segment_cycles(ff.load_analog(args.root))
    blocks = ff.label_blocks(cycles)
    blocks["start"] = blocks["start"].dt.strftime("%m-%d %H:%M")
    blocks["end"] = blocks["end"].dt.strftime("%m-%d %H:%M")
    print(blocks.to_string())
    train, test = ff.cross_day_split(cycles, anomaly=args.anomaly)
    for name, part in [("train (day 1)", train), ("test (day 2)", test)]:
        counts = part["label"].value_counts()
        print(f"{name}: {len(part)} cycles, " + ", ".join(f"{k} {v}" for k, v in counts.items()))
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

    prepare = data_cmds.add_parser("prepare", help="cache resized, background-free samples")
    prepare.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    prepare.add_argument("--cache", default=DEFAULT_CACHE)
    prepare.add_argument("--size", type=int, default=cache.Settings.size)
    prepare.add_argument("--category", action="append", choices=mvtec3d.CATEGORIES)
    prepare.add_argument("--jobs", type=int, default=4)
    prepare.set_defaults(func=data_prepare)

    ff_group = groups.add_parser("ff", help="Future Factories analog data")
    ff_cmds = ff_group.add_subparsers(dest="command", required=True)
    blocks = ff_cmds.add_parser("blocks", help="label blocks and the cross-day split")
    blocks.add_argument("root", nargs="?", default=DEFAULT_FF_ROOT)
    blocks.add_argument("--anomaly", default="NoNose")
    blocks.set_defaults(func=ff_blocks)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
