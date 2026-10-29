"""Command line entry point: `factorylens <group> <command>`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from factorylens import __version__, corruptions, experiments, robustness, sweep
from factorylens.data import cache, ff, mvtec3d, subsets
from factorylens.models import fpfh
from factorylens.models.detectors import (
    BTFDetector,
    Detector,
    FPFHDetector,
    FPFHDetectorConfig,
    RGBDetector,
)
from factorylens.models.memory_bank import BankConfig
from factorylens.models.patchcore import BTF_RGB, PatchCoreConfig

DEFAULT_ROOT = "data/raw/mvtec3d"
DEFAULT_KSHOT = "configs/kshot.json"
DEFAULT_FF_ROOT = "data/raw/ff_analog/Analog Dataset"
DEFAULT_CACHE = "data/processed"
DEFAULT_FEATURES = "outputs/features/fpfh"
PREPROCESS = {"default": cache.Settings(), "btf": cache.BTF_SETTINGS}


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


def rgb_run(args: argparse.Namespace) -> int:
    config = PatchCoreConfig(
        arch=args.arch, layers=tuple(args.layer or PatchCoreConfig.layers), grid=args.grid
    )
    for category in args.category or mvtec3d.CATEGORIES:
        for seed in args.seed or [0]:
            r = experiments.run_rgb(
                args.root,
                args.cache,
                category,
                k=args.k,
                seed=seed,
                model_config=config,
                kshot_path=args.kshot,
                results_dir=args.results,
                scores_dir=None if args.no_scores else args.scores,
                device=args.device,
            )
            print(
                f"{category:12s} k={args.k or 'all'} seed={seed}  "
                f"image AUROC {r['image_auroc']:.3f} "
                f"[{r['image_auroc_ci_low']:.3f}, {r['image_auroc_ci_high']:.3f}]  "
                f"pixel AUROC {r['pixel_auroc']:.3f}  AUPRO {r['aupro']:.3f}  "
                f"fit {r['fit_s']:.0f}s",
                flush=True,
            )
    return 0


def results_table(args: argparse.Namespace) -> int:
    df = experiments.collect(args.results)
    if df.empty:
        print(f"no results in {args.results}")
        return 1
    cols = ["image_auroc", "pixel_auroc", "aupro"]
    group = [c for c in args.by if c in df.columns]
    per_cat = df.groupby([*group, "category"])[cols].mean().unstack("category")
    print(df.groupby(group)[cols].mean().round(3).to_markdown() if group else df[cols].mean())
    if args.per_category:
        for col in cols:
            print(f"\n{col}")
            print(per_cat[col].round(3).to_markdown())
    return 0


def _fpfh_category(job: tuple[str, str, str, str, cache.Settings]) -> str:
    root, cache_dir, features_dir, category, settings = job
    start = time.perf_counter()
    for split in mvtec3d.SPLITS:
        fpfh.patches_for(cache.get(root, cache_dir, category, split, settings), None, features_dir)
    return f"{category:12s} {time.perf_counter() - start:6.1f}s"


def data_fpfh(args: argparse.Namespace) -> int:
    settings = PREPROCESS[args.preprocess]
    jobs = [
        (args.root, args.cache, args.features, c, settings)
        for c in args.category or mvtec3d.CATEGORIES
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for line in pool.map(_fpfh_category, jobs):
            print(line, flush=True)
    return 0


def make_detector(model: str, seed: int, args: argparse.Namespace) -> Detector:
    if model == "rgb":
        return RGBDetector(PatchCoreConfig(seed=seed), device=args.device)
    if model == "rgb-btf":
        return RGBDetector(replace(BTF_RGB, seed=seed), device=args.device)
    if model == "fpfh":
        return FPFHDetector(
            FPFHDetectorConfig(bank=BankConfig(seed=seed)), feature_cache=args.features
        )
    if model == "btf":
        weight = args.fpfh_weight if args.fpfh_weight == "balanced" else float(args.fpfh_weight)
        return BTFDetector(
            replace(BTF_RGB, seed=seed),
            device=args.device,
            feature_cache=args.features,
            fpfh_weight=weight,
        )
    raise ValueError(model)


def run_models(args: argparse.Namespace) -> int:
    settings = PREPROCESS[args.preprocess]
    results = args.results or f"results/{args.model}" + (
        "" if args.preprocess == "default" else f"-{args.preprocess}prep"
    )
    for category in args.category or mvtec3d.CATEGORIES:
        for seed in args.seed or [0]:
            r = experiments.run(
                make_detector(args.model, seed, args),
                args.root,
                args.cache,
                category,
                seed=seed,
                settings=settings,
                results_dir=results,
                scores_dir=None if args.no_scores else f"outputs/scores/{Path(results).name}",
            )
            print(
                f"{args.model:6s} {category:12s} seed={seed}  "
                f"image AUROC {r['image_auroc']:.3f}  pixel AUROC {r['pixel_auroc']:.3f}  "
                f"AUPRO {r['aupro']:.3f}  fit {r['fit_s']:.0f}s",
                flush=True,
            )
    return 0


def run_sweep(args: argparse.Namespace) -> int:
    ks = [None if k == "all" else int(k) for k in (args.k or ["1", "5", "10", "50", "all"])]
    models = tuple(args.model or sweep.MODELS)
    for category in args.category or mvtec3d.CATEGORIES:
        sweep.run_category(
            args.root,
            args.cache,
            category,
            ks=ks,
            seeds=args.seed or [0, 1, 2, 3, 4],
            models=models,
            results_dir=args.results,
            feature_cache=args.features,
            device=args.device,
            log=lambda line: print(line, flush=True),
        )
    return 0


def fit_model(args: argparse.Namespace) -> int:
    from factorylens.inspector import Inspector, InspectorConfig

    config = InspectorConfig(alpha=args.alpha)
    if args.coreset_ratio:
        rgb = replace(config.sweep.rgb, coreset_ratio=args.coreset_ratio)
        config = replace(config, sweep=replace(config.sweep, rgb=rgb))
    splits = {
        s: cache.get(args.root, args.cache, args.category, s, config.sweep.settings)
        for s in ("train", "validation")
    }
    inspector = Inspector(config, device=args.device).fit(splits["train"], splits["validation"])
    out = Path(args.out or f"models/{args.category}.pt")
    inspector.save(out, half=args.half)
    print(
        f"{args.category}: {len(splits['train'])} training and {len(splits['validation'])} "
        f"validation parts, alarm threshold {inspector.threshold:.3f} for alpha {args.alpha}"
    )
    print(f"wrote {out}")
    return 0


def inspect_parts(args: argparse.Namespace) -> int:
    from factorylens.data import mvtec3d
    from factorylens.inspector import Inspector

    inspector = Inspector.load(args.model, device=args.device)
    if len(args.files) % 2:
        print("give the files in pairs: RGB image, then xyz TIFF", file=sys.stderr)
        return 2
    for rgb_path, xyz_path in zip(args.files[::2], args.files[1::2], strict=True):
        result = inspector.inspect_raw(mvtec3d.load_rgb(rgb_path), mvtec3d.load_xyz(xyz_path))
        print(json.dumps({"rgb": rgb_path, **result.as_dict()}))
    return 0


def run_demo(args: argparse.Namespace) -> int:
    from factorylens import demo

    demo.build_app(args.models, args.root, args.cache, max_parts=args.max_parts).launch(
        server_name=args.host, server_port=args.port
    )
    return 0


def serve_model(args: argparse.Namespace) -> int:
    from factorylens import api

    api.serve(args.model, host=args.host, port=args.port)
    return 0


def run_robustness(args: argparse.Namespace) -> int:
    conditions = robustness.all_conditions()
    if args.corruption:
        conditions = [(n, s) for n, s in conditions if n in args.corruption]
    for category in args.category or mvtec3d.CATEGORIES:
        robustness.run_category(
            args.root,
            args.cache,
            category,
            conditions=conditions,
            models=tuple(args.model or robustness.ALL_MODELS),
            seed=args.seed,
            scores_dir=args.scores,
            results_dir=args.results,
            feature_cache=args.features,
            device=args.device,
            log=lambda line: print(line, flush=True),
        )
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

    rgb = groups.add_parser("rgb", help="RGB-only detector (PatchCore)")
    rgb_cmds = rgb.add_subparsers(dest="command", required=True)
    run = rgb_cmds.add_parser("run", help="fit on train, score test, save metrics")
    run.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    run.add_argument("--cache", default=DEFAULT_CACHE)
    run.add_argument("--category", action="append", choices=mvtec3d.CATEGORIES)
    run.add_argument(
        "--k", type=int, choices=subsets.KS, help="k-shot; default: all training samples"
    )
    run.add_argument("--seed", type=int, action="append")
    run.add_argument("--arch", default=PatchCoreConfig.arch)
    run.add_argument("--layer", action="append", help="default: layer2 and layer3")
    run.add_argument("--grid", type=int, help="fixed patch grid side, e.g. 28")
    run.add_argument("--kshot", default=DEFAULT_KSHOT)
    run.add_argument("--results", default="results/rgb")
    run.add_argument("--scores", default="outputs/scores/rgb")
    run.add_argument("--no-scores", action="store_true", help="don't keep per-image scores")
    run.add_argument("--device")
    run.set_defaults(func=rgb_run)

    res = groups.add_parser("results", help="summarise result files")
    res_cmds = res.add_subparsers(dest="command", required=True)
    table = res_cmds.add_parser("table", help="mean metrics, optionally per category")
    table.add_argument("results")
    table.add_argument("--by", action="append", default=[], help="config field to group by")
    table.add_argument("--per-category", action="store_true")
    table.set_defaults(func=results_table)

    feats = data_cmds.add_parser("fpfh", help="precompute FPFH patch features")
    feats.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    feats.add_argument("--cache", default=DEFAULT_CACHE)
    feats.add_argument("--features", default=DEFAULT_FEATURES)
    feats.add_argument("--preprocess", choices=PREPROCESS, default="default")
    feats.add_argument("--category", action="append", choices=mvtec3d.CATEGORIES)
    feats.add_argument("--jobs", type=int, default=4)
    feats.set_defaults(func=data_fpfh)

    runner = groups.add_parser("run", help="fit and evaluate a detector on every category")
    runner.add_argument("model", choices=["rgb", "rgb-btf", "fpfh", "btf"])
    runner.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    runner.add_argument("--cache", default=DEFAULT_CACHE)
    runner.add_argument("--features", default=DEFAULT_FEATURES)
    runner.add_argument("--preprocess", choices=PREPROCESS, default="default")
    runner.add_argument("--category", action="append", choices=mvtec3d.CATEGORIES)
    runner.add_argument("--seed", type=int, action="append")
    runner.add_argument("--results", help="default: results/<model>[-<preprocess>prep]")
    runner.add_argument("--fpfh-weight", default="1", help="btf only: a number or 'balanced'")
    runner.add_argument("--no-scores", action="store_true")
    runner.add_argument("--device")
    runner.set_defaults(func=run_models)

    sw = groups.add_parser("sweep", help="rgb/fpfh/early/late over k-shot and seeds")
    sw.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    sw.add_argument("--cache", default=DEFAULT_CACHE)
    sw.add_argument("--features", default=DEFAULT_FEATURES)
    sw.add_argument("--category", action="append", choices=mvtec3d.CATEGORIES)
    sw.add_argument("--k", action="append", help="1, 5, 10, 50 or all (default: all of them)")
    sw.add_argument("--seed", type=int, action="append")
    sw.add_argument("--model", action="append", choices=[*sweep.MODELS, "rgb_aug"])
    sw.add_argument("--results", default="results/sweep")
    sw.add_argument("--device")
    sw.set_defaults(func=run_sweep)

    fit = groups.add_parser("fit", help="fit an inspector for one product and save it")
    fit.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    fit.add_argument("--category", required=True, choices=mvtec3d.CATEGORIES)
    fit.add_argument("--cache", default=DEFAULT_CACHE)
    fit.add_argument("--alpha", type=float, default=0.1, help="target false alarm rate")
    fit.add_argument("--out", help="default: models/<category>.pt")
    fit.add_argument("--half", action="store_true", help="store the banks as float16")
    fit.add_argument(
        "--coreset-ratio",
        type=float,
        help="share of training patches kept in each bank (default 0.1); see docs/latency.md",
    )
    fit.add_argument("--device")
    fit.set_defaults(func=fit_model)

    ins = groups.add_parser("inspect", help="inspect parts with a saved inspector, in order")
    ins.add_argument("model")
    ins.add_argument("files", nargs="+", help="pairs of RGB image and xyz TIFF")
    ins.add_argument("--device")
    ins.set_defaults(func=inspect_parts)

    srv = groups.add_parser("serve", help="HTTP API around a saved inspector (needs --extra serve)")
    srv.add_argument("model")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(func=serve_model)

    dm = groups.add_parser("demo", help="interactive demo of fitted models (needs --extra demo)")
    dm.add_argument("--models", default="models")
    dm.add_argument("--root", default=DEFAULT_ROOT)
    dm.add_argument("--cache", default=DEFAULT_CACHE)
    dm.add_argument("--host", default="127.0.0.1")
    dm.add_argument("--port", type=int, default=7860)
    dm.add_argument("--max-parts", type=int, help="shorter runs in the 'a sensor fails' view")
    dm.set_defaults(func=run_demo)

    rb = groups.add_parser("robustness", help="the sweep detectors on FactoryLens-C")
    rb.add_argument("root", nargs="?", default=DEFAULT_ROOT)
    rb.add_argument("--cache", default=DEFAULT_CACHE)
    rb.add_argument("--features", default=DEFAULT_FEATURES)
    rb.add_argument("--category", action="append", choices=mvtec3d.CATEGORIES)
    rb.add_argument("--corruption", action="append", choices=list(corruptions.CORRUPTIONS))
    rb.add_argument("--model", action="append", choices=list(robustness.ALL_MODELS))
    rb.add_argument("--scores", default="outputs/scores/robustness")
    rb.add_argument("--seed", type=int, default=0)
    rb.add_argument("--results", default="results/robustness")
    rb.add_argument("--device")
    rb.set_defaults(func=run_robustness)

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
