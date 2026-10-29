"""CPU latency and memory of the inspector on a laptop.

Three parts, each written to its own CSV in results/analysis/:

  - latency_stages.csv: one fitted model loaded on the CPU and given raw test
    scans one at a time: 1, 2 and 4 threads, float16 against float32 banks,
    and the Apple GPU (MPS) for comparison. Milliseconds per stage and peak memory (RSS) of the
    process. Each setting runs in its own process, so the thread count also
    reaches open3d's OpenMP and the peak memory belongs to that setting alone.
  - latency_coreset.csv: the banks thinned to 1-20% of the training patches.
    Image AUROC, false alarm rate on normal parts, bank size and time per
    part for the nearest-neighbour search. Greedy k-center is a prefix
    process (the first m picks don't depend on how many come after), so one
    run at the largest ratio gives every smaller bank exactly.
  - latency_onnx.csv: the backbone exported to ONNX and run with
    onnxruntime, against PyTorch on the same threads, with the largest
    difference in the features.

These are numbers for an M4 MacBook Air with nothing else running, not for
any factory hardware.

usage: OMP_NUM_THREADS=4 uv run --group bench python scripts/benchmark_latency.py all \\
           data/raw/mvtec3d --cache data/processed
       ... coreset data/raw/mvtec3d --categories carrot dowel   (adds to latency_coreset.csv)
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from factorylens.data import cache, mvtec3d
from factorylens.eval.metrics import image_auroc
from factorylens.inspector import Inspector, InspectorConfig
from factorylens.models import fpfh
from factorylens.models.memory_bank import greedy_coreset
from factorylens.models.patchcore import PatchCore
from factorylens.repro import run_info
from factorylens.robustness import points

RATIOS = (0.01, 0.025, 0.05, 0.1, 0.2)
THREADS = (1, 2, 4)
# banks, device, threads
SETTINGS = [
    *[("float16", "cpu", t) for t in THREADS],
    ("float32", "cpu", 4),
    ("float16", "mps", 4),
]


def peak_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3  # bytes on macOS, KiB on Linux


def scans(root: str, category: str, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    samples = mvtec3d.list_samples(root, category, "test")
    pick = np.random.default_rng(0).choice(len(samples), n, replace=False)
    return [
        (mvtec3d.load_rgb(samples[i].rgb_path), mvtec3d.load_xyz(samples[i].xyz_path)) for i in pick
    ]


# -- one process per setting -------------------------------------------------------


def stages(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    imported = peak_mb()
    start = time.perf_counter()
    model = Inspector.load(args.model, device=args.device)
    load_s = time.perf_counter() - start
    loaded = peak_mb()
    parts = scans(args.root, model.meta["category"], args.warmup + args.parts)
    rows = []
    for i, (rgb, xyz) in enumerate(parts):
        model.inspect_raw(rgb, xyz)
        if i >= args.warmup:
            rows.append(model.timings)
    times = pd.DataFrame(rows)
    print(
        json.dumps(
            {
                "median": times.median().to_dict(),
                "p90": times.quantile(0.9).to_dict(),
                "load_s": load_s,
                "peak_mb_imported": imported,
                "peak_mb_loaded": loaded,
                "peak_mb": peak_mb(),
            }
        )
    )


def run_stages(args: argparse.Namespace, models: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for banks, device, threads in SETTINGS:
        if device == "mps" and not torch.backends.mps.is_available():
            continue
        path = models[banks]
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        cmd = [sys.executable, __file__, "stages", args.root, "--model", str(path)]
        cmd += ["--threads", str(threads), "--parts", str(args.parts), "--device", device]
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        row = {
            "banks": banks,
            "device": device,
            "threads": threads,
            "file_mb": path.stat().st_size / 1e6,
        }
        row |= {f"{k}_ms": 1000 * v for k, v in r["median"].items()}
        row["total_p90_ms"] = 1000 * r["p90"]["total"]
        row |= {k: v for k, v in r.items() if k not in ("median", "p90")}
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()})
        rows.append(row)
    return pd.DataFrame(rows)


# -- coreset size --------------------------------------------------------------------


def run_coreset(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    config = InspectorConfig()
    feature_cache = Path("outputs/features/fpfh")
    rows, models = [], {}
    for category in args.categories:
        splits = {
            s: cache.get(args.root, args.cache, category, s, config.sweep.settings)
            for s in ("train", "validation", "test")
        }
        extractor = PatchCore(config.sweep.rgb)
        feats = {
            s: (
                extractor.embed(p.rgb).cpu(),
                fpfh.patches_for(p, config.sweep.fpfh, feature_cache).cpu(),
            )
            for s, p in splits.items()
        }
        flat = [f.reshape(-1, f.shape[-1]) for f in feats["train"]]
        start = time.perf_counter()
        bank = config.sweep.rgb.bank_config()
        picks = [
            greedy_coreset(f, int(len(f) * max(RATIOS)), bank.projection_dim, bank.seed)
            for f in flat
        ]
        print(f"{category}: coreset of {max(RATIOS):.0%} in {time.perf_counter() - start:.0f} s")

        test = splits["test"]
        y = test.is_anomalous
        for ratio in RATIOS:
            model = Inspector(config, device="cpu")
            model.size = (test.rgb.shape[1], test.rgb.shape[2])
            m = [int(len(f) * ratio) for f in flat]
            model.rgb_bank.bank = flat[0][picks[0][: m[0]]].contiguous()
            model.geo_bank.bank = flat[1][picks[1][: m[1]]].contiguous()
            model.calibrate(splits["validation"], *feats["validation"])

            r = model.rgb_bank.score(feats["test"][0], model.size)
            g = model.geo_bank.score(feats["test"][1], model.size)
            (fused, _), _ = model.fusion.fuse(r, g, test.foreground, points(test))
            alarm = fused > model.threshold

            n = min(args.parts, len(test))  # one part at a time, as on the line
            knn = []
            for i in range(n):
                start = time.perf_counter()
                model.rgb_bank.score(feats["test"][0][i : i + 1], model.size)
                model.geo_bank.score(feats["test"][1][i : i + 1], model.size)
                knn.append(time.perf_counter() - start)
            row = {
                "category": category,
                "coreset_ratio": ratio,
                "bank_rgb": m[0],
                "bank_fpfh": m[1],
                "bank_mb": 4 * (m[0] * flat[0].shape[1] + m[1] * flat[1].shape[1]) / 1e6,
                "auroc_rgb": image_auroc(y, r[0]),
                "auroc_fpfh": image_auroc(y, g[0]),
                "auroc": image_auroc(y, fused),
                "false_alarm": float(alarm[~y].mean()),
                "caught": float(alarm[y].mean()),
                "banks_ms": 1000 * float(np.median(knn)),
            }
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
            rows.append(row)
            if category == args.categories[0] and ratio == config.sweep.rgb.coreset_ratio:
                model.meta = {"category": category}
                for banks, half in (("float16", True), ("float32", False)):
                    models[banks] = out_dir / f"{category}_{banks}.pt"
                    model.save(models[banks], half=half)
    return pd.DataFrame(rows), models


# -- ONNX ----------------------------------------------------------------------------


class _Layers(torch.nn.Module):
    """The backbone with a tuple output, which the exporter wants."""

    def __init__(self, backbone: torch.nn.Module, layers: tuple[str, ...]) -> None:
        super().__init__()
        self.backbone, self.layers = backbone, layers

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.backbone(x)
        return tuple(out[name] for name in self.layers)


def timed(fn: Callable[[], object], repeats: int) -> float:
    for _ in range(3):
        fn()
    t = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        t.append(time.perf_counter() - start)
    return float(np.median(t))


def run_onnx(args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    import onnxruntime as ort  # bench group: uv sync --group bench

    config = InspectorConfig().sweep.rgb
    net = _Layers(PatchCore(config, device="cpu").backbone, config.layers).eval()
    size = InspectorConfig().sweep.settings.size
    x = torch.rand(1, 3, size, size)
    path = out_dir / "backbone.onnx"
    torch.onnx.export(net, (x,), path, input_names=["image"], dynamo=True)

    rows = []
    for threads in THREADS:
        torch.set_num_threads(threads)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        feed = {"image": x.numpy()}
        with torch.no_grad():
            ref = [t.numpy() for t in net(x)]
        got = session.run(None, feed)
        pairs = zip(got, ref, strict=True)
        diff = max(float(np.abs(a - b).max() / np.abs(b).max()) for a, b in pairs)
        torch_s = timed(lambda: net(x), args.repeats)
        ort_s = timed(lambda s=session, f=feed: s.run(None, f), args.repeats)
        row = {
            "threads": threads,
            "torch_ms": 1000 * torch_s,
            "onnxruntime_ms": 1000 * ort_s,
            "max_rel_diff": diff,
            # the weights go to a .data file next to the graph
            "onnx_mb": sum(f.stat().st_size for f in out_dir.glob(f"{path.name}*")) / 1e6,
        }
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["all", "coreset", "stages"])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--categories", nargs="+", default=["bagel", "foam", "tire"])
    parser.add_argument("--parts", type=int, default=30, help="parts timed per setting")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20, help="backbone runs per setting")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--work", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    if args.mode == "stages":
        stages(args)
        return

    git = run_info()["git"]
    commit = f"{git['commit'][:7]}{'-dirty' if git['dirty'] else ''}"
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    coreset, models = run_coreset(args, args.work)
    coreset["commit"] = commit
    path = args.out / "latency_coreset.csv"
    if path.exists():  # replace these categories, keep the rest
        before = pd.read_csv(path)
        coreset = pd.concat([before[~before["category"].isin(args.categories)], coreset])
    coreset.round(5).to_csv(path, index=False)
    print(coreset.groupby("coreset_ratio").mean(numeric_only=True).round(4))
    if args.mode == "coreset":
        return
    tables = {"stages": run_stages(args, models), "onnx": run_onnx(args, args.work)}
    for name, table in tables.items():
        table["commit"] = commit
        table.round(5).to_csv(args.out / f"latency_{name}.csv", index=False)


if __name__ == "__main__":
    main()
