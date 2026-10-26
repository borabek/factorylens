"""Fit, save and reload an Inspector for every category, and check what comes back.

For each category:

  - the reloaded model's RGB and FPFH scores against the sweep's full-data
    runs for the same category (they should be identical);
  - the test parts streamed in a random order, as they would arrive on a
    line: fused image AUROC, false alarm rate on normal parts and share of
    defective parts caught at the model's alpha;
  - seconds per part (preprocessed input) and the model file size.

Models are saved with float16 banks (`--half`) unless told otherwise.

usage: OMP_NUM_THREADS=4 uv run python scripts/verify_inspectors.py data/raw/mvtec3d \\
           --cache data/processed
Writes results/analysis/inspectors.csv.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from factorylens import experiments
from factorylens.data import cache
from factorylens.data.mvtec3d import CATEGORIES
from factorylens.eval.metrics import image_auroc
from factorylens.inspector import Inspector, InspectorConfig
from factorylens.repro import run_info


def sweep_scores(results: Path, scores: Path, model: str, category: str) -> np.ndarray:
    runs = experiments.collect(results / model)
    run = runs[runs["k"].isna() & (runs["category"] == category)].iloc[0]
    return np.load(scores / model / Path(run["file"]).with_suffix(".npz").name)["scores"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--full", action="store_true", help="save float32 banks")
    parser.add_argument("--sweep", type=Path, default=Path("results/sweep"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/sweep"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis/inspectors.csv"))
    args = parser.parse_args()

    git = run_info()["git"]
    rows = []
    for category in CATEGORIES:
        config = InspectorConfig()
        splits = {
            s: cache.get(args.root, args.cache, category, s, config.sweep.settings)
            for s in ("train", "validation", "test")
        }
        start = time.perf_counter()
        fitted = Inspector(config).fit(splits["train"], splits["validation"])
        fit_s = time.perf_counter() - start
        path = args.models / f"{category}.pt"
        fitted.save(path, half=not args.full)
        model = Inspector.load(path)

        test = splits["test"]
        y = test.is_anomalous
        rgb = np.empty(len(test))
        geo = np.empty(len(test))
        order = np.random.default_rng(0).permutation(len(test))
        fused, alarm = np.empty(len(test)), np.empty(len(test), bool)
        start = time.perf_counter()
        for i in order:  # arrival order: random, so the window never runs along one defect type
            r = model.inspect(test.rgb[i], test.xyz[i], test.foreground[i])
            rgb[i], geo[i], fused[i], alarm[i] = r.score_rgb, r.score_3d, r.score, r.alarm
        per_part = (time.perf_counter() - start) / len(test)
        ref_rgb = sweep_scores(args.sweep, args.scores, "rgb", category)
        ref_geo = sweep_scores(args.sweep, args.scores, "fpfh", category)
        rows.append(
            {
                "category": category,
                "fit_s": fit_s,
                "ms_per_part": 1000 * per_part,
                "file_mb": path.stat().st_size / 1e6,
                "max_rel_diff_rgb": float(np.max(np.abs(rgb - ref_rgb) / np.abs(ref_rgb))),
                "max_rel_diff_fpfh": float(np.max(np.abs(geo - ref_geo) / np.abs(ref_geo))),
                "auroc": image_auroc(y, fused),
                "false_alarm": float(alarm[~y].mean()),
                "caught": float(alarm[y].mean()),
                "alpha": config.alpha,
                "half": not args.full,
                "commit": f"{git['commit'][:7]}{'-dirty' if git['dirty'] else ''}",
            }
        )
        print(
            {k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True
        )

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(args.out, index=False)
    print(table.drop(columns=["commit"]).mean(numeric_only=True).round(4))


if __name__ == "__main__":
    main()
