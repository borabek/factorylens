"""A sequential test for sensor faults against the median window.

The window rule down-weights a sensor from the median health over recent parts,
with a window and a ramp set by hand. Here each sensor's health signal goes
through `conformal.FaultTest` instead: flagged means weight 0 until a reset,
otherwise weight 1. Its one meaningful setting is `level`, the mean number of
parts between false flags on an exchangeable stream.

Fixed before this script ran on anything but a prototype: epsilon 0.1, level 1000.
The prototype saw five faults at severity 3 on these test parts, so those
conditions are reported apart from the rest (`seen`).

Same streams as scripts/fault_stream.py: the test parts in a random order, clean
for the first half and from a corrupted run for the second. Every corruption at
every severity. Also: streams corrupted from the first part, and clean streams to
compare the false flag rate with the bound t / level.

The health of the validation parts isn't in the saved scores, so it is computed
once from the fitted models and kept in results/analysis/health_validation.csv.

usage: OMP_NUM_THREADS=4 uv run python scripts/sequential_faults.py \\
           --root data/raw/mvtec3d --cache data/processed
Writes results/analysis/sequential_*.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from fault_stream import fuse, stream
from scipy import stats

from factorylens import conformal, corruptions
from factorylens.data import cache
from factorylens.data.mvtec3d import CATEGORIES, valid_points
from factorylens.eval.metrics import image_auroc
from factorylens.inspector import Inspector
from factorylens.models import fpfh

EPSILON, LEVEL = 0.1, 1000.0
GRID = [(e, a) for e in (0.05, 0.1, 0.2) for a in (100.0, 1000.0, 10000.0)]
SEEN = {"gaussian_noise", "defocus_blur", "jpeg", "depth_noise", "quantise"}
ALPHA = 0.1


def validation_health(root: str, cache_dir: str, models: Path, features: str) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        m = Inspector.load(models / f"{category}.pt")
        s = m.config.sweep
        v = cache.get(root, cache_dir, category, "validation", s.settings)
        rgb = m.rgb_bank.score(m.rgb_model.embed(v.rgb), m.size)
        geo = m.geo_bank.score(fpfh.patches_for(v, s.fpfh, features).to(m.rgb_model.device), m.size)
        u_rgb, u_geo = m.fusion.signals(
            rgb, geo, v.foreground, valid_points(v.xyz).sum(axis=(1, 2))
        )
        rows += [{"category": category, "sensor": "rgb", "u": u} for u in u_rgb]
        rows += [{"category": category, "sensor": "xyz", "u": u} for u in u_geo]
        print(category, "validation health done", flush=True)
    return pd.DataFrame(rows)


def sequential(
    s: dict,
    cal_rgb: np.ndarray,
    cal_geo: np.ndarray,
    zr: np.ndarray,
    zg: np.ndarray,
    setting: tuple,
) -> tuple:
    """Weights from two fault tests along the stream, and the fused scores."""
    epsilon, level = setting
    tests = [
        conformal.FaultTest(c, epsilon, level, seed=i) for i, c in enumerate((cal_rgb, cal_geo))
    ]
    wr = np.array([0.0 if tests[0].update(u) else 1.0 for u in s["u_rgb"]])
    wg = np.array([0.0 if tests[1].update(u) else 1.0 for u in s["u_fpfh"]])
    off = (wr == 0) & (wg == 0)
    wr[off], wg[off] = 1.0, 1.0
    return (wr * zr + wg * zg) / np.sqrt(wr**2 + wg**2), wr, wg


def rates(scores: np.ndarray, y: np.ndarray, threshold: float) -> tuple[float, float]:
    alarm = scores > threshold
    return float(alarm[~y].mean()), float(alarm[y].mean())


def measure(
    s: dict,
    onset: int,
    fused: np.ndarray,
    wr: np.ndarray,
    wg: np.ndarray,
    sensor: str,
    threshold: float,
) -> dict:
    y = s["labels"].astype(bool)
    broken, healthy = (wr, wg) if sensor == "rgb" else (wg, wr)
    early = bool((wr[:onset] < 0.5).any() or (wg[:onset] < 0.5).any())
    low = np.flatnonzero(broken[onset:] < 0.5)
    far, caught = rates(fused[onset:], y[onset:], threshold)
    return {
        "auroc_before": image_auroc(y[:onset], fused[:onset]) if onset else np.nan,
        "auroc_after": image_auroc(y[onset:], fused[onset:]),
        "false_alarm_after": far,
        "caught_after": caught,
        "false_flag": early,
        "healthy_off_after": bool((healthy[onset:] < 0.5).any()),
        "noticed": len(low) > 0,
        # only streams with no false flag before the fault, or the delay would read 0
        "delay": float(low[0]) if len(low) and not early else np.nan,
    }


def run(scores: Path, health: pd.DataFrame, seeds: int, grid_seeds: int) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        clean = dict(np.load(scores / f"{category}_clean0_s0.npz"))
        cal = {
            k: g["u"].to_numpy()
            for k, g in health[health["category"] == category].groupby("sensor")
        }
        rv, gv = clean["rgb_val"], clean["fpfh_val"]
        zr_v, zg_v = (rv - rv.mean()) / rv.std(), (gv - gv.mean()) / gv.std()
        limits = {
            "stouffer": conformal.threshold((zr_v + zg_v) / np.sqrt(2), ALPHA),
            "late": conformal.threshold(zr_v + zg_v, ALPHA),
        }
        for name, c in corruptions.CORRUPTIONS.items():
            for severity in range(1, c.severities + 1):
                bad = dict(np.load(scores / f"{category}_{name}{severity}_s0.npz"))
                for seed in range(seeds):
                    s, _, onset = stream(clean, bad, seed)
                    zr = (s["rgb"] - rv.mean()) / rv.std()
                    zg = (s["fpfh"] - gv.mean()) / gv.std()
                    runs = {
                        "late": (None, limits["late"]),
                        "window 4": (4, limits["stouffer"]),
                        "window 16": (16, limits["stouffer"]),
                    }
                    results = {d: (*fuse(s, zr, zg, w), t) for d, (w, t) in runs.items()}
                    settings = GRID if seed < grid_seeds else [(EPSILON, LEVEL)]
                    for e, a in settings:
                        results[f"sequential e={e} A={a:.0f}"] = (
                            *sequential(s, cal["rgb"], cal["xyz"], zr, zg, (e, a)),
                            limits["stouffer"],
                        )
                    for detector, (fused, wr, wg, threshold) in results.items():
                        rows.append(
                            {
                                "category": category,
                                "corruption": name,
                                "severity": severity,
                                "sensor": c.sensor,
                                "seen": name in SEEN and severity == 3,
                                "seed": seed,
                                "detector": detector,
                                **measure(s, onset, fused, wr, wg, c.sensor, threshold),
                            }
                        )
        print(category, "streams done", flush=True)
    return pd.DataFrame(rows)


def clean_bound(scores: Path, health: pd.DataFrame, seeds: int) -> pd.DataFrame:
    """False flags on clean streams, against the bound t / level for one sensor."""
    rows = []
    for category in CATEGORIES:
        clean = dict(np.load(scores / f"{category}_clean0_s0.npz"))
        n = len(clean["labels"])
        y = clean["labels"].astype(bool)
        for sensor, key in (("rgb", "u_rgb"), ("xyz", "u_fpfh")):
            cal = health[(health["category"] == category) & (health["sensor"] == sensor)]["u"]
            flags = []
            for seed in range(seeds):
                order = np.random.default_rng(seed).permutation(n)
                test = conformal.FaultTest(cal.to_numpy(), EPSILON, LEVEL, seed=seed)
                flags.append(any(test.update(u) for u in clean[key][order]))
            normal, defective = clean[key][~y], clean[key][y]
            shift = stats.mannwhitneyu(normal, cal.to_numpy(), alternative="greater").pvalue
            rows.append(
                {
                    "category": category,
                    "sensor": sensor,
                    "parts": n,
                    "false_flag": float(np.mean(flags)),
                    "bound": n / LEVEL,
                    "normal_above_validation_p": float(shift),
                    "defective_median_u": float(np.median(defective)),
                    "normal_median_u": float(np.median(normal)),
                    "validation_median_u": float(cal.median()),
                }
            )
    return pd.DataFrame(rows)


def compare(table: pd.DataFrame, a: str, b: str, metric: str) -> float:
    """Wilcoxon over category means, as everywhere in this project."""
    means = table[table["detector"].isin([a, b])].groupby(["category", "detector"])[metric].mean()
    wide = means.unstack("detector")
    diff = wide[a] - wide[b]
    return float(stats.wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data/raw/mvtec3d")
    parser.add_argument("--cache", default="data/processed")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--features", default="outputs/features/fpfh")
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/robustness-reliable"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--grid-seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    path = args.out / "health_validation.csv"
    if not path.exists():
        validation_health(args.root, args.cache, args.models, args.features).round(5).to_csv(
            path, index=False
        )
    health = pd.read_csv(path)

    table = run(args.scores, health, args.seeds, args.grid_seeds)
    primary = f"sequential e={EPSILON} A={LEVEL:.0f}"
    metrics = {
        "auroc_before": ("auroc_before", "mean"),
        "auroc_after": ("auroc_after", "mean"),
        "false_alarm_after": ("false_alarm_after", "mean"),
        "caught_after": ("caught_after", "mean"),
        "false_flag": ("false_flag", "mean"),
        "healthy_off_after": ("healthy_off_after", "mean"),
        "noticed": ("noticed", "mean"),
        "delay": ("delay", "median"),
    }
    summary = table.groupby(["seen", "detector"]).agg(**metrics).reset_index()
    summary.round(4).to_csv(args.out / "sequential_summary.csv", index=False)
    by = table.groupby(["corruption", "sensor", "detector"]).agg(**metrics).reset_index()
    by.round(4).to_csv(args.out / "sequential_corruptions.csv", index=False)
    cats = table.groupby(["category", "detector"]).agg(**metrics).reset_index()
    cats.round(4).to_csv(args.out / "sequential_categories.csv", index=False)

    tests = []
    unseen = table[~table["seen"]]
    for other in ("window 16", "window 4", "late"):
        for metric in ("auroc_after", "false_alarm_after", "false_flag", "caught_after"):
            tests.append(
                {
                    "versus": other,
                    "metric": metric,
                    "unseen_p": compare(unseen, primary, other, metric),
                }
            )
    pd.DataFrame(tests).round(4).to_csv(args.out / "sequential_tests.csv", index=False)
    bound = clean_bound(args.scores, health, 200)
    bound.round(4).to_csv(args.out / "sequential_bound.csv", index=False)

    keep = ["late", "window 4", "window 16", primary]
    pd.set_option("display.width", 220)
    print(summary[summary["detector"].isin(keep)].round(3).to_string())
    print(summary[~summary["detector"].isin(keep[:3])].round(3).to_string())
    print(pd.DataFrame(tests).round(4).to_string())
    print(bound.round(3).to_string())


if __name__ == "__main__":
    main()
