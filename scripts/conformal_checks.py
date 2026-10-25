"""Where the conformal alarm rate goes wrong, and what fixes it.

Follow-up to `conformal_alarms.py`, from the same saved scores
(outputs/scores/robustness-extent).

1. exchangeability: do the normal test parts score like the validation parts?
   One-sided Mann-Whitney test per category and sensor (test higher), with
   Holm's correction over the 20 tests;
2. recalibration: add m confirmed-good parts from the current run to the
   calibration set (m in {5, 10}) and measure the false alarm rate on the
   remaining normal parts, 200 random draws;
3. strict split: late fusion standardises with the same validation parts it
   calibrates on. Here the validation parts are split in two, one half for the
   z-scores and one for the threshold, 200 random splits;
4. weight-matched calibration for reliability fusion: calibrate each test part
   on validation parts fused with that part's own weights. Tried because the
   fused score's distribution changes with the weights.

usage: uv run python scripts/conformal_checks.py
Writes results/analysis/conformal_checks.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from factorylens import conformal, corruptions
from factorylens.data.mvtec3d import CATEGORIES

ALPHA = 0.1
REPEATS = 200


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values."""
    order = np.argsort(p)
    m = len(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def far(cal: np.ndarray, normal: np.ndarray) -> float:
    return float((normal > conformal.threshold(cal, ALPHA)).mean())


def per_sensor(folder: Path, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        d = np.load(folder / f"{category}_clean0_s0.npz")
        y = d["labels"].astype(bool)
        for sensor in ("rgb", "fpfh"):
            val, normal = d[f"{sensor}_val"], d[sensor][~y]
            row = {
                "category": category,
                "sensor": sensor,
                "check": "exchangeability",
                "shift_p": mannwhitneyu(normal, val, alternative="greater").pvalue,
                "shift_auc": float((normal[:, None] > val[None]).mean()),
                "false_alarm": far(val, normal),
            }
            for m in (5, 10):
                draws = []
                for _ in range(REPEATS):
                    idx = rng.permutation(len(normal))
                    draws.append(far(np.r_[val, normal[idx[:m]]], normal[idx[m:]]))
                row[f"false_alarm_plus{m}"] = float(np.mean(draws))
            rows.append(row)
    t = pd.DataFrame(rows)
    t["shift_p_holm"] = holm(t["shift_p"].to_numpy())
    return t


def strict_split(folder: Path, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        d = np.load(folder / f"{category}_clean0_s0.npz")
        y = d["labels"].astype(bool)
        rv, gv, rt, gt = d["rgb_val"], d["fpfh_val"], d["rgb"], d["fpfh"]
        z = lambda s, ref: (s - ref.mean()) / ref.std()  # noqa: E731
        same = far(z(rv, rv) + z(gv, gv), (z(rt, rv) + z(gt, gv))[~y])
        draws = []
        for _ in range(REPEATS):
            idx = rng.permutation(len(rv))
            a, b = idx[: len(rv) // 2], idx[len(rv) // 2 :]
            cal = z(rv[b], rv[a]) + z(gv[b], gv[a])
            draws.append(far(cal, (z(rt, rv[a]) + z(gt, gv[a]))[~y]))
        rows.append(
            {
                "category": category,
                "sensor": "late",
                "check": "strict split",
                "false_alarm": same,
                "false_alarm_split": float(np.mean(draws)),
            }
        )
    return pd.DataFrame(rows)


def weight_matched(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.npz")):
        name = next((c for c in ["clean", *corruptions.CORRUPTIONS] if f"_{c}" in path.name), None)
        if name is None or name == "clean":
            continue
        category = path.name.split(f"_{name}")[0]
        d = np.load(path)
        y = d["labels"].astype(bool)
        rv, gv = d["rgb_val"], d["fpfh_val"]
        vr, vg = (rv - rv.mean()) / rv.std(), (gv - gv.mean()) / gv.std()
        wr, wg = d["w_rgb_window"], d["w_fpfh_window"]
        fused = d["reliable_window"]
        fixed = fused > conformal.threshold((vr + vg) / np.sqrt(2), ALPHA)
        cal = (wr[:, None] * vr + wg[:, None] * vg) / np.sqrt(wr**2 + wg**2)[:, None]
        k = conformal.rank(len(rv), ALPHA)
        matched = fused > np.sort(cal, axis=1)[:, len(rv) - k]
        rows.append(
            {
                "category": category,
                "sensor": corruptions.CORRUPTIONS[name].sensor,
                "check": "weight-matched",
                "corruption": name,
                "false_alarm": fixed[~y].mean(),
                "false_alarm_matched": matched[~y].mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/robustness-extent"))
    parser.add_argument("--out", type=Path, default=Path("results/analysis/conformal_checks.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    sensors = per_sensor(args.scores, rng)
    split = strict_split(args.scores, rng)
    matched = weight_matched(args.scores)
    table = pd.concat([sensors, split, matched], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out, index=False)

    print(sensors.round(3).to_string(index=False))
    print(
        sensors.groupby("sensor")[["false_alarm", "false_alarm_plus5", "false_alarm_plus10"]]
        .mean()
        .round(3)
    )
    print(split[["false_alarm", "false_alarm_split"]].mean().round(3))
    print(matched.groupby("sensor")[["false_alarm", "false_alarm_matched"]].mean().round(3))


if __name__ == "__main__":
    main()
