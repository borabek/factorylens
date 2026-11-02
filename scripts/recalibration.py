"""Do confirmed-good parts from the current run repair the alarm rate of a broken sensor?

With the camera corrupted, reliability fusion alarms on 19% of good parts against a
10% target, and most of that comes from occlusion (42%), which doesn't move the
background level, so the sensor stays at full weight and its scores shift upwards.
Thresholds come from the validation parts, which never saw the fault.

Fixed before this script ran, from the saved scores of `factorylens robustness`:

  question   an operator confirms k good parts from the current run; their scores join
             the calibration set. What happens to false alarms and to defects caught?
  primary    reliability fusion (16-part window), k = 5, any sensor weights, alpha 0.1,
             all RGB corruptions and severities, 200 random draws of the confirmed parts.
             False alarms and detections on the parts that weren't confirmed.
  success    mean false alarms at or below 12%, while defects caught stay at or above
             the healthy sensor alone (55%). Wilcoxon over the ten category means.
  also       k in {1, 2, 3, 5, 10}, 3D corruptions, clean parts as a control, the extent
             signal, and only parts at full weight (what `confirm_good` accepts today).

These parts aren't exchangeable with the validation parts, which is the whole point
of adding them, so the conformal guarantee doesn't carry over to the mix. This is an
empirical question.

usage: uv run python scripts/recalibration.py
Writes results/analysis/recalibration.csv and recalibration_tests.csv.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from factorylens import conformal, corruptions

ALPHA = 0.1
SIZES = (1, 2, 3, 5, 10)
PRIMARY_K = 5
DETECTORS = {"window": "reliable_window", "extent": "reliable_extent"}
WEIGHTS = {
    "window": ("w_rgb_window", "w_fpfh_window"),
    "extent": ("w_rgb_extent", "w_fpfh_extent"),
}
FILE = re.compile(
    r"(?P<category>[a-z_]+?)_(?P<corruption>clean|"
    + "|".join(corruptions.CORRUPTIONS)
    + r")(?P<severity>\d)_s0\.npz"
)


def rates(calibration: np.ndarray, scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    alarm = scores > conformal.threshold(calibration, ALPHA)
    return float(alarm[~y].mean()), float(alarm[y].mean())


def one_condition(d: dict, draws: int, rng: np.random.Generator) -> list[dict]:
    y = d["labels"].astype(bool)
    rv, gv = d["rgb_val"], d["fpfh_val"]
    zr, zg = (rv - rv.mean()) / rv.std(), (gv - gv.mean()) / gv.std()
    base = (zr + zg) / np.sqrt(2)
    rows = []
    for detector, key in DETECTORS.items():
        s = d[key]
        wr, wg = (d[k] for k in WEIGHTS[detector])
        far, caught = rates(base, s, y)
        rows.append(
            {"detector": detector, "k": 0, "rule": "none", "false_alarm": far, "caught": caught}
        )
        normal = np.flatnonzero(~y)
        full = normal[(wr[normal] == 1) & (wg[normal] == 1)]
        for k in SIZES:
            for rule, pool in (("any", normal), ("full weight", full)):
                # too few left to judge, or too few parts this rule would accept
                if len(pool) < k or len(normal) - k < 5:
                    continue
                result = []
                for _ in range(draws):
                    pick = rng.choice(pool, k, replace=False)
                    rest = np.setdiff1d(np.arange(len(y)), pick)
                    result.append(rates(np.r_[base, s[pick]], s[rest], y[rest]))
                far, caught = np.mean(result, axis=0)
                rows.append(
                    {
                        "detector": detector,
                        "k": k,
                        "rule": rule,
                        "false_alarm": far,
                        "caught": caught,
                    }
                )
    return rows


def healthy_rates(d: dict, sensor: str) -> tuple[float, float]:
    key, cal = ("fpfh", d["fpfh_val"]) if sensor == "rgb" else ("rgb", d["rgb_val"])
    return rates(cal, d[key], d["labels"].astype(bool))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/robustness-extent"))
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(args.scores.glob("*.npz")):
        m = FILE.fullmatch(path.name)
        if m is None:
            continue
        name = m["corruption"]
        sensor = "clean" if name == "clean" else corruptions.CORRUPTIONS[name].sensor
        d = dict(np.load(path))
        info = {
            "category": m["category"],
            "corruption": name,
            "severity": int(m["severity"]),
            "sensor": sensor,
        }
        rows += [{**info, **r} for r in one_condition(d, args.draws, rng)]
        if sensor != "clean":
            far, caught = healthy_rates(d, sensor)
            rows.append(
                {
                    **info,
                    "detector": "healthy sensor",
                    "k": 0,
                    "rule": "none",
                    "false_alarm": far,
                    "caught": caught,
                }
            )
    table = pd.DataFrame(rows)
    table.round(4).to_csv(args.out / "recalibration.csv", index=False)

    keys = ["sensor", "detector", "rule", "k"]
    pd.set_option("display.width", 200)
    print(table.groupby(keys)[["false_alarm", "caught"]].mean().round(3).to_string())

    rgb = table[table["sensor"] == "rgb"]
    print(
        rgb[
            (rgb["detector"] == "window")
            & rgb["k"].isin([0, PRIMARY_K])
            & (rgb["rule"] != "full weight")
        ]
        .pivot_table(index="corruption", columns="k", values=["false_alarm", "caught"])
        .round(3)
        .to_string()
    )

    def category_means(detector: str, rule: str, k: int, metric: str) -> pd.Series:
        t = rgb[(rgb["detector"] == detector) & (rgb["rule"] == rule) & (rgb["k"] == k)]
        return t.groupby("category")[metric].mean()

    tests = []
    for metric in ("false_alarm", "caught"):
        a = category_means("window", "any", PRIMARY_K, metric)
        for label, b in (
            ("no confirmed parts", category_means("window", "none", 0, metric)),
            ("healthy sensor", category_means("healthy sensor", "none", 0, metric)),
        ):
            tests.append(
                {
                    "metric": metric,
                    "versus": label,
                    "primary": a.mean(),
                    "other": b.mean(),
                    "p": stats.wilcoxon(a - b).pvalue,
                }
            )
    result = pd.DataFrame(tests)
    result.round(4).to_csv(args.out / "recalibration_tests.csv", index=False)
    print(result.round(4).to_string())


if __name__ == "__main__":
    main()
