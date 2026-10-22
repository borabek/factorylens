"""Cycle-level fault detectors on the Future Factories robot signals.

Train on normal cycles of one day, score every normal and NoNose cycle of the
other day, report AUROC with a bootstrap CI. Both directions are run (day 1 ->
day 2 is the main one, day 2 -> day 1 the check).

On each test day the normal and NoNose cycles form two contiguous blocks, so
time of day alone separates them perfectly. A detector that scores well could
just be tracking drift. Three things are reported to tell the two apart:

  - the time and level (cycle mean) baselines,
  - every detector on raw and on de-meaned profiles,
  - every robot on its own: a real NoNose detector should point at R03, the
    robot whose gripper load shows the missing part on day 1 (ff_gripper.png).

Outputs:
  results/analysis/ff_detectors.csv
  docs/figures/ff_detectors.png

usage: uv run python scripts/ff/detectors.py "data/raw/ff_analog/Analog Dataset"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from factorylens.data import ff
from factorylens.eval.metrics import bootstrap_ci, image_auroc
from factorylens.models import sensor

sys.path.insert(0, str(Path(__file__).parent))
from figures import INK, INK_2, LABEL_COLORS, MUTED, plt  # noqa: E402

CHANNEL_SETS = {
    "all robots": sensor.ROBOT_CHANNELS,
    "all robots + temperatures": sensor.ROBOT_CHANNELS + sensor.TEMPERATURES,
    **{r: sensor.robot_channels(r) for r in sensor.ROBOTS},
}
DETECTORS = {**sensor.DETECTORS, "level": sensor.Level}


def evaluate(
    df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, direction: str
) -> list[dict[str, object]]:
    labels = test["anomalous"].to_numpy()
    since_start = (test["start"] - test["start"].min()).dt.total_seconds().to_numpy()
    rows: list[dict[str, object]] = [
        {"direction": direction, "channels": "-", "profiles": "-", "detector": "time of day"}
        | auroc(labels, since_start)
    ]
    normal = ~train["anomalous"].to_numpy()
    for name, channels in CHANNEL_SETS.items():
        fit_on = sensor.cycle_profiles(df, train, channels)[normal]
        scored = sensor.cycle_profiles(df, test, channels)
        # R01's joints were not logged in the day-1 normal block
        keep = sensor.usable(fit_on) & sensor.usable(scored)
        fit_on, scored = fit_on[..., keep], scored[..., keep]
        for prep in ("raw", "de-meaned"):
            a, b = (
                (fit_on, scored)
                if prep == "raw"
                else (sensor.demean(fit_on), sensor.demean(scored))
            )
            for det, cls in DETECTORS.items():
                if det == "level" and prep == "de-meaned":
                    continue  # nothing left to see
                scores = cls().fit(a).score(b)
                row = {
                    "direction": direction,
                    "channels": name,
                    "n_channels": int(keep.sum()),
                    "profiles": prep,
                    "detector": det,
                }
                rows.append(row | auroc(labels, scores))
    return rows


def auroc(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    lo, hi = bootstrap_ci(labels, scores)
    return {"auroc": image_auroc(labels, scores), "ci_low": lo, "ci_high": hi}


def figure(
    df: pd.DataFrame, table: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, out: Path
) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.8), width_ratios=[1.1, 1])

    # left: forecast detector per channel set, raw vs de-meaned, day 1 -> day 2
    main = table[(table["direction"] == "day 1 -> day 2") & (table["detector"] == "forecast")]
    names = list(CHANNEL_SETS)
    y = np.arange(len(names))[::-1]
    for prep, color, dy in [("raw", MUTED, 0.14), ("de-meaned", LABEL_COLORS["normal"], -0.14)]:
        part = main[main["profiles"] == prep].set_index("channels").loc[names]
        left.errorbar(
            part["auroc"],
            y + dy,
            xerr=[part["auroc"] - part["ci_low"], part["ci_high"] - part["auroc"]],
            fmt="o",
            ms=4,
            color=color,
            elinewidth=1,
            label=prep,
        )
    left.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    left.text(0.505, len(names) - 0.4, "chance", color=MUTED, fontsize=7.5)
    left.set_yticks(y, names)
    left.set_xlim(0, 1)
    left.set_xlabel("AUROC, normal vs no nose (day 2)")
    left.set_title("forecast-residual detector", loc="left", fontsize=9, color=INK)
    left.legend(frameon=False, fontsize=7.5, loc="lower right")

    # right: where the R03 residual differs, raw profiles
    channels = sensor.robot_channels("R03")
    det = sensor.Forecast().fit(
        sensor.cycle_profiles(df, train, channels)[~train["anomalous"].to_numpy()]
    )
    res = (det.residuals(sensor.cycle_profiles(df, test, channels)) ** 2).sum(axis=2)
    seconds = np.arange(res.shape[1]) + det.lags
    labels = test["anomalous"].to_numpy()
    per_second = np.array([image_auroc(labels, res[:, i]) for i in range(res.shape[1])])
    right.axvspan(90, 97, color=LABEL_COLORS["NoNose"], alpha=0.15, lw=0)
    right.text(99, 0.12, "load pulse\n(gripper figure)", fontsize=7.5, color=INK_2)
    right.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    right.plot(seconds, per_second, color=LABEL_COLORS["normal"], lw=1.2)
    right.set_ylim(0, 1)
    right.set_xlabel("seconds since cycle start")
    right.set_ylabel("AUROC of that second's residual", color=INK_2)
    right.set_title("where in the cycle R03 differs (day 2)", loc="left", fontsize=9, color=INK)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/ff_detectors.csv"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/ff_detectors.png"))
    args = parser.parse_args()

    df = ff.load_analog(args.root)
    cycles = ff.segment_cycles(df)
    day1, day2 = ff.cross_day_split(cycles)
    rows = evaluate(df, day1, day2, "day 1 -> day 2") + evaluate(df, day2, day1, "day 2 -> day 1")
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(args.out, index=False)
    figure(df, table, day1, day2, args.figure)

    wide = table.pivot_table(
        index=["channels", "profiles"],
        columns=["direction", "detector"],
        values="auroc",
        sort=False,
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(wide.round(2).to_string())
    print(table[table["detector"] == "time of day"][["direction", "auroc"]].to_string(index=False))


if __name__ == "__main__":
    main()
