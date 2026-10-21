"""Cycle period from the robot signals alone, and which signals drift over the run.

1. Every gripper-load and base-joint signal is binned to 1 s, the longest
   stretch without gaps is kept, and the period is read off its
   autocorrelation. The answer is compared with the PLC counter, which the
   signals never see.
2. A straight line is fitted to each signal's per-cycle mean over the 29 h,
   normal cycles only, to see which channels drift.

usage: uv run python scripts/ff/periods.py "data/raw/ff_analog/Analog Dataset"
Writes results/analysis/ff_periods.csv, results/analysis/ff_trends.csv and
docs/figures/ff_periodogram.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorylens import ts  # noqa: E402
from factorylens.data import ff  # noqa: E402

ROBOTS = ("R01", "R02", "R03", "R04")
SIGNALS = [f"I_{r}_Gripper_Load" for r in ROBOTS] + [f"M_{r}_SJointAngle_Degree" for r in ROBOTS]
TREND_SIGNALS = SIGNALS + [f"Q_VFD{i}_Temperature" for i in range(1, 5)]
BLUE, ORANGE, INK_2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e1e0d9"


def periods(df: pd.DataFrame, plc_period: float) -> pd.DataFrame:
    rows = []
    for col in SIGNALS:
        series = ts.longest_gap_free(ts.regular(df[col].astype(float), 1.0))
        p = ts.cycle_period(series.to_numpy(), step_s=1.0)
        rows.append(
            {
                "signal": col,
                "hours_used": len(series) / 3600,
                "period_s": p.seconds,
                "acf_at_period": p.strength,
                "diff_to_plc_s": p.seconds - plc_period,
            }
        )
    return pd.DataFrame(rows)


def trends(df: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    ids = ff.cycle_ids(df["Q_Cell_CycleCount"]).to_numpy()
    means = df[TREND_SIGNALS].astype(float).groupby(ids).mean()
    normal = cycles[cycles["label"] == ff.NORMAL]
    per_cycle = means.loc[normal.index].set_index(pd.DatetimeIndex(normal["start"]))
    rows = []
    for col in TREND_SIGNALS:
        slope, r2 = ts.linear_trend(per_cycle[col])
        spread = per_cycle[col].std()
        rows.append(
            {
                "signal": col,
                "slope_per_hour": slope,
                "r2": r2,
                # drift over the whole run in units of the cycle-to-cycle spread
                "drift_over_run_in_sd": slope * 29 / spread if spread > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def figure(df: pd.DataFrame, plc_period: float, out: Path) -> None:
    col = "I_R03_Gripper_Load"
    x = ts.longest_gap_free(ts.regular(df[col].astype(float), 1.0)).to_numpy()
    freqs, power = ts.periodogram(x, step_s=1.0)
    acf = ts.autocorrelation(x, 1200)
    found = ts.cycle_period(x, step_s=1.0)

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.0))
    fig.patch.set_facecolor("#fcfcfb")
    band = (freqs > 1 / 1200) & (freqs < 1 / 20)
    a.semilogy(1000 * freqs[band], power[band], color=BLUE, lw=1.2)
    for k in range(1, 12):
        a.axvline(1000 * k / plc_period, color=ORANGE, lw=0.7, alpha=0.6, ls=(0, (2, 3)))
    a.set_xlabel("frequency (mHz); dashed = multiples of 1 / PLC cycle")
    a.set_ylabel("power", color=INK_2)
    a.set_title("R03 gripper load: periodogram", fontsize=9, color=INK_2)

    lags = np.arange(len(acf))
    b.plot(lags, acf, color=BLUE, lw=1.2)
    b.axvline(found.seconds, color=ORANGE, lw=1.2)
    b.text(found.seconds + 15, 0.9, f"{found.seconds:.1f} s", color=ORANGE, fontsize=8, va="top")
    b.set_xlabel("lag (s)")
    b.set_title("autocorrelation", fontsize=9, color=INK_2)
    for ax in (a, b):
        ax.set_facecolor("#fcfcfb")
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5, colors="#898781")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()

    df = ff.load_analog(args.root)
    cycles = ff.segment_cycles(df)
    plc_period = float(cycles["duration_s"].median())
    out = Path("results/analysis")
    out.mkdir(parents=True, exist_ok=True)

    p = periods(df, plc_period)
    p.to_csv(out / "ff_periods.csv", index=False, float_format="%.4f")
    t = trends(df, cycles)
    t.to_csv(out / "ff_trends.csv", index=False, float_format="%.4f")
    figure(df, plc_period, Path("docs/figures/ff_periodogram.png"))

    print(f"PLC median cycle: {plc_period:.1f} s")
    print(p.round(2).to_string(index=False))
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
