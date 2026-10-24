"""A sensor that breaks in the middle of a run.

The robustness benchmark corrupts a whole test set at once. On a line a
sensor works for a while and then fails. This replays each category's test
parts in a random order as a stream. The first half comes from the clean run
and, from the onset on, the second half from the corrupted run of the same
parts, so labels and part mix are identical on both sides.

For each fusion it reports:

  auroc_after     image AUROC over the parts after the onset
  auroc_before    the same before it (a false switch-off would show here)
  delay           parts after the onset until the broken sensor's weight first
                  drops below 0.5 (only for the windowed versions)
  false_switch    parts before the onset where either sensor's weight was below 0.5

The per-image reliability signals saved by `factorylens robustness` are
enough to replay any window size, so nothing is recomputed.

usage: uv run python scripts/fault_stream.py
Writes results/analysis/fault_stream.csv and docs/figures/fault_stream.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from factorylens import corruptions  # noqa: E402
from factorylens.data.mvtec3d import CATEGORIES  # noqa: E402
from factorylens.eval.metrics import image_auroc  # noqa: E402
from factorylens.fusion import running_median  # noqa: E402

WINDOWS = (1, 4, 8, 16, 32)
RAMP = 3.0
SEVERITY = 3
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"


def stream(clean: dict, bad: dict, seed: int) -> tuple[dict, np.ndarray, int]:
    """Values of every array along a random part order; corrupted from the middle on."""
    n = len(clean["labels"])
    order = np.random.default_rng(seed).permutation(n)
    onset = n // 2
    out = {}
    for key in ("rgb", "fpfh", "u_rgb", "u_fpfh", "labels"):
        a = clean[key][order].astype(float)
        a[onset:] = bad[key][order][onset:]
        out[key] = a
    return out, order, onset


def fuse(s: dict, zr: np.ndarray, zg: np.ndarray, window: int | None) -> tuple:
    if window is None:  # plain late fusion
        return zr + zg, np.ones_like(zr), np.ones_like(zg)
    u_rgb, u_geo = s["u_rgb"], s["u_fpfh"]
    if window > 1:
        idx = np.arange(len(u_rgb))  # the stream is already in arrival order
        u_rgb, u_geo = running_median(u_rgb, window, idx), running_median(u_geo, window, idx)
    wr, wg = np.clip(1 - u_rgb / RAMP, 0, 1), np.clip(1 - u_geo / RAMP, 0, 1)
    off = (wr == 0) & (wg == 0)
    wr[off], wg[off] = 1.0, 1.0
    return (wr * zr + wg * zg) / np.sqrt(wr**2 + wg**2), wr, wg


def run(scores: Path, seeds: int) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        clean = dict(np.load(scores / f"{category}_clean0_s0.npz"))
        zr_ref = clean["rgb_val"].mean(), clean["rgb_val"].std()
        zg_ref = clean["fpfh_val"].mean(), clean["fpfh_val"].std()
        for name, c in corruptions.CORRUPTIONS.items():
            severity = min(SEVERITY, c.severities)
            bad = dict(np.load(scores / f"{category}_{name}{severity}_s0.npz"))
            for seed in range(seeds):
                s, _, onset = stream(clean, bad, seed)
                zr = (s["rgb"] - zr_ref[0]) / zr_ref[1]
                zg = (s["fpfh"] - zg_ref[0]) / zg_ref[1]
                y = s["labels"].astype(bool)
                for window in (None, *WINDOWS):
                    fused, wr, wg = fuse(s, zr, zg, window)
                    broken = wr if c.sensor == "rgb" else wg
                    low = np.flatnonzero(broken[onset:] < 0.5)
                    rows.append(
                        {
                            "category": category,
                            "corruption": name,
                            "sensor": c.sensor,
                            "seed": seed,
                            "fusion": "late" if window is None else f"window {window}",
                            "auroc_before": image_auroc(y[:onset], fused[:onset]),
                            "auroc_after": image_auroc(y[onset:], fused[onset:]),
                            "delay": float(low[0]) if len(low) else np.nan,
                            "never_off": len(low) == 0,
                            "false_switch": int(((wr[:onset] < 0.5) | (wg[:onset] < 0.5)).sum()),
                        }
                    )
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    fig.patch.set_facecolor(SURFACE)
    global_corruptions = ["gaussian_noise", "defocus_blur", "jpeg", "depth_noise", "quantise"]
    t = table[table["corruption"].isin(global_corruptions)]
    names = ["late", *[f"window {w}" for w in WINDOWS]]
    after = t.groupby("fusion")["auroc_after"].mean().loc[names]
    delay = t.groupby("fusion")["delay"].median().loc[names[1:]]
    a.bar(range(len(names)), after, color=[MUTED, *["#0b0b0b"] * len(WINDOWS)])
    a.set_xticks(range(len(names)), ["late", *map(str, WINDOWS)])
    a.set_ylim(0.7, 0.85)
    a.set_xlabel("fusion (window in parts)")
    a.set_ylabel("image AUROC after the fault", color=INK_2)
    a.set_title("after a sensor fails mid-run", loc="left", fontsize=9, color=INK)
    # the price of a window: parts to notice a fault vs switching a healthy sensor off
    # each row is one stream with about 55 clean parts before the fault
    false_rate = table.groupby("fusion")["false_switch"].mean()
    for w in WINDOWS:
        name = f"window {w}"
        b.plot(delay[name], false_rate[name], "o", color=INK)
        b.annotate(
            f"{w}",
            (delay[name], false_rate[name]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    b.set_ylim(bottom=0)
    b.set_xlabel("parts until the broken sensor is down-weighted (median)")
    b.set_ylabel("false switch-offs per stream", color=INK_2)
    b.set_title("reacting fast vs staying calm (labels: window)", loc="left", fontsize=9, color=INK)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e0d9", lw=0.5)
    fig.text(
        0.01,
        0.01,
        "noise, blur, JPEG, depth noise, quantisation at severity 3; 10 categories x 5 orders",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/robustness-reliable"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/fault_stream.csv"))
    parser.add_argument("--figure", type=Path, default=Path("docs/figures/fault_stream.png"))
    args = parser.parse_args()

    table = run(args.scores, args.seeds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out, index=False)
    figure(table, args.figure)
    summary = table.groupby(["sensor", "fusion"]).agg(
        before=("auroc_before", "mean"),
        after=("auroc_after", "mean"),
        delay=("delay", "median"),
        never_off=("never_off", "mean"),
        false_switch=("false_switch", "sum"),
    )
    print(summary.round(3).to_string())
    print(
        table.groupby(["corruption", "fusion"])["auroc_after"].mean().unstack().round(3).to_string()
    )


if __name__ == "__main__":
    main()
