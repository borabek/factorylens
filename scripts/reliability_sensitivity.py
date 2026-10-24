"""How much the reliability fusion depends on its two settings.

`reliable` and `reliable_window` were run with a ramp of 3 SDs and a window
of 16 parts, both fixed before any result was seen. The saved per-image
scores and reliability signals are enough to recompute the fused image scores
for other settings, so this sweeps ramp in {1, 2, 3, 5, 8} and window in
{1, 4, 8, 16, 32, 64} without touching the features. Image AUROC only: the
pixel maps were not saved.

usage: uv run python scripts/reliability_sensitivity.py
Writes results/analysis/reliability_sensitivity.csv.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from factorylens import corruptions
from factorylens.eval.metrics import image_auroc
from factorylens.fusion import running_median

RAMPS = (1.0, 2.0, 3.0, 5.0, 8.0)
WINDOWS = (1, 4, 8, 16, 32, 64)
NAME = re.compile(
    r"(?P<category>.+?)_(?P<corruption>"
    + "|".join(["clean", *corruptions.CORRUPTIONS])
    + r")(?P<severity>\d)_s0\.npz"
)


def fused(d: np.lib.npyio.NpzFile, ramp: float, window: int) -> np.ndarray:
    zr = (d["rgb"] - d["rgb_val"].mean()) / d["rgb_val"].std()
    zg = (d["fpfh"] - d["fpfh_val"].mean()) / d["fpfh_val"].std()
    u_rgb, u_geo = d["u_rgb"], d["u_fpfh"]  # per image, relative to the validation worst
    if window > 1:
        order = np.random.default_rng(0).permutation(len(zr))
        u_rgb, u_geo = running_median(u_rgb, window, order), running_median(u_geo, window, order)
    wr, wg = np.clip(1 - u_rgb / ramp, 0, 1), np.clip(1 - u_geo / ramp, 0, 1)
    off = (wr == 0) & (wg == 0)
    wr[off], wg[off] = 1.0, 1.0
    return np.asarray((wr * zr + wg * zg) / np.sqrt(wr**2 + wg**2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores/robustness-reliable"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/analysis/reliability_sensitivity.csv")
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.scores.glob("*.npz")):
        m = NAME.fullmatch(path.name)
        assert m, path.name
        d = np.load(path)
        sensor = (
            "clean"
            if m["corruption"] == "clean"
            else corruptions.CORRUPTIONS[m["corruption"]].sensor
        )
        for ramp in RAMPS:
            for window in WINDOWS:
                rows.append(
                    {
                        "category": m["category"],
                        "corruption": m["corruption"],
                        "severity": int(m["severity"]),
                        "sensor": sensor,
                        "ramp": ramp,
                        "window": window,
                        "auroc": image_auroc(d["labels"], fused(d, ramp, window)),
                    }
                )
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out, index=False)
    summary = table.groupby(["sensor", "ramp", "window"])["auroc"].mean().unstack("window")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
