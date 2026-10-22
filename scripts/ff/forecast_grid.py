"""How much the forecast detector's result depends on its settings.

`detectors.py` reports one setting (5 lags, 6 Fourier pairs, ridge alpha 1),
chosen before any result was seen. This reruns it over a grid of lags,
harmonics and alpha, for R03 and for all robots (de-meaned), in both day
directions. If the R03 result only held for a few settings, it would be a
tuning artefact.

usage: uv run python scripts/ff/forecast_grid.py "data/raw/ff_analog/Analog Dataset"
Writes results/analysis/ff_forecast_grid.csv.
"""

import argparse
import itertools
from pathlib import Path

import pandas as pd

from factorylens.data import ff
from factorylens.eval.metrics import image_auroc
from factorylens.models import sensor

LAGS = (1, 3, 5, 10, 20)
HARMONICS = (0, 3, 6, 12)
ALPHAS = (0.1, 1.0, 10.0)
SETS = {"R03": (sensor.robot_channels("R03"), False), "all robots": (sensor.ROBOT_CHANNELS, True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/analysis/ff_forecast_grid.csv"))
    args = parser.parse_args()

    df = ff.load_analog(args.root)
    day1, day2 = ff.cross_day_split(ff.segment_cycles(df))
    rows = []
    for direction, (train, test) in {
        "day 1 -> day 2": (day1, day2),
        "day 2 -> day 1": (day2, day1),
    }.items():
        labels = test["anomalous"].to_numpy()
        for name, (channels, demean) in SETS.items():
            fit_on = sensor.cycle_profiles(df, train, channels)[~train["anomalous"].to_numpy()]
            scored = sensor.cycle_profiles(df, test, channels)
            keep = sensor.usable(fit_on) & sensor.usable(scored)
            fit_on, scored = fit_on[..., keep], scored[..., keep]
            if demean:
                fit_on, scored = sensor.demean(fit_on), sensor.demean(scored)
            for lags, harmonics, alpha in itertools.product(LAGS, HARMONICS, ALPHAS):
                det = sensor.Forecast(lags=lags, harmonics=harmonics, alpha=alpha).fit(fit_on)
                rows.append(
                    {
                        "direction": direction,
                        "channels": name,
                        "lags": lags,
                        "harmonics": harmonics,
                        "alpha": alpha,
                        "auroc": image_auroc(labels, det.score(scored)),
                    }
                )
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.out, index=False)
    summary = table.groupby(["channels", "direction"])["auroc"].describe()
    print(summary[["count", "min", "25%", "50%", "75%", "max"]].round(3).to_string())
    print(table.groupby(["channels", "harmonics"])["auroc"].median().unstack().round(3))


if __name__ == "__main__":
    main()
