"""How much of the FF label can be predicted from things that only track time?

The multimodal (image) window comes from the first and last batch JSON
timestamps. Those are local time while the analog file is UTC; NSF-MAP aligned
the two with a 4h59m51.1s shift.

usage: uv run python scripts/ff/confound_check.py "data/raw/ff_analog/Analog Dataset"
"""

import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from factorylens.data import ff

SHIFT = pd.Timedelta(hours=4, minutes=59, seconds=51.1)
IMAGE_WINDOW = (
    pd.Timestamp("2023-12-11 11:50:52.119915", tz="UTC") + SHIFT,
    pd.Timestamp("2023-12-12 11:25:42.410629", tz="UTC") + SHIFT,
)


def main(root: str) -> None:
    df = ff.load_analog(root)
    cycles = ff.segment_cycles(df)
    cycle_of_row = ff.cycle_ids(df["Q_Cell_CycleCount"]).values
    means = df[["Q_VFD1_Temperature", "Q_VFD2_Temperature"]].groupby(cycle_of_row).mean()
    cycles = cycles.join(means.rename(columns=lambda c: c.split("_")[1].lower()))
    cycles["hours"] = (cycles["start"] - cycles["start"].min()).dt.total_seconds() / 3600

    lo, hi = IMAGE_WINDOW
    inside = cycles[(cycles["start"] >= lo) & (cycles["end"] <= hi)]
    print("image window (UTC):", lo, "->", hi)
    print("\ncycles inside the image window by day and label:")
    print(inside.groupby(["day", "label"]).size().to_string())

    for name, part in [("all analog cycles", cycles), ("image window only", inside)]:
        print(f"\n{name}: n={len(part)}, anomalous={int(part['anomalous'].sum())}")
        for feat in ["hours", "vfd1", "vfd2"]:
            auc = roc_auc_score(part["anomalous"], part[feat])
            print(f"  AUROC from {feat:5s}: {max(auc, 1 - auc):.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
