"""How much of the FF label can be predicted from things that only track time?

Uses the analog dataset (10 Hz, labelled). The multimodal window comes from the
first and last batch JSON timestamps; those are local time, the analog file is
UTC, and NSF-MAP aligned them with a 4h59m51.1s shift.

usage: python scripts/ff/confound_check.py "data/raw/ff_analog/Analog Dataset"
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

root = Path(sys.argv[1])
shift = pd.Timedelta(hours=4, minutes=59, seconds=51.1)
mm_start = pd.Timestamp("2023-12-11 11:50:52.119915", tz="UTC") + shift
mm_end = pd.Timestamp("2023-12-12 11:25:42.410629", tz="UTC") + shift


def load(name, cols):
    df = pd.read_csv(root / name, usecols=["_time", *cols], dtype={"Description": str})
    df["_time"] = pd.to_datetime(df["_time"], format="ISO8601", utc=True)
    return df.sort_values("_time").reset_index(drop=True)


cyc = load("FFCell_CycleManagement.csv", ["Q_Cell_CycleCount", "Description"])
conv = load("Conveyor_Signals.csv", ["Q_VFD1_Temperature", "Q_VFD2_Temperature"])
df = cyc.merge(conv, on="_time", how="left")
df["label"] = df["Description"].fillna("normal")
df["run"] = (df["Q_Cell_CycleCount"] != df["Q_Cell_CycleCount"].shift()).cumsum()

runs = df.groupby("run").agg(
    start=("_time", "min"),
    end=("_time", "max"),
    label=("label", lambda s: s.value_counts().idxmax()),
    vfd1=("Q_VFD1_Temperature", "mean"),
    vfd2=("Q_VFD2_Temperature", "mean"),
)
runs = runs[(runs["end"] - runs["start"]).dt.total_seconds() > 60]
runs["anomalous"] = (runs["label"] != "normal").astype(int)
runs["hours"] = (runs["start"] - runs["start"].min()).dt.total_seconds() / 3600
runs["day"] = runs["start"].dt.day

inside = runs[(runs["start"] >= mm_start) & (runs["end"] <= mm_end)]
print("multimodal window (UTC):", mm_start, "->", mm_end)
print("\ncycles inside the image window, by day and label:")
print(inside.groupby(["day", "label"]).size().to_string())

for name, part in [("all analog cycles", runs), ("image window only", inside)]:
    print(f"\n{name}: n={len(part)}, anomalous={part['anomalous'].sum()}")
    for feat in ["hours", "vfd1", "vfd2"]:
        auc = roc_auc_score(part["anomalous"], part[feat])
        print(f"  AUROC from {feat:5s}: {max(auc, 1 - auc):.3f}")
