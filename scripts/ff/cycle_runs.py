"""Split the FF analog recording into physical assembly cycles and list their labels.

usage: python scripts/ff/cycle_runs.py "data/raw/ff_analog/Analog Dataset"
"""
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
df = pd.read_csv(root / "FFCell_CycleManagement.csv", dtype={"Description": str})
df["_time"] = pd.to_datetime(df["_time"], format="ISO8601", utc=True)
df = df.sort_values("_time").reset_index(drop=True)
df["label"] = df["Description"].fillna("normal")

# the PLC counter restarts during the day, so a physical cycle is a run of
# consecutive rows with the same counter value, not a unique counter value
run_id = (df["Q_Cell_CycleCount"] != df["Q_Cell_CycleCount"].shift()).cumsum()
runs = df.groupby(run_id).agg(
    counter=("Q_Cell_CycleCount", "first"),
    start=("_time", "min"),
    end=("_time", "max"),
    rows=("label", "size"),
    labels=("label", lambda s: ",".join(f"{k}:{v}" for k, v in s.value_counts().items())),
    n_labels=("label", "nunique"),
)
runs["dur_s"] = (runs["end"] - runs["start"]).dt.total_seconds().round(1)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 70)
pd.set_option("display.max_rows", 500)
print(runs[["counter", "start", "dur_s", "rows", "labels"]].to_string())
print("\nruns:", len(runs), " mixed-label runs:", (runs["n_labels"] > 1).sum())
print(runs["dur_s"].describe().to_string())

runs["label"] = runs["labels"].str.split(":").str[0]
runs = runs[runs["dur_s"] > 60]  # drop counter-reset stubs
block = (runs["label"] != runs["label"].shift()).cumsum()
blocks = runs.groupby(block).agg(
    label=("label", "first"), cycles=("label", "size"), start=("start", "min"), end=("end", "max"))
print("\nlabel blocks (full cycles only):")
print(blocks.to_string())
print("\ncycles per label:")
print(runs["label"].value_counts().to_string())
