"""Future Factories analog dataset (11-12 Dec 2023 run): loading, cycles and splits.

Seven CSVs sampled at about 10 Hz with identical timestamps. The anomaly label
lives in the `Description` column: empty for a normal cycle, otherwise the
parts that were taken off the tray ("NoNose", "NoNose,NoBody2", ...).

The PLC cycle counter resets to 0 after every interruption, so the same
counter value appears several times during the day. A physical cycle is
therefore a run of consecutive rows with one counter value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FILES = (
    "FFCell_CycleManagement.csv",
    "Conveyor_Signals.csv",
    "FFCell_SafetyManagement.csv",
    "R01_Data.csv",
    "R02_Data.csv",
    "R03_Data.csv",
    "R04_Data.csv",
)
LABEL_FILE = FILES[0]
NORMAL = "normal"

# Day 1 holds blocks 1-3 (normal, NoNose, NoNose+NoBody2; block 3 runs 16 min
# past midnight), day 2 holds blocks 4-6 (all three parts missing, normal, NoNose).
# Normal and NoNose cycles exist on both days, which is what makes a
# train-on-day-1 / test-on-day-2 split possible at all.
DAY_BOUNDARY = pd.Timestamp("2023-12-12T00:00:00Z")

# Conveyor drives 2-4 report about 1% of temperatures as what look like
# misread registers: values from 1e-30 to 1e38, many negative. Drive 1 stays
# between 68 and 94 F, so anything outside 40-140 F is treated as missing.
TEMPERATURE_RANGE_F = (40.0, 140.0)

# R01's six joint angles read exactly 0 through the whole day-1 normal block
# (22% of day 1) while its gripper load keeps moving, so the arm was working
# and the angles were not logged. A row with every joint of an arm at exactly
# zero is treated as missing for that arm.
JOINTS = ("S", "L", "U", "R", "B", "T")


def load_analog(root: str | Path, files: tuple[str, ...] = FILES) -> pd.DataFrame:
    """All signals in one frame indexed by UTC time, plus a `label` column."""
    root = Path(root)
    frames = []
    index = None
    for name in files:
        df = pd.read_csv(root / name, dtype={"Description": "string"})
        times = pd.to_datetime(df.pop("_time"), format="ISO8601", utc=True)
        if index is None:
            index = times
        elif not times.equals(index):
            raise ValueError(f"{name} has different timestamps from {files[0]}")
        description = df.pop("Description")
        if name == LABEL_FILE:
            df["label"] = description.fillna(NORMAL)
        frames.append(df.set_axis(pd.DatetimeIndex(times, name="time")))
    out = pd.concat(frames, axis=1)
    lo, hi = TEMPERATURE_RANGE_F
    for col in [c for c in out.columns if c.endswith("_Temperature")]:
        out[col] = out[col].where(out[col].between(lo, hi))
    return mask_joint_dropouts(out).sort_index()


def mask_joint_dropouts(df: pd.DataFrame) -> pd.DataFrame:
    """Set an arm's joint angles to NaN where all of them read exactly 0."""
    df = df.copy()
    robots = sorted({c.split("_")[1] for c in df.columns if c.endswith("JointAngle_Degree")})
    for robot in robots:
        cols = [f"M_{robot}_{j}JointAngle_Degree" for j in JOINTS]
        if not set(cols) <= set(df.columns):
            continue
        dead = (df[cols] == 0).all(axis=1)
        df.loc[dead, cols] = np.nan
    return df


def cycle_ids(counter: pd.Series) -> pd.Series:
    """Number consecutive runs of the same counter value 0, 1, 2, ..."""
    return (counter != counter.shift()).cumsum() - 1


def _majority(labels: pd.Series) -> str:
    return str(labels.value_counts().idxmax())


def segment_cycles(df: pd.DataFrame, min_duration_s: float = 60.0) -> pd.DataFrame:
    """One row per physical cycle, ordered in time.

    Runs shorter than `min_duration_s` are the stubs left when the counter is
    reset (counter 0 for a few seconds) and are dropped.
    """
    ids = cycle_ids(df["Q_Cell_CycleCount"])
    grouped = df.assign(_t=df.index).groupby(ids)
    cycles = pd.DataFrame(
        {
            "counter": grouped["Q_Cell_CycleCount"].first(),
            "start": grouped["_t"].min(),
            "end": grouped["_t"].max(),
            "rows": grouped.size(),
            "label": grouped["label"].agg(_majority),
            "n_labels": grouped["label"].nunique(),
        }
    )
    cycles["duration_s"] = (cycles["end"] - cycles["start"]).dt.total_seconds()
    cycles = cycles[cycles["duration_s"] >= min_duration_s].copy()
    cycles["anomalous"] = cycles["label"] != NORMAL
    cycles["day"] = (cycles["start"] >= DAY_BOUNDARY).map({False: 1, True: 2})
    # fill_value matters: with pandas' string dtype the first comparison against
    # a shifted NA is NA, and that cycle would silently drop out of every block
    changed = cycles["label"] != cycles["label"].shift(fill_value="")
    cycles["block"] = changed.cumsum()
    cycles.index.name = "cycle"
    return cycles


def label_blocks(cycles: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of cycles with the same label."""
    return cycles.groupby("block").agg(
        label=("label", "first"),
        cycles=("label", "size"),
        start=("start", "min"),
        end=("end", "max"),
    )


def cross_day_split(
    cycles: pd.DataFrame, anomaly: str = "NoNose"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on day 1, test on day 2, keeping only normal and `anomaly` cycles.

    Both days contain both classes, so a detector cannot score well just by
    telling the days apart.
    """
    keep = cycles[cycles["label"].isin([NORMAL, anomaly])]
    train, test = keep[keep["day"] == 1], keep[keep["day"] == 2]
    for name, part in [("train", train), ("test", test)]:
        if part["anomalous"].nunique() < 2:
            raise ValueError(f"{name} split has only one class for anomaly={anomaly!r}")
    return train, test
