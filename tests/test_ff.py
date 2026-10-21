import pandas as pd
import pytest
from fakes import make_ff_frame as make_frame
from fakes import write_ff_csvs as write_csvs

from factorylens.data import ff


def test_counter_resets_give_separate_cycles():
    cycles = ff.segment_cycles(make_frame(), min_duration_s=60)
    assert list(cycles["counter"]) == [1, 2, 1, 2]
    assert list(cycles["label"]) == ["normal", "NoNose", "normal", "NoNose"]
    assert list(cycles["day"]) == [1, 1, 2, 2]
    assert (cycles["duration_s"] == 99).all()


def test_short_stubs_are_dropped():
    cycles = ff.segment_cycles(make_frame(), min_duration_s=1)
    assert 0 in cycles["counter"].values
    assert 0 not in ff.segment_cycles(make_frame())["counter"].values


def test_majority_label_wins_inside_a_cycle():
    df = make_frame()
    df.iloc[:10, df.columns.get_loc("label")] = "NoNose"  # first cycle mostly normal
    cycles = ff.segment_cycles(df)
    assert cycles.iloc[0]["label"] == "normal"
    assert cycles.iloc[0]["n_labels"] == 2


def test_first_cycle_keeps_its_block_with_string_dtype():
    # read_csv(dtype="string") gives pandas' string dtype, where comparing with a
    # shifted NA returns NA instead of True
    df = make_frame()
    df["label"] = df["label"].astype("string")
    cycles = ff.segment_cycles(df)
    assert cycles["block"].tolist() == [1, 2, 3, 4]
    assert ff.label_blocks(cycles)["cycles"].sum() == len(cycles)


def test_blocks_group_consecutive_labels():
    df = make_frame()
    df["label"] = ff.NORMAL
    blocks = ff.label_blocks(ff.segment_cycles(df))
    assert len(blocks) == 1 and blocks.iloc[0]["cycles"] == 4


def test_cross_day_split():
    train, test = ff.cross_day_split(ff.segment_cycles(make_frame()))
    assert (train["day"] == 1).all() and (test["day"] == 2).all()
    assert train["anomalous"].tolist() == [False, True]
    with pytest.raises(ValueError, match="only one class"):
        ff.cross_day_split(ff.segment_cycles(make_frame()), anomaly="NoBody1")


def test_load_analog_joins_files(tmp_path):
    write_csvs(tmp_path, make_frame())
    df = ff.load_analog(tmp_path, files=ff.FILES[:2])
    assert {"Q_Cell_CycleCount", "signal_1", "label"} <= set(df.columns)
    assert df.index.tz is not None and df.index.is_monotonic_increasing
    assert df["label"].value_counts()["NoNose"] == 200


def test_load_analog_refuses_misaligned_files(tmp_path):
    df = make_frame()
    write_csvs(tmp_path, df)
    other = pd.read_csv(tmp_path / ff.FILES[1])
    other.loc[0, "_time"] = "2023-12-11T09:59:59.000Z"
    other.to_csv(tmp_path / ff.FILES[1], index=False)
    with pytest.raises(ValueError, match="different timestamps"):
        ff.load_analog(tmp_path, files=ff.FILES[:2])


def test_load_analog_masks_impossible_temperatures(tmp_path):
    df = make_frame()
    write_csvs(tmp_path, df)
    other = pd.read_csv(tmp_path / ff.FILES[1])
    other = other.rename(columns={"signal_1": "Q_VFD2_Temperature"})
    other.loc[5, "Q_VFD2_Temperature"] = -2.98e38
    other.loc[6, "Q_VFD2_Temperature"] = 3.1e-31
    other.loc[7, "Q_VFD2_Temperature"] = 1.1e15
    other.to_csv(tmp_path / ff.FILES[1], index=False)
    loaded = ff.load_analog(tmp_path, files=ff.FILES[:2])
    assert loaded["Q_VFD2_Temperature"].isna().sum() == 3
    assert loaded["Q_VFD2_Temperature"].dropna().eq(68.0).all()
