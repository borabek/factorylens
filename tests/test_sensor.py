import numpy as np
import pandas as pd
import pytest
from fakes import make_ff_frame

from factorylens.data import ff
from factorylens.models import sensor


def profiles(n, seconds=60, channels=3, missing_pulse=False, seed=0):
    """Cycles with a load pulse at 20-25 s; anomalies drop the pulse."""
    rng = np.random.default_rng(seed)
    t = np.arange(seconds)
    base = np.stack([np.sin(2 * np.pi * t / seconds * (c + 1)) for c in range(channels)], axis=-1)
    out = np.repeat(base[None], n, axis=0) + rng.normal(0, 0.05, (n, seconds, channels))
    out += rng.normal(0, 0.3, (n, 1, channels))  # cycle-to-cycle offsets, like real drift
    if not missing_pulse:
        out[:, 20:25, 0] += 3.0
    return out


@pytest.mark.parametrize("name", ["profile_z", "forecast"])
def test_detectors_find_missing_pulses_once_offsets_are_removed(name):
    det = sensor.DETECTORS[name]().fit(sensor.demean(profiles(30)))
    normal = det.score(sensor.demean(profiles(10, seed=1)))
    broken = det.score(sensor.demean(profiles(10, missing_pulse=True, seed=2)))
    assert normal.shape == (10,)
    assert broken.min() > normal.max()


def test_offsets_hide_the_pulse_without_demeaning():
    det = sensor.ProfileZ().fit(profiles(30))
    normal = det.score(profiles(10, seed=1))
    broken = det.score(profiles(10, missing_pulse=True, seed=2))
    assert broken.min() < normal.max()


def test_iforest_finds_whole_cycle_outliers():
    # it splits on one random feature at a time, so a change confined to one
    # 10 s bin of one channel is easy to miss; a cycle that is off everywhere is not
    det = sensor.IForest().fit(profiles(30))
    normal = det.score(profiles(10, seed=1))
    broken = det.score(profiles(10, seed=2) + 2.0)
    assert broken.min() > np.median(normal)
    assert broken.mean() > normal.mean() + 0.05


def test_level_ignores_shape_but_demean_removes_offsets():
    train = profiles(30)
    shifted = profiles(10, seed=4) + 5.0
    assert sensor.Level().fit(train).score(shifted).min() > 10
    np.testing.assert_allclose(sensor.demean(shifted).mean(axis=1), 0, atol=1e-12)
    no_pulse = sensor.Level().fit(train).score(profiles(10, missing_pulse=True, seed=2))
    assert np.median(no_pulse) < 2  # five seconds of a missing pulse barely move the mean


def test_forecast_residuals_are_small_on_normal_cycles():
    det = sensor.Forecast(lags=3, harmonics=4).fit(profiles(30))
    res = det.residuals(profiles(5, seed=3))
    assert res.shape == (5, 60 - 3, 3)
    assert np.abs(res).mean() < 0.5


def test_forecast_without_phase_terms():
    # lags only: the plain autoregressive model
    det = sensor.Forecast(lags=2, harmonics=0).fit(profiles(10))
    assert det.score(profiles(3, seed=5)).shape == (3,)


def test_cycle_profiles_from_the_fake_line():
    df = make_ff_frame()
    df["I_R01_Gripper_Load"] = np.arange(len(df), dtype=float)
    cycles = ff.segment_cycles(df)
    out = sensor.cycle_profiles(df, cycles, ("I_R01_Gripper_Load",), seconds=100)
    assert out.shape == (4, 100, 1)
    assert not np.isnan(out).any()  # the 100th second is missing, filled from its neighbour
    assert out[0, 0, 0] == 0 and out[1, 0, 0] == 100


def test_standardiser_handles_flat_channels():
    flat = np.ones((4, 10, 2))
    z = sensor.Standardiser.fit(flat)(flat)
    assert np.isfinite(z).all() and np.allclose(z, 0)


def test_profile_z_does_not_divide_by_zero():
    train = np.zeros((5, 10, 2))
    det = sensor.ProfileZ().fit(train)
    assert np.isfinite(det.score(np.ones((2, 10, 2)))).all()
    assert isinstance(pd.Series(det.score(train)).sum(), float)
