import numpy as np
import pandas as pd
import pytest

from factorylens import ts


def cycles(period_s=295.0, hours=6, step_s=1.0, noise=0.5, seed=0):
    """A pulse every `period_s` seconds plus noise and a slow drift."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, hours * 3600, step_s)
    phase = (t % period_s) / period_s
    x = (phase < 0.05).astype(float) * 5 + rng.normal(0, noise, len(t)) + t / 3600
    index = pd.date_range("2023-12-11", periods=len(t), freq=pd.Timedelta(seconds=step_s))
    return pd.Series(x, index=index)


@pytest.mark.parametrize("period", [180.0, 295.0, 400.0])
def test_cycle_period_finds_the_fundamental(period):
    found = ts.cycle_period(cycles(period).to_numpy(), step_s=1.0)
    assert found.seconds == pytest.approx(period, rel=0.01)
    assert found.strength > 0.5


def test_largest_periodogram_peak_is_a_harmonic_for_pulses():
    # the reason cycle_period uses the autocorrelation: for a short pulse per
    # cycle the top spectral peak is not the fundamental
    freqs, power = ts.periodogram(cycles(295.0).to_numpy(), step_s=1.0)
    band = (freqs > 1 / 1200) & (freqs < 1 / 30)
    top = 1 / freqs[band][np.argmax(power[band])]
    assert abs(top - 295.0) > 20
    # but every strong peak sits on a multiple of 1/295 s
    strong = freqs[band][power[band] > 0.2 * power[band].max()]
    assert np.allclose(strong * 295.0, np.round(strong * 295.0), atol=0.05)


def test_pure_noise_has_no_strong_period():
    rng = np.random.default_rng(1)
    assert ts.cycle_period(rng.normal(0, 1, 6 * 3600), step_s=1.0).strength < 0.1


def test_too_short_series():
    with pytest.raises(ValueError):
        ts.cycle_period(np.ones(20), step_s=1.0, min_s=30, max_s=60)


def test_regular_and_gap_handling():
    s = cycles(hours=1, step_s=0.1)
    s = s.drop(s.index[10000:12000])  # a 200 s hole
    r = ts.regular(s, 1.0)
    assert r.isna().sum() == 200
    longest = ts.longest_gap_free(r)
    assert longest.notna().all() and len(longest) == 3600 - 1200
    assert len(ts.longest_gap_free(pd.Series([np.nan, np.nan]))) == 0


def test_linear_trend_recovers_the_drift():
    slope, r2 = ts.linear_trend(cycles(noise=0.0))
    assert slope == pytest.approx(1.0, rel=0.05)  # +1 per hour
    flat_slope, flat_r2 = ts.linear_trend(pd.Series(1.0, index=cycles(hours=1).index))
    assert flat_slope == pytest.approx(0, abs=1e-9) and flat_r2 == 0.0
    assert 0 < r2 < 1
