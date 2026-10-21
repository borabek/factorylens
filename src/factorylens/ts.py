"""Small time-series helpers for the Future Factories signals.

The analog recording is sampled at roughly 10 Hz with jitter and a few long
gaps. Spectral methods want an evenly spaced series without holes, so the
signals are binned onto a regular grid and only gap-free stretches are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal


def regular(series: pd.Series, step_s: float = 1.0) -> pd.Series:
    """Mean of `series` in bins of `step_s` seconds; empty bins stay NaN."""
    return series.resample(pd.Timedelta(seconds=step_s)).mean()


def longest_gap_free(series: pd.Series) -> pd.Series:
    """The longest run of consecutive non-NaN values."""
    ok = series.notna().to_numpy()
    if not ok.any():
        return series.iloc[:0]
    # start/end of every run of True
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    i = int(np.argmax(ends - starts))
    return series.iloc[starts[i] : ends[i]]


@dataclass(frozen=True)
class Period:
    seconds: float
    strength: float  # autocorrelation at that lag, 1 = perfectly repeating


def periodogram(
    x: np.ndarray, step_s: float, max_s: float = 1200.0
) -> tuple[np.ndarray, np.ndarray]:
    """Welch periodogram of the detrended series, segments 8x the longest period of interest."""
    x = signal.detrend(np.asarray(x, dtype=float))
    nperseg = min(len(x), int(8 * max_s / step_s))
    return signal.welch(x, fs=1 / step_s, nperseg=nperseg, window="hann")


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = signal.detrend(np.asarray(x, dtype=float))
    x = x - x.mean()
    n = len(x)
    spec = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spec * np.conj(spec))[: max_lag + 1]
    return np.asarray(acf / acf[0])


def cycle_period(
    x: np.ndarray, step_s: float, min_s: float = 30.0, max_s: float = 1200.0
) -> Period:
    """Fundamental period between min_s and max_s, from the autocorrelation.

    The largest periodogram peak is the wrong tool for this: a short pulse
    once per cycle has harmonics as strong as the fundamental, and on
    synthetic data the top peak lands on half the period. The autocorrelation
    only peaks at whole multiples of the period, and the first multiple is
    the highest because cycle-to-cycle jitter wears the later ones down.
    """
    lo, hi = int(min_s / step_s), int(max_s / step_s)
    if len(x) < 2 * hi:
        raise ValueError("series too short for the requested period range")
    acf = autocorrelation(x, hi)
    i = lo + int(np.argmax(acf[lo : hi + 1]))
    # parabolic interpolation between lags
    shift = 0.0
    if lo < i < hi:
        a, b, c = acf[i - 1 : i + 2]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return Period((i + shift) * step_s, float(acf[i]))


def linear_trend(series: pd.Series) -> tuple[float, float]:
    """Slope per hour and the fraction of variance a straight line explains (R^2)."""
    s = series.dropna()
    hours = (s.index - s.index[0]).total_seconds().to_numpy() / 3600
    slope, intercept = np.polyfit(hours, s.to_numpy(), 1)
    fitted = slope * hours + intercept
    ss_res = float(((s.to_numpy() - fitted) ** 2).sum())
    ss_tot = float(((s.to_numpy() - s.mean()) ** 2).sum())
    return float(slope), 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
