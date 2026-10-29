"""Split-conformal alarm thresholds.

An anomaly score only becomes a decision with a threshold. Split conformal
prediction sets it from a calibration set of normal parts so that, if new
normal parts are exchangeable with the calibration parts, the probability of
a false alarm is at most alpha, whatever the detector.

With n calibration scores and k = floor(alpha * (n + 1)), alarm when a score
is above the k-th largest calibration score. The false alarm rate is then
k / (n + 1) <= alpha on average over calibration sets. For one fixed
calibration set it is random: Beta(k, n + 1 - k) distributed (Vovk 2012),
which is what decides how many normal parts a line needs before the promised
rate means anything. With k = 0 (n < 1/alpha - 1) the rule never alarms.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def rank(n: int, alpha: float) -> int:
    """How many calibration scores may lie above the threshold."""
    return math.floor(alpha * (n + 1))


def threshold(calibration: np.ndarray, alpha: float) -> float:
    """Alarm for scores strictly above this; inf if n is too small for alpha."""
    k = rank(len(calibration), alpha)
    if k < 1:
        return math.inf
    return float(np.sort(calibration)[len(calibration) - k])


def p_values(calibration: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Conformal p-value of each score: share of calibration scores at least as large (+1)."""
    cal = np.asarray(calibration)
    above = (cal[None, :] >= np.asarray(scores)[:, None]).sum(axis=1)
    return np.asarray((1 + above) / (len(cal) + 1))


def false_alarm_quantiles(n: int, alpha: float, q: np.ndarray | list[float]) -> np.ndarray:
    """Quantiles of the false alarm rate for one calibration set of size n."""
    k = rank(n, alpha)
    if k < 1:
        return np.zeros(len(q))
    return np.asarray(stats.beta.ppf(q, k, n + 1 - k))


def alarm_rate_range(n: int, alpha: float, m: int, level: float = 0.99) -> tuple[float, float]:
    """Range for the share of alarms among m normal parts, with n calibration parts.

    The count is binomial given the false alarm rate, and the rate is
    Beta(k, n + 1 - k) over calibration sets, so the count is beta-binomial.
    A share outside this range is unlikely if the parts are still
    exchangeable with the calibration set.
    """
    k = rank(n, alpha)
    if k < 1:
        return 0.0, 0.0
    tail = (1 - level) / 2
    lo, hi = np.asarray(stats.betabinom.ppf([tail, 1 - tail], m, k, n + 1 - k))
    return float(lo / m), float(hi / m)


def min_calibration(alpha: float, tolerance: float, confidence: float = 0.9) -> int:
    """Smallest n for which the false alarm rate stays below alpha + tolerance
    with the given probability over calibration sets."""
    for n in range(1, 100_000):
        if rank(n, alpha) >= 1 and false_alarm_quantiles(n, alpha, [confidence])[0] <= (
            alpha + tolerance
        ):
            return n
    raise ValueError("no n below 100000 is enough")
