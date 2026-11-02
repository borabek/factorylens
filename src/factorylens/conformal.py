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

`FaultTest` asks a different question of the same idea: has a stream of values
stopped looking like its calibration values? Each new value gets a conformal
p-value against the calibration values and everything seen so far. If the
stream is exchangeable those p-values are independent and uniform (Vovk 2005),
so a betting function that integrates to 1 over [0, 1] has mean 1 on them, and
the Shiryaev-Roberts statistic R = (1 + R) * bet(p) has R_t - t as a martingale.
Doob's inequality then bounds a false flag within t values by t / level, and
the mean time to a false flag is at least `level`.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
from scipy import stats


def rank(n: int, alpha: float) -> int:
    """How many calibration scores may lie above the threshold."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
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


def bet(p: float, epsilon: float) -> float:
    """Power betting function: integrates to 1 on [0, 1], large when p is small."""
    return float(epsilon * p ** (epsilon - 1))


class FaultTest:
    """Flags a stream once its values stop looking like the calibration values.

    Only large values count as evidence, and only their ranks matter, so the
    scale of the signal needs no tuning. A flag stays up until a new test starts.
    """

    def __init__(
        self,
        calibration: np.ndarray | list[float],
        epsilon: float = 0.1,
        level: float = 1000.0,
        seed: int = 0,
    ) -> None:
        if not 0 < epsilon < 1:
            raise ValueError(f"epsilon must be between 0 and 1, got {epsilon}")
        if level <= 1:
            raise ValueError(f"level must be above 1, got {level}")
        self.epsilon, self.level = epsilon, level
        self.bag = sorted(float(v) for v in calibration)
        self.statistic = 0.0
        self.flagged = False
        self._rng = np.random.default_rng(seed)

    def update(self, value: float) -> bool:
        """Take the next value; True once the stream has been flagged."""
        if self.flagged:
            return True
        value = float(value)
        low, high = bisect.bisect_left(self.bag, value), bisect.bisect_right(self.bag, value)
        ties = high - low + 1  # the new value ties with itself
        # 1 - random is in (0, 1], so the p-value is never exactly 0
        p = (len(self.bag) - high + (1 - self._rng.random()) * ties) / (len(self.bag) + 1)
        self.statistic = (1 + self.statistic) * bet(p, self.epsilon)
        self.bag.insert(high, value)
        self.flagged = self.statistic >= self.level
        return self.flagged
