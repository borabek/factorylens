import math

import numpy as np
import pytest

from factorylens import conformal


def test_threshold_is_the_kth_largest_calibration_score():
    cal = np.arange(1.0, 20.0)  # n = 19, alpha = 0.1 -> k = 2
    assert conformal.threshold(cal, 0.1) == 18.0
    assert conformal.threshold(cal, 0.01) == math.inf  # k = 0: never alarm


def test_p_values_match_the_threshold_rule():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=39), rng.normal(size=500)
    t = conformal.threshold(cal, 0.1)
    p = conformal.p_values(cal, test)
    np.testing.assert_array_equal(p <= 0.1, test > t)


def test_false_alarm_rate_holds_on_average():
    rng = np.random.default_rng(1)
    n, alpha = 19, 0.1
    rates = []
    for _ in range(2000):
        t = conformal.threshold(rng.normal(size=n), alpha)
        rates.append((rng.normal(size=200) > t).mean())
    assert abs(np.mean(rates) - conformal.rank(n, alpha) / (n + 1)) < 0.01
    lo, hi = conformal.false_alarm_quantiles(n, alpha, [0.05, 0.95])
    inside = np.mean((np.array(rates) >= lo - 0.02) & (np.array(rates) <= hi + 0.02))
    assert inside > 0.85


def test_more_calibration_parts_tighten_the_rate():
    small = np.diff(conformal.false_alarm_quantiles(20, 0.1, [0.05, 0.95]))[0]
    large = np.diff(conformal.false_alarm_quantiles(500, 0.1, [0.05, 0.95]))[0]
    assert large < small / 3


def test_min_calibration():
    n = conformal.min_calibration(0.05, 0.02, confidence=0.9)
    assert conformal.false_alarm_quantiles(n, 0.05, [0.9])[0] <= 0.07
    assert (
        conformal.false_alarm_quantiles(n - 1, 0.05, [0.9])[0] > 0.07
        or conformal.rank(n - 1, 0.05) < 1
    )
    with pytest.raises(ValueError):
        conformal.min_calibration(0.05, -0.05)
