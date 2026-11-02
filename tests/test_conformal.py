import math

import numpy as np
import pytest
from scipy import integrate

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


def test_alarm_rate_range_covers_simulated_runs():
    rng = np.random.default_rng(0)
    n, m, alpha = 30, 100, 0.1
    lo, hi = conformal.alarm_rate_range(n, alpha, m, level=0.9)
    inside = []
    for _ in range(2000):
        t = conformal.threshold(rng.normal(size=n), alpha)
        rate = (rng.normal(size=m) > t).mean()
        inside.append(lo <= rate <= hi)
    assert np.mean(inside) >= 0.88  # a discrete range covers at least the level, up to noise
    assert conformal.alarm_rate_range(5, 0.1, 50) == (0.0, 0.0)  # k = 0: never alarms


@pytest.mark.parametrize("alpha", [0, 1, 1.5, -0.1])
def test_alpha_outside_zero_to_one_is_refused(alpha):
    with pytest.raises(ValueError, match="between 0 and 1"):
        conformal.threshold(np.arange(22.0), alpha)


def test_the_bet_has_mean_one_on_uniform_p_values():
    assert integrate.quad(conformal.bet, 0, 1, args=(0.1,))[0] == pytest.approx(1)


def test_fault_test_p_values_are_uniform_on_an_exchangeable_stream():
    # one bet per value is all a run of R = (1 + R) * bet(p) sees, so recover p from it
    rng = np.random.default_rng(0)
    ps = []
    for seed in range(300):
        t = conformal.FaultTest(rng.normal(size=20), level=1e300, seed=seed)
        for x in rng.normal(size=10):
            before = t.statistic
            t.update(x)
            ps.append(((t.statistic / (1 + before)) / t.epsilon) ** (1 / (t.epsilon - 1)))
    counts = np.histogram(ps, bins=10, range=(0, 1))[0]
    assert counts.min() > 0.8 * len(ps) / 10 and counts.max() < 1.2 * len(ps) / 10


def test_fault_test_keeps_its_false_flag_bound():
    rng = np.random.default_rng(0)
    t_max, level = 60, 200
    flags = 0
    runs = 1500
    for seed in range(runs):
        test = conformal.FaultTest(rng.normal(size=30), level=level, seed=seed)
        flags += any(test.update(x) for x in rng.normal(size=t_max))
    assert flags / runs <= t_max / level


def test_fault_test_ignores_scale_and_notices_a_shift():
    rng = np.random.default_rng(0)
    calibration, clean = rng.normal(size=30), rng.normal(size=40)
    a, b = conformal.FaultTest(calibration), conformal.FaultTest(calibration * 1000 + 5)
    for x in clean:
        a.update(x)
        b.update(x * 1000 + 5)
    assert a.statistic == pytest.approx(b.statistic)  # only ranks matter
    assert not a.flagged
    delay = next(i for i in range(100) if a.update(20.0 + i))
    assert delay < 15


def test_fault_test_only_treats_large_values_as_evidence():
    t = conformal.FaultTest(np.arange(30.0))
    assert not any(t.update(-100.0 - i) for i in range(200))


def test_fault_test_flag_stays_up_and_settings_are_checked():
    t = conformal.FaultTest(np.arange(30.0), level=2)
    while not t.update(1000.0):
        pass
    assert t.update(-1000.0) and t.flagged
    with pytest.raises(ValueError, match="epsilon"):
        conformal.FaultTest([1.0], epsilon=1.5)
    with pytest.raises(ValueError, match="level"):
        conformal.FaultTest([1.0], level=0.5)
