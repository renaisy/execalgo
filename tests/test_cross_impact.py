"""v1.4 cross-impact: synthetic recovery + ratio properties (T27)."""

import numpy as np
import pytest

from execalgo.cross_impact import cross_response, spillover_label, spillover_ratio
from execalgo.realdata import N_SESSION_MINUTES


class _Day:
    def __init__(self, dt, signed, mid):
        self.date = dt
        self.minute_signed = signed
        self.minute_mid = mid


def _make_day(dt, seed, cross_amp=0.0):
    """One day: A has own impact; B has own impact + optional spillover from A."""
    rng = np.random.default_rng(seed)
    eps = rng.choice([-1.0, 1.0], N_SESSION_MINUTES)
    # A's mid: random walk + own impact (eps pushes A's price)
    mid_a = 100.0 * np.exp(np.cumsum(
        rng.standard_normal(N_SESSION_MINUTES) * 1e-3 + 0.5e-3 * eps))
    # B's signs (independent of A)
    eps_b = rng.choice([-1.0, 1.0], N_SESSION_MINUTES)
    # B's mid: random walk + own impact + spillover from A
    mid_b = 200.0 * np.exp(np.cumsum(
        rng.standard_normal(N_SESSION_MINUTES) * 1e-3
        + 0.5e-3 * eps_b + cross_amp * eps))
    day_a = _Day(dt, eps, mid_a)
    day_b = _Day(dt, eps_b, mid_b)
    return day_a, day_b


def _synthetic_pair(n_days, cross_amp, seed=7):
    from datetime import date, timedelta
    days_a, days_b = [], []
    base = date(2025, 6, 2)
    for i in range(n_days):
        a, b = _make_day(base + timedelta(days=i), seed + i, cross_amp)
        days_a.append(a)
        days_b.append(b)
    return days_a, days_b


def test_t27_cross_response_computable():
    days_a, days_b = _synthetic_pair(20, cross_amp=0.005)
    result = cross_response(days_a, days_b, max_lag=60)
    assert result["r_cross"].shape == (60,)
    assert np.isfinite(result["r_cross"]).all()
    assert np.isfinite(result["r_own_a"]).all()


def test_t27_zero_spillover_when_independent():
    days_a, days_b = _synthetic_pair(30, cross_amp=0.0, seed=11)
    result = cross_response(days_a, days_b, max_lag=60)
    ratio = spillover_ratio(result)
    assert ratio < 0.1, f"独立品种溢出应可忽略, got {ratio:.3f}"
    assert spillover_label(ratio) in ("可忽略", "弱溢出")


def test_t27_strong_spillover_when_coupled():
    days_a, days_b = _synthetic_pair(30, cross_amp=0.01, seed=3)
    result = cross_response(days_a, days_b, max_lag=60)
    ratio = spillover_ratio(result)
    assert ratio > 0.3, f"强耦合品种溢出应显著, got {ratio:.3f}"
    assert spillover_label(ratio) in ("强溢出", "中等溢出")


def test_own_response_positive():
    """Own response should be positive at short lags (impact pushes own price)."""
    from datetime import date
    rng = np.random.default_rng(5)
    eps = rng.choice([-1.0, 1.0], N_SESSION_MINUTES)
    mid = 100.0 * np.exp(np.cumsum(0.5e-3 * eps + rng.standard_normal(N_SESSION_MINUTES) * 0.5e-3))
    day_a = _Day(date(2025, 6, 4), eps, mid)
    result = cross_response([day_a], [day_a], max_lag=30)
    assert result["r_own_a"][0] > 0
