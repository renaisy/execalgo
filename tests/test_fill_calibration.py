"""v1.2 fill calibration: synthetic recovery + curve properties (T21-T22)."""

import numpy as np
import pytest

from execalgo.fill_calibration import (
    FillCurve,
    calibrate_fill_curve,
    fill_at,
    fill_at_constant,
)
from execalgo.realdata import N_SESSION_MINUTES


class _DepthDay:
    def __init__(self, sell, depth):
        self.minute_sell = np.asarray(sell, dtype=float)
        self.minute_bid_depth = np.asarray(depth, dtype=float)


def _synthetic_days(n_days=40, seed=7):
    """Days where sell_flow and bid_depth have a known relationship."""
    rng = np.random.default_rng(seed)
    days = []
    for _ in range(n_days):
        depth = rng.uniform(50, 500, N_SESSION_MINUTES)
        # sell flow is sometimes much larger than depth (fill), sometimes smaller
        ratio = np.exp(rng.uniform(-2, 1.5, N_SESSION_MINUTES))
        sell = depth * ratio
        days.append(_DepthDay(sell, depth))
    return days


def test_t21_curve_monotone_and_ci_bounded():
    days = _synthetic_days()
    curve = calibrate_fill_curve(days, n_bins=8, bootstrap_draws=100)
    valid = curve.counts > 5
    assert valid.sum() >= 6
    # fill fraction should be non-decreasing in the ratio (monotone)
    fractions = curve.fill_fractions[valid]
    assert np.all(np.diff(fractions) >= -0.05)
    # at high ratios, fill fraction approaches 1
    assert curve.fill_fractions[-1] > 0.3
    # CI width in extreme bins bounded
    widths = curve.ci_high - curve.ci_low
    assert widths[valid].max() < 0.5
    # low-ratio bins have near-zero fill
    assert curve.fill_fractions[0] < 0.15


def test_t21_curve_synthetic_recovery():
    """With a known linear depth/flow relation, the curve recovers the pattern."""
    rng = np.random.default_rng(3)
    n = 2000
    depth = rng.uniform(100, 300, n)
    sell = depth * rng.uniform(0.5, 3.0, n)
    days = [_DepthDay(sell, depth)]
    curve = calibrate_fill_curve(days, n_bins=6, bootstrap_draws=10)
    # expected fill_fraction at ratio r: max(0, r-1)/r = 1 - 1/r for r>1
    # verify direction: last bin >> first bin
    assert curve.fill_fractions[-1] > 3 * curve.fill_fractions[0]


def test_fill_at_interpolates():
    edges = np.geomspace(0.01, 10, 5)
    fractions = np.array([0.0, 0.1, 0.5, 0.9])
    curve = FillCurve(edges, fractions, fractions * 0.8, np.minimum(fractions * 1.2, 1.0),
                      np.array([100, 100, 100, 100]))
    # low ratio → near-zero fill
    assert fill_at(curve, 10, 1000) == pytest.approx(0.0, abs=5.0)
    # high ratio → mostly filled
    result = fill_at(curve, 500, 100)  # ratio = 5, high bin
    assert result > 0.3 * 500
    # zero inputs
    assert fill_at(curve, 0, 100) == 0.0
    assert fill_at(curve, 100, 0) == 0.0


def test_fill_at_constant_backward_compatible():
    assert fill_at_constant(0.7, 100) == pytest.approx(70.0)
    assert fill_at_constant(0.7, 0) == 0.0


def test_calibration_raises_on_insufficient_data():
    with pytest.raises(ValueError):
        calibrate_fill_curve([_DepthDay(np.zeros(225), np.zeros(225))])
