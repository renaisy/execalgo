"""v2.4 minute-layer upgrades: L1 queue basis, isotonic fill curve, auto
passive fraction, per-minute spread."""

from datetime import date

import numpy as np
import pytest

from execalgo.execution_model import MakerPolicy, replay_day_book
from execalgo.fill_calibration import _isotonic, calibrate_fill_curve, fill_at
from execalgo.model import fit_square_root
from execalgo.realdata import N_SESSION_MINUTES, TradingDay


def syn_day(bid_l1=None, minute_hs=None):
    price = 100.0
    day = TradingDay(
        date(2025, 6, 4),
        np.full(N_SESSION_MINUTES, price), np.ones(N_SESSION_MINUTES),
        N_SESSION_MINUTES, 1.0)
    day.minute_sell = np.full(N_SESSION_MINUTES, 50.0)
    day.minute_bid_depth = np.full(N_SESSION_MINUTES, 60.0)
    if bid_l1 is not None:
        day.minute_bid_l1 = np.full(N_SESSION_MINUTES, bid_l1)
    if minute_hs is not None:
        day.minute_half_spread = np.full(N_SESSION_MINUTES, minute_hs)
    return day


def test_isotonic_is_monotone_and_fits_means():
    y = np.array([0.0, 0.5, 0.2, 0.9, np.nan, 0.7])
    w = np.array([10, 5, 20, 3, 0, 1])
    out = _isotonic(y, w)
    assert np.isfinite(out).all()
    good = np.arange(len(y)) != 4
    assert (np.diff(out[good]) >= -1e-12).all()      # monotone non-decreasing
    # the violators 0.5/0.2 pool to their weighted mean
    pooled = (0.5 * 5 + 0.2 * 20) / 25
    assert out[1] == pytest.approx(pooled)
    assert out[2] == pytest.approx(pooled)
    # the NaN bin interpolates between neighbours
    assert out[4] == pytest.approx((out[3] + out[5]) / 2)


def test_l1_basis_curve_moves_ratio_right():
    """Same days: the L1 curve's ratio grid sits ~8x higher (24.7 vs 197.7
    mean depth) and its low-ratio region fills nothing, matching overflow
    against the true queue."""
    days = [syn_day(bid_l1=25.0) for _ in range(6)]
    # stagger flows so the ratio distribution is non-degenerate
    for i, d in enumerate(days):
        d.minute_sell = np.full(N_SESSION_MINUTES, 10.0 + 20.0 * i)
    l5 = calibrate_fill_curve(days, queue_basis="l5", isotonic=False,
                              bootstrap_draws=2)
    l1 = calibrate_fill_curve(days, queue_basis="l1", isotonic=False,
                              bootstrap_draws=2)
    assert l1.queue_basis == "l1" and l5.queue_basis == "l5"
    # D_l1=25 vs D_l5=60 in this synthetic: ratio grid must sit higher
    assert l1.ratio_edges[0] > l5.ratio_edges[0]
    # against the same flow, L1 depth yields at least as much fill
    for flow in (30.0, 60.0, 120.0):
        assert fill_at(l1, flow, 25.0) >= fill_at(l5, flow, 60.0) - 1e-9


def test_isotonic_curve_is_usable_in_fill_at():
    days = [syn_day(bid_l1=25.0) for _ in range(6)]
    for i, d in enumerate(days):
        d.minute_sell = np.full(N_SESSION_MINUTES, 5.0 + 25.0 * i)
    curve = calibrate_fill_curve(days, isotonic=True, bootstrap_draws=2)
    assert curve.isotonic is not None
    assert (np.diff(curve.isotonic) >= -1e-12).all()
    # fill_at returns the monotone table's value times the flow
    ratio = 40.0 / 25.0
    idx = int(np.clip(np.searchsorted(curve.ratio_edges, ratio) - 1,
                      0, len(curve.fill_fractions) - 1))
    assert fill_at(curve, 40.0, 25.0) == pytest.approx(curve.isotonic[idx] * 40.0)


def test_auto_passive_fraction_uncaps_posting():
    """pi="auto" posts the full slice up to the physical fill capacity —
    no 0.5 policy ceiling."""
    kernel, _ = fit_square_root()
    day = syn_day()
    q = np.full(N_SESSION_MINUTES, 1.0)
    cap = MakerPolicy(passive_fraction=0.5, queue_capture=1.0)
    auto = MakerPolicy(passive_fraction="auto", queue_capture=1.0)
    r_cap = replay_day_book(q, day, 0.01, kernel, policy=cap)
    r_auto = replay_day_book(q, day, 0.01, kernel, policy=auto)
    assert r_cap["passive_fraction"] == pytest.approx(0.5, abs=0.02)
    # beta=1 with 50 lots flow vs 1 lot slices: auto fills ~everything
    assert r_auto["passive_fraction"] > 0.95
    assert r_auto["is_bps"] < r_cap["is_bps"]


def test_minute_spread_mode_uses_per_minute_halves():
    kernel, _ = fit_square_root()
    day = syn_day(minute_hs=2.0)  # double the session constant 1.0
    q = np.full(N_SESSION_MINUTES, 1.0)
    pol = MakerPolicy(passive_fraction=0.0)
    r_day = replay_day_book(q, day, 0.01, kernel, policy=pol)
    r_min = replay_day_book(q, day, 0.01, kernel,
                            policy=MakerPolicy(passive_fraction=0.0,
                                               spread_mode="minute"))
    # pure taker pays the (doubled) minute spread on everything
    assert r_min["is_bps"] == pytest.approx(r_day["is_bps"] + (2.0 - 1.0) / 100 * 1e4)
    assert r_min["spread_capture_bps"] == pytest.approx(-2.0 / 100 * 1e4, abs=0.1)
