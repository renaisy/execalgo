"""v1.0 maker/taker execution model: fill mechanics and rules (T19/T20)."""

from datetime import date

import numpy as np
import pytest

from execalgo.execution_model import MakerPolicy, replay_day_book
from execalgo.model import fit_square_root
from execalgo.realdata import N_SESSION_MINUTES, TradingDay

N = N_SESSION_MINUTES
PRICE = 100.0
SPREAD = 2.0  # full spread => half_spread = 1.0


def _day(volume=None, sell=None, buy=None):
    rng = np.random.default_rng(5)
    volume = np.full(N, 100.0) if volume is None else volume
    sell = volume / 2 if sell is None else sell
    buy = volume - sell if buy is None else buy
    return TradingDay(
        date=date(2025, 6, 4),
        minute_vwap=np.full(N, PRICE),
        minute_volume=np.asarray(volume, dtype=float),
        coverage=int((np.asarray(volume) > 0).sum()),
        half_spread=SPREAD / 2,
        night_volume=0.0,
        minute_signed=np.asarray(buy) - np.asarray(sell),
        minute_mid=np.full(N, PRICE),
        minute_buy=np.asarray(buy, dtype=float),
        minute_sell=np.asarray(sell, dtype=float),
    )


def _kernel():
    return fit_square_root()[0]


def test_t19_all_taker_pays_half_spread():
    day = _day()
    quantities = np.full(N, 10.0)
    result = replay_day_book(quantities, day, lam=0.0, kernel=_kernel(),
                             policy=MakerPolicy(passive_fraction=0.0))
    # pure taker: cost = mid + half_spread -> is_bps = spread/2 / price
    assert result["is_bps"] == pytest.approx(1.0 / PRICE * 1e4, rel=1e-9)
    assert result["passive_fraction"] == pytest.approx(0.0)
    assert result["spread_capture_bps"] == pytest.approx(-1.0 / PRICE * 1e4, rel=1e-9)


def test_t19_full_passive_with_flow_captures_spread():
    day = _day(sell=np.full(N, 100.0), buy=np.full(N, 0.0))
    quantities = np.full(N, 5.0)   # small slices: flow covers everything
    result = replay_day_book(quantities, day, lam=0.0, kernel=_kernel(),
                             policy=MakerPolicy(passive_fraction=1.0))
    # minute 0 is taker-only by the T20 rule: (N-1)/N fills passively
    assert result["passive_fraction"] == pytest.approx((N - 1) / N)
    expected = ((N - 1) * (-1.0) + 1.0) / N / PRICE * 1e4
    assert result["is_bps"] == pytest.approx(expected, rel=1e-9)


def test_t19_queue_capture_limits_passive_fills():
    day = _day(sell=np.full(N, 10.0), buy=np.full(N, 90.0))
    quantities = np.full(N, 100.0)
    result = replay_day_book(quantities, day, lam=0.0, kernel=_kernel(),
                             policy=MakerPolicy(passive_fraction=0.5, queue_capture=0.7))
    # passive cap per minute = 0.7 * 10 = 7 lots of the 50 posted; minute 0
    # (taker-only) contributes none
    assert result["passive_fraction"] == pytest.approx((N - 1) * 7.0 / (N * 100.0))
    assert 0.0 < result["passive_fraction"] < 0.5


def test_t20_opening_minute_is_taker_only():
    day = _day(sell=np.full(N, 1000.0), buy=np.full(N, 0.0))
    quantities = np.full(N, 10.0)
    result = replay_day_book(quantities, day, lam=0.0, kernel=_kernel(),
                             policy=MakerPolicy(passive_fraction=1.0))
    # minute 0 cannot fill passively: (N-1) minutes earn the half spread,
    # one minute pays it
    expected_bps = ((N - 1) * (-1.0) + 1.0) / N / PRICE * 1e4
    assert result["is_bps"] == pytest.approx(expected_bps, rel=1e-6)
    assert result["passive_fraction"] == pytest.approx((N - 1) / N)


def test_t20_untradable_minutes_roll_forward():
    volume = np.full(N, 100.0)
    volume[100:103] = 0.0                       # a three-minute halt
    day = _day(volume=volume)
    quantities = np.full(N, 10.0)
    result = replay_day_book(quantities, day, lam=0.0, kernel=_kernel(),
                             policy=MakerPolicy(passive_fraction=0.5, queue_capture=1.0))
    # every share still executes (halt slices deferred into the next minute)
    assert result["executed_fraction"] == pytest.approx(1.0)
    assert np.isfinite(result["is_bps"])
    # deferred lots price at the deferral minute: with sell flow 50/min,
    # minute 103 wants 40 (10 + 30 rolled) and fills min(0.5*40, 50)=20
    # passive + 20 taker; the three halted minutes lose their passive fills
    expected_passive = (N - 1 - 3) * 5.0 + 15.0    # minute 0 stays taker; the
    # deferred 30 lots add 15 passive on top of minute 103's own 5
    assert result["passive_fraction"] == pytest.approx(expected_passive / (N * 10.0))


def test_t19_passive_fraction_band():
    rng = np.random.default_rng(7)
    day = _day(sell=rng.uniform(30, 70, N), buy=rng.uniform(30, 70, N))
    quantities = np.full(N, 20.0)
    result = replay_day_book(quantities, day, lam=0.0, kernel=_kernel(),
                             policy=MakerPolicy(passive_fraction=0.5, queue_capture=0.7))
    assert 0.0 < result["passive_fraction"] <= 0.5 + 1e-12
    assert -2.0 / PRICE * 1e4 <= result["spread_capture_bps"] <= 1.0 / PRICE * 1e4


def test_impact_residual_unchanged_by_policy():
    """Same quantities -> same impact add-on under taker-only and passive."""
    day = _day()
    quantities = np.full(N, 10.0)
    taker = replay_day_book(quantities, day, lam=5.0, kernel=_kernel(),
                            policy=MakerPolicy(passive_fraction=0.0))
    passive = replay_day_book(quantities, day, lam=5.0, kernel=_kernel(),
                              policy=MakerPolicy(passive_fraction=1.0,
                                                 queue_capture=1.0))
    # impact enters both fills identically; the spread terms differ by the
    # full spread on N-1 minutes (minute 0 is taker under both policies)
    assert taker["is_bps"] - passive["is_bps"] == pytest.approx(
        SPREAD * (N - 1) / N / PRICE * 1e4, rel=1e-9)
