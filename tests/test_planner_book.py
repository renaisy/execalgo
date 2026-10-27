"""v2.6 integration: rolling replan with tick-level fills."""

from datetime import date

import numpy as np
import pytest

from execalgo.bookdata import BookTicks
from execalgo.model import fit_square_root
from execalgo.planner_book import execute_with_book
from execalgo.replanner import N_MINUTES, ReplanConfig, RollingPlanner
from execalgo.tick_sim import LadderPolicy

DT = 1.0 / 120.0


def u_profile():
    t = (np.arange(N_MINUTES) + 0.5) / N_MINUTES
    d = np.exp(-4.0 * t) + 0.6 * np.exp(-4.0 * (1.0 - t)) + 0.4
    return d / d.sum()


def synth_ticks(n_minutes=30, seed=0, empty_slots=()):
    """n_minutes of ~0.5 s snapshots: two-sided book, mean-reverting prints."""
    rng = np.random.default_rng(seed)
    n = n_minutes * 30
    i = np.arange(n)
    last = 100.0 + 3.0 * np.sin(i / 50.0) + 0.5 * np.sin(i / 7.0)
    prev = np.roll(last, 1)
    bid = last - 1.0
    ask = last + 1.0
    t = i * DT
    slot = np.floor(t).astype(int)
    keep = ~np.isin(slot, list(empty_slots))
    five = lambda base, arr: np.column_stack([arr + k * base for k in range(5)])
    vol = np.where(rng.random(n) < 0.3, rng.integers(1, 8, n).astype(float), 0.0)
    sign = np.where(last < prev, -1.0, np.where(last > prev, 1.0, 0.0))
    sign[vol <= 0] = 0.0
    return BookTicks(
        date=date(2025, 6, 4), t=t[keep], slot=slot[keep],
        last=last[keep], bid1=bid[keep], ask1=ask[keep],
        bid_vol1=np.full(keep.sum(), 20.0), ask_vol1=np.full(keep.sum(), 20.0),
        bid_vols=np.tile(np.array([20.0, 10, 10, 10, 10]), (keep.sum(), 1)),
        ask_vols=np.tile(np.array([20.0, 10, 10, 10, 10]), (keep.sum(), 1)),
        bid_pxs=five(-1.0, bid[keep]),
        ask_pxs=five(+1.0, ask[keep]),
        vol_inc=vol[keep], sign=sign[keep],
    )


@pytest.fixture(scope="module")
def kernel():
    return fit_square_root()[0]


def make_planner(kernel):
    profile = u_profile()
    return RollingPlanner(
        0.2055, kernel, profile,
        config=ReplanConfig(interval_minutes=15, grid_size=151, max_order=12),
    )


def test_planner_completes_with_book_fills(kernel):
    """The planner's full parent size fills through the tick book; fills,
    passive/taker split, and shortfall are all consistent."""
    ticks = synth_ticks(30)
    share = 200.0
    res = execute_with_book(make_planner(kernel), ticks, share,
                            LadderPolicy(mode="passive_fallback"))
    assert res.shortfall == pytest.approx(0.0)
    assert res.passive_qty + res.taker_qty == pytest.approx(share)
    assert res.fractions.sum() == pytest.approx(1.0, abs=1e-9)
    assert 0.9 < res.avg_price / 100.0 < 1.1
    assert res.replans >= 1


def test_same_trajectory_taker_costs_more(kernel):
    """Arm-vs-arm on the SAME trajectory: the passive fallback undercuts the
    pure taker by roughly the spread times the passive share."""
    ticks = synth_ticks(30, seed=3)
    share = 200.0
    pol_passive = LadderPolicy(mode="passive_fallback")
    pol_taker = LadderPolicy(mode="taker")
    r1 = execute_with_book(make_planner(kernel), ticks, share, pol_passive)
    r2 = execute_with_book(make_planner(kernel), ticks, share, pol_taker)
    # same planner + same realized volume -> identical trajectory
    np.testing.assert_allclose(r1.fractions, r2.fractions, atol=1e-12)
    assert r2.passive_fraction == pytest.approx(0.0)
    assert r1.avg_price < r2.avg_price
    # the saving is bounded by the spread (intra-minute drift moves both
    # arms' escalation prices, so the exact split is not spread x passive)
    spread = float(np.median(ticks.spread))
    assert 0.0 < (r2.avg_price - r1.avg_price) < spread


def test_sell_side_parent_mirrors(kernel):
    """side=-1 sells into the bid book and still completes the parent."""
    ticks = synth_ticks(30, seed=5)
    res = execute_with_book(make_planner(kernel), ticks, 200.0,
                            LadderPolicy(mode="passive_fallback", side=-1))
    assert res.shortfall == pytest.approx(0.0)
    assert res.avg_price < 100.0 + 2.0  # fills cluster at/below the mid
    assert res.is_bps(ticks, side=-1) == pytest.approx(
        -(res.avg_price - ticks.mid[0]) / ticks.mid[0] * 1e4)


def test_empty_minutes_roll_forward(kernel):
    """A tick-less minute defers its quantity into the next minute instead
    of dropping it on the floor."""
    ticks = synth_ticks(10, seed=1, empty_slots=(5,))
    bounds = ticks.minute_bounds()
    assert bounds[5] == bounds[6]  # minute 5 empty by construction
    share = 100.0
    res = execute_with_book(make_planner(kernel), ticks, share,
                            LadderPolicy(mode="passive_fallback"))
    assert res.shortfall == pytest.approx(0.0)
    assert sum(f.qty for f in res.fills) == pytest.approx(share)
