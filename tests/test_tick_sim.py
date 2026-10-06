"""Synthetic-scenario tests for the tick-level execution simulator (v2.4).

Each test builds a minimal BookTicks (one or two minutes, 0.5 s snapshots)
with hand-set queues and prints, then asserts the fill physics: FIFO queue
depletion with front-of-queue overshoot, trade-through immediate fills,
timeout escalation with multi-level book walking, microprice gate blocking,
cancel guard, and pending rollforward.
"""

from datetime import date

import numpy as np
import pytest

from execalgo.bookdata import BookTicks
from execalgo.tick_sim import Fill, LadderPolicy, simulate_day

DT = 1.0 / 120.0  # 0.5 s in minutes


def make_ticks(specs, slot=0):
    """specs: list of dicts with optional keys
    last, bid, ask, bvol, avol, vol, sign, ask_px (5,), ask_vol (5,).
    Defaults give an empty book step of one snapshot."""
    n = len(specs)
    arr = lambda key, default: np.array(
        [s.get(key, default) for s in specs], dtype=float)
    five = lambda key, default: np.array(
        [s.get(key, [default] * 5) for s in specs], dtype=float)
    t = np.arange(n) * DT
    return BookTicks(
        date=date(2025, 3, 10), t=t, slot=np.full(n, slot, dtype=int),
        last=arr("last", 100.0), bid1=arr("bid", 100.0), ask1=arr("ask", 110.0),
        bid_vol1=arr("bvol", 10.0), ask_vol1=arr("avol", 10.0),
        bid_vols=five("bid_vol", 10.0), ask_vols=five("ask_vol", 10.0),
        bid_pxs=five("bid_px", 100.0),
        ask_pxs=np.array([s.get("ask_px", [110.0, 111.0, 112.0, 113.0, 114.0])
                          for s in specs], dtype=float),
        vol_inc=arr("vol", 0.0), sign=arr("sign", 0.0),
    )


def sched(qty_by_slot):
    q = np.zeros(225)
    for slot, qty in qty_by_slot.items():
        q[slot] = qty
    return q


def test_queue_depletion_fifo_with_overshoot():
    """Queue of 10 ahead; sells 4+4 exhaust partially, then 2 clears it and
    the overshoot plus the next print fill our 6 lots at the front."""
    specs = [{}]
    specs += [{"last": 100.0, "vol": 4.0, "sign": -1.0},
              {"last": 100.0, "vol": 4.0, "sign": -1.0},
              {"last": 100.0, "vol": 2.0, "sign": -1.0},   # queue -> 0, overshoot 0
              {"last": 100.0, "vol": 5.0, "sign": -1.0},   # front fill 5
              {"last": 100.0, "vol": 3.0, "sign": -1.0}]   # front fill 1
    ticks = make_ticks(specs)
    res = simulate_day(ticks, sched({0: 6.0}),
                       LadderPolicy(mode="passive_fallback"))
    assert res.passive_fraction == pytest.approx(1.0)
    assert res.avg_price == pytest.approx(100.0)
    assert res.shortfall == pytest.approx(0.0)
    qty_at = [f.qty for f in res.fills]
    assert qty_at == pytest.approx([5.0, 1.0])


def test_trade_through_fills_everything_at_limit():
    specs = [{}]
    specs += [{"last": 100.0, "vol": 2.0, "sign": -1.0}]
    specs += [{"last": 95.0, "bid": 95.0, "vol": 1.0, "sign": -1.0}]  # through
    specs += [{"last": 95.0, "bid": 95.0, "vol": 1.0, "sign": -1.0}]
    ticks = make_ticks(specs)
    res = simulate_day(ticks, sched({0: 8.0}),
                       LadderPolicy(mode="passive_fallback"))
    assert res.passive_fraction == pytest.approx(1.0)
    assert res.avg_price == pytest.approx(100.0)  # our limit, not 95
    assert res.fills[0].reason == "through"


def test_timeout_escalation_walks_levels():
    """No flow at all: after the timeout the remainder lifts L1 then L2."""
    specs = [{} for _ in range(20)]  # 10 s of idle book
    ticks = make_ticks(specs)
    pol = LadderPolicy(mode="ladder", passive_timeout=5.0)
    res = simulate_day(ticks, sched({0: 16.0}), pol)
    assert res.passive_fraction == pytest.approx(0.0)
    assert res.escalations == 1
    # 10 lots @ 110 + 6 lots @ 111
    assert res.avg_price == pytest.approx((10 * 110.0 + 6 * 111.0) / 16.0)
    assert all(f.reason == "escalate" for f in res.fills)


def test_taker_mode_walks_at_minute_start():
    specs = [{} for _ in range(5)]
    specs[0]["ask_vol"] = [10.0, 10.0, 10.0, 10.0, 10.0]
    ticks = make_ticks(specs)
    res = simulate_day(ticks, sched({0: 16.0}), LadderPolicy(mode="taker"))
    assert res.passive_fraction == pytest.approx(0.0)
    assert res.avg_price == pytest.approx((10 * 110.0 + 6 * 111.0) / 16.0)
    assert res.fills[0].reason == "start"


def test_gate_blocks_and_completes_patiently():
    """Ask-heavy book all minute: the microprice gate never opens, the whole
    slice completes as taker at the LAST snapshot of the minute."""
    specs = [{"bvol": 5.0, "avol": 20.0} for _ in range(120)]  # frac ~ -0.6
    ticks = make_ticks(specs)
    pol = LadderPolicy(mode="ladder", passive_timeout=30.0, gate="microprice")
    res = simulate_day(ticks, sched({0: 10.0}), pol)
    assert res.passive_fraction == pytest.approx(0.0)
    assert res.escalations == 1
    assert res.fills[0].t == pytest.approx(ticks.t[-1])  # minute end, not 30 s


def test_cancel_guard_abandons_into_patient_escalation():
    """Book balanced at placement, then microprice pins to the bid (frac
    < -0.8): the guard cancels, recovery never comes, and the remainder
    escalates at the deadline (minute end here) — not into the falling book
    at the flip."""
    specs = [{} for _ in range(20)]
    for k in range(10, 20):
        specs[k] = {"bvol": 1.0, "avol": 40.0}  # frac ~ -0.95, never recovers
    ticks = make_ticks(specs)
    pol = LadderPolicy(mode="ladder", passive_timeout=60.0, cancel_guard=True)
    res = simulate_day(ticks, sched({0: 6.0}), pol)
    assert res.guard_cancels == 1
    assert res.requotes == 0
    assert res.passive_fraction == pytest.approx(0.0)
    assert res.escalations == 1
    assert res.fills[0].venue == "taker"
    assert res.fills[0].t == pytest.approx(ticks.t[-1])


def test_guard_requote_recovers():
    """After a guard cancel the microprice recovers within the same minute:
    the order re-posts with a fresh queue and fills passively."""
    specs = [{} for _ in range(40)]
    specs[10] = {"bvol": 1.0, "avol": 40.0}   # flip -> cancel
    specs[11] = {"bvol": 1.0, "avol": 40.0}
    specs[12] = {"bvol": 20.0, "avol": 5.0}   # recovered -> re-quote here
    specs[20] = {"last": 100.0, "vol": 30.0, "sign": -1.0}  # exhausts queue
    ticks = make_ticks(specs)
    pol = LadderPolicy(mode="ladder", passive_timeout=60.0, cancel_guard=True)
    res = simulate_day(ticks, sched({0: 6.0}), pol)
    assert res.guard_cancels == 1
    assert res.requotes == 1
    assert res.passive_fraction == pytest.approx(1.0)
    assert res.avg_price == pytest.approx(100.0)


def test_pending_rolls_forward_over_empty_minute():
    specs = [{} for _ in range(10)]
    ticks = make_ticks(specs, slot=1)  # minute 0 has no snapshots
    res = simulate_day(ticks, sched({0: 4.0, 1: 2.0}),
                       LadderPolicy(mode="passive_fallback"))
    # 6 lots idle book: escalation at minute end walks the (infinite) book
    assert res.shortfall == pytest.approx(0.0)
    assert sum(f.qty for f in res.fills) == pytest.approx(6.0)
    assert res.minutes[0].want == pytest.approx(6.0)  # 4 rolled + 2


def test_cancellation_clamp_shrinks_queue_ahead():
    """The observed L1 queue collapses without any trade (cancellations):
    the clamp pulls our queue-ahead down and a small later print fills us."""
    specs = [{}]
    specs += [{"bvol": 2.0}]                                   # queue 10 -> 2 (cancels)
    specs += [{"last": 100.0, "vol": 3.0, "sign": -1.0}]       # fills 1 lot at front
    ticks = make_ticks(specs)
    res = simulate_day(ticks, sched({0: 1.0}),
                       LadderPolicy(mode="passive_fallback"))
    assert res.passive_fraction == pytest.approx(1.0)
    assert res.avg_price == pytest.approx(100.0)


def test_day_totals_are_consistent():
    specs = [{}]
    specs += [{"last": 100.0, "vol": 50.0, "sign": -1.0}]
    specs += [{} for _ in range(10)]
    ticks = make_ticks(specs)
    res = simulate_day(ticks, sched({0: 10.0}),
                       LadderPolicy(mode="passive_fallback"))
    assert res.passive_qty + res.taker_qty == pytest.approx(res.want_total)
    assert res.shortfall == pytest.approx(0.0)
