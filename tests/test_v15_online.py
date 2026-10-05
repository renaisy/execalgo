"""v1.5 online service primitives: step/snapshot/restore/decision (T30-T32)."""

import numpy as np
import pytest

from execalgo.decision import ExecutionDecision, make_decision
from execalgo.model import fit_square_root
from execalgo.planner_state import restore, snapshot
from execalgo.replanner import N_MINUTES, ReplanConfig, RollingPlanner


def _profile():
    t = (np.arange(N_MINUTES) + 0.5) / N_MINUTES
    d = np.exp(-4.0 * t) + 0.6 * np.exp(-4.0 * (1 - t)) + 0.4
    return d / d.sum()


def _planner(kernel, **kw):
    return RollingPlanner(0.2055, kernel, _profile(),
                          config=ReplanConfig(interval_minutes=15), **kw)


def test_t30_step_matches_execute(kernel=None):
    """T30: the step-based execute is identical to the pre-refactor loop."""
    kernel = kernel or fit_square_root()[0]
    rng = np.random.default_rng(7)
    day = _profile() * rng.uniform(800, 1200, N_MINUTES)

    p1 = _planner(kernel)
    result_loop = p1.execute(day)

    p2 = _planner(kernel)
    fractions = np.zeros(N_MINUTES)
    filled, pending = 0.0, None
    for start in range(0, N_MINUTES, 15):
        state = p2.step(day, start, filled, fractions, pending)
        fractions, filled, pending = state["fractions"], state["filled"], state["pending"]
        if state["done"]:
            break
    result_step = fractions / fractions.sum()

    np.testing.assert_allclose(result_loop, result_step, rtol=0, atol=1e-14)
    assert p1.replans == p2.replans


def test_t31_snapshot_restore_continuation():
    """T31: snapshot at minute 60, restore, continue == uninterrupted run."""
    kernel = fit_square_root()[0]
    rng = np.random.default_rng(11)
    day = _profile() * rng.uniform(800, 1200, N_MINUTES)

    # uninterrupted
    p_full = _planner(kernel)
    full = p_full.execute(day)

    # half-run then snapshot
    p_half = _planner(kernel)
    fractions = np.zeros(N_MINUTES)
    filled, pending = 0.0, None
    for start in range(0, 60, 15):
        state = p_half.step(day, start, filled, fractions, pending)
        fractions, filled, pending = state["fractions"], state["filled"], state["pending"]
        if state["done"]:
            break

    snap = snapshot(p_half, start=60, fractions=fractions,
                    filled=filled, pending=pending, minute_volume=day)

    # "crash" + restore + continue
    p_rest, start_r, fractions_r, filled_r, pending_r, vol_r = restore(snap)
    for start2 in range(start_r, N_MINUTES, 15):
        state = p_rest.step(day, start2, filled_r, fractions_r, pending_r)
        fractions_r, filled_r, pending_r = (
            state["fractions"], state["filled"], state["pending"])
        if state["done"]:
            break
    resumed = fractions_r / fractions_r.sum()

    np.testing.assert_allclose(full, resumed, rtol=1e-10, atol=1e-12)


def test_t31_snapshot_roundtrip():
    """Snapshot fields survive a to_dict/from_dict roundtrip."""
    kernel = fit_square_root()[0]
    p = _planner(kernel)
    snap = snapshot(p, start=30, fractions=np.full(N_MINUTES, 0.004),
                    filled=0.9, pending=None, minute_volume=np.ones(N_MINUTES))
    assert snap.start == 30
    assert snap.filled == 0.9
    assert len(snap.fractions) == N_MINUTES
    assert len(snap.kernel_weights) == 3

    p2, start, fractions, filled, pending, vol = restore(snap)
    assert start == 30
    assert filled == 0.9
    assert p2.eta == p.eta
    assert p2.benchmark == p.benchmark


def test_t32_decision_format():
    """T32: ExecutionDecision matches the reference architecture's schema."""
    d = make_decision(
        bar_ts="2025-06-04 09:30:00",
        fraction=0.008,
        parent_qty=3500,
        parent_filled=1200,
        side=1,
        reason="replan",
        latency_ms=42.5,
    )
    assert isinstance(d, ExecutionDecision)
    assert d.action == "buy"
    assert d.child_qty == pytest.approx(28.0)  # 0.008 * 3500
    assert d.parent_remaining == pytest.approx(2272.0)  # 3500 - 1200 - 28
    assert d.bar_ts == "2025-06-04 09:30:00"
    assert d.reason == "replan"
    record = d.to_dict()
    for key in ("bar_ts", "action", "side", "child_qty",
                "parent_remaining", "reason", "latency_ms"):
        assert key in record
