"""v1.1.1 remediation regression tests."""

import numpy as np
import pytest

from execalgo import ExecParams, participation_upper
from execalgo.discretize import make_grid
from execalgo.model import fit_square_root


def test_participation_upper_hand_check():
    """v <= cap * Vbar * p(t) * N * share / X, at the TWAP anchor minute."""
    kernel, _ = fit_square_root()
    params = ExecParams(share=1.0, grid_size=151)
    grid = params.make_grid()
    profile = np.full(225, 1.0 / 225)        # flat profile
    mean_volume = 70_000.0
    share_lots = 3_500.0

    upper = participation_upper(0.08, profile, mean_volume, grid, share_lots)
    # every minute: cap * 70000 * (1/225) * 225 / 3500 = cap * 70000 / 3500
    expected = 0.08 * mean_volume * (1.0 / 225) * 225 / share_lots
    assert np.allclose(upper, expected, rtol=1e-10)


def test_participation_upper_share_factor():
    """The solver-share factor scales the ceiling linearly."""
    kernel, _ = fit_square_root()
    params = ExecParams(share=2.0, grid_size=151)
    grid = params.make_grid()
    profile = np.full(225, 1.0 / 225)
    base = participation_upper(0.1, profile, 50_000.0, grid, 1000.0, share=1.0)
    doubled = participation_upper(0.1, profile, 50_000.0, grid, 1000.0, share=2.0)
    assert np.allclose(doubled, 2.0 * base)


def test_fallback_constrained_segment_offset_live(kernel=None):
    """The _constrained_segment fallback must actually pass the carry offset
    and risk to continue_branch — not solve a bare risk-neutral problem."""
    from unittest.mock import patch
    from execalgo.replanner import N_MINUTES, ReplanConfig, RollingPlanner
    kernel = kernel or fit_square_root()[0]
    minutes = np.arange(N_MINUTES)
    t = (minutes + 0.5) / N_MINUTES
    density = np.exp(-4.0 * t) + 0.6 * np.exp(-4.0 * (1.0 - t)) + 0.4
    profile = density / density.sum()

    planner = RollingPlanner(0.2055, kernel, profile,
                             risk=0.05,
                             config=ReplanConfig(interval_minutes=15))
    fractions = np.zeros(N_MINUTES)
    fractions[:15] = 0.01
    phi = 1.0 - fractions.sum()

    captured = {}
    original = "execalgo.replanner.continue_branch"

    def spy_grid(*args, **kwargs):
        captured["risk"] = kwargs.get("risk", 0.0)
        captured["offset"] = kwargs.get("offset", None)
        from execalgo.branch import continue_branch as real_cb
        return real_cb(*args, **kwargs)

    with patch(original, side_effect=spy_grid):
        planner._constrained_segment(profile * 1000.0, fractions, 15,
                                     N_MINUTES - 15, phi)
    h = (N_MINUTES - 15) / N_MINUTES
    assert captured["risk"] == pytest.approx(0.05 * float(np.sqrt(phi)) * h**0.5)
    assert captured["offset"] is not None
    assert np.abs(np.nan_to_num(captured["offset"])).max() > 0.005


def test_loader_no_night_session(tmp_path):
    """A product with no night rows: night_volume is 0, day volume unchanged."""
    import pandas as pd
    import pyarrow.feather as feather
    from execalgo.realdata import iter_day_sessions, N_SESSION_MINUTES as N

    rows = []
    day = "2025-06-04"
    price = 100.0
    rows.append((f"{day} 03:59:59.500", price, 5000.0))
    for minute in range(4):
        t = pd.Timestamp(f"{day} 04:{minute:02d}:30.000")
        rows.append((t.strftime("%Y-%m-%d %H:%M:%S.%f"), price + minute,
                     5000.0 + 100.0 * (minute + 1)))
    frame = pd.DataFrame(rows, columns=["time", "lastPrice", "volume"])
    frame["bidPrice1"] = frame["lastPrice"] - 1.0
    frame["askPrice1"] = frame["lastPrice"] + 1.0
    for i in range(1, 6):
        frame[f"bidVolume{i}"] = 10
        frame[f"askVolume{i}"] = 10
    path = tmp_path / "NONIGHT_Comdty_2025.feather"
    feather.write_feather(frame, path)

    days = list(iter_day_sessions(path, min_coverage=2))
    assert len(days) == 1
    assert days[0].night_volume == 0.0
    assert days[0].minute_volume.sum() == 400.0


def test_book_model_terminal_halt_forced_completion():
    """A day ending on untradable minutes forces the tail taker (P2-8)."""
    from datetime import date
    from execalgo.execution_model import MakerPolicy, replay_day_book
    from execalgo.model import fit_square_root
    from execalgo.realdata import N_SESSION_MINUTES as N, TradingDay

    volume = np.full(N, 100.0)
    volume[-5:] = 0.0                        # the day ends halted
    day = TradingDay(date(2025, 6, 4), np.full(N, 100.0), volume,
                     int((volume > 0).sum()), 1.0, 0.0,
                     np.zeros(N), np.full(N, 100.0), np.full(N, 50.0), np.full(N, 50.0))
    result = replay_day_book(np.full(N, 10.0), day, lam=0.0,
                             kernel=fit_square_root()[0],
                             policy=MakerPolicy(passive_fraction=0.5))
    # all shares execute (forced taker at the last tradable minute)
    assert result["executed_fraction"] == pytest.approx(1.0)
    # the forced tail is taker: 50 halted lots don't fill passively
    # (the ceiling for passive fraction is (N-6)*5 / (N*10))
    ceiling = (N - 6) * 5.0 / (N * 10.0)
    assert result["passive_fraction"] <= ceiling + 1e-9
