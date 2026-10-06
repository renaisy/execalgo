"""Tick-level book loader: session mapping, volume increments, book invariants."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
pyarrow_feather = pytest.importorskip("pyarrow.feather")

from execalgo.bookdata import load_day_ticks, minute_book_stats


def _write_synthetic_ticks(path):
    """One synthetic trading day in vendor format with full L5 quotes.

    Day session 09:00-09:03 Beijing (file 04:00-04:03), one 0.5 s snapshot
    per half-minute plus a pre-open anchor; volume cumulative within the
    session; print prices alternate up/down for the tick rule.
    """
    rows = []
    day = "2025-06-04"
    base = 10_000.0
    rows.append((f"{day} 03:59:59.500", 100.0, base))
    for minute in range(4):
        for half, sec in ((0, "00.500"), (1, "01.000")):
            price = 100.0 + minute + (2.0 if half else 0.0)
            rows.append((f"{day} 04:{minute:02d}:{sec}", price,
                         base + 50.0 * (2 * minute + half + 1)))
    # a night row of the PREVIOUS evening must not leak into the day load
    rows.append(("2025-06-03 16:00:30.000", 100.0, 50.0))
    frame = pd.DataFrame(rows, columns=["time", "lastPrice", "volume"])
    frame = frame.sort_values("time").reset_index(drop=True)
    for lvl in range(1, 6):
        frame[f"bidPrice{lvl}"] = frame["lastPrice"] - 1.0 - (lvl - 1)
        frame[f"askPrice{lvl}"] = frame["lastPrice"] + 1.0 + (lvl - 1)
        frame[f"bidVolume{lvl}"] = 20 + lvl
        frame[f"askVolume{lvl}"] = 30 + lvl
    # one limit-locked degenerate row in the middle of the day session
    frame.loc[frame["time"].str.startswith("2025-06-04 04:02:00"), "bidPrice1"] = 0.0
    # AND a degenerate opening stretch (locked open): leading rows have no
    # earlier sane book to forward-fill from (v2.4: the 2025-04-07 case)
    lead = frame["time"].str.startswith("2025-06-04 04:00")
    frame.loc[lead, "bidPrice1"] = 0.0
    pyarrow_feather.write_feather(frame, path)
    return frame


def _load(tmp_path, min_snapshots=4):
    path = tmp_path / "SYN_Comdty_2025.feather"
    _write_synthetic_ticks(path)
    return load_day_ticks(path, "2025-06-04", min_snapshots=min_snapshots)


def test_load_day_ticks_session_mapping(tmp_path):
    """Only the day session's snapshots load (night excluded, anchor dropped
    after anchoring the volume diff), t is on the 225-minute timeline."""
    ticks = _load(tmp_path)
    assert ticks.date == date(2025, 6, 4)
    # 4 minutes x 2 snapshots; the 03:59 anchor is not a session minute
    assert len(ticks.t) == 8
    np.testing.assert_allclose(np.unique(ticks.slot), [0, 1, 2, 3])
    # the open auction (anchor -> first snapshot) survives the diff
    assert ticks.vol_inc[0] == 50.0
    assert ticks.vol_inc.sum() == pytest.approx(400.0)
    # 0.5 s within a minute, ~1 min across minute boundaries
    diffs = np.diff(ticks.t)
    np.testing.assert_allclose(diffs[::2], [1.0 / 120.0] * 4)
    np.testing.assert_allclose(diffs[1::2], [59.5 / 60.0] * 3)


def test_tick_rule_signing(tmp_path):
    """Print prices step +2/-1 alternately: signs follow the tick rule with
    carry through flat prints; zero-volume snapshots carry sign 0."""
    ticks = _load(tmp_path)
    # prints: 100,102,101,103,102,104,103,105 -> diffs (vs anchor 100)
    # 0,+2,-1,+2,-1,+2,-1,+2
    signs = ticks.sign
    assert signs[0] == 0.0                       # flat vs anchor, no carry
    assert signs[1] == pytest.approx(1.0)        # +2 up-tick
    assert signs[2] == pytest.approx(-1.0)       # -1 down-tick
    assert (ticks.sign[ticks.vol_inc <= 0] == 0.0).all()


def test_book_invariants_microprice_and_l1(tmp_path):
    ticks = _load(tmp_path)
    mic = ticks.microprice
    assert ((mic >= ticks.bid1 - 1e-9) & (mic <= ticks.ask1 + 1e-9)).all()
    # ask L1 heavier than bid L1 by construction -> micro below mid
    assert (mic <= ticks.mid + 1e-9).all()
    assert (ticks.bid_vol1 <= ticks.bid_vols.sum(axis=1) + 1e-9).all()
    assert (ticks.ask_vol1 <= ticks.ask_vols.sum(axis=1) + 1e-9).all()
    np.testing.assert_allclose(ticks.spread, 2.0)


def test_degenerate_book_forward_filled(tmp_path):
    """The bid=0 rows must not poison prices: mid-row gaps forward-fill from
    the previous sane book, the locked opening backfills from the first sane
    quote, and the opening mid is finite."""
    ticks = _load(tmp_path)
    assert (ticks.bid1 > 0).all()
    assert np.isfinite(ticks.mid).all()
    assert np.isfinite(ticks.mid[0])
    lock_idx = np.flatnonzero(ticks.t >= 2.0)[0]  # snapshot at 04:02:00
    assert ticks.bid1[lock_idx] == pytest.approx(ticks.bid1[lock_idx - 1])
    # the locked opening takes the first sane book (the 04:00:30 row)
    assert ticks.bid1[0] == pytest.approx(ticks.bid1[2])


def test_minute_bounds_and_stats(tmp_path):
    ticks = _load(tmp_path)
    bounds = ticks.minute_bounds()
    assert len(bounds) == 226
    assert bounds[0] == 0 and bounds[4] == len(ticks.t)
    stats = minute_book_stats(ticks)
    # one 50-lot down-tick print per minute from minute 1 onward
    np.testing.assert_allclose(stats["sell_flow"][:4], [0.0, 50.0, 50.0, 50.0])
    np.testing.assert_allclose(stats["bid_l1"][:4], 21.0)       # 20 + 1
    np.testing.assert_allclose(stats["half_spread"][:4], 1.0)


def test_min_snapshots_guard(tmp_path):
    with pytest.raises(ValueError, match="snapshots"):
        _load(tmp_path, min_snapshots=100)
