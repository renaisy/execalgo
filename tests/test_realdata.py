"""Real-data layer: session mapping, loader quirks, calibration, replay invariants."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

pyarrow = pytest.importorskip("pyarrow")
pyarrow_feather = pytest.importorskip("pyarrow.feather")

from execalgo.model import fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    SESSION_MINUTES,
    TradingDay,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    kernel_transpose_mass,
    replay_day,
    window_stats,
)


def flat_day(price=100.0, volume=None):
    volume = np.ones(N_SESSION_MINUTES) if volume is None else volume
    return TradingDay(date(2025, 6, 4), np.full(N_SESSION_MINUTES, price), volume,
                      N_SESSION_MINUTES, 0.0)


def test_session_timeline_excludes_breaks():
    assert N_SESSION_MINUTES == 225
    minutes = set(SESSION_MINUTES.tolist())
    assert 540 in minutes and 899 in minutes          # 09:00 open, 15:00 close edge
    assert 614 in minutes and 615 not in minutes      # 10:15 break starts
    assert 629 not in minutes and 630 in minutes      # 10:30 resumes
    assert 689 in minutes and 690 not in minutes      # 11:30 lunch starts
    assert 809 not in minutes and 810 in minutes      # 13:30 resumes


def _write_synthetic_feather(path):
    """Two synthetic trading days in vendor format: file time = Beijing - 5h,
    volume cumulative within the day; each day's night session (file hours
    16-19 + the post-midnight hour 0 of the next file-date evening) precedes
    it and one day carries a limit-locked (bid=0) degenerate row."""
    rows = []
    evenings = ("2025-06-03", "2025-06-04")
    for evening, day, base_volume, price in zip(
        evenings,
        ("2025-06-04", "2025-06-05"),
        (10_000.0, 20_000.0),
        (100.0, 110.0),
    ):
        # night 21:00-23:00 (file 16:00-18:00): +300 lots
        for hour in (16, 17):
            rows.append((f"{evening} {hour}:00:30.000", price, 0.0 if hour == 16 else 150.0))
            rows.append((f"{evening} {hour}:30:30.000", price, 150.0 if hour == 16 else 300.0))
        # post-midnight 00:30 Beijing next morning (file 19:30 same evening): +100
        rows.append((f"{evening} 19:30:30.000", price, 400.0))
        # pre-open anchor (Beijing 08:59:59.5 -> file 03:59:59.5)
        rows.append((f"{day} 03:59:59.500", price, base_volume))
        # day session 09:00-09:03 Beijing -> file 04:00-04:03, 100 lots/minute;
        # prints alternate above/below the running mid to exercise the sign rule
        for minute in range(4):
            t = pd.Timestamp(f"{day} 04:{minute:02d}:30.000")
            rows.append((t.strftime("%Y-%m-%d %H:%M:%S.%f"), price + minute + 2 * (minute % 2),
                         base_volume + 100.0 * (minute + 1)))
        # one row inside the lunch break (no volume change)
        rows.append((f"{day} 07:10:00.000", price, base_volume + 400.0))
    frame = pd.DataFrame(rows, columns=["time", "lastPrice", "volume"])
    frame = frame.sort_values("time").reset_index(drop=True)
    # quotes are built around each row's own print (mid = lastPrice); with
    # the loader's previous-mid signing this makes the sign a tick rule and
    # the alternating prints yield buy/sell/buy by design
    frame["bidPrice1"] = frame["lastPrice"] - 1.0
    frame["askPrice1"] = frame["lastPrice"] + 1.0
    # degenerate limit-locked row on day 1's DAY session (v1.6 P1-9: was
    # placed in the night which the loader strips before the book filter,
    # making the test unable to fail): bid quoted at 0
    lock = frame["time"].str.startswith("2025-06-04 04:02:")
    frame.loc[lock, "bidPrice1"] = 0.0
    # depth columns: L1=50+i, L2..L5=10 each (L5 sum = 90+i per snapshot),
    # varying by row index so minute means are distinguishable
    for i in range(1, 6):
        frame[f"bidVolume{i}"] = (50 + i + frame.index % 4 if i == 1 else 10)
        frame[f"askVolume{i}"] = (50 + i + frame.index % 3 if i == 1 else 10)
    pyarrow_feather.write_feather(frame, path)
    return frame


def test_loader_timezone_and_cumulative_volume(tmp_path):
    path = tmp_path / "SYN_Comdty_2025.feather"
    _write_synthetic_feather(path)
    days = list(iter_day_sessions(path, min_coverage=2))
    assert [d.date for d in days] == [date(2025, 6, 4), date(2025, 6, 5)]
    for day, base in zip(days, (10_000.0, 20_000.0)):
        # exactly 400 lots traded (night total must not leak into day volume)
        assert day.minute_volume.sum() == 400.0
        assert day.coverage == 4
        assert day.half_spread == 1.0
    # per-minute VWAPs are the prints (alternating above/below mid by design)
    np.testing.assert_allclose(days[0].minute_vwap[:4], [100.0, 103.0, 102.0, 105.0])
    np.testing.assert_allclose(days[1].minute_vwap[:4], [110.0, 113.0, 112.0, 115.0])


def test_loader_night_session_attribution(tmp_path):
    """Night volume (21:00-23:00 + post-midnight hour) belongs to the next
    trading day; the post-midnight tail must not be dropped."""
    path = tmp_path / "SYN_Comdty_2025.feather"
    _write_synthetic_feather(path)
    days = list(iter_day_sessions(path, min_coverage=2))
    # each evening traded 400 night lots (300 before midnight + 100 after)
    assert days[0].night_volume == 400.0
    assert days[1].night_volume == 400.0
    # the post-midnight hour (file 19:30 = Beijing 00:30) is inside that 400:
    # it landed in neither the day bars nor the void between sessions
    assert days[0].minute_volume.sum() == 400.0      # day volume unchanged


def test_loader_buy_sell_signing(tmp_path):
    """Mid-rule signing: prints above the running mid are buys, below sells;
    the first print of the day carries sign 0 (counts in volume only)."""
    path = tmp_path / "SYN_Comdty_2025.feather"
    _write_synthetic_feather(path)
    days = list(iter_day_sessions(path, min_coverage=2))
    day = days[0]
    np.testing.assert_allclose(day.minute_buy[:4], [0.0, 100.0, 0.0, 100.0])
    np.testing.assert_allclose(day.minute_sell[:4], [0.0, 0.0, 100.0, 0.0])
    np.testing.assert_allclose(day.minute_signed[:4], [0.0, 100.0, -100.0, 100.0])
    assert day.minute_mid is not None and np.isfinite(day.minute_mid).all()


def test_loader_degenerate_book_filtered(tmp_path):
    """Limit-locked bid=0 quotes must not poison the spread (median over
    two-sided valid quotes only)."""
    path = tmp_path / "SYN_Comdty_2025.feather"
    frame = _write_synthetic_feather(path)
    days = list(iter_day_sessions(path, min_coverage=2))
    # day 1's evening contains a bid=0 row; the valid quotes have spread 2
    assert days[0].half_spread == 1.0
    assert days[1].half_spread == 1.0


def test_calibration_internal_consistency():
    kernel, _ = fit_square_root()
    stats = {
        "mean_volume": 50_000.0, "volume_std_over_mean": 0.15,
        "mean_price": 100.0, "window_sigma": 1.0,
        "mean_half_spread": 0.2,
    }
    calib = calibrate(stats, kernel, participation=0.05, c_imp=1.0)
    i_g = kernel_transpose_mass(kernel)
    total = stats["window_sigma"] * np.sqrt(0.05)
    assert calib["total_per_share"] == pytest.approx(total)
    # solver eta reproduces the slippage/impact split: eta = s/(1-s) * I_G
    assert calib["eta_solver"] == pytest.approx(
        calib["slippage_share"] / (1 - calib["slippage_share"]) * i_g, rel=1e-9
    )
    # and the split is floored at the paper's canonical balance
    assert calib["slippage_share"] >= 0.13
    # TWAP per-share cost implied by the scales matches the target
    twap_cost = calib["eta_p"] * calib["share"] + calib["lambda"] * i_g
    assert twap_cost == pytest.approx(total, rel=1e-9)


def test_replay_flat_price_twap_matches_calibration():
    kernel, _ = fit_square_root()
    stats = {"mean_volume": 50_000.0, "volume_std_over_mean": 0.1,
             "mean_price": 100.0, "window_sigma": 1.0, "mean_half_spread": 0.2}
    calib = calibrate(stats, kernel, participation=0.05)
    day = flat_day(price=100.0)
    quantities = np.full(N_SESSION_MINUTES, calib["share"] / N_SESSION_MINUTES)
    result = replay_day(quantities, day, calib["lambda"], calib["eta_p"], kernel)
    # flat price removes timing; TWAP cost = slippage + impact = total target
    assert result["is_bps"] == pytest.approx(calib["total_bps"], rel=0.05)


def test_replay_profile_matching_has_zero_vwap_slippage_without_impact():
    """Strong invariant: with no impact and no slippage, executing exactly in
    proportion to realized volume fills at the day VWAP, by construction."""
    kernel, _ = fit_square_root()
    rng = np.random.default_rng(3)
    day = flat_day(price=100.0, volume=rng.uniform(0.5, 3.0, N_SESSION_MINUTES))
    day = TradingDay(day.date, 100.0 + np.arange(N_SESSION_MINUTES) * 0.01,
                     day.minute_volume, N_SESSION_MINUTES)
    quantities = day.minute_volume / day.minute_volume.sum() * 123.0
    result = replay_day(quantities, day, lam=0.0, eta_p=0.0, kernel=kernel)
    assert abs(result["vwap_bps"]) < 1e-9


def test_profile_normalization_real():
    kernel, _ = fit_square_root()
    rng = np.random.default_rng(5)
    days = [
        TradingDay(date(2025, 6, i + 1), np.full(N_SESSION_MINUTES, 100.0),
                   rng.uniform(0.5, 2.0, N_SESSION_MINUTES), N_SESSION_MINUTES, 0.1)
        for i in range(3)
    ]
    profile = build_volume_profile(days)
    assert profile.sum() == pytest.approx(1.0)
    assert profile.shape == (N_SESSION_MINUTES,)
    stats = window_stats(days)
    assert stats["mean_volume"] == pytest.approx(
        np.mean([d.minute_volume.sum() for d in days])
    )


def test_loader_depth_aggregation(tmp_path):
    """L1-L5 depth columns are summed in Arrow and aggregated per minute."""
    path = tmp_path / "SYN_Comdty_2025.feather"
    _write_synthetic_feather(path)
    days = list(iter_day_sessions(path, min_coverage=2))
    for day in days:
        assert day.minute_bid_depth is not None
        assert day.minute_ask_depth is not None
        assert day.minute_bid_depth.shape == (N_SESSION_MINUTES,)
        # day-session minutes have nonzero depth (L1~50-57 + L2-L5=40 => 90-97)
        assert day.minute_bid_depth[:4].min() > 80
        assert day.minute_ask_depth[:4].min() > 80
        # silent minutes have depth 0 (no snapshots)
        assert day.minute_bid_depth[10] == 0.0
