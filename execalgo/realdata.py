"""Real futures market data: loading, session mapping, calibration, replay backtest.

Data source: the L5 order-book snapshot feather files under /data/ticker_data
used by the Model-LightGBM project (one row per ~0.5 s, columns time /
lastPrice / volume with volume cumulative within the trading day and the
night session of the previous evening already included).

Two vendor quirks, both verified empirically on CU 2024/2025 by comparing
hour-of-day activity against the SHFE session schedule:

* file timestamps are Beijing time MINUS 5 hours (day session 09:00-15:00
  appears as 04:00-10:00, night session 21:00-01:00 as 16:00-20:00);
* `volume` is cumulative per trading day, so per-minute volume must be the
  positive diff, never the raw column.

The execution window is the day session with scheduled breaks removed
(09:00-10:15, 10:30-11:30, 13:30-15:00 -> 225 trading minutes), mapped onto
the normalized [0, 1] horizon of the perturbative solver.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .model import MultiExponential

# Beijing = file + 5h; the day session spans file-time hours 04..09, plus the
# single ~03:59 pre-open snapshot that anchors the cumulative-volume diff (the
# open-auction volume would otherwise be lost with the first row's diff).
# File hours 16..20 of an evening are the night session (21:00-01:00
# Beijing) that belongs to the NEXT trading day.
FILE_TO_BEIJING = pd.Timedelta(hours=5)
_DAY_FILE_HOURS = ("03", "04", "05", "06", "07", "08", "09")
_NIGHT_FILE_HOURS = ("16", "17", "18", "19", "20")

# SHFE-style day session minute-of-day (Beijing), breaks excluded.
SESSION_MINUTES = np.array(
    [m for m in range(540, 615)]      # 09:00-10:15
    + [m for m in range(630, 690)]    # 10:30-11:30
    + [m for m in range(810, 900)],   # 13:30-15:00
    dtype=int,
)
N_SESSION_MINUTES = SESSION_MINUTES.size  # 225
_SLOT_OF_MINUTE = {int(m): i for i, m in enumerate(SESSION_MINUTES)}
MIN_COVERAGE = 210  # drop days with fewer active minutes (halts / bad data)

_TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"


def _slots_array(values, slots):
    """Scatter filtered per-minute aggregates onto the 225-slot timeline."""
    array = np.zeros(N_SESSION_MINUTES)
    array[slots] = np.asarray(values, dtype=float)
    return array


def _ohlc_array(bars, agg, slots, fallback):
    """Scatter per-minute OHLC (from lastPrice) and forward-fill gaps."""
    column = {"first": "open_px", "max": "high_px", "min": "low_px", "last": "close_px"}[agg]
    array = np.full(N_SESSION_MINUTES, np.nan)
    if column in bars.columns:
        array[slots] = bars[column].to_numpy()
    array[0] = np.nan_to_num(array[0], nan=fallback[0])
    return pd.Series(array).ffill().to_numpy()


@dataclass
class TradingDay:
    """One day session on the 225-minute execution timeline."""

    date: date
    minute_vwap: np.ndarray    # (225,) per-minute market VWAP price, forward-filled
    minute_volume: np.ndarray  # (225,) per-minute volume in lots
    coverage: int              # minutes with actual trades
    half_spread: float = 0.0   # mean (ask1 - bid1) / 2 over the session, price units
    night_volume: float = 0.0  # preceding night-session volume, attributed here
    minute_signed: object = None   # (225,) net signed volume (mid rule), lots
    minute_mid: object = None      # (225,) last mid price of each minute
    minute_buy: object = None      # (225,) gross buy-signed volume, lots
    minute_sell: object = None     # (225,) gross sell-signed volume, lots
    minute_bid_depth: object = None  # (225,) mean bid L1-L5 depth per minute, lots
    minute_ask_depth: object = None  # (225,) mean ask L1-L5 depth per minute, lots
    minute_open: object = None       # (225,) first lastPrice of each minute
    minute_high: object = None       # (225,) max lastPrice of each minute
    minute_low: object = None        # (225,) min lastPrice of each minute
    minute_close: object = None      # (225,) last lastPrice of each minute


def iter_day_sessions(
    path, start: Optional[date] = None, end: Optional[date] = None,
    min_coverage: int = MIN_COVERAGE,
) -> Iterator[TradingDay]:
    """Yield day sessions from one vendor feather file within [start, end]."""
    depth_cols = [f"{side}Volume{level}" for side in ("bid", "ask")
                  for level in range(1, 6)]
    all_columns = ["time", "lastPrice", "volume", "bidPrice1", "askPrice1"] + depth_cols
    table = feather.read_table(path, columns=all_columns, memory_map=True)

    # sum the 10 depth columns into 2 (bid_depth / ask_depth) in Arrow before
    # pandas — avoids materializing 10 int64 columns (~835 MB/year-file)
    from functools import reduce
    bid_sum = reduce(pc.add, [table.column(f"bidVolume{i}") for i in range(1, 6)])
    ask_sum = reduce(pc.add, [table.column(f"askVolume{i}") for i in range(1, 6)])
    table = table.append_column("bid_depth", bid_sum)
    table = table.append_column("ask_depth", ask_sum)
    table = table.drop_columns(depth_cols)

    time_col = table.column("time")
    hours = pc.utf8_slice_codeunits(time_col, start=11, stop=13)
    wanted = set(_DAY_FILE_HOURS) | set(_NIGHT_FILE_HOURS)
    in_session = pc.is_in(hours, value_set=pa.array(sorted(wanted)))
    sub = table.filter(in_session).to_pandas()

    sub = sub.dropna(subset=["lastPrice"])
    sub["datetime"] = pd.to_datetime(sub["time"], format=_TIME_FMT, errors="coerce")
    sub = sub.dropna(subset=["datetime"]).sort_values("datetime")
    sub = sub.drop_duplicates(subset=["datetime"], keep="last")

    # Beijing clock. The vendor volume counter is cumulative across the whole
    # trading session (evening night part through next day's close), so diffs
    # must chain over a SESSION key, not the calendar date: evening rows
    # (hour >= 21) belong to the NEXT trading day's session.
    sub["datetime"] = sub["datetime"] + FILE_TO_BEIJING
    sub["day"] = sub["datetime"].dt.strftime("%Y-%m-%d")
    beijing_hour = sub["datetime"].dt.hour
    next_day = (sub["datetime"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    sub["session"] = np.where(beijing_hour >= 21, next_day, sub["day"])
    sub["minute_of_day"] = sub["datetime"].dt.hour * 60 + sub["datetime"].dt.minute

    sub["vol_diff"] = sub.groupby("session")["volume"].diff().clip(lower=0.0)
    # the first day-session row after the night part carries the whole night
    # total in its cumulative column: zero it there (night volume is booked
    # separately below)
    night_rows = (beijing_hour >= 21) | (beijing_hour <= 1)
    anchor = (~night_rows) & night_rows.shift(fill_value=True)
    sub.loc[anchor, "vol_diff"] = 0.0
    sub = sub[sub["vol_diff"] > 0.0]

    # Night-session volume per file-date (Beijing >= 21:00 of that evening,
    # including the post-midnight hour which is still file-hour <= 20); it
    # belongs to the NEXT trading day.
    # the night session spans Beijing 21:00 of one evening through 01:00 of
    # the next calendar morning. The pre-midnight part carries the evening's
    # date; the post-midnight tail carries the NEXT day's date string, so it
    # is keyed by the morning's date — and a Friday evening's post-midnight
    # tail lands on a Saturday (or Sunday) string that no day session will
    # ever look up, so it must be re-attributed to the FOLLOWING trading
    # day's evening (the one it actually belongs to by session).
    beijing_hour = sub["datetime"].dt.hour
    evening = sub[beijing_hour >= 21]
    postmid = sub[beijing_hour == 0]
    evening_by_day = evening.groupby("day")["vol_diff"].sum().to_dict()
    night_dates = sorted(evening_by_day)
    # re-key each post-midnight group onto the *next evening date* in the
    # night_dates sequence (a Friday-evening tail → next Monday's key is
    # wrong; it belongs to the Friday session, so the volume should join
    # the Friday evening's total — i.e. the same night session). Simplest
    # correct attribution: append each post-midnight group's volume onto
    # the PRECEDING evening date's total (the session it is part of).
    postmid_by_day = {}
    for day_key, group in postmid.groupby("day"):
        prior_evenings = [d for d in night_dates if d < day_key]
        if prior_evenings:
            evening_by_day[prior_evenings[-1]] = evening_by_day.get(prior_evenings[-1], 0.0) + float(
                group["vol_diff"].sum()
            )
        # else: orphaned post-midnight (no preceding evening in this file);
        # drop it rather than mis-attribute

    # v2.2 P0-2: record each evening's LAST traded price (the price known
    # just before the 09:00 open) — night rows are about to be stripped
    night_last_price = {}
    night_all = sub[(beijing_hour >= 21) | (beijing_hour == 0)]
    for evening_date, grp in night_all.groupby("session"):
        if len(grp):
            night_last_price[evening_date] = float(grp["lastPrice"].iloc[-1])

    night_mask = (beijing_hour >= 21) | (beijing_hour == 0)

    sub = sub[~night_mask]
    bid = sub["bidPrice1"].astype(float)
    ask = sub["askPrice1"].astype(float)
    # degenerate books (limit-locked side quoted at 0) poison the mid and the
    # spread: keep only two-sided quotes with a sane spread
    valid_book = (bid > 0.0) & (ask > bid)
    mid = pd.Series(np.where(valid_book, (ask + bid) / 2.0, np.nan), index=sub.index)
    half_spread = pd.Series(
        np.where(valid_book, (ask - bid) / 2.0, np.nan), index=sub.index
    )
    half_spread[half_spread > 0.02 * mid] = np.nan
    sub["half_spread"] = half_spread
    spread_by_day = sub.groupby("day")["half_spread"].median()
    sub["mid"] = mid
    sub["notional"] = sub["lastPrice"].astype(float) * sub["vol_diff"]
    # trade signing (mid rule, Lee-Ready style): +1 when the print is above
    # the PREVIOUS midpoint, -1 below, tick-rule carry when exactly at it
    prev_mid = sub["mid"].shift(1).ffill()
    raw_sign = np.sign(sub["lastPrice"].astype(float) - prev_mid)
    sub["sign"] = raw_sign.where(raw_sign != 0.0).ffill().fillna(0.0)
    sub["signed_volume"] = sub["sign"] * sub["vol_diff"]
    sub["buy_volume"] = sub["vol_diff"].where(sub["sign"] > 0, 0.0)
    sub["sell_volume"] = sub["vol_diff"].where(sub["sign"] < 0, 0.0)

    for day, group in sub.groupby("day"):
        if start is not None and day < start.isoformat():
            continue
        if end is not None and day > end.isoformat():
            continue
        bars = group.groupby("minute_of_day").agg(
            notional=("notional", "sum"), volume=("vol_diff", "sum"),
            signed=("signed_volume", "sum"),
            buy=("buy_volume", "sum"), sell=("sell_volume", "sum"),
            bid_depth=("bid_depth", "mean"), ask_depth=("ask_depth", "mean"),
            open_px=("lastPrice", "first"), high_px=("lastPrice", "max"),
            low_px=("lastPrice", "min"), close_px=("lastPrice", "last"),
        )
        bars = bars[bars.index.isin(_SLOT_OF_MINUTE)]
        if bars.empty:
            continue
        vwap = (bars["notional"] / bars["volume"]).to_numpy()
        slots = np.array([_SLOT_OF_MINUTE[int(m)] for m in bars.index])

        minute_signed = np.zeros(N_SESSION_MINUTES)
        minute_signed[slots] = bars["signed"].to_numpy()
        # minute-end mid price: last snapshot mid within each minute
        minute_mid = np.full(N_SESSION_MINUTES, np.nan)
        mid_by_minute = group.assign(mid_val=mid).groupby("minute_of_day")["mid_val"].last()
        known = mid_by_minute[mid_by_minute.index.isin(_SLOT_OF_MINUTE)]
        minute_mid[[_SLOT_OF_MINUTE[int(m)] for m in known.index]] = known.to_numpy()
        minute_mid[0] = np.nan_to_num(minute_mid[0], nan=float(known.iloc[0]))
        minute_mid = pd.Series(minute_mid).ffill().to_numpy()

        minute_vwap = np.full(N_SESSION_MINUTES, np.nan)
        minute_vwap[slots] = vwap
        # v2.2 P0-2 (actually implementing the v1.6 claim): lead-fill the
        # open from the night session's LAST traded price — known before 09:00,
        # so no look-ahead. Falls back to the first day print only when no
        # night session exists in this file.
        series = pd.Series(minute_vwap).ffill()
        if pd.isna(series.iloc[0]) or slots[0] != 0:
            prior = [d for d in night_dates if d < day]
            if prior:
                minute_vwap[0] = night_last_price.get(day, vwap[0])
            else:
                minute_vwap[0] = vwap[0]
        series = pd.Series(minute_vwap).ffill()
        minute_volume = np.zeros(N_SESSION_MINUTES)
        minute_volume[slots] = bars["volume"].to_numpy()
        coverage = int((minute_volume > 0).sum())
        if coverage < min_coverage:
            continue
        night_volume = 0.0
        prior = [d for d in night_dates if d < day]
        if prior:
            night_volume = float(evening_by_day[prior[-1]])
        yield TradingDay(
            date.fromisoformat(day),
            series.to_numpy(),
            minute_volume,
            coverage,
            float(np.nan_to_num(spread_by_day.get(day, 0.0))),
            night_volume,
            minute_signed,
            minute_mid,
            _slots_array(bars["buy"].to_numpy(), slots),
            _slots_array(bars["sell"].to_numpy(), slots),
            _slots_array(bars["bid_depth"].to_numpy(), slots),
            _slots_array(bars["ask_depth"].to_numpy(), slots),
            _ohlc_array(bars, "first", slots, series.to_numpy()),
            _ohlc_array(bars, "max", slots, series.to_numpy()),
            _ohlc_array(bars, "min", slots, series.to_numpy()),
            _ohlc_array(bars, "last", slots, series.to_numpy()),
        )


def build_volume_profile(days) -> np.ndarray:
    """Average normalized day-session volume density across days (length 225)."""
    total = np.zeros(N_SESSION_MINUTES)
    for day in days:
        total += day.minute_volume
    if total.sum() <= 0:
        raise ValueError("no volume in the supplied days")
    return total / total.sum()


def window_stats(days) -> dict:
    """Per-window market statistics used by the calibration."""
    vols = np.array([day.minute_volume.sum() for day in days])
    prices = np.concatenate([day.minute_vwap for day in days])
    rets = np.concatenate(
        [np.diff(np.log(day.minute_vwap)) for day in days]
    )
    minute_vol = float(np.std(rets))
    return {
        "mean_volume": float(vols.mean()),
        "volume_std_over_mean": float(vols.std() / vols.mean()),
        "mean_price": float(prices.mean()),
        "window_sigma": float(minute_vol * np.sqrt(N_SESSION_MINUTES) * prices.mean()),
        "mean_half_spread": float(np.mean([d.half_spread for d in days])),
    }


def kernel_transpose_mass(kernel: MultiExponential) -> float:
    """I_G = int int_{s<t} G(t-s) ds dt = int_0^1 (1-u) G(u) du."""
    from .discretize import make_grid

    grid = make_grid(601, kernel)
    ones = np.ones(grid.size)
    return float(grid.weights @ (grid.past @ ones))


def calibrate(stats, kernel, participation=0.05, c_imp=1.0):
    """Map real-data statistics onto the solver and the replay backtest.

    Price model with a reference-rate normalization (the deformation must be
    centered on the order's own rate scale or the eps-series loses validity):

        displacement(t) = Lambda * (v(t)/X)^(1-eps),   slippage per share = eta_p * v(t).

    Substituting v = X*u (u sums to 1) reduces the shape problem to the
    canonical solver at share 1 with

        eta_solver = (eta_p * X) / (Lambda * I_G),

    so only this one dimensionless ratio shapes the trajectory. Calibration:

    * level — the square-root law: trading X = participation * V of a window
      with volume V and volatility sigma_w costs c_imp * sigma_w *
      sqrt(participation) per share in total, which fixes Lambda;
    * split — the slippage (linear) share of that total cost is set by the
      realized half-spread versus the impact level, both from the data.

    Returns the solver parameters (share=1, eta for ExecParams) and the
    price-unit scales (Lambda, eta_p) plus the real order size X.
    """
    share = participation * stats["mean_volume"]
    i_g = kernel_transpose_mass(kernel)
    total_per_share = c_imp * stats["window_sigma"] * np.sqrt(participation)
    spread_share = stats["mean_half_spread"] / total_per_share

    # Floor the slippage share at the paper's canonical split. Below it the
    # benchmark give-back dominates the slippage penalty and the unconstrained
    # VWAP optimum enters the benchmark-gaming regime (negative terminal
    # rate), which is not an executable schedule for a buy parent.
    paper_share, paper_eta = 0.1, 0.65
    paper_slippage_share = (
        paper_eta * paper_share**2
        / (paper_eta * paper_share**2 + paper_share**1.5 * i_g)
    )
    slippage_share = float(np.clip(spread_share / 2.0, paper_slippage_share, 0.5))

    # total = slippage + impact for a TWAP reference; with rates ~1 the
    # solver-unit TWAP cost is eta_solver + 1 * I_G, and eta_solver is the
    # slippage/(impact-per-rate) ratio eta_p X / Lambda by the substitution.
    # Total per share = (1 + eta_solver) * Lambda * I_G ... solved directly:
    #   per-share cost = eta_p * X + Lambda * I_G   (TWAP rate v = X)
    # with eta_p X = (s/(1-s)) * Lambda * I_G  =>  Lambda = total * (1-s) / I_G.
    lam = total_per_share * (1.0 - slippage_share) / i_g
    eta_p = slippage_share * total_per_share / share
    eta_solver = eta_p * share / lam
    return {
        "share": float(share),
        "participation": float(participation),
        "eta_solver": float(eta_solver),
        "lambda": float(lam),
        "eta_p": float(eta_p),
        "total_per_share": float(total_per_share),
        "slippage_share": float(slippage_share),
        "half_spread_bps": float(stats["mean_half_spread"] / stats["mean_price"] * 1e4),
        "total_bps": float(total_per_share / stats["mean_price"] * 1e4),
    }


def replay_day(
    quantities: np.ndarray,
    day: TradingDay,
    lam: float,
    eta_p: float,
    kernel: MultiExponential,
    eps: float = 0.5,
    volume_scale: float = 1.0,
) -> dict:
    """Execute a per-minute quantity schedule against one real day session.

    Fills happen at the real per-minute market VWAP plus our own simulated
    transient impact (displacement Lambda * (v/X)^(1-eps) propagated by the
    kernel, causal across minutes) and linear slippage.

    `volume_scale` multiplies the impact scale Lambda only: on a day whose
    volume runs at s_vol times the train average, the square-root law says
    impact scales like 1/sqrt(s_vol). Default 1.0 keeps the v0.1/v0.2
    (window-level calibration) behaviour unchanged.

    Benchmarks, model-consistent with the algorithms' objective: arrival =
    first minute VWAP; day VWAP = the realized volume-weighted price WITH our
    impact residual priced into the tape (our trades do move the benchmark at
    the participation we assume). Returns slippage in bps.
    """
    n = N_SESSION_MINUTES
    h = 1.0 / n
    quantities = np.asarray(quantities, dtype=float)
    share = quantities.sum()
    if share <= 0:
        raise ValueError("schedule must trade a positive quantity")
    rate = quantities / h                       # lots per normalized time (integral = share)
    relative = rate / share                     # v/X, O(1) for TWAP-like schedules

    centers = (np.arange(n) + 0.5) * h
    lag = centers[:, None] - centers[None, :]
    decay = np.tril(kernel(np.abs(lag)), k=-1)
    residual = lam * volume_scale * (decay @ (relative ** (1.0 - eps))) * h
    fills = day.minute_vwap + residual + eta_p * rate

    cost = float((fills * quantities).sum() / share)
    arrival = float(day.minute_vwap[0])
    market_volume = day.minute_volume.sum()
    vwap = float(((day.minute_vwap + residual) * day.minute_volume).sum() / market_volume)
    return {
        "is_bps": (cost - arrival) / arrival * 1e4,
        "vwap_bps": (cost - vwap) / vwap * 1e4,
        "arrival": arrival,
        "vwap": vwap,
    }


def day_volume_density(day: TradingDay) -> np.ndarray:
    """The realized volume density of one day (for oracle strategies)."""
    return day.minute_volume / day.minute_volume.sum()


def solver_risk_scale(stats, calib) -> float:
    """Convert a real risk aversion into solver units.

    The real objective's risk term is lambda_real * sigma_w^2 * X^2 *
    int (1 - C_u)^2 dt; dividing by the impact scale X * Lambda gives the
    solver-unit coefficient lambda_real * sigma_w^2 * share / Lambda. A
    frontier scan multiplies this scale by O(0.1..10) coefficients.
    """
    return float(stats["window_sigma"] ** 2 * calib["share"] / calib["lambda"])
