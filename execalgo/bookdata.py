"""Tick-level L5 order-book snapshots for one day session (v2.4).

Loads the ~0.5 s vendor snapshots for a single trading day at full L5
resolution — the granularity the minute-bar pipeline (realdata.py) aggregates
away. The tick simulator (tick_sim.py) consumes these arrays to model queue
position, trade-through fills, book walking, and adverse-selection gates.

Session conventions are shared with realdata.py: file time = Beijing - 5 h
(day session = file hours 04..09 plus the ~03:59 pre-open anchor row), volume
is cumulative within the trading day so per-snapshot increments are positive
diffs, and the 225 execution minutes exclude the scheduled breaks.

Snapshot fields:
    t        minutes on the 225-minute execution timeline, [0, 225)
    slot     int minute slot 0..224
    last     last traded price
    bid1/ask1 best quotes; bid_vol1/ask_vol1 the L1 queues
    bid_vols/ask_vols   (n, 5) resting volume per level
    bid_pxs/ask_pxs     (n, 5) prices per level (taker walks these)
    vol_inc  traded volume increment attributed to this snapshot
    sign     tick-rule sign of the print (+1 buy / -1 sell / 0 no trade)
"""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .realdata import FILE_TO_BEIJING, N_SESSION_MINUTES, SESSION_MINUTES, _SLOT_OF_MINUTE, _TIME_FMT

_MINUTES_SET = set(int(m) for m in SESSION_MINUTES)


@dataclass
class BookTicks:
    """One day session of ~0.5 s L5 book snapshots on the execution timeline."""

    date: date
    t: np.ndarray            # (n,) minutes on the 225-minute timeline
    slot: np.ndarray         # (n,) int minute slot
    last: np.ndarray         # (n,) last traded price
    bid1: np.ndarray         # (n,) best bid price
    ask1: np.ndarray         # (n,) best ask price
    bid_vol1: np.ndarray     # (n,) resting volume at the best bid (the queue)
    ask_vol1: np.ndarray
    bid_vols: np.ndarray     # (n, 5) resting volume per bid level
    ask_vols: np.ndarray     # (n, 5) resting volume per ask level
    bid_pxs: np.ndarray      # (n, 5) price per bid level
    ask_pxs: np.ndarray      # (n, 5) price per ask level
    vol_inc: np.ndarray      # (n,) volume increment at this snapshot
    sign: np.ndarray         # (n,) tick-rule sign of the driving print

    @property
    def mid(self) -> np.ndarray:
        return (self.bid1 + self.ask1) / 2.0

    @property
    def spread(self) -> np.ndarray:
        return self.ask1 - self.bid1

    @property
    def microprice(self) -> np.ndarray:
        """Depth-weighted fair price; > mid when the bid queue is heavier."""
        total = self.bid_vol1 + self.ask_vol1
        safe = np.where(total > 0, total, 1.0)
        return (self.bid1 * self.ask_vol1 + self.ask1 * self.bid_vol1) / safe

    def minute_bounds(self) -> np.ndarray:
        """(226,) snapshot index boundaries of each execution minute."""
        return np.searchsorted(self.slot, np.arange(N_SESSION_MINUTES + 1))


def load_day_ticks(path, day, min_snapshots: int = 5000) -> BookTicks:
    """Load one trading day's day-session snapshots at native resolution.

    `day` may be a datetime.date or an ISO string. Raises ValueError when the
    day has too few snapshots (halt / missing file section), mirroring the
    minute loader's coverage guard.
    """
    if isinstance(day, str):
        day = date.fromisoformat(day)
    day_str = day.isoformat()

    px_cols = [f"{side}Price{lvl}" for side in ("bid", "ask") for lvl in range(1, 6)]
    vol_cols = [f"{side}Volume{lvl}" for side in ("bid", "ask") for lvl in range(1, 6)]
    table = feather.read_table(
        path, columns=["time", "lastPrice", "volume"] + px_cols + vol_cols,
        memory_map=True,
    )

    time_col = table.column("time")
    day_prefix = pc.utf8_slice_codeunits(time_col, start=0, stop=10)
    hours = pc.utf8_slice_codeunits(time_col, start=11, stop=13)
    # file hours 03..09 = Beijing 08..14 (03 anchors the cumulative-volume diff)
    wanted_hours = pa.array(["03", "04", "05", "06", "07", "08", "09"])
    keep = pc.and_(pc.equal(day_prefix, pa.scalar(day_str)),
                   pc.is_in(hours, value_set=wanted_hours))
    sub = table.filter(keep).to_pandas()

    sub = sub.dropna(subset=["lastPrice"])
    if sub.empty:
        raise ValueError(f"no snapshots for {day_str} in {path}")
    sub["datetime"] = pd.to_datetime(sub["time"], format=_TIME_FMT, errors="coerce")
    sub = sub.dropna(subset=["datetime"]).sort_values("datetime")
    sub = sub.drop_duplicates(subset=["datetime"], keep="last")

    sub["datetime"] = sub["datetime"] + FILE_TO_BEIJING
    sub["minute_of_day"] = sub["datetime"].dt.hour * 60 + sub["datetime"].dt.minute

    # diff BEFORE the minute filter: the ~03:59:59.5 pre-open anchor (minute
    # 539) is excluded from the session but anchors the cumulative-volume
    # diff so the open-auction volume is not lost with the first row; the
    # print-price tick rule uses the same anchor chain
    sub["vol_inc"] = sub["volume"].diff().clip(lower=0.0)
    sub.loc[sub.index[0], "vol_inc"] = 0.0  # pre-open anchor carries no trade
    raw_sign = np.sign(sub["lastPrice"].astype(float).diff())

    sub = sub[sub["minute_of_day"].isin(_MINUTES_SET)]

    sub = sub[sub["minute_of_day"].isin(_MINUTES_SET)]
    if len(sub) < min_snapshots:
        raise ValueError(
            f"{day_str}: only {len(sub)} in-session snapshots (min {min_snapshots})"
        )

    # degenerate books (locked/one-sided quotes) poison prices and queues:
    # forward-fill the last sane book over them rather than dropping trades
    bid1 = sub["bidPrice1"].astype(float)
    ask1 = sub["askPrice1"].astype(float)
    valid = (bid1 > 0.0) & (ask1 > bid1)
    for col in px_cols + vol_cols:
        sub.loc[~valid, col] = np.nan
        sub[col] = sub[col].ffill()

    # tick rule on consecutive print prices; carry the sign through flat
    # prints and zero out non-trade snapshots
    sub["sign"] = raw_sign.where(raw_sign != 0.0).ffill().fillna(0.0)
    sub.loc[sub["vol_inc"] <= 0.0, "sign"] = 0.0

    last = sub["lastPrice"].astype(float)
    within = (sub["datetime"].dt.second + sub["datetime"].dt.microsecond / 1e6) / 60.0
    slot = np.array([_SLOT_OF_MINUTE[int(m)] for m in sub["minute_of_day"]])
    t = slot + within.to_numpy(dtype=float)

    def levels(names):
        return np.column_stack([sub[c].to_numpy(dtype=np.float64) for c in names])

    return BookTicks(
        date=day,
        t=t,
        slot=slot,
        last=last.to_numpy(),
        bid1=sub["bidPrice1"].to_numpy(dtype=float),
        ask1=sub["askPrice1"].to_numpy(dtype=float),
        bid_vol1=sub["bidVolume1"].to_numpy(dtype=float),
        ask_vol1=sub["askVolume1"].to_numpy(dtype=float),
        bid_vols=levels(vol_cols[:5]),
        ask_vols=levels(vol_cols[5:]),
        bid_pxs=levels(px_cols[:5]),
        ask_pxs=levels(px_cols[5:]),
        vol_inc=sub["vol_inc"].to_numpy(dtype=float),
        sign=sub["sign"].to_numpy(dtype=float),
    )


def minute_book_stats(ticks: BookTicks) -> dict:
    """Per-minute aggregates for cross-checking the tick view against the
    minute-bar pipeline (sell flow, L1 depth, median spread)."""
    bounds = ticks.minute_bounds()
    sell = np.where(ticks.sign < 0, ticks.vol_inc, 0.0)
    out = {
        "sell_flow": np.zeros(N_SESSION_MINUTES),
        "bid_l1": np.zeros(N_SESSION_MINUTES),
        "ask_l1": np.zeros(N_SESSION_MINUTES),
        "half_spread": np.full(N_SESSION_MINUTES, np.nan),
        "micro_bias": np.full(N_SESSION_MINUTES, np.nan),
    }
    micro = ticks.microprice
    for k in range(N_SESSION_MINUTES):
        i0, i1 = bounds[k], bounds[k + 1]
        if i1 <= i0:
            continue
        out["sell_flow"][k] = sell[i0:i1].sum()
        out["bid_l1"][k] = ticks.bid_vol1[i0:i1].mean()
        out["ask_l1"][k] = ticks.ask_vol1[i0:i1].mean()
        out["half_spread"][k] = float(np.median(ticks.spread[i0:i1]) / 2.0)
        out["micro_bias"][k] = float(np.median(micro[i0:i1] - ticks.mid[i0:i1]))
    return out
