"""Rolling replan with tick-level micro execution (v2.6).

Bridges the RollingPlanner (trajectory layer: participation caps, impact
carry, revision damping, incremental replans) to the tick simulator
(micro layer: queue position, escalation, book walking).

`execute_with_book` runs the planner's replan loop while routing each
minute's quantity through the real ~0.5 s book instead of the minute-bar
cost model:

* at each replan the planner sees the realized volume only up to that
  minute (the same causality `execute()` relies on — step() reads
  minute_volume[:start] only);
* the interval's per-minute quantities execute via the tick simulator's
  per-minute policy; untradable minutes roll forward; every minute
  completes, so the trajectory's quantity completion matches `execute()`;
* fills are recorded at real book prices — the accounting the minute
  replay model approximates with minute VWAP +/- half spread.

Impact is NOT re-priced here (the tick simulator does not move the tape);
for cost attribution use the arm-vs-arm differences on the same trajectory,
where impact is common-mode.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .bookdata import BookTicks
from .replanner import N_MINUTES, RollingPlanner
from .tick_sim import Fill, LadderPolicy, MinuteResult, _simulate_minute, _walk_book


@dataclass
class BookExecution:
    """Result of one day executed planner-and-book."""

    fills: List[Fill]
    minutes: List[MinuteResult]
    fractions: np.ndarray          # (225,) trajectory the planner produced
    want_total: float
    passive_qty: float
    taker_qty: float
    shortfall: float
    avg_price: float
    replans: int
    fallbacks: int

    @property
    def passive_fraction(self) -> float:
        return self.passive_qty / self.want_total if self.want_total > 0 else 0.0

    def is_bps(self, ticks: BookTicks, side: int = +1) -> float:
        arrival = float(ticks.mid[0])
        return side * (self.avg_price - arrival) / arrival * 1e4

    def vwap_bps(self, ticks: BookTicks, side: int = +1) -> float:
        tape = float((ticks.last * ticks.vol_inc).sum() / ticks.vol_inc.sum())
        return side * (self.avg_price - tape) / tape * 1e4


def execute_with_book(planner: RollingPlanner, ticks: BookTicks, share: float,
                      policy: LadderPolicy, today=None,
                      night_volume: Optional[float] = None) -> BookExecution:
    """Run the planner's replan loop with tick-level fills for one day.

    `share` is the parent size in lots; `planner` should be fresh (its
    `execute()` is never called internally). `today`/`night_volume` prime
    the forecaster on the first replan, exactly as `execute()` does.
    """
    if share <= 0:
        raise ValueError("share must be positive")
    bounds = ticks.minute_bounds()
    # realized volume per minute, revealed to the planner only as it passes
    realized = np.array([ticks.vol_inc[bounds[k]:bounds[k + 1]].sum()
                         for k in range(N_MINUTES)])
    minute_volume = np.zeros(N_MINUTES)

    fractions = np.zeros(N_MINUTES)
    filled_frac = 0.0            # planner bookkeeping (fraction of parent)
    pending = None               # planner's damping anchor
    pending_qty = 0.0            # unfilled lots rolling to the next minute
    fills: List[Fill] = []
    minutes: List[MinuteResult] = []
    done = False

    planner._today = today
    if night_volume is not None and planner.forecaster is not None:
        planner.forecaster.prime(night_volume=night_volume)

    start = 0
    while start < N_MINUTES and not done:
        state = planner.step(minute_volume, start, filled_frac, fractions,
                             pending)
        fractions = state["fractions"]
        pending = state["pending"]
        take = min(N_MINUTES - start, planner.config.interval_minutes)
        for k in range(start, start + take):
            minute_volume[k] = realized[k]  # reveal AFTER the replan that
            # ends before k; for minutes inside the current interval the
            # planner has already committed, so this ordering is causal
            want = fractions[k] * share + pending_qty
            i0, i1 = bounds[k], bounds[k + 1]
            if want <= 1e-9:
                continue
            if i1 <= i0:
                pending_qty = want  # untradable minute: roll forward
                continue
            res = _simulate_minute(ticks, i0, i1, want, k, policy)
            fills.extend(res.fills)
            minutes.append(res)
            pending_qty = max(0.0, want - sum(f.qty for f in res.fills))
        filled_frac = float(fractions[: start + take].sum())
        if state["done"]:
            done = True
        start += take

    if pending_qty > 1e-9:
        # day-end completion burst at the last snapshot (forced deadline)
        j = len(ticks.t) - 1
        res = MinuteResult(slot=N_MINUTES - 1, want=pending_qty)
        _walk_book(ticks, j, pending_qty, ticks.t[j], res.slot, "deadline",
                   res.fills, policy.side)
        fills.extend(res.fills)
        minutes.append(res)
        pending_qty = 0.0

    filled_qty = sum(f.qty for f in fills)
    passive_qty = sum(f.qty for f in fills if f.venue == "passive")
    notional = sum(f.price * f.qty for f in fills)
    return BookExecution(
        fills=fills,
        minutes=minutes,
        fractions=fractions,
        want_total=share,
        passive_qty=passive_qty,
        taker_qty=filled_qty - passive_qty,
        shortfall=share - filled_qty,
        avg_price=notional / filled_qty if filled_qty > 0 else float("nan"),
        replans=planner.replans,
        fallbacks=planner.fallbacks,
    )
