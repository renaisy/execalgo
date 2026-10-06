"""Tick-level execution simulator with L1 queue physics (v2.4; v2.5 adds
sell-side symmetry via LadderPolicy.side).

Routes a per-minute quantity schedule through the ~0.5 s book snapshots of
one day (bookdata.BookTicks), replacing the minute-bar maker/taker split of
execution_model.py with explicit micro-structure:

* queue position — a passive order joins the tail of the same-side L1
  queue; the queue ahead is consumed by opposite-signed aggressor volume
  (tick rule) while the best quote stays at our price, and clamped down by
  observed queue shrinkage (cancellations ahead of us);
* trade-through — a print strictly through our limit (or the same-side best
  quote moving beyond it) fills the full remaining size at our price;
* front-of-queue partial fills — once the queue ahead is exhausted, later
  aggressor prints at our level fill our remaining size directly;
* book walking — the taker remainder crosses level by level using the real
  L1-L5 prices and volumes of the opposite side (execution_model charges
  exactly one half-spread regardless of size);
* ladder escalation — a resting order times out after `passive_timeout`
  seconds and the remainder crosses the spread;
* adverse-selection guards — a microprice gate blocks placement when the
  depth-weighted fair price sits on the away side (falls-into-us regime),
  and a cancel guard pulls the order when the microprice pins at the side
  that would fill us into an adverse move.

Conventions follow the Model-LightGBM reference simulator (queue-depletion
fill model, strict trade-through, microprice/OBI guards); improvements are
the FIFO partial fills, the cancellation clamp, multi-level walking, and
both-direction symmetry.

Limitations (documented, deliberate): our own orders do not perturb the
observed book (fine at 5% participation where a minute's slice ~16 lots is
below the mean L1 queue ~25 lots), and market impact is not re-priced here —
the tick simulator measures the micro layer only; replay_day() remains the
impact-aware cost model.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .bookdata import BookTicks

ADVERSE_HORIZON_MIN = 5.0  # drift measurement window after each fill


@dataclass
class Fill:
    price: float
    qty: float
    t: float            # minutes on the 225-minute execution timeline
    venue: str          # "passive" | "taker"
    reason: str         # queue | through | start | escalate | deadline
    slot: int


@dataclass
class MinuteResult:
    slot: int
    want: float
    passive_qty: float = 0.0
    taker_qty: float = 0.0
    shortfall: float = 0.0
    fills: List[Fill] = field(default_factory=list)
    gate_skips: int = 0
    guard_cancels: int = 0
    requotes: int = 0
    escalations: int = 0


@dataclass
class DayResult:
    fills: List[Fill]
    minutes: List[MinuteResult]
    want_total: float
    passive_qty: float
    taker_qty: float
    shortfall: float
    avg_price: float
    gate_skips: int
    guard_cancels: int
    requotes: int
    escalations: int

    @property
    def passive_fraction(self) -> float:
        return self.passive_qty / self.want_total if self.want_total > 0 else 0.0

    def adverse_drift_bps(self, ticks: BookTicks, venue: str,
                          side: int = +1) -> float:
        """Mean post-fill 5-min mid drift in bps over `venue` fills,
        normalized to the order's favor direction: negative = the mid moved
        against us after the fill (adverse selection) for EITHER side.
        """
        mids = ticks.mid
        drifts = []
        for f in self.fills:
            if f.venue != venue:
                continue
            j = int(np.searchsorted(ticks.t, f.t + ADVERSE_HORIZON_MIN))
            j = min(j, len(mids) - 1)
            drifts.append(side * (mids[j] - f.price) / f.price * 1e4)
        return float(np.mean(drifts)) if drifts else float("nan")


@dataclass
class LadderPolicy:
    """Micro-execution policy for one parent schedule.

    mode:
      "taker"            — cross immediately at minute start, walk the book
      "passive_fallback" — rest at the bid for the whole minute, remainder
                           walks the book at minute end (the tick-granularity
                           analogue of the v1.2 minute-bar maker/taker split)
      "ladder"           — gate (optional) → rest → cancel guard (optional)
                           → timeout escalation

    Gate and guard thresholds are in units of the HALF-SPREAD because the
    microprice deviation (micro - mid) is confined to [-s/2, +s/2]; fractions
    transfer across contracts with different tick sizes. The microprice
    fraction is micro_frac = (micro - mid) / (spread / 2) in [-1, 1]:
    positive = bid queue heavier = mildly favorable for a resting buy.

    A blocked gate waits (re-checking) until the minute ends rather than
    escalating at the timeout: paying the spread because the book is
    momentarily adverse is exactly the behaviour the gate exists to avoid.
    The guard cancels only when the microprice is pinned nearly at the bid
    (guard_frac near 1), the falling-knife regime — mirroring the reference
    architecture's OBI-swing guard but in spread-normalized units.
    """

    mode: str = "ladder"
    side: int = +1                  # +1 buy parent, -1 sell parent (v2.5)
    passive_timeout: float = 30.0   # seconds resting before escalation
    gate: Optional[str] = None      # None | "microprice"
    gate_frac: float = 0.0          # post when micro_frac >= gate_frac (buys;
                                    # mirrored: <= -gate_frac for sells)
    gate_retry: float = 5.0         # seconds between gate re-checks while blocked
    cancel_guard: bool = False
    guard_frac: float = 0.8         # microprice pinned at the adverse side
    max_requotes: int = 1           # re-posts after a guard cancel


def _walk_book(ticks: BookTicks, j: int, qty: float, t: float, slot: int,
               reason: str, fills: List[Fill], side: int = +1) -> None:
    """Cross the spread level by level for `qty` lots at snapshot j: a buy
    lifts the ask, a sell hits the bid."""
    pxs = ticks.ask_pxs[j] if side > 0 else ticks.bid_pxs[j]
    vols = ticks.ask_vols[j] if side > 0 else ticks.bid_vols[j]
    spread = float(ticks.ask1[j] - ticks.bid1[j])
    left = qty
    for lvl in range(pxs.size):
        take = min(left, float(vols[lvl]))
        if take > 0 and pxs[lvl] > 0:
            fills.append(Fill(float(pxs[lvl]), take, t, "taker", reason, slot))
            left -= take
        if left <= 1e-9:
            return
    # beyond L5: penalized completion at the worst seen level plus one spread
    ref = float(ticks.ask1[j]) if side > 0 else float(ticks.bid1[j])
    worst = float(pxs[-1]) if pxs[-1] > 0 else ref
    fills.append(Fill(worst + side * spread, left, t, "taker",
                      reason + "_l5", slot))


def _micro_frac(ticks: BookTicks, j: int) -> float:
    """Microprice deviation in half-spread units, in [-1, 1]."""
    total = ticks.bid_vol1[j] + ticks.ask_vol1[j]
    if total <= 0:
        return 0.0
    micro = (ticks.bid1[j] * ticks.ask_vol1[j]
             + ticks.ask1[j] * ticks.bid_vol1[j]) / total
    half = (ticks.ask1[j] - ticks.bid1[j]) / 2.0
    if half <= 0:
        return 0.0
    return float(np.clip((micro - ticks.mid[j]) / half, -1.0, 1.0))


def _gate_ok(ticks: BookTicks, j: int, policy: LadderPolicy) -> bool:
    """Directional microprice gate: a resting BUY wants the bid queue at
    least as heavy as the ask (micro_frac >= gate_frac); a resting SELL
    wants the mirror image (micro_frac <= -gate_frac)."""
    frac = _micro_frac(ticks, j)
    if policy.side > 0:
        return frac >= policy.gate_frac
    return frac <= -policy.gate_frac


def _guard_adverse(ticks: BookTicks, j: int, policy: LadderPolicy) -> bool:
    """True when the microprice is pinned at the side that would fill us
    into an adverse move (the bid for resting buys, the ask for sells)."""
    frac = _micro_frac(ticks, j)
    if policy.side > 0:
        return frac < -policy.guard_frac
    return frac > policy.guard_frac


def _rest_fill(ticks: BookTicks, i1: int, place_j: int, qty: float,
               deadline_t: float, policy: LadderPolicy, slot: int,
               res: MinuteResult) -> None:
    """Run one resting order from snapshot place_j until filled, canceled,
    or the deadline; escalate any remainder across the spread.

    A buy rests at the bid and is consumed by sell-signed prints; a sell
    rests at the ask and is consumed by buy-signed prints. The deadline
    stays anchored at the ORIGINAL placement + timeout across requotes, so
    a guard-cancelled-and-reposted order cannot extend its passive window.
    """
    side = policy.side
    if side > 0:
        level_px, level_vol = ticks.bid1, ticks.bid_vol1
        aggressor_sign = -1.0
    else:
        level_px, level_vol = ticks.ask1, ticks.ask_vol1
        aggressor_sign = +1.0

    limit = float(level_px[place_j])
    if (limit <= 0) or (side > 0 and ticks.ask1[place_j] <= 0):
        _walk_book(ticks, place_j, qty, ticks.t[place_j], slot,
                   "escalate", res.fills, side)
        res.escalations += 1
        return
    pos = float(level_vol[place_j])  # queue ahead of us
    remaining = qty
    attempts = 0
    resting = True

    j = place_j + 1
    while j < i1 and remaining > 1e-9:
        t_j = ticks.t[j]
        if t_j >= deadline_t:
            break
        if not resting:
            # canceled by the guard: wait for the microprice to recover,
            # then re-post with a fresh queue (within the re-quote budget)
            if attempts < policy.max_requotes and _gate_ok(ticks, j, policy):
                attempts += 1
                res.requotes += 1
                resting = True
                limit = float(level_px[j])
                pos = float(level_vol[j])
            j += 1
            continue
        # cancel guard: microprice pinned at the adverse side (falling knife
        # for buys, rising knife for sells)
        if policy.cancel_guard and _guard_adverse(ticks, j, policy):
            res.guard_cancels += 1
            resting = False
            j += 1
            continue
        # trade-through: price printed or quoted strictly THROUGH our limit
        through = (ticks.last[j] < limit - 1e-9 or level_px[j] < limit - 1e-9) \
            if side > 0 else (ticks.last[j] > limit + 1e-9
                              or level_px[j] > limit + 1e-9)
        if through:
            res.fills.append(Fill(limit, remaining, t_j, "passive", "through", slot))
            return
        if abs(level_px[j] - limit) < 1e-9:
            if ticks.sign[j] == aggressor_sign and ticks.vol_inc[j] > 0:
                aggr = float(ticks.vol_inc[j])
                if pos > 0:
                    hit = min(pos, aggr)
                    pos -= hit
                    front_fill = aggr - hit  # overshoot fills us at the front
                else:
                    front_fill = aggr
                if front_fill > 0:
                    take = min(remaining, front_fill)
                    res.fills.append(Fill(limit, take, t_j, "passive", "queue", slot))
                    remaining -= take
            # observed queue shrank without trades -> cancellations ahead
            pos = min(pos, float(level_vol[j]))
        j += 1

    if remaining > 1e-9:
        esc_j = min(j, i1 - 1)
        _walk_book(ticks, esc_j, remaining, ticks.t[esc_j], slot,
                   "escalate", res.fills, side)
        res.escalations += 1


def _simulate_minute(ticks: BookTicks, i0: int, i1: int, want: float, slot: int,
                     policy: LadderPolicy) -> MinuteResult:
    res = MinuteResult(slot=slot, want=want)

    if policy.mode == "taker":
        _walk_book(ticks, i0, want, ticks.t[i0], slot, "start", res.fills,
                   policy.side)
        return res

    timeout = 60.0 if policy.mode == "passive_fallback" else policy.passive_timeout

    # choose the placement snapshot (gate may delay or block posting)
    place_j = i0
    if policy.gate == "microprice":
        j = i0
        while j < i1:
            if _gate_ok(ticks, j, policy):
                break
            res.gate_skips += 1
            j = int(np.searchsorted(ticks.t, ticks.t[j] + policy.gate_retry / 60.0))
        place_j = j
        if place_j >= i1:
            # gate never opened within the minute: taker completion at the end
            # (patient — the point of the gate is to avoid paying the spread
            # into momentary adversity, not to force it)
            _walk_book(ticks, i1 - 1, want, ticks.t[i1 - 1], slot,
                       "escalate", res.fills, policy.side)
            res.escalations += 1
            return res

    deadline_t = ticks.t[place_j] + timeout / 60.0
    _rest_fill(ticks, i1, place_j, want, deadline_t, policy, slot, res)
    return res


def simulate_day(ticks: BookTicks, quantities, policy: LadderPolicy) -> DayResult:
    """Execute a (225,) per-minute lot schedule against one day of ticks.

    Minutes without snapshots roll their quantity forward (the tick analogue
    of the minute model's T20 rollforward); any day-end remainder is forced
    through the book at the last snapshot. Every minute completes, so the
    day's shortfall is zero unless the book itself is degenerate at the end.
    """
    quantities = np.asarray(quantities, dtype=float)
    bounds = ticks.minute_bounds()
    pending = 0.0
    minutes: List[MinuteResult] = []

    for slot in range(len(quantities)):
        want = quantities[slot]
        i0, i1 = bounds[slot], bounds[slot + 1]
        if want <= 1e-9 and pending <= 1e-9:
            continue
        if i1 <= i0:
            pending += want  # untradable minute: roll forward
            continue
        res = _simulate_minute(ticks, i0, i1, want + pending, slot, policy)
        pending = res.shortfall = _shortfall(res, want + pending)
        minutes.append(res)

    if pending > 1e-9:
        j = len(ticks.t) - 1
        res = MinuteResult(slot=len(quantities) - 1, want=pending)
        _walk_book(ticks, j, pending, ticks.t[j], res.slot, "deadline",
                   res.fills, policy.side)
        res.shortfall = _shortfall(res, pending)
        minutes.append(res)

    fills = [f for m in minutes for f in m.fills]
    want_total = float(quantities.sum())
    passive_qty = sum(f.qty for f in fills if f.venue == "passive")
    taker_qty = sum(f.qty for f in fills if f.venue == "taker")
    notional = sum(f.price * f.qty for f in fills)
    filled_qty = passive_qty + taker_qty
    return DayResult(
        fills=fills,
        minutes=minutes,
        want_total=want_total,
        passive_qty=passive_qty,
        taker_qty=taker_qty,
        shortfall=want_total - filled_qty,
        avg_price=notional / filled_qty if filled_qty > 0 else float("nan"),
        gate_skips=sum(m.gate_skips for m in minutes),
        guard_cancels=sum(m.guard_cancels for m in minutes),
        requotes=sum(m.requotes for m in minutes),
        escalations=sum(m.escalations for m in minutes),
    )


def _shortfall(res: MinuteResult, want: float) -> float:
    return max(0.0, want - sum(f.qty for f in res.fills))
