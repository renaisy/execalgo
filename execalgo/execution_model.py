"""Maker/taker execution against the real book (v1.0).

Replaces the linear slippage eta * v of the replay with an L5-consistent
fill model:

* each minute's slice splits into a passive part posted at the bid and a
  taker part crossing the spread;
* the passive part fills against the minute's opposite-side gross flow,
  scaled by a queue-capture ratio beta (we are not at the front of the
  queue), the unfilled remainder executes as taker in the same minute;
* the opening-auction minute is taker-only (no book to queue in before the
  open print) and quantities in untradable minutes roll forward.

The transient impact residual (propagator) is unchanged: it prices our
footprint in the tape, while the spread terms are private execution costs.
Reported metrics include the realized passive fraction and the spread
capture in bp, which T19 checks against experience bands.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import MultiExponential
from .realdata import N_SESSION_MINUTES, TradingDay


@dataclass(frozen=True)
class MakerPolicy:
    passive_fraction: object = 0.5  # pi: share posted passively; "auto" = no
                                    # policy cap, the calibrated fill curve is
                                    # the only limit (v2.4)
    queue_capture: float = 0.7       # beta: fallback constant (no fill_curve)
    auction_taker_only: bool = True  # T20: no passive in the opening minute
    fill_curve: object = None        # v1.2: calibrated FillCurve overrides beta
    spread_mode: str = "day"         # "day" | "minute": per-minute median
                                     # half spread instead of one daily number


def replay_day_book(
    quantities: np.ndarray,
    day: TradingDay,
    lam: float,
    kernel: MultiExponential,
    policy: MakerPolicy = MakerPolicy(),
    eps: float = 0.5,
    volume_scale: float = 1.0,
) -> dict:
    """Execute a buy schedule with passive/taker splitting on one real day.

    Fills: passive at mid - half_spread, taker at mid + half_spread, both on
    top of the impact residual. Benchmarks follow replay_day's convention.
    """
    n = N_SESSION_MINUTES
    h = 1.0 / n
    quantities = np.asarray(quantities, dtype=float)
    share = quantities.sum()
    if share <= 0:
        raise ValueError("schedule must trade a positive quantity")
    rate = quantities / h
    relative = rate / share

    centers = (np.arange(n) + 0.5) * h
    lag = centers[:, None] - centers[None, :]
    decay = np.tril(kernel(np.abs(lag)), k=-1)
    residual = lam * volume_scale * (decay @ (relative ** (1.0 - eps))) * h

    sell_flow = (day.minute_sell if day.minute_sell is not None
                 else np.zeros(n))
    bid_depth = day.minute_bid_depth  # None for synthetic days without depth
    half_spread = float(day.half_spread)
    # v2.4: per-minute spread when the day carries it (falls back to the
    # session constant on synthetic / legacy days)
    if policy.spread_mode == "minute" and day.minute_half_spread is not None:
        hs = np.asarray(day.minute_half_spread, dtype=float)
        minute_half_spread = np.where(hs > 0, hs, half_spread)
    else:
        minute_half_spread = np.full(n, half_spread)

    pi_cap = None if policy.passive_fraction == "auto" \
        else float(policy.passive_fraction)

    pending = 0.0                     # rolled-forward quantity (T20)
    passive_filled_total = 0.0
    taker_total = 0.0
    cost_notional = 0.0
    executed = 0.0
    spread_notional = 0.0
    last_tradable = -1
    for k in range(n):
        want = quantities[k] + pending
        tradable = day.minute_volume[k] > 0 or k == 0
        if not tradable:
            pending = want            # defer into the next tradable minute
            continue
        last_tradable = k
        pending = 0.0
        passive_cap = 0.0
        if not (policy.auction_taker_only and k == 0):
            if policy.fill_curve is not None and bid_depth is not None:
                # v1.2: calibrated curve — fill depends on flow AND depth
                from .fill_calibration import fill_at
                cap = fill_at(policy.fill_curve, sell_flow[k], bid_depth[k])
                passive_cap = cap if pi_cap is None else min(pi_cap * want, cap)
            else:
                # v1.0 fallback: constant queue capture ratio
                cap = policy.queue_capture * sell_flow[k]
                passive_cap = cap if pi_cap is None else min(pi_cap * want, cap)
        passive = passive_cap
        taker = want - passive
        if taker < 0.0:               # flow alone over-filled the slice
            passive, taker = want, 0.0
        mid = day.minute_vwap[k] + residual[k]
        hs_k = minute_half_spread[k]
        # deferred lots are priced at THIS minute's blended fill, where they
        # actually execute
        cost_notional += (mid - hs_k) * passive + (mid + hs_k) * taker
        spread_notional += (passive - taker) * hs_k
        passive_filled_total += passive
        taker_total += taker
        executed += want

    if pending > 0.0 and last_tradable >= 0:
        # the day ends on untradable minutes: force the tail into the last
        # tradable minute as a taker burst (the realistic forced completion)
        mid = day.minute_vwap[last_tradable] + residual[last_tradable]
        hs_k = minute_half_spread[last_tradable]
        cost_notional += (mid + hs_k) * pending
        spread_notional -= hs_k * pending
        taker_total += pending
        executed += pending
        pending = 0.0

    if executed <= 0.0:
        raise ValueError("no tradable minute in the session")
    cost = float(cost_notional / executed)
    arrival = float(day.minute_vwap[0])
    market_volume = day.minute_volume.sum()
    # same benchmark convention as replay_day: our impact residual moves the
    # tape the VWAP is computed from
    vwap = float(((day.minute_vwap + residual) * day.minute_volume).sum()
                 / market_volume)

    return {
        "is_bps": (cost - arrival) / arrival * 1e4,
        "vwap_bps": (cost - vwap) / vwap * 1e4,
        "passive_fraction": float(passive_filled_total / executed),
        # spread captured net of spread paid, at the minute spreads actually
        # faced (v2.4: was (passive - taker) * daily half_spread)
        "spread_capture_bps": float(spread_notional / (executed * vwap) * 1e4),
        "executed_fraction": float(executed / share),
        "arrival": arrival,
        "vwap": vwap,
    }
