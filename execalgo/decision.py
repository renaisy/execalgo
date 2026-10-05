"""Execution decision records aligned with the reference architecture (v1.5).

Mirrors the schema of services/vwap-execution's decision stream:
bar_ts, action, side, child_qty, parent_remaining, reason, latency_ms, ts.
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class ExecutionDecision:
    """One per-bar execution instruction, ready for Redis XADD."""

    bar_ts: str                 # bar timestamp (ISO string)
    action: str                 # "buy" / "sell" / "hold"
    side: int                   # +1 buy, -1 sell, 0 hold
    child_qty: float            # lots to execute this bar
    parent_remaining: float     # lots remaining after this decision
    reason: str                 # "replan" / "fallback" / "final"
    latency_ms: float           # planner step time
    ts: float                   # wall-clock unix timestamp

    def to_dict(self):
        return asdict(self)


def make_decision(bar_ts: str, fraction: float, parent_qty: float,
                  parent_filled: float, side: int = 1,
                  reason: str = "replan",
                  latency_ms: float = 0.0) -> ExecutionDecision:
    """Build a decision from a planner step's per-minute fraction."""
    child = fraction * parent_qty
    remaining = max(parent_qty - parent_filled - child, 0.0)
    action = "buy" if side > 0 else ("sell" if side < 0 else "hold")
    return ExecutionDecision(
        bar_ts=bar_ts,
        action=action,
        side=side,
        child_qty=round(child, 1),
        parent_remaining=round(remaining, 1),
        reason=reason,
        latency_ms=round(latency_ms, 2),
        ts=time.time(),
    )
