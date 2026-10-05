"""Planner state snapshot/restore for crash-resume (v1.5).

Serializes the minimum state needed to resume a RollingPlanner from a
mid-day interruption: config identity, kernel coefficients, training
profile, forecaster primed state, and day progress (fractions executed so
far, filled total, damping anchor, realized volume prefix). The LU cache
is excluded — it rebuilds on demand.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .model import MultiExponential
from .replanner import N_MINUTES, ReplanConfig, RollingPlanner


@dataclass
class PlannerSnapshot:
    """Minimum resumable state (excludes performance caches)."""

    # config identity
    eta: float
    risk: float
    benchmark: str
    config: dict
    kernel_weights: list
    kernel_rates: list
    profile: list
    train_mean_volume: Optional[float]

    # day progress
    today: Optional[str]
    start: int                          # next replan minute
    fractions: list                     # (225,) per-minute filled fractions
    filled: float                       # cumulative filled
    pending: Optional[list]             # damping anchor (normalized tail)
    minute_volume_prefix: list          # realized volume up to `start`
    replans: int
    fallbacks: int

    # forecaster primed state
    forecaster_night_volume: Optional[float] = None


def snapshot(planner: RollingPlanner, start: int, fractions: np.ndarray,
             filled: float, pending: Optional[np.ndarray],
             minute_volume: np.ndarray) -> PlannerSnapshot:
    """Capture the planner's resumable state mid-execution."""
    return PlannerSnapshot(
        eta=planner.eta,
        risk=planner.risk,
        benchmark=planner.benchmark,
        config={
            "interval_minutes": planner.config.interval_minutes,
            "grid_size": planner.config.grid_size,
            "max_order": planner.config.max_order,
            "gamma_bucket": planner.config.gamma_bucket,
            "scale_clip": list(planner.config.scale_clip),
            "carry_impact": planner.config.carry_impact,
            "revision_damping": planner.config.revision_damping,
        },
        kernel_weights=planner.kernel.weights.tolist(),
        kernel_rates=planner.kernel.rates.tolist(),
        profile=planner.profile.tolist(),
        train_mean_volume=planner.train_mean_volume,
        today=planner._today.isoformat() if planner._today else None,
        forecaster_night_volume=(
            planner.forecaster._live_night
            if planner.forecaster is not None and hasattr(planner.forecaster, "_live_night")
            else None
        ),
        start=start,
        fractions=np.asarray(fractions, dtype=float).tolist(),
        filled=filled,
        pending=pending.tolist() if pending is not None else None,
        minute_volume_prefix=np.asarray(minute_volume[:start], dtype=float).tolist(),
        replans=planner.replans,
        fallbacks=planner.fallbacks,
    )


def restore(snap: PlannerSnapshot,
            forecaster=None) -> tuple:
    """Rebuild a planner + day state from a snapshot.

    Returns (planner, start, fractions, filled, pending, minute_volume).
    The caller reconstitutes the full minute_volume by appending live bars
    after `start`.
    """
    kernel = MultiExponential(
        weights=np.array(snap.kernel_weights),
        rates=np.array(snap.kernel_rates),
    )
    config = ReplanConfig(
        interval_minutes=snap.config["interval_minutes"],
        grid_size=snap.config["grid_size"],
        max_order=snap.config["max_order"],
        gamma_bucket=snap.config["gamma_bucket"],
        scale_clip=tuple(snap.config["scale_clip"]),
        carry_impact=snap.config["carry_impact"],
        revision_damping=snap.config["revision_damping"],
    )
    planner = RollingPlanner(
        snap.eta, kernel, np.array(snap.profile),
        risk=snap.risk, config=config,
        train_mean_volume=snap.train_mean_volume,
        benchmark=snap.benchmark,
        tail_forecaster=forecaster,
    )
    from datetime import date as _date
    planner._today = _date.fromisoformat(snap.today) if snap.today else None
    if forecaster is not None and snap.forecaster_night_volume is not None:
        forecaster.prime(night_volume=snap.forecaster_night_volume)
    planner.replans = snap.replans
    planner.fallbacks = snap.fallbacks

    minute_volume = np.zeros(N_MINUTES)
    minute_volume[: len(snap.minute_volume_prefix)] = snap.minute_volume_prefix
    pending = np.array(snap.pending) if snap.pending is not None else None
    return (
        planner,
        snap.start,
        np.array(snap.fractions, dtype=float),
        snap.filled,
        pending,
        minute_volume,
    )
