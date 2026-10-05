"""Intraday volume-profile forecasting (v0.6).

The reference project's Bayesian layer (`bayes_volume.py` in
Model-LightGBM/services/vwap-execution) adapts the LEVEL: prior curve share
x observed/expected session volume. What it lacks is day-conditional SHAPE,
which is exactly the channel the v0.3 rolling replanner identified as the
missing gain driver. This module supplies it with a prefix-conditional
kernel estimator:

    p_hat(t) = sum_k w_k * p_k(t) / sum_k w_k,

where day weights w_k come from the similarity of the realized morning
prefix (normalized cumulative curve), the open-auction share, the preceding
night share and the weekday between the live day and each training day.
The prediction is a whole-day normalized density, so its tail restriction
carries the correct remaining mass (the unnormalized p~ = p * h subtlety
that time consistency forced in v0.3).

Forecaster API: `tail(realized_volume_prefix, from_minute)` returns the
day-density values of the remaining minutes (sums to the predicted
remaining fraction, not to one).
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

import numpy as np

from .realdata import N_SESSION_MINUTES


@dataclass(frozen=True)
class ForecastConfig:
    shape_bandwidth: float = 0.06     # RMS distance on cumulative prefix curves
    auction_bandwidth: float = 0.10   # |auction share difference|
    night_bandwidth: float = 0.50     # |log night-ratio difference|
    weekday_bonus: float = 1.5        # weight multiplier for same weekday
    auction_minutes: int = 3
    min_weight: float = 1e-3


def _as_matrix(train_volumes):
    matrix = np.asarray(train_volumes, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != N_SESSION_MINUTES:
        raise ValueError(f"train_volumes must be (n_days, {N_SESSION_MINUTES})")
    return matrix


class PrefixForecaster:
    """Kernel conditioning of the whole-day volume shape on the live prefix."""

    def __init__(
        self,
        train_volumes,
        train_dates: Optional[Sequence[date]] = None,
        train_night_volumes: Optional[Sequence[float]] = None,
        config: ForecastConfig = ForecastConfig(),
    ):
        self.config = config
        volumes = _as_matrix(train_volumes)
        totals = volumes.sum(axis=1)
        keep = totals > 0
        volumes, totals = volumes[keep], totals[keep]
        if volumes.shape[0] < 5:
            raise ValueError("need at least 5 training days")
        self.shapes = volumes / totals[:, None]                 # (n, 225), rows sum to 1
        self.cumulative = np.cumsum(self.shapes, axis=1)
        self.auction = self.shapes[:, : config.auction_minutes].sum(axis=1)
        self.night_ratio = None
        if train_night_volumes is not None:
            nights = np.asarray(train_night_volumes, dtype=float)[keep]
            self.night_ratio = np.log1p(nights) - np.log1p(totals)
        self.weekday = (
            np.array([d.weekday() for d in train_dates])[keep]
            if train_dates is not None else None
        )
        self.prior = self.shapes.mean(axis=0)

    def _weights(self, realized, upto, today: Optional[date]):
        cfg = self.config
        prefix = realized[:upto]
        total = prefix.sum()
        if total <= 0:
            return None
        normalized = prefix / total
        c_live = np.cumsum(normalized)
        c_train = self.cumulative[:, :upto]
        # compare prefix-normalized cumulative curves on both sides (the live
        # prefix cumulative ends at 1, so the train side must be rescaled the
        # same way or every day looks equidistant)
        c_train = c_train / c_train[:, -1][:, None]
        dist = np.sqrt(np.mean((c_train - c_live[None, :]) ** 2, axis=1))
        weights = np.exp(-0.5 * (dist / cfg.shape_bandwidth) ** 2)

        # auction share compared within the prefix on both sides (a share of
        # the prefix is not comparable to a share of the whole day)
        auction_live = normalized[: cfg.auction_minutes].sum()
        train_prefix = self.shapes[:, :upto].sum(axis=1)
        train_auction = np.divide(
            self.shapes[:, : cfg.auction_minutes].sum(axis=1), train_prefix,
            out=np.zeros_like(train_prefix), where=train_prefix > 0,
        )
        weights = weights * np.exp(
            -0.5 * ((train_auction - auction_live) / cfg.auction_bandwidth) ** 2
        )

        if self.night_ratio is not None and getattr(self, "_live_night", None) is not None:
            night_live = np.log1p(self._live_night) - np.log1p(total + self._live_night)
            weights = weights * np.exp(
                -0.5 * ((self.night_ratio - night_live) / cfg.night_bandwidth) ** 2
            )

        if self.weekday is not None and today is not None:
            weights = weights * np.where(
                self.weekday == today.weekday(), cfg.weekday_bonus, 1.0
            )

        weights = np.maximum(weights, cfg.min_weight)
        return weights / weights.sum()

    def prime(self, night_volume: Optional[float] = None):
        """Register day-level information available before the open."""
        self._live_night = night_volume

    def tail(self, realized_volume, from_minute, today: Optional[date] = None):
        """Day-density values over minutes [from_minute, 225).

        The returned vector is the whole-day normalized prediction restricted
        to the tail (it sums to the predicted remaining fraction), keeping the
        unnormalized-mass semantics the planner's cross term requires.
        """
        realized = np.asarray(realized_volume, dtype=float)
        weights = self._weights(realized, from_minute, today)
        if weights is None:
            shape = self.prior
        else:
            shape = weights @ self.shapes
        return shape[from_minute:]


class AverageForecaster:
    """The unconditional train-average shape (the v0.3 baseline)."""

    def __init__(self, train_volumes):
        volumes = _as_matrix(train_volumes)
        mean = volumes.mean(axis=0)
        self.shape = mean / mean.sum()

    def prime(self, night_volume=None):
        pass

    def tail(self, realized_volume, from_minute, today=None):
        return self.shape[from_minute:]


class OracleForecaster:
    """Clairvoyant bound: uses the day's realized full-day shape.

    For upper-bound diagnostics only — never admissible in production.
    """

    def __init__(self):
        self._shape = None

    def prime(self, night_volume=None, day_volume=None):
        if day_volume is not None:
            total = float(np.sum(day_volume))
            self._shape = np.asarray(day_volume, dtype=float) / total

    def tail(self, realized_volume, from_minute, today=None):
        return self._shape[from_minute:]


def tail_distance(predicted_tail, realized_tail) -> float:
    """Total-variation distance between predicted and realized tail shapes."""
    predicted = np.asarray(predicted_tail, dtype=float)
    realized = np.asarray(realized_tail, dtype=float)
    predicted = predicted / max(predicted.sum(), 1e-12)
    realized = realized / max(realized.sum(), 1e-12)
    return float(0.5 * np.abs(predicted - realized).sum())
