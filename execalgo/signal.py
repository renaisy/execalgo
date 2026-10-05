"""Causal price-direction signals for alpha-aware execution (v1.3).

The alpha term rewards executing at nodes where the price is expected to
be higher over the remaining horizon:

    J(v) = ... - int v(t) * alpha(t) dt

which puts -alpha(t) in the gradient constant and +alpha(t) on the solve
RHS (same side as the VWAP cross term). This module computes the signal
from past prices only (causal) and scales it to the solver's cost units.
"""

import numpy as np

from .realdata import N_SESSION_MINUTES


def momentum_signal(day, lookback_minutes: int = 15) -> np.ndarray:
    """Causal rolling return from minute_vwap, zero before warm-up.

    At minute t the signal is P(t)/P(t - lookback) - 1. The first
    `lookback_minutes` entries are zero (no history yet). Uses the
    forward-filled minute_vwap so the signal is always finite.
    """
    prices = np.asarray(day.minute_vwap, dtype=float)
    lookback = min(lookback_minutes, N_SESSION_MINUTES)
    signal = np.zeros(N_SESSION_MINUTES)
    base = np.maximum(prices[: N_SESSION_MINUTES - lookback], 1e-12)
    signal[lookback:] = prices[lookback:] / base - 1.0
    return signal


def signal_ic(day, lookback_minutes: int = 15,
              forward_minutes: int = 15) -> float:
    """Information coefficient: corr(rolling return, forward return).

    Causal: the forward return at t uses P(t+forward)/P(t) - 1, available
    only for t < N - forward. Returns nan when either leg is degenerate.
    """
    signal = momentum_signal(day, lookback_minutes)
    prices = np.asarray(day.minute_vwap, dtype=float)
    forward = np.zeros(N_SESSION_MINUTES)
    n = N_SESSION_MINUTES
    forward[: n - forward_minutes] = (
        prices[forward_minutes:] / np.maximum(prices[: n - forward_minutes], 1e-12) - 1.0
    )
    valid = (signal != 0.0) & np.isfinite(signal) & np.isfinite(forward)
    if valid.sum() < 10:
        return float("nan")
    return float(np.corrcoef(signal[valid], forward[valid])[0, 1])


def solver_alpha(signal: np.ndarray, price_level: float,
                 lambda_scale: float) -> np.ndarray:
    """Scale a return-space signal to the solver's cost units.

    The solver's costs are dimensionless (share-normalized), calibrated at
    the level Lambda (price units per sqrt(rate)). A price move of
    alpha_real * P corresponds to alpha_real * P / Lambda in solver units:
    at that scale a move equal to the full impact cost has alpha ~ 1.
    """
    if lambda_scale <= 0.0:
        return np.zeros_like(signal, dtype=float)
    return np.asarray(signal, dtype=float) * price_level / lambda_scale
