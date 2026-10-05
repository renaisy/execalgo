"""v1.3 signal module: causality, scaling, IC (T24)."""

import numpy as np
import pytest

from execalgo.realdata import N_SESSION_MINUTES, TradingDay
from execalgo.signal import momentum_signal, signal_ic, solver_alpha


def _day(prices=None, seed=7):
    rng = np.random.default_rng(seed)
    if prices is None:
        prices = 100.0 * np.exp(np.cumsum(rng.standard_normal(N_SESSION_MINUTES) * 1e-3))
    return TradingDay(
        __import__("datetime").date(2025, 6, 4), prices,
        np.ones(N_SESSION_MINUTES), N_SESSION_MINUTES, 1.0, 0.0,
        np.zeros(N_SESSION_MINUTES), prices, np.full(N_SESSION_MINUTES, 50.0),
        np.full(N_SESSION_MINUTES, 50.0), np.full(N_SESSION_MINUTES, 500.0),
        np.full(N_SESSION_MINUTES, 500.0),
    )


def test_momentum_causal_and_zero_warmup():
    day = _day()
    signal = momentum_signal(day, lookback_minutes=15)
    assert signal.shape == (N_SESSION_MINUTES,)
    assert np.all(signal[:15] == 0.0)
    assert np.any(signal[15:] != 0.0)


def test_momentum_correctness_on_trend():
    """A monotonically rising price gives positive momentum after warm-up."""
    prices = np.linspace(100.0, 110.0, N_SESSION_MINUTES)
    day = _day(prices=prices)
    signal = momentum_signal(day, lookback_minutes=15)
    assert np.all(signal[15:] > 0.0)
    # and falling gives negative
    day_down = _day(prices=prices[::-1].copy())
    signal_down = momentum_signal(day_down, lookback_minutes=15)
    assert np.all(signal_down[15:] < 0.0)


def test_solver_alpha_scaling():
    """A 1bp signal at price 100000 and Lambda=57 gives ~0.175 solver units."""
    signal = np.full(N_SESSION_MINUTES, 1e-4)  # 1bp
    alpha = solver_alpha(signal, 100000.0, 57.0)
    assert alpha[0] == pytest.approx(100000.0 * 1e-4 / 57.0, rel=1e-10)
    # zero lambda → all zero (guard)
    assert np.all(solver_alpha(signal, 100.0, 0.0) == 0.0)


def test_t24_signal_ic_computed_and_finite():
    """T24: the IC is computable from any non-degenerate day."""
    rng = np.random.default_rng(3)
    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal(N_SESSION_MINUTES) * 5e-3))
    day = _day(prices=prices)
    ic = signal_ic(day, lookback_minutes=15, forward_minutes=15)
    assert np.isfinite(ic)
    assert -1.0 <= ic <= 1.0


def test_t24_signal_ic_strong_trend():
    """With a very strong drift the momentum IC is positive."""
    drift = np.full(N_SESSION_MINUTES, 1e-2)  # 1% per minute
    prices = 100.0 * np.exp(np.cumsum(drift))
    day = _day(prices=prices)
    ic = signal_ic(day, lookback_minutes=15, forward_minutes=15)
    assert ic > 0.0
