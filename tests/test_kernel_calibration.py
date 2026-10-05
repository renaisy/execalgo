"""v0.5 kernel calibration: synthetic recovery with known kernel and sign memory (T14)."""

import numpy as np
import pytest

from execalgo.kernel_calibration import (
    CalibrationConfig,
    deconvolve_kernel,
    estimate_kernel,
    estimate_response,
    fit_kernel,
    kernel_persistence,
)
from execalgo.model import MultiExponential, fit_square_root
from execalgo.realdata import N_SESSION_MINUTES

N = N_SESSION_MINUTES
TRUE_KERNEL = MultiExponential(
    weights=np.array([0.6, 0.4]), rates=np.array([0.10, 0.012])
)  # minute scale: half-lives ~7 and ~58 minutes
CONFIG = CalibrationConfig(max_lag=90)
FIT_RANGE = 80  # lags used for recovery checks (tail-closure safe zone)


class _SyntheticDay:
    def __init__(self, minute_signed, minute_mid):
        self.minute_signed = minute_signed
        self.minute_mid = minute_mid
        self.minute_volume = np.abs(minute_signed)


def _simulate(persistence, n_days=80, seed=7):
    """Pure propagator DGP: m_t = m_0 + sum G(t-s) x_s, minute resolution.

    `persistence` IS the lag-one sign autocorrelation C(1).
    """
    rng = np.random.default_rng(seed)
    days = []
    for _ in range(n_days):
        signs = np.empty(N)
        signs[0] = rng.choice([-1.0, 1.0])
        for t in range(1, N):
            if rng.random() < persistence:
                signs[t] = signs[t - 1]
            else:
                signs[t] = rng.choice([-1.0, 1.0])
        volumes = rng.poisson(40.0, N).astype(float) + 1.0
        x = signs * volumes
        states = np.zeros(2)
        mid = np.empty(N)
        for t in range(N):
            states = states * np.exp(-TRUE_KERNEL.rates) + TRUE_KERNEL.weights * x[t]
            mid[t] = 100.0 + states.sum()
        days.append(_SyntheticDay(x, mid))
    return days


def _recovery_error(days):
    estimate = estimate_response(days, CONFIG)
    g = deconvolve_kernel(estimate, CONFIG)
    truth = TRUE_KERNEL(estimate["lags"].astype(float))
    return float(np.abs(g[:FIT_RANGE] / g[0] - truth[:FIT_RANGE] / truth[0]).max())


def test_t14_recovery_independent_signs():
    assert _recovery_error(_simulate(persistence=0.0)) < 0.10


def test_t14_deconvolution_corrects_sign_memory():
    # the raw response decays much faster than the kernel under persistence
    days = _simulate(persistence=0.5)
    estimate = estimate_response(days, CONFIG)
    truth = TRUE_KERNEL(estimate["lags"].astype(float))
    raw = estimate["response"] / estimate["response"][0]
    raw_error = np.abs(raw[:FIT_RANGE] - truth[:FIT_RANGE] / truth[0]).max()
    assert raw_error > 0.15
    # deconvolution recovers it
    assert _recovery_error(days) < 0.10


def test_t14_strong_persistence():
    assert _recovery_error(_simulate(persistence=0.8, n_days=160)) < 0.15


def test_t14_fitted_kernel_monotone_and_close():
    kernel, diagnostics, _, g = estimate_kernel(_simulate(persistence=0.5), CONFIG)
    grid = np.linspace(0.01, 90, 400)
    values = kernel(grid)
    assert (np.diff(values) < 0).all()                 # completely monotone
    assert (kernel.weights > 0).all() and (kernel.rates > 0).all()
    assert diagnostics["kernel_positive"]
    # shape error against the truth over the fit range
    truth = TRUE_KERNEL(grid)
    assert np.abs(values / values[0] - truth / truth[0]).max() < 0.15


def test_t15_front_load_orders_with_kernel_persistence():
    """Faster kernel decay must reduce the optimal front-loading (T15).

    The slow scale stops at 0.75: slower decay strengthens the benchmark
    give-back and crosses into the positive-baseline (benchmark-gaming)
    regime of the asymptotic VWAP path.
    """
    from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm
    from execalgo.vwap import u_shaped

    paper, _ = fit_square_root()  # normalized-time rates (T = 225 min)
    fast = MultiExponential(paper.weights, paper.rates * 1.6)
    slow = MultiExponential(paper.weights, paper.rates * 0.75)

    params = [ExecParams(share=1.0, eta=0.2055, kernel=k) for k in (fast, paper, slow)]
    profile = u_shaped(params[1].make_grid(), sharpness=1.0)

    is_fracs = [ISAlgorithm(p).solve().front_load_fraction() for p in params]
    vwap_fracs = [VWAPAlgorithm(p, profile).solve().front_load_fraction() for p in params]
    assert is_fracs[0] < is_fracs[1] < is_fracs[2]
    assert vwap_fracs[0] < vwap_fracs[1] < vwap_fracs[2]
    # and persistence itself orders as constructed
    minutes = float(N)
    assert (kernel_persistence(fast, minutes) <
            kernel_persistence(paper, minutes) <
            kernel_persistence(slow, minutes))
