"""Transient-impact kernel calibration from L5 snapshots (v0.5).

Identification follows the propagator framework (Bouchaud et al.): with the
mid-price driven by signed flow, m_t = m_0 + sum_{s<t} G(t-s) eps_s q_s, the
empirical response function

    R(tau) = < (m_{t+tau} - m_t) * q_t > / < q_t^2 >      (pooled OLS)

is NOT the kernel when trade signs are autocorrelated:

    R(tau)/g0 = G(tau) + sum_{l>=1} C(l) [G(tau+l) - G(l)]
                        + sum_{l=1..tau} C(l) G(tau-l),

with C(l) = <eps_t eps_{t+l}> the sign autocorrelation (signs scaled to unit
RMS). Order-flow persistence in Chinese futures is strong, so fitting the
kernel to R directly would systematically overstate persistence. Both R and
C are measured at one-minute resolution; the kernel is then recovered by
solving the linear system above with truncation at max_lag (G assumed zero
beyond), and a three-exponential family with log-parametrized weights and
rates (guaranteeing complete monotonicity) is fitted to the recovered shape.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from .model import MultiExponential
from .realdata import N_SESSION_MINUTES


@dataclass(frozen=True)
class CalibrationConfig:
    max_lag: int = 120            # minutes; G assumed zero beyond
    min_lag: int = 1
    n_exponentials: int = 3
    bootstrap_draws: int = 200


def _stack(days, field):
    return np.array([getattr(d, field) for d in days], dtype=float)


def estimate_response(days, config: CalibrationConfig = CalibrationConfig()):
    """Sign-based response R(tau) and sign autocorrelation C(l), minute scale.

    R(tau) = mean over signed minutes of eps_t * (m_{t+tau} - m_{t-1}),
    measured from the PRE-trade mid so the trade's own impact enters
    positively. Sign-based weighting (rather than pooled OLS on raw
    volumes) keeps the estimator stable under persistent order flow, where
    same-sign runs make volume-weighted products heavy-tailed; the level is
    per sign-event and the execution calibration absorbs the scale. The
    only contamination is same-sign flow after t, which the deconvolution
    removes.
    """
    signed = _stack(days, "minute_signed")
    mid = _stack(days, "minute_mid")
    n_days, n = signed.shape
    lags = np.arange(config.min_lag, config.max_lag + 1)

    r_num = np.zeros(lags.size)
    r_cnt = np.zeros(lags.size)     # per-lag denominator (v1.1.1: the
    c_num = np.zeros(config.max_lag)  # numerator is lag-truncated, so the
    c_cnt = np.zeros(config.max_lag)  # count must be too — a global count
    for i in range(n_days):           # dilutes R(tau) by (n - tau)/n)
        q, m = signed[i], mid[i]
        eps = np.where(q != 0.0, np.sign(q), 0.0)
        pre = np.concatenate([m[:1], m[:-1]])          # m_{t-1}, held at t=0
        for j, tau in enumerate(lags):
            mask = eps[: n - tau] != 0.0
            r_num[j] += float(np.dot(eps[: n - tau][mask],
                                     (m[tau:] - pre[: n - tau])[mask]))
            r_cnt[j] += int(mask.sum())
        # autocorrelation on the MINUTE grid: silent minutes stay in place
        # (zero sign) and each lag counts only its overlapping signed pairs,
        # so C(l) is a minute-lag correlation matching the lags the
        # deconvolution applies it to
        for l in range(1, config.max_lag + 1):
            a, b = eps[:-l], eps[l:]
            both = (a != 0.0) & (b != 0.0)
            c_num[l - 1] += float(np.dot(a[both], b[both]))
            c_cnt[l - 1] += int(both.sum())
    response = r_num / np.maximum(r_cnt, 1.0)
    autocorr = c_num / np.maximum(c_cnt, 1.0)
    return {"lags": lags, "response": response, "autocorr": autocorr}


def deconvolve_kernel(estimate, config: CalibrationConfig = CalibrationConfig()):
    """Recover G(tau) from R and C via the full propagator linear system.

    With the pre-trade response convention the expectation of the response
    under sign autocorrelation C is (verified against Monte Carlo):

        R(tau)/E[q] = G(tau)
            + sum_{l=1..tau} C(l) G(tau - l)            (same-sign flow after t)
            - sum_{k>=1} C(k) [G(k - 1) - G(k + tau)]   (relaxation of the
                                                          pre-trade impact state)

    The relaxation term makes the raw response decay FASTER than the kernel
    under persistent flow. Two closures make the finite system consistent:
    G(0) by smooth back-extrapolation 2G(1)-G(2) (exact for the exponential
    family), and the beyond-L tail of the last sum by the fitted
    parametric family, iterated. Returns G(1..L).
    """
    r = estimate["response"]
    c = estimate["autocorr"]
    l_max = config.max_lag
    k_max = min(c.size, 400)
    lags = np.arange(1, l_max + 1, dtype=float)

    def solve(g0, tail):
        a = np.zeros((l_max, l_max))
        rhs = r + tail
        for row, tau in enumerate(range(1, l_max + 1)):
            a[row, tau - 1] += 1.0
            for l in range(1, tau + 1):
                if tau - l >= 1:
                    a[row, tau - l - 1] += c[l - 1]
                else:
                    rhs[row] -= c[l - 1] * g0        # forward l=tau: +C(tau) G(0)
            for k in range(1, k_max + 1):
                if k - 1 >= 1:
                    a[row, k - 2] -= c[k - 1]
                else:
                    rhs[row] += c[k - 1] * g0        # relaxation k=1: -C(1) G(0)
                if k + tau <= l_max:
                    a[row, k + tau - 1] += c[k - 1]
        return np.linalg.solve(a, rhs)

    tail = np.zeros(l_max)
    g0 = max(r[0], 1e-9)
    # warm-start the parametric tail from the raw response shape (positive
    # and smooth; flatter than G but far better than zero)
    family, _ = fit_kernel(np.maximum(r, 1e-9), lags, config)
    tail = np.array([
        sum(c[k - 1] * family(float(tau + k))
            for k in range(1, k_max + 1) if tau + k > l_max)
        for tau in range(1, l_max + 1)
    ])
    for _ in range(10):
        for _ in range(30):
            g = solve(g0, tail)
            target = max(2.0 * g[0] - g[1], 0.0)
            if abs(target - g0) < 1e-10 * (1.0 + abs(target)):
                break
            g0 = 0.5 * g0 + 0.5 * target
        try:
            family, _ = fit_kernel(np.maximum(g, 1e-9), lags, config)
        except Exception as error:  # noqa: BLE001 - keep the stale tail, loudly
            warnings.warn(f"parametric tail fit failed ({error}); "
                          "deconvolution keeps the previous tail closure")
            break
        new_tail = np.array([
            sum(c[k - 1] * family(float(tau + k))
                for k in range(1, k_max + 1) if tau + k > l_max)
            for tau in range(1, l_max + 1)
        ])
        if np.abs(new_tail - tail).max() < 1e-10 * (1.0 + np.abs(tail).max()):
            tail = new_tail
            break
        tail = 0.6 * tail + 0.4 * new_tail
    return solve(g0, tail)


def fit_kernel(g_values, lags, config: CalibrationConfig = CalibrationConfig()):
    """Fit sum a_i exp(-rho_i tau) to the recovered kernel shape.

    The fit targets the shape g(tau)/g(1) (level is absorbed by the
    calibration's Lambda); log-parametrization keeps the fitted kernel
    completely monotone.
    """
    target = np.asarray(g_values, dtype=float)
    keep = target > 0
    lags = np.asarray(lags, dtype=float)[keep]
    values = target[keep] / target[0]
    k = config.n_exponentials

    def residual(log_params):
        weights = np.exp(np.clip(log_params[:k], -30.0, 5.0))
        rates = np.exp(np.clip(log_params[k:], -12.0, 4.0))
        model = (weights[:, None] * np.exp(-rates[:, None] * lags[None, :])).sum(0)
        return model / values - 1.0

    start = np.log([1.0 / k] * k + list(np.geomspace(0.02, 3.0, k)))
    solution = least_squares(
        residual, start, method="trf",
        bounds=(np.log([1e-12] * k + [1e-5] * k), np.log([1e2] * k + [1e1] * k)),
        max_nfev=20000,
    )
    weights = np.exp(solution.x[:k])
    rates = np.exp(solution.x[k:])
    order = np.argsort(rates)
    kernel = MultiExponential(weights[order], rates[order])
    deviation = float(np.abs(residual(solution.x)).max())
    # the fit targets the shape g/g(1) only; the level is absorbed by the
    # execution calibration's Lambda, so the returned kernel is left in
    # shape units (g_fit(1) == 1 by construction of the target)
    return kernel, deviation


def bootstrap_response_bands(days, config: CalibrationConfig = CalibrationConfig(),
                             draws: Optional[int] = None, seed: int = 11):
    """Day-block bootstrap of the normalized response R(tau)/R(1)."""
    rng = np.random.default_rng(seed)
    days = list(days)
    n = len(days)
    draws = draws or config.bootstrap_draws
    samples = []
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        estimate = estimate_response([days[i] for i in pick], config)
        response = estimate["response"]
        if response[0] > 0:
            samples.append(response / response[0])
    samples = np.array(samples)
    return {
        "low": np.percentile(samples, 5, axis=0),
        "high": np.percentile(samples, 95, axis=0),
        "draws": len(samples),
    }


def kernel_persistence(kernel: MultiExponential, horizon: float = 1.0) -> float:
    """int_0^horizon G — the persistence proxy for front-loading comparisons."""
    grid = np.linspace(0.0, horizon, 2001)
    return float(np.trapezoid(kernel(grid), grid))


def estimate_kernel(days, config: CalibrationConfig = CalibrationConfig()):
    """End-to-end kernel calibration with diagnostics.

    The last few lags are the most tail-closure sensitive, so the family is
    fitted on lags 1..max_lag - 10 and extrapolates smoothly through them.
    """
    estimate = estimate_response(days, config)
    g = deconvolve_kernel(estimate, config)          # G(1..max_lag), minute lags
    if not np.all(g[:10] > 0):
        raise ValueError(
            "deconvolved kernel is non-positive at short lags; the response/"
            "autocorrelation inputs do not identify a decaying kernel here"
        )
    fit_length = max(config.max_lag - 10, 10)
    lags = np.arange(1, config.max_lag + 1)
    kernel, deviation = fit_kernel(g[:fit_length], lags[:fit_length], config)
    diagnostics = {
        "max_fit_deviation": deviation,
        "response_at_one": float(estimate["response"][0]),
        "autocorr_at_one": float(estimate["autocorr"][0]),
        "kernel_persistence_minutes": kernel_persistence(kernel, float(config.max_lag)),
        "kernel_positive": bool(np.all(g[:fit_length] > 0)),
        "recovered_lags": fit_length,
    }
    return kernel, diagnostics, estimate, g
