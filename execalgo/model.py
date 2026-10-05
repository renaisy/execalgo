"""Propagator kernels.

Multi-exponential kernels G(t) = sum_i a_i exp(-rho_i |t|), the
least-squares fit of a three-term instance to the square-root decay kernel
t^{-1/2} over the horizon, and the secular equation whose roots are the
modes of the linear baseline.

Ported from gordonritter/perturbative-impact (MIT license), the numerical
supplement of "Perturbation theory for nonlinear transient impact".
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import least_squares


@dataclass(frozen=True)
class MultiExponential:
    """The kernel G(t) = sum_i weights[i] exp(-rates[i] |t|).

    Rates are stored in increasing order, so index 0 is the slowest mode.
    """

    weights: np.ndarray
    rates: np.ndarray

    def __call__(self, lag):
        lag = np.abs(np.asarray(lag, dtype=float))
        exponent = -self.rates[:, None] * lag.reshape(1, -1)
        return (self.weights[:, None] * np.exp(exponent)).sum(0).reshape(lag.shape)

    @property
    def at_zero(self):
        """G(0+) = sum_i a_i."""
        return float(self.weights.sum())

    def modes(self, eta, slippage_free_weight=1.0):
        """Roots kappa_j of the secular equation.

        The equation is

            eta + lam sum_i a_i rho_i / (rho_i^2 - z) = 0,

        cleared of denominators into a polynomial of degree n in z = kappa^2.
        """
        lam = slippage_free_weight
        rates = self.rates
        product = Polynomial([1.0])
        for rate in rates:
            product = product * Polynomial([rate**2, -1.0])
        polynomial = eta * product
        for j, (weight, rate) in enumerate(zip(self.weights, rates)):
            cofactor = Polynomial([1.0])
            for i, other in enumerate(rates):
                if i != j:
                    cofactor = cofactor * Polynomial([other**2, -1.0])
            polynomial = polynomial + lam * weight * rate * cofactor
        return np.sqrt(np.sort(polynomial.roots().real))


@lru_cache(maxsize=8)
def _fit_square_root_cached(lower_endpoint, sample_size):
    # the nonlinear fit is deterministic for given arguments, so cache it
    return _fit_square_root_impl(lower_endpoint, sample_size)


def fit_square_root(lower_endpoint=0.02, sample_size=400):
    """Fit three exponentials to t^{-1/2} (cached per arguments)."""
    return _fit_square_root_cached(lower_endpoint, sample_size)


def _fit_square_root_impl(lower_endpoint, sample_size):
    """Fit three exponentials to t^{-1/2} on [lower_endpoint, 1] in relative error.

    Returns the fitted kernel and the largest relative deviation over the
    sample. Weights and rates are parametrized by their logarithms, which keeps
    both positive and so keeps the fitted kernel completely monotone.
    """
    sample = np.geomspace(lower_endpoint, 1.0, sample_size)
    target = sample**-0.5

    def relative_error(log_parameters):
        weights = np.exp(log_parameters[:3])
        rates = np.exp(log_parameters[3:])
        model = (weights[:, None] * np.exp(-rates[:, None] * sample[None, :])).sum(0)
        return model / target - 1.0

    start = np.log([2.0, 1.0, 0.5, 2.0, 15.0, 90.0])
    solution = least_squares(relative_error, start, method="lm", max_nfev=20000)
    weights = np.exp(solution.x[:3])
    rates = np.exp(solution.x[3:])
    order = np.argsort(rates)
    kernel = MultiExponential(weights[order], rates[order])
    return kernel, float(np.abs(relative_error(solution.x)).max())
