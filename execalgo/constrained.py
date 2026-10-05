"""Box-constrained solves on the bordered KKT system (v1.0).

Active-set method for

    min 1/2 v' A v - b' v   s.t.   w' v = share,   0 <= v <= cap,

where A is the linear cost operator (or a Newton Hessian) and b carries the
cross-term / offset right-hand side. The long-only clamp of v0.1-v0.6 is the
lower bound alone — solved here structurally, which removes the asymptotic
path's positive-baseline restriction in the benchmark-gaming regime — and
the upper bound realizes participation caps. Termination is a verified KKT
point: grad J = ell * 1 + mu with mu >= 0 on lower-active nodes, <= 0 on
upper-active ones, zero on the free set.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import lu_factor, lu_solve

TOL = 1e-10


@dataclass
class BoxSolution:
    rate: np.ndarray
    multiplier: float            # equality multiplier ell
    active_lower: np.ndarray
    active_upper: np.ndarray
    kkt_residual: float          # max KKT violation at return
    iterations: int


def _bordered(matrix, weights):
    n = matrix.shape[0]
    bordered = np.zeros((n + 1, n + 1))
    bordered[:n, :n] = matrix
    bordered[:n, n] = -1.0
    bordered[n, :n] = weights
    return bordered


def solve_box_linear(a, weights, b, share, upper: Optional[np.ndarray] = None,
                     max_iter: int = 80):
    """Exact active-set solve of the box-constrained bordered system."""
    n = a.shape[0]
    upper = None if upper is None else np.asarray(upper, dtype=float)
    if upper is not None and float(weights @ upper) < share - 1e-9:
        raise ValueError("infeasible: share exceeds the capped supply w'cap")
    lower_mask = np.zeros(n, dtype=bool)
    upper_mask = np.zeros(n, dtype=bool)

    iterations = 0
    rate = np.zeros(n)
    ell = 0.0
    for iterations in range(1, max_iter + 1):
        free = ~(lower_mask | upper_mask)
        fixed = np.where(upper_mask, upper if upper is not None else 0.0, 0.0)
        free_idx = np.flatnonzero(free)
        fixed_idx = np.flatnonzero(~free)
        share_free = share - float(weights[fixed_idx] @ fixed[fixed_idx])

        sub = np.ix_(free_idx, free_idx)
        rhs_free = b[free_idx] - a[np.ix_(free_idx, fixed_idx)] @ fixed[fixed_idx]
        bordered = _bordered(a[sub], weights[free_idx])
        try:
            solution = lu_solve(lu_factor(bordered),
                                np.concatenate([rhs_free, [share_free]]))
        except np.linalg.LinAlgError:
            raise RuntimeError("active-set submatrix singular")

        rate = fixed.copy()
        rate[free_idx] = solution[:-1]
        ell = float(solution[-1])

        # primal violations on the free set -> activate
        violate_low = free & (rate < -TOL)
        violate_up = (
            free & (rate > upper + TOL) if upper is not None
            else np.zeros(n, dtype=bool)
        )
        if violate_low.any() or violate_up.any():
            lower_mask |= violate_low
            if upper is not None:
                upper_mask |= violate_up
            continue

        # dual (KKT) violations on the active set -> release
        gradient = a @ rate - b
        mu = gradient - ell
        release_low = lower_mask & (mu < -TOL)
        release_up = upper_mask & (mu > TOL)
        if release_low.any() or release_up.any():
            # release the worst offenders only, to avoid cycling
            candidates = np.flatnonzero(release_low | release_up)
            worst = candidates[np.argmax(np.abs(mu[candidates]))]
            lower_mask[worst] = upper_mask[worst] = False
            continue

        kkt = _kkt_residual(a, weights, b, share, rate, ell, lower_mask, upper_mask, upper)
        return BoxSolution(rate, ell, lower_mask, upper_mask, kkt, iterations)

    kkt = _kkt_residual(a, weights, b, share, rate, ell, lower_mask, upper_mask, upper)
    return BoxSolution(rate, ell, lower_mask, upper_mask, kkt, iterations)


def _kkt_residual(a, weights, b, share, rate, ell, lower_mask, upper_mask, upper):
    """Max violation of the KKT system (primal feasibility + dual signs)."""
    gradient = a @ rate - b
    mu = gradient - ell
    primal = max(
        float(-min(rate.min(), 0.0)),
        float(max((rate - upper).max(), 0.0)) if upper is not None else 0.0,
        abs(float(weights @ rate) - float(share)),
    )
    dual_low = float(-mu[lower_mask].min()) if lower_mask.any() else 0.0  # mu >= 0
    dual_up = float(mu[upper_mask].max()) if upper_mask.any() else 0.0    # mu <= 0
    dual_free = float(np.abs(mu[~(lower_mask | upper_mask)]).max()) \
        if (~(lower_mask | upper_mask)).any() else 0.0
    return float(max(primal, dual_low, dual_up, dual_free))


def newton_step_boxed(hessian, residual, weights, rate, lower_mask, upper_mask,
                      upper: Optional[np.ndarray] = None):
    """One constrained Newton step on the free set.

    Solves [H_ff, -1; w_f, 0] [step; d_ell] = [F_f; 0] so that the CALLER's
    convention (rate -= step, multiplier -= d_ell) is the Newton update —
    matching the unbounded path's sign convention. Active nodes keep
    step = 0 (projected Newton); masks are updated by the caller.
    """
    n = hessian.shape[0]
    free = ~(lower_mask | upper_mask)
    free_idx = np.flatnonzero(free)
    bordered = _bordered(hessian[np.ix_(free_idx, free_idx)], weights[free_idx])
    rhs = residual[free_idx]
    solution = np.linalg.solve(bordered, np.concatenate([rhs, [0.0]]))
    step = np.zeros(n)
    step[free_idx] = solution[:-1]
    return step, float(solution[-1])
