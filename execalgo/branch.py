"""The nonlinear solution branch, followed by Newton continuation in eps.

The discrete optimality system at parameter eps is

    2 eta v + P f(v;eps) + diag(f_x(v;eps)) ( Q v - q ) = ell 1,   w^T v = share,

with f(x;eps) = x^{1-eps}. For the arrival-price objective q = 0; for the
VWAP objective q = (X/M) Q mu is the benchmark cross term. Starting from
the linear baseline at eps = 0, the parameter is raised in small steps and
each step is corrected by Newton's method from the previous solution. The
reduced Hessian can be monitored after each converged step to check local
minimality along the sampled branch.

Adapted from gordonritter/perturbative-impact (MIT license).
"""

import numpy as np
from scipy.linalg import null_space

from .constrained import newton_step_boxed, solve_box_linear
from .impact import impact, impact_curvature, impact_slope


def stationarity(grid, eta, eps, rate, multiplier, share, cross=None, risk=0.0,
                 offset=None):
    """The residual of the optimality system, with the constraint appended.

    `offset` is an eps-independent constant that IS the gradient constant:
    it enters the interior with a PLUS sign here and is SUBTRACTED from
    the solve RHS in continue_branch. NOTE the sign convention differs from
    series.coefficients' `offset` parameter (which is RHS-side): to inject
    a gradient constant g, pass offset=g to branch and offset=-g to series.
    Keep the two straight — a flip front-loads into own impact.
    """
    if cross is None:
        cross = 0.0
    interior = (
        2.0 * eta * rate
        + grid.past @ impact(rate, eps)
        + impact_slope(rate, eps) * ((grid.future @ rate) - cross)
        - multiplier
    )
    if risk:
        interior = interior + grid.risk_operator(risk) @ rate - grid.risk_vector(risk, share)
    if offset is not None:
        interior = interior + np.asarray(offset, dtype=float)
    return np.concatenate([interior, [grid.weights @ rate - share]])


def hessian(grid, eta, eps, rate, cross=None, risk=0.0):
    """Derivative in v of the left-hand side of the optimality system."""
    slope = impact_slope(rate, eps)
    curvature = impact_curvature(rate, eps)
    matrix = (
        2.0 * eta * np.eye(grid.size)
        + grid.past * slope[None, :]
        + np.diag(curvature * (grid.future @ rate))
        + slope[:, None] * grid.future
    )
    if cross is not None:
        matrix = matrix - np.diag(curvature * cross)
    if risk:
        matrix = matrix + grid.risk_operator(risk)
    return matrix


def _bordered(grid, matrix):
    size = grid.size
    bordered = np.zeros((size + 1, size + 1))
    bordered[:size, :size] = matrix
    bordered[:size, size] = -1.0
    bordered[size, :size] = grid.weights
    return bordered


def continue_branch(
    grid,
    eta,
    share,
    eps_target,
    eps_step=0.01,
    max_newton=100,
    monitor_hessian=False,
    cross=None,
    risk=0.0,
    upper=None,
    offset=None,
):
    """Follow the branch and optionally record the reduced-Hessian minimum.

    The returned trace contains ``(eps, eigenvalue)`` at the baseline and
    after every converged continuation step. It is empty when monitoring is
    disabled.

    With `upper` set (participation cap in solver-rate units) the path runs
    projected active-set Newton on the box 0 <= v <= upper; the unbounded
    path is untouched. Even without `upper`, a negative unconstrained
    baseline now falls back to the box solve (structural long-only) instead
    of returning a non-executable schedule.

    `offset` is an eps-independent constant entering the GRADIENT with a
    plus sign (see `stationarity`); the solve RHS therefore gets it
    subtracted, mirroring `series.coefficients`' offset convention.
    """
    if cross is None:
        cross = np.zeros(grid.size)
    operator = grid.cost_operator(eta, risk)
    # gradient constant -offset => RHS +offset with sign flipped relative
    # to how the risk vector enters (risk gradient constant is negative so
    # it enters the RHS positively; a positive gradient constant like the
    # carry I_past enters the RHS negatively)
    rhs = cross + grid.risk_vector(risk, share)
    if offset is not None:
        rhs = rhs - np.asarray(offset, dtype=float)
    # The eps = 0 baseline solves A v = ell 1 + q + risk vector with
    # w^T v = share, i.e. v = A^{-1}(q + r) + ell A^{-1} 1 with ell fixed
    # by the share constraint.
    particular = np.linalg.solve(operator, rhs)
    direction = np.linalg.solve(operator, np.ones(grid.size))
    multiplier = (share - grid.weights @ particular) / (grid.weights @ direction)
    rate = particular + multiplier * direction

    lower_mask = np.zeros(grid.size, dtype=bool)
    upper_mask = np.zeros(grid.size, dtype=bool)
    if upper is not None or rate.min() <= 0.0:
        box = solve_box_linear(operator, grid.weights, rhs, share, upper)
        rate, multiplier = box.rate, box.multiplier
        lower_mask, upper_mask = box.active_lower, box.active_upper

    tangent = None
    hessian_trace = []
    if monitor_hessian:
        tangent = null_space(np.sqrt(grid.weights)[None, :])
        hessian_trace.append(
            (0.0, reduced_hessian_min(grid, eta, 0.0, rate, tangent=tangent,
                                      cross=cross, risk=risk))
        )

    # Any requested ceiling must keep the box machinery armed for the whole
    # path even when the eps = 0 baseline is interior: the square-root optimum
    # front-loads harder than the linear one, so "slack at eps=0, binding at
    # eps_target" is a reachable regime. Likewise, a time-varying offset
    # (alpha) can push local rates toward the v=0 singular manifold even when
    # the baseline is interior — the box keeps the Jacobian finite there.
    bounded = bool(lower_mask.any() or upper_mask.any() or upper is not None
                  or offset is not None)
    ceiling = None if upper is None else np.asarray(upper, dtype=float)

    # The square-root impact slope diverges at rate 0, which the box forces
    # exactly at active lower nodes; regularize the nonlinear evaluations on
    # a tiny floor (active rows leave the free-set system anyway, so this
    # only tames near-boundary free nodes and the column blocks).
    slope_floor = 1e-9

    def floored_stationarity(v, m):
        return stationarity(
            grid, eta, eps, np.maximum(v, slope_floor), m, share, cross, risk, offset
        )

    steps = list(np.arange(eps_step, eps_target + 1e-9, eps_step))
    if not steps and eps_target > 0:
        steps = [eps_target]               # eps_target < eps_step: one step
    elif steps and steps[-1] < eps_target - 1e-12:
        steps.append(eps_target)           # land exactly on the requested target
    for eps in steps:
        eps = round(float(eps), 10)
        converged = False
        for _ in range(max_newton):
            if bounded:
                residual = floored_stationarity(rate, multiplier)
            else:
                residual = stationarity(grid, eta, eps, rate, multiplier, share,
                                        cross, risk, offset)
            residual_norm = float(np.linalg.norm(residual))

            # v2.2 fix a: on the bounded path, the FULL residual includes bound
            # multipliers at active nodes (the KKT condition, generically
            # O(1)), so it can never reach 1e-11 — converge on the free-set
            # residual + equality row instead. On the unbounded path the two
            # coincide. We evaluate the correct merit once the free set is
            # known for this iteration.
            if bounded:
                free_idx = np.flatnonzero(~(lower_mask | upper_mask))
                if free_idx.size:
                    free_residual = np.r_[residual[free_idx], residual[-1]]
                    merit_now = 0.5 * float(np.linalg.norm(free_residual)) ** 2
                else:
                    merit_now = 0.5 * residual[-1] ** 2  # only the equality row
            else:
                merit_now = 0.5 * residual_norm ** 2

            if merit_now < 0.5e-22:  # sqrt(2*merit) < 1e-11
                converged = True
                break

            if bounded:
                step, delta_multiplier = newton_step_boxed(
                    hessian(grid, eta, eps, np.maximum(rate, slope_floor), cross, risk),
                    residual, grid.weights, rate, lower_mask, upper_mask, ceiling,
                )
            else:
                step = np.linalg.solve(
                    _bordered(grid, hessian(grid, eta, eps, rate, cross, risk)), residual
                )
                delta_multiplier = step[grid.size]
                step = step[: grid.size]

            # v2.1 fix 2: Armijo sufficient-decrease on the squared residual
            # for the bordered Newton direction, grad(phi)·(-step) = -||F||²
            # < 0, so small damping is always accepted — this removes the
            # damping-floor stall of the previous plain-norm comparison
            phi_ref = 0.5 * residual_norm ** 2

            damping = 1.0
            if bounded:
                free_idx = np.flatnonzero(~(lower_mask | upper_mask))

                def merit_sq(v, m):
                    full = floored_stationarity(v, m)
                    return 0.5 * float(np.linalg.norm(
                        np.r_[full[free_idx], full[-1]])) ** 2
            else:
                def merit_sq(v, m):
                    full = stationarity(grid, eta, eps, v, m, share,
                                        cross, risk, offset)
                    return 0.5 * float(np.linalg.norm(full)) ** 2

            phi_ref = merit_sq(rate, multiplier)
            while damping > 1e-10:
                trial_rate = rate - damping * step
                if bounded and ceiling is not None:
                    trial_rate = np.clip(trial_rate, 0.0, ceiling)
                elif bounded:
                    trial_rate = np.maximum(trial_rate, 0.0)
                trial_multiplier = multiplier - damping * delta_multiplier
                if bounded:
                    deficit = share - float(grid.weights @ trial_rate)
                    w_free = grid.weights[free_idx]
                    trial_rate[free_idx] += deficit * w_free / float(w_free @ w_free)
                    if ceiling is not None:
                        trial_rate = np.clip(trial_rate, 0.0, ceiling)
                    else:
                        trial_rate = np.maximum(trial_rate, 0.0)
                # Armijo: phi(trial) <= (1 - c·damping)·phi(ref)
                if merit_sq(trial_rate, trial_multiplier) <= (1 - 2e-4 * damping) * phi_ref:
                    break
                damping /= 2.0

            # v2.1 fix 4: bottom-out safeguard — if damping hit the floor,
            # take the projected (clipped) step anyway; the active-set update
            # below will absorb any bound violations
            if damping <= 1e-10:
                damping = 1.0  # take full step; clip handles feasibility

            rate_old = rate.copy()
            rate = rate - damping * step
            if bounded:
                if ceiling is not None:
                    rate = np.clip(rate, 0.0, ceiling)
                else:
                    rate = np.maximum(rate, 0.0)
                deficit = share - float(grid.weights @ rate)
                free_idx = np.flatnonzero(~(lower_mask | upper_mask))
                w_free = grid.weights[free_idx]
                rate[free_idx] += deficit * w_free / float(w_free @ w_free)
                rate = np.clip(rate, 0.0, ceiling) if ceiling is not None \
                    else np.maximum(rate, 0.0)
            multiplier = multiplier - damping * delta_multiplier

            # v2.3 P0-3: actual displacement AFTER clip + redistribution
            # (frozen fixed points have zero actual displacement but nonzero
            # nominal steps; a redistribution jump with a tiny nominal step
            # is also caught now)
            displacement_norm = float(np.linalg.norm(rate - rate_old))

            if bounded:
                # activate on primal contact
                hits_low = rate <= 1e-12
                hits_up = (
                    rate >= ceiling - 1e-12 if ceiling is not None
                    else np.zeros_like(hits_low)
                )
                lower_mask |= hits_low
                if ceiling is not None:
                    upper_mask |= hits_up
                # v2.2 fix c: dual release on the UNFLOORED gradient. The
                # floored gradient at v=0 gives impact_slope ~ 1.6e4, so mu
                # at frozen nodes is always hugely positive and the release
                # condition mu < -tol can never fire — nodes freeze
                # irreversibly and the deficit redistribution creates spike
                # schedules. The unfloored gradient is the correct KKT
                # multiplier signal (it is finite in the limit v→0 for the
                # release direction because only the sign matters).
                mu = stationarity(
                    grid, eta, eps, rate, multiplier, share, cross, risk, offset
                )[: grid.size]
                release_low = lower_mask & ~hits_up & (mu < -1e-9)
                release_up = upper_mask & ~hits_low & (mu > 1e-9)
                if release_low.any():
                    worst = np.flatnonzero(release_low)[np.argmin(mu[release_low])]
                    lower_mask[worst] = False
                if release_up.any():
                    worst = np.flatnonzero(release_up)[np.argmax(mu[release_up])]
                    upper_mask[worst] = False

            # v2.2 fix b: secondary convergence on the ACTUAL displacement
            # (after clip + redistribution), not the nominal step — frozen
            # fixed points have zero displacement but nonzero nominal steps
            if displacement_norm < 1e-13 * (1.0 + np.linalg.norm(rate)):
                converged = True
                break

        if not converged:
            raise RuntimeError(f"Newton corrector failed to converge at eps={eps}")
        if monitor_hessian:
            hessian_trace.append(
                (eps, reduced_hessian_min(grid, eta, eps, rate, tangent=tangent,
                                          cross=cross, risk=risk))
            )

    # v2.2 fix d: quality gate — a converged schedule that deviates wildly
    # from the eps=0 linear baseline signals a degenerate solve (the v2.1
    # spike pathology). We compute the cost functional at eps=0 and compare.
    if offset is not None:
        # re-solve the alpha-free baseline for comparison
        plain_rhs = cross + grid.risk_vector(risk, share)
        plain_particular = np.linalg.solve(operator, plain_rhs)
        plain_direction = np.linalg.solve(operator, np.ones(grid.size))
        plain_mult = (share - grid.weights @ plain_particular) / (grid.weights @ plain_direction)
        plain_rate = plain_particular + plain_mult * plain_direction
        plain_cost = float(grid.weights @ (plain_rate * (operator @ plain_rate))) / 2.0
        # the actual solution's cost under the same (alpha-free) operator
        actual_cost = float(grid.weights @ (rate * (operator @ rate))) / 2.0
        if actual_cost > 2.5 * plain_cost + 1e-6:
            raise RuntimeError(
                f"Quality gate: solution cost {actual_cost:.4f} > 2.5x alpha-free "
                f"baseline {plain_cost:.4f} — degenerate solve (rate max {rate.max():.1f} "
                f"vs baseline max {plain_rate.max():.1f})"
            )

    return rate, multiplier, np.asarray(hessian_trace, dtype=float).reshape(-1, 2)


def reduced_hessian_min(grid, eta, eps, rate, tangent=None, cross=None, risk=0.0):
    """Smallest eigenvalue of the Hessian restricted to the constraint tangent space.

    The similarity transform by the square root of the quadrature weights
    makes the weighted Hessian symmetric, so the reduced eigenvalues are real.
    """
    root_weights = np.sqrt(grid.weights)
    scaled = root_weights[:, None] * hessian(grid, eta, eps, rate, cross, risk) / root_weights[
        None, :
    ]
    scaled = 0.5 * (scaled + scaled.T)
    if tangent is None:
        tangent = null_space(root_weights[None, :])
    return float(np.linalg.eigvalsh(tangent.T @ scaled @ tangent).min())


def residual_norm(grid, eta, eps, rate, multiplier, share, cross=None, risk=0.0):
    return float(
        np.linalg.norm(
            stationarity(grid, eta, eps, rate, multiplier, share, cross, risk)
        )
    )
