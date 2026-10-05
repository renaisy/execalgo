"""Series machinery: share conservation, degenerate cross term, convergence."""

import numpy as np
import pytest

from execalgo.branch import continue_branch
from execalgo.discretize import ConstrainedSolver
from execalgo.series import coefficients, optimal_truncation_order, partial_sum
from execalgo.vwap import benchmark_cross, u_shaped


def _engine(params, cross=None):
    grid = params.make_grid()
    return grid, ConstrainedSolver(grid, params.eta), cross


def test_share_conservation_every_order(small_params):
    """Only v_0 carries the block; every correction integrates to zero."""
    params = small_params
    grid, solver, _ = _engine(params)
    series = coefficients(grid, solver, params.share, params.max_order)
    for m in range(1, params.max_order + 1):
        assert grid.weights @ series[m] == pytest.approx(0.0, abs=1e-12)


def test_zero_cross_equals_none(small_params):
    """cross = 0 vector is the IS objective, identical to cross = None."""
    params = small_params
    grid, solver, _ = _engine(params)
    cross = benchmark_cross(grid, u_shaped(grid), params.share)
    plain = coefficients(grid, solver, params.share, params.max_order)
    zeroed = coefficients(grid, solver, params.share, params.max_order, 0.0 * cross)
    for m in range(params.max_order + 1):
        assert np.allclose(plain[m], zeroed[m], rtol=0, atol=1e-14)


def test_partial_sums_converge_inside_radius(small_params):
    """For eps below the convergence radius the series converges to the branch."""
    params = small_params
    grid, solver, _ = _engine(params)
    series = coefficients(grid, solver, params.share, params.max_order)
    eps = 0.3  # < radius ~0.45
    rate, _, _ = continue_branch(grid, params.eta, params.share, eps)
    low = partial_sum(series, eps, 4)
    high = partial_sum(series, eps, params.max_order)
    assert grid.norm(high - rate) < grid.norm(low - rate)
    assert grid.norm(high - rate) < 1e-3


def test_optimal_truncation_order_is_small_term(small_params):
    """Beyond the radius, the smallest-term order is single-digit, and the
    asymptotically truncated series is far closer to the branch than v_0."""
    params = small_params
    grid, solver, _ = _engine(params)
    series = coefficients(grid, solver, params.share, params.max_order)
    order = optimal_truncation_order(grid, series, params.eps)
    assert 5 <= order <= 15
    rate, _, _ = continue_branch(grid, params.eta, params.share, params.eps)
    truncated = partial_sum(series, params.eps, order)
    assert grid.norm(truncated - rate) < 0.05 * grid.norm(series[0] - rate)
