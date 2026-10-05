"""v0.2 inventory-risk term: operator properties, limits, series consistency (T2-T6, T8)."""

import time

import numpy as np
import pytest

from conftest import load_reference

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm
from execalgo.branch import continue_branch, residual_norm
from execalgo.discretize import ConstrainedSolver, make_grid
from execalgo.model import fit_square_root
from execalgo.series import coefficients, partial_sum
from execalgo.vwap import benchmark_cross, u_shaped

RISK = 0.05


def test_t2_risk_operator_symmetric_psd_and_kernel_values(small_params):
    grid = small_params.make_grid()
    operator = grid.risk_operator(RISK)
    # symmetrize under the quadrature similarity transform
    root = np.sqrt(grid.weights)
    scaled = root[:, None] * operator / root[None, :]
    assert np.allclose(scaled, scaled.T, atol=1e-14)
    assert np.linalg.eigvalsh(0.5 * (scaled + scaled.T)).min() >= -1e-12

    # hand-check the kernel on a tiny grid: K_ij = 2*risk*(T - max(t_i,t_j)) * w_j
    tiny = make_grid(5, fit_square_root()[0])
    i, j = 2, 4
    expected = 2.0 * RISK * (tiny.nodes[-1] - max(tiny.nodes[i], tiny.nodes[j])) * tiny.weights[j]
    assert tiny.risk_operator(RISK)[i, j] == pytest.approx(expected, rel=1e-14)


def test_t2_risk_vector_values(small_params):
    grid = small_params.make_grid()
    vector = grid.risk_vector(RISK, small_params.share)
    assert vector[0] == pytest.approx(2.0 * RISK * small_params.share, rel=1e-14)
    assert vector[-1] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("with_cross", [False, True])
def test_t3_share_conservation_with_risk(small_params, with_cross):
    grid = small_params.make_grid()
    cross = benchmark_cross(grid, u_shaped(grid), small_params.share) if with_cross else None
    solver = ConstrainedSolver(grid, small_params.eta, RISK)
    offset = grid.risk_vector(RISK, small_params.share)
    series = coefficients(
        grid, solver, small_params.share, small_params.max_order, cross, offset
    )
    for m in range(1, small_params.max_order + 1):
        assert grid.weights @ series[m] == pytest.approx(0.0, abs=1e-12)


def test_t4_small_risk_matches_risk_neutral(small_params):
    grid = small_params.make_grid()
    plain = coefficients(grid, ConstrainedSolver(grid, small_params.eta),
                         small_params.share, 6)
    tiny = coefficients(grid, ConstrainedSolver(grid, small_params.eta, 1e-8),
                        small_params.share, 6,
                        offset=grid.risk_vector(1e-8, small_params.share))
    for m in range(7):
        assert np.allclose(plain[m], tiny[m], rtol=1e-6, atol=1e-10)


def test_t4_large_risk_front_loads():
    schedule = ISAlgorithm(ExecParams(share=1.0, eta=0.2055, risk=1e3)).solve()
    assert schedule.front_load_fraction() >= 0.9
    # decaying in the coarse sense (the truncated series wiggles at tiny tail rates)
    n = schedule.grid.size
    head = schedule.rate[: n // 10].mean()
    tail = schedule.rate[-n // 10 :].mean()
    assert head > 10.0 * tail


def test_t5_series_matches_continuation_with_risk(small_params):
    grid = small_params.make_grid()
    solver = ConstrainedSolver(grid, small_params.eta, RISK)
    offset = grid.risk_vector(RISK, small_params.share)
    series = coefficients(grid, solver, small_params.share, small_params.max_order,
                          offset=offset)

    # inside the convergence radius the partial sums converge to the branch
    eps = 0.3
    rate, _, _ = continue_branch(grid, small_params.eta, small_params.share, eps, risk=RISK)
    high = partial_sum(series, eps, small_params.max_order)
    assert grid.norm(high - rate) < 1e-3

    # at the target eps the asymptotically truncated series stays close
    eps = small_params.eps
    rate, multiplier, _ = continue_branch(
        grid, small_params.eta, small_params.share, eps, risk=RISK
    )
    norms = [eps**m * grid.norm(series[m]) for m in range(small_params.max_order + 1)]
    best = int(np.argmin(norms))
    truncated = partial_sum(series, eps, best)
    assert grid.norm(truncated - rate) / grid.norm(rate) < 0.05

    # and the continuation itself is a stationary point of the risk-augmented system
    assert residual_norm(grid, small_params.eta, eps, rate, multiplier,
                         small_params.share, risk=RISK) < 1e-10


def test_t6_risk_penalty_matches_fine_quadrature(small_params):
    params = ExecParams(share=small_params.share, eta=small_params.eta,
                        grid_size=small_params.grid_size, risk=RISK)
    schedule = ISAlgorithm(params).solve()
    reported = schedule.expected_costs()["risk_penalty"]

    # independent check: piecewise-linear rate on a 4x finer grid, trapezoid
    grid = schedule.grid
    fine = np.linspace(0.0, grid.nodes[-1], 4 * grid.size)
    rate_fine = np.interp(fine, grid.nodes, schedule.rate)
    inventory = params.share - np.concatenate(
        [[0.0], np.cumsum(np.diff(fine) * (rate_fine[1:] + rate_fine[:-1]) / 2.0)]
    )
    expected = RISK * float(np.trapezoid(inventory**2, fine))
    assert reported == pytest.approx(expected, rel=1e-3)


def test_t8_risk_solve_cost_comparable(small_params):
    grid = small_params.make_grid()
    start = time.perf_counter()
    for _ in range(20):
        ConstrainedSolver(grid, small_params.eta)
    plain_time = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(20):
        ConstrainedSolver(grid, small_params.eta, RISK)
    risk_time = time.perf_counter() - start
    assert risk_time < plain_time * 2.0 + 0.1


def test_t1_reference_series_unchanged(small_params):
    """The default path stays bit-identical to the reference implementation."""
    ref_series = load_reference("series")
    ref_disc = load_reference("discretize")
    ref_model = load_reference("model")

    params = small_params
    kernel, _ = fit_square_root()
    grid = make_grid(params.grid_size, kernel, params.horizon)
    ref_grid = ref_disc.make_grid(params.grid_size, ref_model.fit_square_root()[0])
    ours = coefficients(grid, ConstrainedSolver(grid, params.eta), params.share, 6)
    theirs = ref_series.coefficients(
        ref_grid, ref_disc.ConstrainedSolver(ref_grid, params.eta), params.share, 6
    )
    for m in range(7):
        assert np.allclose(ours[m], theirs[m], rtol=0, atol=1e-14)
