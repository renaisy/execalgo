"""v1.0 constrained solver: active-set KKT correctness (T18)."""

import numpy as np
import pytest

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm
from execalgo.constrained import solve_box_linear
from execalgo.discretize import ConstrainedSolver, make_grid
from execalgo.model import fit_square_root
from execalgo.series import coefficients
from execalgo.vwap import benchmark_cross, u_shaped


@pytest.fixture(scope="module")
def grid():
    return make_grid(151, fit_square_root()[0])


def test_t18_interior_matches_plain_border(grid):
    """No active bounds: identical to the unconstrained bordered solve."""
    share, eta = 1.0, 0.2055
    plain, multiplier_plain = ConstrainedSolver(grid, eta).solve(np.zeros(grid.size), share)
    box = solve_box_linear(grid.cost_operator(eta), grid.weights,
                           np.zeros(grid.size), share, upper=None)
    assert np.allclose(box.rate, plain, rtol=0, atol=1e-10)
    assert box.multiplier == pytest.approx(multiplier_plain, abs=1e-10)
    assert not box.active_lower.any()
    assert box.kkt_residual < 1e-8


def test_t18_long_only_structural(grid):
    """The benchmark-gaming regime (negative unconstrained baseline) now
    solves structurally: feasible, share-exact, KKT-clean."""
    share, eta = 1.0, 0.10                     # low eta pushed the tail negative in v0.2
    cross = benchmark_cross(grid, u_shaped(grid, sharpness=8.0), share)
    operator = grid.cost_operator(eta)
    unconstrained = ConstrainedSolver(grid, eta).solve(cross, share)[0]
    assert unconstrained.min() < 0.0           # the regime is live in this cell

    box = solve_box_linear(operator, grid.weights, cross, share, upper=None)
    assert box.rate.min() >= -1e-12
    assert grid.weights @ box.rate == pytest.approx(share, abs=1e-10)
    assert box.kkt_residual < 1e-8
    assert box.active_lower.any()


def test_t18_upper_bound_binding(grid):
    share, eta = 1.0, 0.2055
    upper = np.full(grid.size, 1.6)            # binds against the front-load
    box = solve_box_linear(grid.cost_operator(eta), grid.weights,
                           np.zeros(grid.size), share, upper=upper)
    assert box.rate.max() <= 1.6 + 1e-9
    assert grid.weights @ box.rate == pytest.approx(share, abs=1e-10)
    assert box.kkt_residual < 1e-8
    assert box.active_upper.any()


def test_t18_infeasible_raises(grid):
    with pytest.raises(ValueError):
        solve_box_linear(grid.cost_operator(0.2055), grid.weights,
                         np.zeros(grid.size), 1.0, upper=np.full(grid.size, 0.001))


def test_t18_constrained_continuation_respects_cap(grid):
    cap = np.full(grid.size, 1.8)
    params = ExecParams(share=1.0, eta=0.2055, eps=0.5,
                        grid_size=grid.size, max_order=20, upper_rate=cap)
    schedule = ISAlgorithm(params).solve(method="continuation")
    assert schedule.rate.max() <= 1.8 + 1e-8
    assert schedule.grid.weights @ schedule.rate == pytest.approx(1.0, abs=1e-8)
    assert (schedule.rate >= -1e-12).all()


def test_t18_cap_slack_at_baseline_binding_at_target(grid):
    """P0 regression: a ceiling slack at eps=0 must still bind at eps=0.5.

    The square-root optimum front-loads harder than the linear baseline, so
    a cap between the two peaks is reachable; the continuation must respect
    it (the boxed machinery stays armed whenever `upper` is given).
    """
    import dataclasses

    eta = 0.2055
    baseline = ConstrainedSolver(grid, eta).solve(np.zeros(grid.size), 1.0)[0]
    free = ISAlgorithm(ExecParams(share=1.0, eta=eta, eps=0.5,
                                  grid_size=grid.size,
                                  max_order=20)).solve(method="continuation")
    cap_level = float((baseline.max() + free.rate.max()) / 2.0)
    assert baseline.max() < cap_level < free.rate.max()   # the regime is live

    capped = ISAlgorithm(dataclasses.replace(
        free.params, upper_rate=np.full(grid.size, cap_level)
    )).solve(method="continuation")
    assert capped.rate.max() <= cap_level + 1e-8
    assert grid.weights @ capped.rate == pytest.approx(1.0, abs=1e-8)
    assert capped.rate.max() < free.rate.max()            # the cap actually binds


def test_gaming_regime_now_solves_via_contination(grid):
    """v0.2's ValueError path: the VWAP optimum in the gaming regime is now
    executable through the constrained continuation."""
    profile = u_shaped(grid, sharpness=8.0)
    params = ExecParams(share=1.0, eta=0.10, eps=0.5, grid_size=grid.size, max_order=20)
    with pytest.raises(ValueError):
        VWAPAlgorithm(params, profile).solve()          # asymptotic still guards
    schedule = VWAPAlgorithm(params, profile).solve(method="continuation")
    assert schedule.rate.min() >= -1e-12
    assert grid.weights @ schedule.rate == pytest.approx(1.0, abs=1e-8)


def test_series_unchanged_without_bounds(grid):
    """Regression guard: the interior asymptotic path matches the reference
    implementation bit-for-bit (the v1.0 changes must stay inert here)."""
    from conftest import load_reference

    ref_series = load_reference("series")
    ref_disc = load_reference("discretize")
    ref_model = load_reference("model")

    ours = coefficients(grid, ConstrainedSolver(grid, 0.2055), 1.0, 8)
    ref_grid = ref_disc.make_grid(grid.size, ref_model.fit_square_root()[0])
    theirs = ref_series.coefficients(
        ref_grid, ref_disc.ConstrainedSolver(ref_grid, 0.2055), 1.0, 8
    )
    for m in range(9):
        assert np.allclose(ours[m], theirs[m], rtol=0, atol=1e-14)
