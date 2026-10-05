"""Acceptance: the IS path reproduces the reference implementation and the paper's numbers."""

import numpy as np
import pytest

from conftest import load_reference

from execalgo.branch import continue_branch
from execalgo.discretize import ConstrainedSolver, make_grid
from execalgo.model import fit_square_root
from execalgo.series import coefficients, optimal_truncation_order, partial_sum

from execalgo.diagnostics import front_load_split, radius_estimate, truncation_study


def test_kernel_fit_deviation():
    _, deviation = fit_square_root()
    assert deviation == pytest.approx(0.0173, abs=0.0015)  # paper: ~1.7%


def test_series_matches_reference_bitwise(small_params):
    ref_series = load_reference("series")
    ref_disc = load_reference("discretize")
    ref_model = load_reference("model")

    params = small_params
    kernel, _ = fit_square_root()
    grid = make_grid(params.grid_size, kernel, params.horizon)
    ref_grid = ref_disc.make_grid(params.grid_size, ref_model.fit_square_root()[0])

    ours = coefficients(grid, ConstrainedSolver(grid, params.eta), params.share, params.max_order)
    theirs = ref_series.coefficients(
        ref_grid, ref_disc.ConstrainedSolver(ref_grid, params.eta), params.share, params.max_order
    )
    for m in range(params.max_order + 1):
        assert np.allclose(ours[m], theirs[m], rtol=0, atol=1e-14)


def test_paper_diagnostics(params):
    grid = make_grid(params.grid_size, params.resolve_kernel(), params.horizon)
    solver = ConstrainedSolver(grid, params.eta)
    series = coefficients(grid, solver, params.share, params.max_order)
    optimizer, _, _ = continue_branch(grid, params.eta, params.share, params.eps)

    study = truncation_study(grid, series, optimizer, params.eps)
    assert 7 <= study["best_order"] <= 11          # paper: m* = 9
    assert study["improvement"] > 30.0             # paper: ~43x
    assert optimal_truncation_order(grid, series, params.eps) in range(7, 12)

    radius = radius_estimate(grid, series)
    assert radius["radius"] == pytest.approx(0.446, abs=0.01)   # paper: ~0.446
    assert radius["growth"] == pytest.approx(2.24, abs=0.02)    # paper: ~2.24

    split = front_load_split(grid, series[1])
    assert split["is_front_loaded"]                 # Theorem 5.5: first-order correction front-loads
    assert split["front"] == pytest.approx(-split["back"], rel=1e-6)  # v1 is antisymmetric


def test_continuation_stationarity(params):
    grid = make_grid(params.grid_size, params.resolve_kernel(), params.horizon)
    rate, multiplier, _ = continue_branch(grid, params.eta, params.share, params.eps)
    from execalgo.branch import residual_norm

    assert residual_norm(grid, params.eta, params.eps, rate, multiplier, params.share) < 1e-10
    assert grid.weights @ rate == pytest.approx(params.share, abs=1e-12)
    assert rate.min() > 0
