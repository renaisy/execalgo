"""VWAP benchmark: cross-term structure, optimality, limits, simulator consistency."""

import numpy as np
import pytest

from execalgo import (
    ISAlgorithm,
    VWAPAlgorithm,
    profile_schedule,
    twap_schedule,
)
from execalgo.simulator import MarketSim
from execalgo.vwap import benchmark_cross, from_samples, twap, u_shaped


def test_profile_normalization(params):
    grid = params.make_grid()
    for profile in (u_shaped(grid), twap(grid), u_shaped(grid, sharpness=4.0, floor=0.5)):
        assert grid.weights @ profile == pytest.approx(1.0, abs=1e-14)
        assert profile.min() > 0


def test_cross_term_structure(params):
    """q = X (Q p) is nonnegative and decreasing: early impact persists into
    more market-volume-weighted benchmark time."""
    grid = params.make_grid()
    cross = benchmark_cross(grid, twap(grid), params.share)
    assert cross.min() >= 0
    assert np.all(np.diff(cross) <= 1e-12)
    assert cross[0] > cross[-1]


def test_from_samples_recovers_density(params):
    grid = params.make_grid()
    times = np.linspace(0.0, params.horizon, 24)
    volumes = 1.0 + 3.0 * np.exp(-8.0 * times)
    profile = from_samples(grid, times, volumes)
    assert grid.weights @ profile == pytest.approx(1.0, abs=1e-12)
    assert profile[0] > profile[grid.size // 2]  # still front-heavy


def test_vwap_optimal_wins_on_vwap_metric(params):
    grid = params.make_grid()
    profile = u_shaped(grid, sharpness=8.0)
    schedules = [
        ISAlgorithm(params).solve(),
        VWAPAlgorithm(params, profile).solve(),
        twap_schedule(params),
        profile_schedule(params, profile),
    ]
    vwap_costs = [s.expected_costs(profile)["vwap"] for s in schedules]
    assert vwap_costs[1] == min(vwap_costs)          # VWAP algorithm best on its own benchmark
    assert vwap_costs[1] < vwap_costs[0]             # beats the IS trajectory
    assert vwap_costs[1] < vwap_costs[2]             # beats TWAP
    assert vwap_costs[1] < vwap_costs[3]             # beats naive VWAP-profile

    is_costs = [s.expected_costs(profile)["is"] for s in schedules]
    assert is_costs[0] == min(is_costs)              # IS algorithm best on arrival price


def test_vwap_solution_front_loads_more(params):
    grid = params.make_grid()
    profile = u_shaped(grid, sharpness=8.0)
    is_fraction = ISAlgorithm(params).solve().front_load_fraction()
    vwap_fraction = VWAPAlgorithm(params, profile).solve().front_load_fraction()
    assert 0.5 < is_fraction < vwap_fraction


def test_long_only_enforcement(params):
    """The unconstrained VWAP optimum dips negative at the close (benchmark
    gaming: selling into the close pushes the VWAP benchmark down). Schedules
    handed to execution must be long-only with the block size preserved."""
    grid = params.make_grid()
    profile = u_shaped(grid, sharpness=8.0)
    for method in ("asymptotic", "continuation"):
        schedule = VWAPAlgorithm(params, profile).solve(method=method)
        assert schedule.rate.min() >= 0.0
        assert grid.weights @ schedule.rate == pytest.approx(params.share, abs=1e-12)
        edges, quantities = schedule.slices(30)
        assert (quantities >= 0).all()
    # clamping preserves the cost ordering that makes the algorithm optimal
    costs = VWAPAlgorithm(params, profile).solve().expected_costs(profile)["vwap"]
    assert costs < twap_schedule(params).expected_costs(profile)["vwap"]


def test_vwap_continuation_certificate(small_params):
    """The asymptotic VWAP solution nearly satisfies the nonlinear optimality system."""
    params = small_params
    grid = params.make_grid()
    profile = u_shaped(grid, sharpness=8.0)
    algo = VWAPAlgorithm(params, profile)
    asymptotic = algo.solve()
    reference = algo.solve(method="continuation")
    # the certificate is the free-set merit of the RETURNED (clamped)
    # schedule: the continuation itself converges to ~1e-15, and the clamp's
    # renormalization adds O(0.1%) free-set non-constancy
    assert reference.stationarity_residual < 5e-3
    assert grid.norm(asymptotic.rate - reference.rate) / grid.norm(reference.rate) < 0.03


def test_slices_sum_to_share(params):
    grid = params.make_grid()
    profile = u_shaped(grid, sharpness=8.0)
    for schedule in (
        ISAlgorithm(params).solve(),
        VWAPAlgorithm(params, profile).solve(),
        twap_schedule(params),
    ):
        edges, quantities = schedule.slices(30)
        assert quantities.sum() == pytest.approx(params.share, abs=1e-12)
        assert (quantities > 0).all()


def test_simulator_matches_expectations(small_params):
    """MC means agree with the deterministic model within sampling noise."""
    params = small_params
    grid = params.make_grid()
    profile = u_shaped(grid, sharpness=8.0)
    sim = MarketSim(profile, participation=20.0, sigma=0.30, seed=11)
    for schedule in (
        ISAlgorithm(params).solve(),
        VWAPAlgorithm(params, profile).solve(),
        profile_schedule(params, profile),
    ):
        run = sim.run(schedule, n_slices=60, n_paths=8000)
        standard_error = run["is_paths"].std() / np.sqrt(8000)
        assert abs(run["is_paths"].mean() - run["expected_is"]) < 6 * standard_error + 0.01
        vwap_se = run["vwap_paths"].std() / np.sqrt(8000)
        assert abs(run["vwap_paths"].mean() - run["expected_vwap"]) < 6 * vwap_se + 0.005
