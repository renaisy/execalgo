"""v0.3 rolling re-planner: degeneracy consistency, carry offset, cache reuse (T9-T11)."""

import numpy as np
import pytest

from execalgo.model import fit_square_root
from execalgo.replanner import (
    N_MINUTES,
    ReplanConfig,
    RollingPlanner,
    impact_carry_offset,
)


def u_shaped_grid():
    """A smooth front-loaded 225-minute profile, U-shape in trading time."""
    minutes = np.arange(N_MINUTES)
    t = (minutes + 0.5) / N_MINUTES
    density = np.exp(-4.0 * t) + 0.6 * np.exp(-4.0 * (1.0 - t)) + 0.4
    return density / density.sum()


@pytest.fixture(scope="module")
def kernel():
    return fit_square_root()[0]


def _flat_volume_day(profile, level=1.0):
    """A synthetic day whose realized minute volumes follow the train profile."""
    return profile * level * 1000.0


def _day_cost(kernel, profile, frac, eta=0.2055):
    """Minute-level model cost of a fraction schedule (day-time, lambda=1)."""
    n = N_MINUTES
    h = 1.0 / n
    rate = frac * n
    centers = (np.arange(n) + 0.5) * h
    lag = centers[:, None] - centers[None, :]
    past = np.tril(kernel(np.abs(lag)), k=-1)
    future = np.triu(kernel(np.abs(lag)), k=1)
    root = np.sqrt(rate)
    return (
        eta * float(rate @ rate) * h
        + float(rate @ (past @ root)) * h
        - float(root @ (future @ profile)) * h
    )


def test_t9_no_new_information_matches_single_shot(kernel):
    """Time consistency under unchanged information.

    The first replan solves the identical problem (bit-level agreement).
    Later replans drift along numerically flat directions of the objective —
    the front-loading direction has cost curvature of order 1e-4 — so the
    honest consistency statement is cost equivalence plus bounded drift,
    with the drift shrinking as the replan interval grows.
    """
    profile = u_shaped_grid()
    day = _flat_volume_day(profile)
    one_shot = RollingPlanner(
        0.2055, kernel, profile,
        config=ReplanConfig(interval_minutes=N_MINUTES),
    ).execute(day)

    for interval, drift_bound, cost_bound in ((15, 0.45, 0.02), (45, 0.25, 0.005)):
        rolling = RollingPlanner(
            0.2055, kernel, profile,
            config=ReplanConfig(interval_minutes=interval),
        )
        adaptive = rolling.execute(day)
        assert adaptive.sum() == pytest.approx(1.0)
        # the active carry offset trips the positivity guard on some replans;
        # since v1.1 those fall back to the CONSTRAINED continuation (still
        # optimal under the long-only box), so only the count cap matters
        assert rolling.fallbacks <= N_MINUTES // interval
        if interval == 15:
            assert np.allclose(one_shot[:15], adaptive[:15], atol=1e-10)
        assert np.abs(one_shot - adaptive).sum() < drift_bound
        cost_static = _day_cost(kernel, profile, one_shot)
        cost_rolling = _day_cost(kernel, profile, adaptive)
        assert abs(cost_rolling - cost_static) / cost_static < cost_bound


def test_t9_carry_offset_enters_subproblem(kernel):
    """The I_past offset makes the static tail stationary for the remaining problem.

    Time consistency (principle of optimality): the static solution's tail,
    mapped into the sub-window canonical variables, must satisfy the
    remaining-window optimality system WITH the carry offset — and fails it
    by the omitted gradient term without. Also asserts the offset wiring is
    live (the two solves differ substantially). Found by the v1.1 review:
    the offset target times were wired in minutes instead of day-normalized
    time, making the term nearly constant and shape-inert since v0.3.
    """
    profile = u_shaped_grid()
    day = _flat_volume_day(profile)
    one_shot = RollingPlanner(
        0.2055, kernel, profile,
        config=ReplanConfig(interval_minutes=N_MINUTES),
    ).execute(day)

    planners = {
        True: RollingPlanner(0.2055, kernel, profile,
                             config=ReplanConfig(interval_minutes=15)),
        False: RollingPlanner(0.2055, kernel, profile,
                              config=ReplanConfig(interval_minutes=15,
                                                  carry_impact=False)),
    }
    fractions = np.zeros(N_MINUTES)
    fractions[:15] = one_shot[:15]
    phi = 1.0 - fractions.sum()
    h = (N_MINUTES - 15) / N_MINUTES

    solves = {}
    for carry, planner in planners.items():
        planner._today = None
        segment, rate, sub_grid, cross = planner._solve_segment(
            day, fractions, 15, N_MINUTES - 15, phi, return_rate=True)
        solves[carry] = (rate, sub_grid, cross)

    # wiring live: the two solves differ substantially
    assert np.abs(solves[True][0] - solves[False][0]).max() > 0.1

    # static tail mapped into the sub-window canonical rate
    sub_grid, cross = solves[True][1], solves[True][2]
    v_tail = one_shot[15:] * N_MINUTES            # day-time rate
    u_static = np.interp(
        sub_grid.nodes * h,
        (np.arange(N_MINUTES - 15) + 0.5) / N_MINUTES,
        v_tail * h / phi,
    )
    from execalgo.replanner import impact_carry_offset
    from execalgo.branch import stationarity

    eta_j = 0.2055 * round(float(np.sqrt(phi)) / 0.05) * 0.05 / h**1.5
    u_safe = np.maximum(u_static, 1e-9)
    base = stationarity(sub_grid, eta_j, 0.5, u_safe, 0.0, 1.0, cross=cross)
    base = base[: sub_grid.size]
    offset = impact_carry_offset(
        kernel, fractions, 15, 15 / N_MINUTES + sub_grid.nodes * h, phi, h)

    def spread(gradient):
        return float(np.max(np.abs(gradient - gradient.mean())))

    # the objective's linear term +int v*I_past puts +I_past in the gradient,
    # so stationarity requires base + I_past ≈ const (v1.1.1 review:
    # the code previously wired it with the opposite sign into the RHS)
    with_offset = spread(base + offset)
    without_offset = spread(base)
    assert with_offset < 0.3 * without_offset      # measured ~0.10 vs 0.53
    assert np.abs(offset).max() > 0.1              # the term is material

    # and the solve WITH carry must be closer to the static tail than without
    # (the correct sign makes the re-solve consistent with the static optimum;
    # the wrong sign moves it further away)
    u_with = solves[True][0]
    u_without = solves[False][0]
    assert np.abs(u_with - u_safe).sum() < np.abs(u_without - u_safe).sum()


def test_t10_impact_carry_offset_hand_check(kernel):
    """Two executed minutes, hand-computed offset on a small target grid."""
    executed = np.zeros(N_MINUTES)
    executed[0] = 0.02
    executed[1] = 0.03
    phi, h = 0.95, 210.0 / N_MINUTES
    targets = np.array([0.05, 0.5, 0.95])

    displacement = np.sqrt(np.array([0.02, 0.03]) * N_MINUTES) / N_MINUTES
    centers = np.array([0.5, 1.5]) / N_MINUTES
    expected = np.zeros(3)
    for i, t in enumerate(targets):
        total = 0.0
        for j, c in enumerate(centers):
            if t > c:
                total += kernel(t - c) * displacement[j]
        expected[i] = total / np.sqrt(phi * h)

    offset = impact_carry_offset(kernel, executed, 2, targets, phi, h)
    np.testing.assert_allclose(offset, expected, rtol=1e-12)


def test_t11_lu_cache_reuse_across_days(kernel):
    profile = u_shaped_grid()
    planner = RollingPlanner(
        0.2055, kernel, profile, train_mean_volume=1000.0,
        config=ReplanConfig(interval_minutes=15),
    )
    day = _flat_volume_day(profile)
    planner.execute(day)
    cache_after_first = len(planner._cache)
    replans_after_first = planner.replans
    assert cache_after_first > 0

    # an identical day hits the same (remaining, gamma bucket) keys: no new LUs
    planner.execute(day)
    assert len(planner._cache) == cache_after_first
    assert planner.replans == 2 * replans_after_first


def test_level_conditioning_changes_plan(kernel):
    """A heavy-volume day must shift the plan (s_vol channel is alive)."""
    profile = u_shaped_grid()
    planner = RollingPlanner(
        0.2055, kernel, profile, train_mean_volume=1000.0,
        config=ReplanConfig(interval_minutes=15),
    )
    normal = planner.execute(_flat_volume_day(profile))
    heavy = planner.execute(_flat_volume_day(profile, level=1.5))
    assert np.abs(normal - heavy).sum() > 1e-3
    assert heavy.sum() == pytest.approx(1.0)
