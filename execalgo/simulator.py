"""Evaluation of an execution schedule: deterministic expectations and Monte Carlo.

The deterministic layer prices the model exactly as the optimizer sees it:
fills at arrival price plus impact residual plus slippage, so the expected
slippage of the optimal schedule is reproducible analytically.

The Monte Carlo layer adds an exogenous mid-price noise (Brownian, by
default) on top of the same impact dynamics and prices both benchmarks
path by path:

* realized IS per share      = average fill price - S(0);
* realized VWAP slippage     = average fill price - market VWAP,

where the market VWAP is the volume-weighted average of the *same* noisy
mid plus our impact residual. Market impact itself is treated as
exogenous and common to both sides, so it cancels from the comparison;
what the noise layer measures is benchmark risk, i.e. how much of the
realized slippage is luck rather than expectation.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .algorithms import ExecutionSchedule
from .impact import impact


@dataclass
class MarketSim:
    """Monte Carlo market simulator around the transient-impact model."""

    profile: np.ndarray                 # normalized market volume density on the grid
    participation: float = 20.0         # M / X, market volume in units of our block
    s0: float = 100.0                   # arrival (opening) price
    sigma: float = 0.30                 # Brownian vol of the mid over the horizon
    seed: Optional[int] = 7
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def run(self, schedule: ExecutionSchedule, n_slices=60, n_paths=4000):
        """Evaluate a schedule: analytic expectations plus MC slippage paths.

        Returns a dict with keys:

        expected_is, expected_vwap — deterministic per-share slippage;
        is_paths, vwap_paths       — realized per-share slippage per path;
        fill_edge                  — the slice boundary times used.
        """
        params = schedule.params
        edges, quantities = schedule.slices(n_slices)
        widths = np.diff(edges)
        rates = quantities / widths
        starts = edges[:-1]
        centers = starts + widths / 2.0

        # Impact residual from the piecewise-constant slice rates, computed
        # with the same grid quadrature the optimizer uses (the kernel is
        # singular at short lags, so a slice-level midpoint rule would
        # systematically understate the transient cost), then interpolated
        # to the slice centers where fills and the benchmark are priced.
        nodes = schedule.grid.nodes
        slice_index = np.clip(np.searchsorted(edges, nodes, side="right") - 1, 0, n_slices - 1)
        step_rate = rates[slice_index]
        residual = np.interp(
            centers, nodes, schedule.grid.past @ impact(step_rate, params.eps)
        )

        # Market volume per slice, renormalized onto the slice partition.
        market_mass = np.empty(n_slices)
        cumulative_market = np.cumsum(schedule.grid.weights * self.profile)
        for k in range(n_slices):
            lo = np.interp(edges[k], nodes, cumulative_market)
            hi = np.interp(edges[k + 1], nodes, cumulative_market)
            market_mass[k] = hi - lo
        market_mass = market_mass * (self.participation * params.share / market_mass.sum())

        deterministic = schedule.expected_costs(self.profile)
        increments = self._rng.standard_normal((n_paths, n_slices)) * self.sigma * np.sqrt(
            widths
        )
        walk = np.cumsum(increments, axis=1) - increments / 2.0
        mid = self.s0 + walk + residual[None, :]

        fills = mid + params.eta * rates[None, :]
        cost = (fills * quantities[None, :]).sum(axis=1) / params.share
        benchmark = (mid * market_mass[None, :]).sum(axis=1) / market_mass.sum()

        return {
            "expected_is": deterministic["is"] / params.share,
            "expected_vwap": deterministic["vwap"] / params.share,
            "is_paths": cost - self.s0,
            "vwap_paths": cost - benchmark,
            "fill_edges": edges,
        }
