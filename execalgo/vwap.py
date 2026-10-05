"""Market volume profiles and the VWAP benchmark cross term.

Slippage against a VWAP benchmark differs from slippage against the
arrival price by one controllable term: our own impact residual also
moves the benchmark, and the give-back is weighted by the market volume
density. Writing p(t) for the normalized market volume density
(integral one) and X for our share block, the controllable part of the
benchmark-relative cost is

    J_VWAP(v) = J_IS(v) - int f(v(s)) q(s) ds,      q = X * Q p,

with (Q h)(s) = int_s^T G(t - s) h(t) dt. Note that the market size M
cancels: the cross term depends only on the *shape* of the intraday
volume profile, not on how much the market trades in total.

`grid.future` is the trapezoidal image of Q, so the discrete q is
`share * (grid.future @ profile)` where `profile` integrates to one
under the same quadrature.
"""

import numpy as np


def _normalized(grid, density):
    """Scale a positive density so that it integrates to one on the grid."""
    total = grid.weights @ density
    if total <= 0.0:
        raise ValueError("market volume profile must be positive somewhere")
    return density / total


def u_shaped(grid, sharpness=8.0, open_weight=1.0, close_weight=1.0, floor=0.2):
    """A normalized U-shaped intraday volume density on the grid.

    Exponential bumps at the open and the close on top of a flat floor,
    the standard first model of intraday volume. `sharpness` is the decay
    rate of the bumps in units of 1/horizon; the weights control their
    relative heights.
    """
    nodes = grid.nodes
    horizon = nodes[-1]
    density = (
        open_weight * np.exp(-sharpness * nodes)
        + close_weight * np.exp(-sharpness * (horizon - nodes))
        + floor
    )
    return _normalized(grid, density)


def from_samples(grid, times, volumes):
    """Interpolate external volume data onto the grid and normalize.

    `times` and `volumes` are equal-length arrays of (timestamp, volume)
    observations covering the horizon, e.g. one bar per five minutes of
    the session. The density is piecewise linear between observations.
    """
    times = np.asarray(times, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if times.shape != volumes.shape or times.ndim != 1 or times.size < 2:
        raise ValueError("times and volumes must be 1-D arrays of equal length >= 2")
    density = np.interp(grid.nodes, times, volumes)
    return _normalized(grid, density)


def twap(grid):
    """The flat (uniform) volume density: the naive TWAP benchmark."""
    return _normalized(grid, np.ones(grid.size))


def benchmark_cross(grid, profile, share):
    """The VWAP benchmark vector q = X * (Q p) entering the optimality system."""
    return share * (grid.future @ np.asarray(profile, dtype=float))
