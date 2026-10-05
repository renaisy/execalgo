"""Execution algorithms against the two price benchmarks.

ISAlgorithm minimizes the expected implementation shortfall against the
arrival price — the objective of the paper. VWAPAlgorithm minimizes the
expected slippage against the market VWAP, which under the same price
model differs by one linear "cross term" (see vwap.py), so both share
the perturbative solver and differ only in a right-hand-side vector.

Two solve methods are offered:

* 'asymptotic' — the perturbative series truncated at the asymptotically
  optimal order (the smallest term). No reference solution needed; this
  is the production path (one LU factorization plus back-substitutions).
* 'continuation' — Newton continuation of the nonlinear optimality system
  in eps; slower but exact, used as a certificate and for study.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .branch import continue_branch, stationarity
from .discretize import ConstrainedSolver, Grid, make_grid
from .impact import impact
from .model import MultiExponential, fit_square_root
from .series import coefficients, optimal_truncation_order, partial_sum
from .vwap import benchmark_cross

Profile = Union[np.ndarray, object]  # density array on the grid, or grid -> density


@dataclass(frozen=True)
class ExecParams:
    """Problem parameters shared by both algorithms.

    Defaults are the paper's canonical configuration. `share` X is the
    block size, `eta` the slippage coefficient of eta int v^2, `eps` the
    impact nonlinearity (1/2 = square-root law), `kernel` the transient
    decay (defaults to the three-exponential fit of t^{-1/2}), and `risk`
    the inventory-risk aversion of risk * int (X - int_0^t v)^2 dt in
    solver units (0 = risk-neutral, the paper's objective).

    `upper_rate` (v1.0) is a per-node participation ceiling in solver-rate
    units (share-normalized): with it set the solve runs the constrained
    active-set path; see `participation_upper` for building it from a
    market profile.
    """

    share: float = 0.1
    horizon: float = 1.0
    eta: float = 0.65
    eps: float = 0.5
    grid_size: int = 601
    max_order: int = 44
    kernel: Optional[MultiExponential] = None
    risk: float = 0.0
    upper_rate: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None  # per-node expected return (solver units)

    def make_grid(self) -> Grid:
        return make_grid(self.grid_size, self.resolve_kernel(), self.horizon)

    def resolve_kernel(self):
        return self.kernel if self.kernel is not None else fit_square_root()[0]


def participation_upper(cap, market_profile, mean_volume, grid: Grid,
                        share_lots: float, share: float = 1.0) -> np.ndarray:
    """Per-node solver-rate ceiling from a per-minute participation cap.

    `market_profile` is the normalized (225,) day volume density,
    `mean_volume` the train-period average day volume in lots, `share_lots`
    the parent order in lots, `share` the SOLVER share the ceiling is
    expressed against (the rate integrates to `share`, so a minute with
    rate v executes v / (225 * share) of the block). Capping that at `cap`
    of the expected market volume Vbar * p gives

        v <= cap * Vbar * p(t) * 225 * share / X.

    The default share=1.0 matches the canonical real-data solve; passing
    any other ExecParams.share requires mirroring it here.
    """
    profile = np.asarray(market_profile, dtype=float)
    minute_times = (np.arange(profile.size) + 0.5) / profile.size
    density = np.interp(grid.nodes, minute_times, profile)
    return cap * density * mean_volume * profile.size * share / share_lots


@dataclass
class ExecutionSchedule:
    """A trading-rate trajectory on the grid, resamplable into order slices."""

    params: ExecParams
    grid: Grid
    rate: np.ndarray
    label: str
    method: str = "asymptotic"
    order: Optional[int] = None
    stationarity_residual: Optional[float] = None
    clamped: bool = False

    def slices(self, count):
        """Integrate the trajectory into `count` equal-time order slices.

        Returns (edges, quantities): the slice boundaries and the number of
        shares to execute in each slice. The last quantity absorbs the
        quadrature remainder so the quantities always sum to X exactly.
        """
        params = self.params
        edges = np.linspace(0.0, params.horizon, count + 1)
        nodes = self.grid.nodes
        cumulative = np.empty(count + 1)
        cumulative[0] = 0.0
        for k in range(1, count + 1):
            inside = (nodes >= edges[k - 1]) & (nodes <= edges[k])
            piece_nodes = np.concatenate([[edges[k - 1]], nodes[inside], [edges[k]]])
            piece_rate = np.interp(piece_nodes, nodes, self.rate)
            cumulative[k] = cumulative[k - 1] + np.trapezoid(piece_rate, piece_nodes)
        quantities = np.diff(cumulative)
        quantities[-1] = params.share - quantities[:-1].sum()
        return edges, quantities

    def front_load_fraction(self):
        """The fraction of the block executed in the first half of the horizon."""
        nodes = self.grid.nodes
        half = nodes <= nodes[-1] / 2.0
        executed = self.grid.weights[half] @ self.rate[half]
        return float(executed / self.params.share)

    def expected_costs(self, profile=None):
        """Deterministic expected costs under the transient-impact model.

        Returns a dict with, in shares-times-price units:

        is           — slippage against the arrival price (implementation
                       shortfall), eta int v^2 + int int v G f(v);
        vwap         — slippage against the market VWAP when `profile` (a
                       normalized market volume density on the grid) is given:
                       the IS cost minus the benchmark give-back term;
        risk_penalty — risk * int (X - int_0^t v)^2 dt, the inventory-risk
                       term of the risk-averse objective.
        """
        params = self.params
        eps = params.eps
        rate = self.rate
        slippage = params.eta * self.grid.inner(rate, rate)
        transient = float(
            self.grid.weights @ (rate * (self.grid.past @ impact(rate, eps)))
        )
        total = slippage + transient
        result = {"is": total}
        if profile is not None:
            give_back = params.share * float(
                self.grid.weights @ (impact(rate, eps) * (self.grid.future @ profile))
            )
            result["vwap"] = total - give_back
        if params.risk:
            # cumulative trapezoid of the executed quantity, consistent with
            # the grid quadrature: C(t_i) = sum_{j<i} w_j v_j + w_i v_i / 2
            cumulative = np.cumsum(self.grid.weights * rate) - self.grid.weights * rate / 2.0
            inventory = params.share - cumulative
            result["risk_penalty"] = params.risk * float(
                self.grid.weights @ (inventory**2)
            )
        return result


class _PerturbativeSolver:
    """Shared machinery: grid, one LU factorization, series, continuation."""

    def __init__(self, params: ExecParams, cross: Optional[np.ndarray] = None):
        self.params = params
        self.grid = params.make_grid()
        if cross is not None and cross.shape != self.grid.nodes.shape:
            raise ValueError("cross term must live on the solver grid")
        self.solver = ConstrainedSolver(self.grid, params.eta, params.risk)
        self.cross = None if cross is None else np.asarray(cross, dtype=float)

    def series(self):
        offset = None
        if self.params.risk or self.params.alpha is not None:
            offset = self.grid.risk_vector(self.params.risk, self.params.share) \
                if self.params.risk else np.zeros(self.grid.size)
            if self.params.alpha is not None:
                # alpha rewards executing where expected return is high:
                # it enters the gradient constant as -alpha, so the solve
                # RHS (which negates the gradient constant) gets +alpha
                offset = offset + self.params.alpha
        return coefficients(
            self.grid,
            self.solver,
            self.params.share,
            self.params.max_order,
            self.cross,
            offset,
        )

    def continuation(self, monitor_hessian=False):
        # continue_branch's `offset` is the GRADIENT constant; alpha enters
        # the gradient as -alpha, so the continuation offset is -alpha
        cont_offset = None
        if self.params.alpha is not None:
            cont_offset = -self.params.alpha
        return continue_branch(
            self.grid,
            self.params.eta,
            self.params.share,
            self.params.eps,
            monitor_hessian=monitor_hessian,
            cross=self.cross,
            risk=self.params.risk,
            upper=self.params.upper_rate,
            offset=cont_offset,
        )

    def schedule(self, label, method, rate, order=None, residual=None, clamped=False):
        return ExecutionSchedule(
            self.params, self.grid, rate, label, method, order, residual, clamped
        )

    def solve(self, label, method="asymptotic") -> ExecutionSchedule:
        params = self.params
        if method == "asymptotic":
            if params.upper_rate is not None:
                raise ValueError(
                    "participation caps need the constrained path: solve(method='continuation')"
                )
            series = self.series()
            order = optimal_truncation_order(self.grid, series, params.eps)
            rate = partial_sum(series, params.eps, order)
            rate, clamped = self._long_only(rate)
            return self.schedule(label, method, rate, order, clamped=clamped)
        if method == "continuation":
            rate, multiplier, _ = self.continuation()
            # certify the RETURNED trajectory; when clamping engaged, the
            # pre-clamp residual would describe a different schedule. The
            # sqrt impact slope is singular at the clamp's exact zeros, so
            # the residual is evaluated on the same 1e-9 floor the bounded
            # continuation uses.
            rate, clamped = self._long_only(rate)
            # certificate: free-set stationarity of the returned trajectory.
            # At clamped (bound-active) nodes the full gradient is balanced by
            # the long-only bound and does not vanish, so the interior spread
            # is the meaningful merit — identical to the full residual for
            # interior solutions.
            # v1.6 P2-11: include alpha in the certificate and exclude
            # cap-active nodes from the "interior" set (their gradient
            # legitimately differs from ell by the bound multiplier)
            cert_offset = -params.alpha if params.alpha is not None else None
            full = stationarity(
                self.grid, params.eta, params.eps, np.maximum(rate, 1e-9),
                multiplier, params.share, self.cross, params.risk,
                cert_offset,
            )
            interior = rate > 1e-6
            if params.upper_rate is not None:
                interior = interior & (rate < params.upper_rate - 1e-8)
            if interior.any():
                gradient = full[: self.grid.size][interior]
                residual = float(np.max(np.abs(gradient - gradient.mean())))
            else:
                residual = float("nan")
            return self.schedule(label, method, rate, None, residual, clamped)
        raise ValueError(f"unknown method {method!r}")

    def _long_only(self, rate):
        """Clamp a negative tail onto zero and renormalize, when present.

        The unconstrained VWAP optimum can turn slightly negative at the
        horizon's end: selling into the close pushes the benchmark down,
        which is genuine benchmark gaming rather than an executable order
        for a buy parent. Long-only enforcement is the practical remedy;
        the effect on expected cost is negligible whenever the dip is small.
        """
        if rate.min() >= 0.0:
            return rate, False
        clamped = np.clip(rate, 0.0, None)
        clamped *= self.params.share / (self.grid.weights @ clamped)
        return clamped, True


class ISAlgorithm:
    """Optimal execution benchmarked against the arrival price (IS)."""

    def __init__(self, params: ExecParams = ExecParams()):
        self._engine = _PerturbativeSolver(params)

    def solve(self, method="asymptotic") -> ExecutionSchedule:
        return self._engine.solve("IS", method)


class VWAPAlgorithm:
    """Optimal execution benchmarked against the market VWAP.

    `market_profile` is the normalized intraday volume density — either an
    array on a grid of `params.grid_size` nodes, or a callable grid ->
    density such as `lambda grid: vwap.u_shaped(grid)`.
    """

    def __init__(self, params: ExecParams, market_profile: Profile):
        grid = params.make_grid()
        profile = (
            market_profile(grid) if callable(market_profile) else market_profile
        )
        profile = np.asarray(profile, dtype=float)
        if profile.shape != grid.nodes.shape:
            raise ValueError(
                "market profile array must have grid_size entries; "
                "pass a callable grid -> density instead for custom sizes"
            )
        self._profile = profile
        self._engine = _PerturbativeSolver(
            params, benchmark_cross(grid, profile, params.share)
        )

    @property
    def profile(self):
        return self._profile

    def solve(self, method="asymptotic") -> ExecutionSchedule:
        return self._engine.solve("VWAP", method)


def twap_schedule(params: ExecParams) -> ExecutionSchedule:
    """The time-weighted baseline: a constant rate over the horizon."""
    grid = params.make_grid()
    rate = np.full(grid.size, params.share / params.horizon)
    return ExecutionSchedule(params, grid, rate, "TWAP", "uniform")


def profile_schedule(params: ExecParams, profile: np.ndarray) -> ExecutionSchedule:
    """The naive VWAP strategy: trade in proportion to market volume."""
    grid = params.make_grid()
    rate = params.share * np.asarray(profile, dtype=float)
    return ExecutionSchedule(params, grid, rate, "VWAP-profile", "uniform")
