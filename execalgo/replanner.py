"""Intraday rolling re-planning (v0.3).

Every `interval_minutes`, the remaining-window problem is re-solved on the
canonical perturbative solver, conditioned on what the day has shown:

* remaining share phi = (X - filled) / X and sub-window length h =
  remaining/225: normalizing the sub-window to [0, 1] rescales rates by 1/h,
  so the canonical slippage/risk balance shifts to

      eta_j   = eta_solver * gamma / sqrt(h),
      risk_j  = risk_solver * gamma * sqrt(h),      gamma = sqrt(phi * s_vol)

  (a shorter window makes slippage relatively more expensive and risk
  accrual smaller);
* the impact state built by earlier own trades enters the remaining
  optimality system as a constant offset I_past / sqrt(phi * h) — without
  it re-planning would not be time-consistent with the static solution
  under unchanged information;
* the day's volume level so far re-scales the impact belief,
  s_vol = V_real(m) / E[V(m)] (square-root law: Lambda_day = Lambda /
  sqrt(s_vol)), and gamma carries it together with phi, which lets the LU
  factorization be cached per (window length, gamma bucket).

The tail shape prediction is v1 = the train-period average tail. The
normalized give-back cross term is level-free by construction, so volume
information reaches the plan only through the impact scale s_vol.

The planner is unit-free: it consumes per-minute market volumes and emits
the fraction of the parent order to execute in each of the 225 minutes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .branch import continue_branch
from .discretize import ConstrainedSolver, make_grid
from .model import MultiExponential
from .series import coefficients, optimal_truncation_order, partial_sum

N_MINUTES = 225  # matches realdata.N_SESSION_MINUTES


class _ScaledKernel:
    """The day-normalized kernel restricted to a sub-window of relative length h."""

    def __init__(self, kernel, scale):
        self._kernel = kernel
        self._scale = scale

    def __call__(self, lag):
        return self._kernel(np.asarray(lag) * self._scale)


@dataclass(frozen=True)
class ReplanConfig:
    interval_minutes: int = 15
    grid_size: int = 301
    max_order: int = 20
    gamma_bucket: float = 0.05   # LU cache quantization of gamma
    scale_clip: tuple = (0.6, 1.67)
    carry_impact: bool = True
    # Revision damping: blend each re-solve with the previous plan. Successive
    # re-solves wander along numerically flat directions of the objective;
    # minute-level convexities (slippage, chunked impact) turn that wandering
    # into real cost. Damping toward the previous plan removes the wandering
    # cost while keeping genuine adaptation.
    revision_damping: float = 0.5


def impact_carry_offset(kernel, executed_fraction, upto_minute, target_times, phi, sub_length):
    """The residual-displacement offset for the remaining problem.

    `executed_fraction` holds the fraction of the parent order executed per
    minute (indices < upto_minute); `target_times` are absolute
    day-normalized times of the remaining sub-grid nodes. The displacement
    uses the full-order reference scale, so price units cancel and the
    canonical offset is I_past / sqrt(phi * h) with h the sub-window length.
    """
    centers = (np.arange(upto_minute) + 0.5) / N_MINUTES
    displacement = np.sqrt(executed_fraction[:upto_minute] * N_MINUTES) / N_MINUTES
    lag = np.asarray(target_times, dtype=float)[:, None] - centers[None, :]
    residual = np.where(lag > 0.0, kernel(np.abs(lag)), 0.0)
    return (residual @ displacement) / np.sqrt(phi * sub_length)


def _minute_fractions(sub_grid, rate, remaining):
    """Integrate a sub-window rate into `remaining` per-minute fractions."""
    union_x = np.concatenate([[0.0], sub_grid.nodes, [1.0]])
    union_y = np.interp(union_x, sub_grid.nodes, rate)
    increments = np.diff(union_x) * (union_y[1:] + union_y[:-1]) / 2.0
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    edges = np.linspace(0.0, 1.0, remaining + 1)
    at_edges = np.interp(edges, union_x, cumulative)
    return np.diff(at_edges)


class RollingPlanner:
    """Re-solves the remaining-window execution problem every K minutes."""

    def __init__(
        self,
        eta_solver: float,
        kernel: MultiExponential,
        train_profile: np.ndarray,
        risk: float = 0.0,
        config: ReplanConfig = ReplanConfig(),
        train_mean_volume: Optional[float] = None,
        benchmark: str = "vwap",
        tail_forecaster=None,
    ):
        if train_profile.size != N_MINUTES:
            raise ValueError(f"train_profile must have {N_MINUTES} minute slots")
        if benchmark not in ("vwap", "is"):
            raise ValueError("benchmark must be 'vwap' or 'is'")
        self.benchmark = benchmark
        self.forecaster = tail_forecaster
        self.eta = float(eta_solver)
        self.risk = float(risk)
        self.kernel = kernel
        self.profile = np.asarray(train_profile, dtype=float)
        self.config = config
        self.train_mean_volume = train_mean_volume
        self._cum_profile = np.cumsum(self.profile)
        self._cache = {}
        self._today = None
        self.replans = 0
        self.fallbacks = 0

    # -- helpers ---------------------------------------------------------

    def _level_estimate(self, minute_volume, upto):
        """s_vol = realized volume so far / its train expectation, clipped."""
        if self.train_mean_volume is None or upto == 0:
            return 1.0
        expected = self._cum_profile[upto - 1] * self.train_mean_volume
        if expected <= 0.0:
            return 1.0
        lo, hi = self.config.scale_clip
        return float(np.clip(minute_volume[:upto].sum() / expected, lo, hi))

    def _solver(self, remaining, gamma):
        bucket = round(gamma / self.config.gamma_bucket) * self.config.gamma_bucket
        key = (remaining, round(bucket, 4))
        h = remaining / N_MINUTES
        if key not in self._cache:
            sub_grid = make_grid(
                self.config.grid_size,
                _ScaledKernel(self.kernel, h),
            )
            self._cache[key] = (
                ConstrainedSolver(
                    sub_grid,
                    self.eta * bucket / h**1.5,
                    self.risk * bucket * np.sqrt(h),
                ),
                sub_grid,
            )
        return self._cache[key]

    def _tail_density(self, sub_grid, from_minute, minute_volume):
        """The market density on the sub-window, p~ = p * h, NOT normalized.

        The give-back cross term of the remaining problem involves the
        market's remaining volume share P_tail, so the sub-canonical cross
        vector is Q_h[p~]; normalizing p~ would rescale q by 1/P_tail and
        break time consistency with the static solution. With a forecaster
        attached (v0.6), p is the prefix-conditioned prediction instead of
        the train average.
        """
        remaining = N_MINUTES - from_minute
        h = remaining / N_MINUTES
        if self.forecaster is not None:
            # the forecaster speaks per-minute fractions; density = fraction * N
            minute_density = self.forecaster.tail(minute_volume, from_minute, self._today) * N_MINUTES
        else:
            minute_density = self.profile[from_minute:] * N_MINUTES
        day_density = np.interp(
            sub_grid.nodes * remaining, np.arange(remaining) + 0.5, minute_density
        )
        return day_density * h

    # -- main loop -------------------------------------------------------

    def execute(self, minute_volume: np.ndarray, night_volume: Optional[float] = None,
                today=None) -> np.ndarray:
        """Plan the whole day, seeing `minute_volume` only up to each replan.

        `night_volume` and `today` are day-level priors handed to the
        forecaster (if attached) before the first replan. Returns the
        per-minute fraction of the parent order (sums to one).
        """
        minute_volume = np.asarray(minute_volume, dtype=float)
        self._today = today
        if self.forecaster is not None:
            self.forecaster.prime(night_volume=night_volume)
        cfg = self.config
        fractions = np.zeros(N_MINUTES)
        filled = 0.0
        pending = None  # previous plan for the current remaining window

        for start in range(0, N_MINUTES, cfg.interval_minutes):
            state = self.step(minute_volume, start, filled, fractions, pending)
            fractions = state["fractions"]
            filled = state["filled"]
            pending = state["pending"]
            if state["done"]:
                break

        return fractions / fractions.sum()

    def step(self, minute_volume, start, filled, fractions, pending=None,
             today=None, night_volume=None):
        """One incremental replan (v1.5 online interface).

        Returns a dict with updated 'fractions', 'filled', 'pending', and
        a 'done' flag. The `execute()` loop is a thin wrapper around this
        method so batch and online behavior are identical by construction.
        Pass `today`/`night_volume` on the FIRST call of each day (they
        prime the forecaster); subsequent calls within the day can omit them.
        """
        cfg = self.config
        if today is not None:
            self._today = today
        if night_volume is not None and self.forecaster is not None:
            self.forecaster.prime(night_volume=night_volume)
        remaining = N_MINUTES - start
        phi = 1.0 - filled
        if phi <= 1e-8:
            return {"fractions": fractions, "filled": filled,
                    "pending": pending, "done": True}
        self.replans += 1
        fractions = fractions.copy()
        try:
            segment = self._solve_segment(minute_volume, fractions, start, remaining, phi)
            if pending is not None and cfg.revision_damping < 1.0:
                alpha = cfg.revision_damping
                tail_prev = pending[-remaining:]
                segment = alpha * segment + (1.0 - alpha) * tail_prev
                segment = segment / segment.sum()
            take = min(remaining, cfg.interval_minutes)
            pending = segment[take:]
            mass = pending.sum()
            pending = pending / mass if mass > 0 else None
            fractions[start : start + take] = segment[:take] * phi
            filled += fractions[start : start + take].sum()
        except ValueError:
            self.fallbacks += 1
            segment = self._constrained_segment(minute_volume, fractions,
                                                start, remaining, phi)
            if segment is None:
                fractions[start:] = phi / remaining
                return {"fractions": fractions, "filled": 1.0,
                        "pending": None, "done": True}
            take = min(remaining, cfg.interval_minutes)
            tail = segment[take:]
            tail_mass = float(tail.sum())
            pending = tail / tail_mass if tail_mass > 0 else None
            fractions[start : start + take] = segment[:take] * phi
            filled += fractions[start : start + take].sum()
        return {"fractions": fractions, "filled": filled,
                "pending": pending, "done": False}

    # fallback solves fire up to once per replan; a coarser grid keeps the
    # whole-day rolling cost bounded with negligible schedule-quality loss
    FALLBACK_GRID = 151

    def _constrained_segment(self, minute_volume, fractions, start, remaining, phi):
        """Long-only-constrained continuation for the guard-tripping replan."""
        cfg = self.config
        s_vol = self._level_estimate(minute_volume, start)
        gamma = float(np.sqrt(phi * s_vol))
        h = remaining / N_MINUTES
        cache_key = ("fallback", remaining)
        if cache_key not in self._cache:
            self._cache[cache_key] = make_grid(
                self.FALLBACK_GRID, _ScaledKernel(self.kernel, h))
        sub_grid = self._cache[cache_key]
        if self.benchmark == "vwap":
            cross = sub_grid.future @ self._tail_density(sub_grid, start, minute_volume)
        else:
            cross = np.zeros(sub_grid.size)
        # the carry gradient constant is +I_past (positive linear objective
        # term), matching the sign convention in _solve_segment
        gradient_offset = None
        if cfg.carry_impact and start > 0:
            gradient_offset = impact_carry_offset(
                self.kernel, fractions, start,
                start / N_MINUTES + sub_grid.nodes * h, phi, h)
        risk_scaled = self.risk * gamma * np.sqrt(h)
        try:
            rate, _, _ = continue_branch(
                sub_grid, self.eta * gamma / h**1.5, 1.0, 0.5,
                eps_step=0.05, cross=cross, risk=risk_scaled,
                upper=None, offset=gradient_offset,
            )
        except Exception:
            return None
        rate = np.clip(rate, 0.0, None)
        mass = float(sub_grid.weights @ rate)
        if not (mass > 1e-12):
            return None
        rate *= 1.0 / mass
        return _minute_fractions(sub_grid, rate, remaining)

    def _solve_segment(self, minute_volume, fractions, start, remaining, phi,
                       return_rate=False):
        cfg = self.config
        s_vol = self._level_estimate(minute_volume, start)
        gamma = float(np.sqrt(phi * s_vol))
        (solver, sub_grid) = self._solver(remaining, gamma)

        if self.benchmark == "vwap":
            cross = sub_grid.future @ self._tail_density(sub_grid, start, minute_volume)
        else:
            cross = np.zeros(sub_grid.size)
        h = remaining / N_MINUTES
        offset = sub_grid.risk_vector(self.risk * gamma * np.sqrt(h), 1.0)
        if cfg.carry_impact and start > 0:
            # The objective's linear term is +int v*I_past, so its gradient
            # constant is +I_past and the baseline RHS gets MINUS I_past
            # (contrast the risk vector, whose gradient constant is negative
            # and enters the RHS positively). The wrong sign front-loads into
            # our own residual impact and drives the positivity guard rate up.
            offset = offset - impact_carry_offset(
                self.kernel, fractions, start,
                start / N_MINUTES + sub_grid.nodes * h, phi, h,
            )

        series = coefficients(
            sub_grid, solver, 1.0, cfg.max_order, cross=cross, offset=offset
        )
        order = optimal_truncation_order(sub_grid, series, 0.5)
        rate = partial_sum(series, 0.5, order)
        rate = np.clip(rate, 0.0, None)
        mass = float(sub_grid.weights @ rate)
        if not (mass > 1e-12) or np.isnan(mass):
            raise ValueError("degenerate sub-solve: rate collapsed to zero or NaN")
        rate *= 1.0 / mass
        if return_rate:
            return _minute_fractions(sub_grid, rate, remaining), rate, sub_grid, cross
        return _minute_fractions(sub_grid, rate, remaining)
