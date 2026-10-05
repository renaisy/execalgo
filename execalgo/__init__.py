"""Optimal execution against arrival-price (IS) and VWAP benchmarks.

Both algorithms minimize the expected cost of trading a fixed share block
over a fixed horizon under Gatheral's transient-impact (propagator) model
with nonlinear impact f(x; eps) = x^(1-eps), solved by the perturbative
expansion of Ritter, "Perturbation theory for nonlinear transient impact".
"""

from .algorithms import (
    ExecParams,
    ExecutionSchedule,
    ISAlgorithm,
    VWAPAlgorithm,
    participation_upper,
    profile_schedule,
    twap_schedule,
)
from .model import MultiExponential, fit_square_root

__all__ = [
    "ExecParams",
    "ExecutionSchedule",
    "ISAlgorithm",
    "VWAPAlgorithm",
    "participation_upper",
    "twap_schedule",
    "profile_schedule",
    "MultiExponential",
    "fit_square_root",
]
