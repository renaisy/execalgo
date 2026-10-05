"""Diagnostics: truncation quality, convergence radius, front-loading.

These are the study tools behind the paper's headline numbers, exposed on
top of the same solver objects the algorithms use.
"""

from typing import Optional

import numpy as np

from .discretize import Grid
from .series import partial_sum


def truncation_study(grid, series, optimizer, eps, max_order=None):
    """Errors of the partial sums against a reference (continued) optimizer."""
    if max_order is None:
        max_order = len(series) - 1
    errors = np.array(
        [
            grid.norm(partial_sum(series, eps, order) - optimizer)
            for order in range(max_order + 1)
        ]
    )
    return {
        "errors": errors,
        "best_order": int(np.argmin(errors)),
        "baseline_error": float(errors[0]),
        "best_error": float(errors.min()),
        "improvement": float(errors[0] / errors.min()),
    }


def radius_estimate(grid, series, fit_range=(20, None)):
    """Geometric growth factor of ||v_m|| and the implied convergence radius.

    A straight line through log||v_m|| over the fitted range has slope
    log(1/R), so the growth factor is exp(slope) and the radius its
    reciprocal.
    """
    low, high = fit_range[0], fit_range[1] or len(series) - 1
    orders = np.arange(low, high + 1)
    norms = np.array([grid.norm(series[m]) for m in range(low, high + 1)])
    slope = np.polyfit(orders, np.log(norms), 1)[0]
    return {"growth": float(np.exp(slope)), "radius": float(np.exp(-slope))}


def front_load_split(grid: Grid, rate: np.ndarray) -> dict:
    """Mass of a trajectory in the first vs the second half of the horizon."""
    nodes = grid.nodes
    half = nodes <= nodes[-1] / 2.0
    front = float(grid.weights[half] @ rate[half])
    back = float(grid.weights[~half] @ rate[~half])
    return {
        "front": front,
        "back": back,
        "front_fraction": front / (front + back),
        "is_front_loaded": front > back,
    }
