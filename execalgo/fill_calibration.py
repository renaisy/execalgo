"""Passive fill-probability calibration from L5 depth (v1.2).

Physical model: a buy order posted at the bid joins the queue behind the
standing bid depth D. Aggressive sell flow F consumes the queue from the
front; our order starts filling once cumulative F exceeds D. For a small
posted quantity (q << D) the fill amount is approximately

    passive_fill ≈ max(0, F − D)

so the fraction of the sell flow available to a new passive order is

    fill_fraction(F, D) = max(0, F − D) / max(F, 1).

The calibration estimates the average fill_fraction as a function of the
flow-to-depth ratio r = F / D, binned over the training period, with a
day-block bootstrap for confidence bands. The resulting `FillCurve`
replaces the constant `queue_capture` in `MakerPolicy`.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .realdata import N_SESSION_MINUTES


@dataclass(frozen=True)
class FillCurve:
    """Non-parametric fill_fraction(flow/depth) curve with bootstrap CIs."""

    ratio_edges: np.ndarray       # (n_bins+1,) bin edges on the ratio grid
    fill_fractions: np.ndarray    # (n_bins,) mean fill fraction per bin
    ci_low: np.ndarray            # (n_bins,) bootstrap 5th percentile
    ci_high: np.ndarray           # (n_bins,) bootstrap 95th percentile
    counts: np.ndarray            # (n_bins,) minute-observations per bin


def _minute_stats(days):
    """Stack (sell_flow, bid_depth) pairs across all training minutes."""
    sells, depths = [], []
    for day in days:
        if day.minute_sell is None or day.minute_bid_depth is None:
            continue
        sell = np.asarray(day.minute_sell, dtype=float)
        depth = np.asarray(day.minute_bid_depth, dtype=float)
        active = sell > 0
        sells.append(sell[active])
        depths.append(depth[active])
    if not sells:
        return np.array([]), np.array([])
    return np.concatenate(sells), np.concatenate(depths)


def _binned_fractions(sell, depth, edges):
    """Mean fill_fraction per ratio bin."""
    ratio = sell / np.maximum(depth, 1.0)
    fill = np.maximum(0.0, sell - depth) / np.maximum(sell, 1.0)
    fractions = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for i in range(len(edges) - 1):
        mask = (ratio >= edges[i]) & (ratio < edges[i + 1])
        if mask.any():
            fractions[i] = fill[mask].mean()
            counts[i] = int(mask.sum())
    return fractions, counts


def calibrate_fill_curve(train_days, n_bins: int = 10,
                         bootstrap_draws: int = 200,
                         seed: int = 11) -> FillCurve:
    """Estimate the fill_fraction(ratio) curve from training days.

    Bins the flow/depth ratio on a log-spaced grid, computes the mean
    fill fraction per bin, and bootstraps (resampling days with
    replacement) for 90% confidence bands.
    """
    sell, depth = _minute_stats(train_days)
    if sell.size < 50:
        raise ValueError("not enough training minutes with depth for calibration")

    ratio = sell / np.maximum(depth, 1.0)
    lo, hi = np.percentile(ratio, [2, 98])
    edges = np.geomspace(max(lo, 1e-3), max(hi, lo * 1.1), n_bins + 1)

    fractions, counts = _binned_fractions(sell, depth, edges)

    rng = np.random.default_rng(seed)
    days_list = [d for d in train_days
                 if d.minute_sell is not None and d.minute_bid_depth is not None]
    boot_fractions = np.full((bootstrap_draws, n_bins), np.nan)
    for b in range(bootstrap_draws):
        pick = rng.integers(0, len(days_list), len(days_list))
        bs, bd = _minute_stats([days_list[i] for i in pick])
        bf, _ = _binned_fractions(bs, bd, edges)
        boot_fractions[b] = bf

    valid = ~np.isnan(boot_fractions)
    ci_low = np.where(valid.any(axis=0),
                      np.nanpercentile(boot_fractions, 5, axis=0), 0.0)
    ci_high = np.where(valid.any(axis=0),
                       np.nanpercentile(boot_fractions, 95, axis=0), 1.0)

    return FillCurve(edges, fractions, ci_low, ci_high, counts)


def fill_at(curve: FillCurve, sell_flow: float, bid_depth: float) -> float:
    """Expected passive fill volume at (sell_flow, bid_depth) from the curve.

    Returns the fill AMOUNT (in lots), not the fraction: the caller uses it
    directly as the passive cap.
    """
    if sell_flow <= 0 or bid_depth <= 0:
        return 0.0
    ratio = sell_flow / max(bid_depth, 1.0)
    # v1.6 P2-13: use edge-based binning to match calibration bins
    idx = int(np.clip(np.searchsorted(curve.ratio_edges, ratio) - 1,
                      0, len(curve.fill_fractions) - 1))
    fraction = curve.fill_fractions[idx]
    if np.isnan(fraction):
        return 0.0
    return fraction * sell_flow


def fill_at_constant(queue_capture: float, sell_flow: float) -> float:
    """v1.0 fallback: constant beta * sell_flow."""
    return queue_capture * max(sell_flow, 0.0)
