"""Survival-table aggregation for the v0.4 sensitivity scan (T13).

Pure helpers: given one record per scan cell (paired-diff conclusions and
front-loading fractions), compute per-conclusion survival rates — the
fraction of valid cells where the conclusion keeps its expected sign —
plus median t-statistics and the weakest cell, for honest robustness
reporting.
"""

from typing import Optional

import numpy as np

CONCLUSIONS = (
    "naive_minus_vwapopt",
    "twap_minus_isopt",
    "twap_minus_vwapopt",
)


def survival_table(cells) -> dict:
    """Aggregate scan cells into per-conclusion robustness statistics.

    Each cell dict carries: `pairs` {conclusion: (mean, t) or None},
    `front` {"VWAP-opt": f, "IS-opt": f} (optional), `boundary` flag, and
    its axis coordinates for the weakest-cell report.
    """
    table = {}
    for conclusion in CONCLUSIONS:
        valid = [
            cell for cell in cells
            if not cell.get("boundary") and cell["pairs"].get(conclusion) is not None
        ]
        means = [cell["pairs"][conclusion][0] for cell in valid]
        ts = [cell["pairs"][conclusion][1] for cell in valid]
        surviving = [cell for cell in valid if cell["pairs"][conclusion][0] > 0]
        weakest = None
        if valid:
            weakest_cell = min(valid, key=lambda c: c["pairs"][conclusion][0])
            weakest = {
                "axes": _axes(weakest_cell),
                "mean": weakest_cell["pairs"][conclusion][0],
                "t": weakest_cell["pairs"][conclusion][1],
            }
        table[conclusion] = {
            "n_cells": len(valid),
            "n_surviving": len(surviving),
            "survival_rate": len(surviving) / len(valid) if valid else None,
            "median_t": float(np.median(ts)) if ts else None,
            "median_bp": float(np.median(means)) if means else None,
            "weakest": weakest,
        }

    front_valid = [c for c in cells if not c.get("boundary") and c.get("front")]
    if front_valid:
        ok = [c for c in front_valid
              if 0.5 < c["front"]["IS-opt"] < c["front"]["VWAP-opt"]]
        table["front_ordering"] = {
            "n_cells": len(front_valid),
            "n_surviving": len(ok),
            "survival_rate": len(ok) / len(front_valid),
        }
    return table


def robust_verdict(table: dict, threshold: float = 0.8) -> dict:
    """The T13 verdict: a conclusion is robust if it survives >= threshold of cells."""
    verdicts = {}
    for conclusion in CONCLUSIONS:
        rate = table[conclusion]["survival_rate"]
        verdicts[conclusion] = (
            None if rate is None else ("robust" if rate >= threshold else "fragile")
        )
    if "front_ordering" in table:
        verdicts["front_ordering"] = (
            "robust" if table["front_ordering"]["survival_rate"] >= threshold else "fragile"
        )
    return verdicts


def _axes(cell) -> dict:
    return {
        key: cell.get(key)
        for key in ("product", "participation", "c_imp", "kernel_scale")
    }
