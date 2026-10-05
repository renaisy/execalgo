"""Shared replay/table helpers for the analysis scripts (v1.1 P2-13).

One implementation of the paired-t helper, the four-strategy replay block,
and the summary table printer, so the six scripts stop drifting on
volume_scale / participation caps / formatting conventions.
"""

import pathlib
import sys
from datetime import date

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = pathlib.Path("/data/ticker_data")
TRAIN_RANGE = (date(2024, 7, 1), date(2024, 12, 31))
TEST_RANGE = (date(2025, 1, 1), date(2025, 6, 30))
N_SESSION_MINUTES = 225
MINUTE_TIMES = (np.arange(N_SESSION_MINUTES) + 0.5) / N_SESSION_MINUTES


def paired_t(diffs):
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return mean, (mean / se if se > 0 else float("inf"))


def replay_strategies(strategies, test_days, calib, kernel, stats,
                      volume_scale=None):
    """Replay per-minute fraction schedules over test days.

    `strategies`: {name: (225,) fraction of the parent order}. With
    `volume_scale="day_adaptive"` each day's impact scale follows the
    square-root law (sqrt of train/test day volume); None keeps the
    window-level calibration. Returns {name: [replay_day dicts]}.
    """
    from execalgo.realdata import replay_day

    rows = {name: [] for name in strategies}
    for day in test_days:
        scale = 1.0
        if volume_scale == "day_adaptive":
            scale = float(np.sqrt(stats["mean_volume"] / day.minute_volume.sum()))
        for name, frac in strategies.items():
            rows[name].append(
                replay_day(frac * calib["share"], day, calib["lambda"],
                           calib["eta_p"], kernel, volume_scale=scale)
            )
    return rows


def summarize_replays(rows, names=None):
    """mean/std of is_bps and vwap_bps per strategy."""
    names = names or list(rows)
    table = {}
    for name in names:
        is_bps = np.array([r["is_bps"] for r in rows[name]])
        vwap_bps = np.array([r["vwap_bps"] for r in rows[name]])
        table[name] = {
            "is_mean": float(is_bps.mean()), "is_std": float(is_bps.std()),
            "vwap_mean": float(vwap_bps.mean()), "vwap_std": float(vwap_bps.std()),
        }
    return table


def print_strategy_table(table, metric_pair=("is", "vwap")):
    left, right = metric_pair
    print(f"{'策略':<14}{'IS口径bp':>18}{'VWAP口径bp':>18}")
    for name, row in table.items():
        print(f"{name:<14}{row[f'{left}_mean']:>10.2f}±{row[f'{left}_std']:<6.1f}"
              f"{row[f'{right}_mean']:>10.2f}±{row[f'{right}_std']:<6.1f}")


def day_volume_scale(day, stats):
    """Day-adaptive impact scale (square-root law against the train mean)."""
    return float(np.sqrt(stats["mean_volume"] / day.minute_volume.sum()))
