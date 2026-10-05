"""Cross-product impact spillover estimation (v1.4).

Estimates how trading in product A moves product B's mid-price:
    R_AB(tau) = E[eps_A(t) * (log mid_B(t+tau) - log mid_B(t-1))]  in bps.

The own response R_AA is computed alongside for normalization. All responses
use log-mid differences so cross-product magnitudes are directly comparable.
"""

from typing import Dict, List

import numpy as np

from .realdata import N_SESSION_MINUTES


def _date_index(days):
    return {d.date: i for i, d in enumerate(days)}


def _aligned_pairs(days_a, days_b):
    """Yield (signed_a, logmid_a, logmid_b) for dates present in both."""
    idx_b = _date_index(days_b)
    for day_a in days_a:
        if day_a.date in idx_b:
            day_b = days_b[idx_b[day_a.date]]
            if day_a.minute_signed is None or day_b.minute_mid is None:
                continue
            eps = np.where(day_a.minute_signed != 0,
                           np.sign(day_a.minute_signed), 0.0)
            log_a = np.log(np.maximum(day_a.minute_mid, 1e-12))
            log_b = np.log(np.maximum(day_b.minute_mid, 1e-12))
            yield eps, log_a, log_b


def cross_response(days_a, days_b, max_lag: int = 120) -> Dict[str, np.ndarray]:
    """Cross-product response R_AB plus own responses R_AA and R_BB.

    Returns dict with 'lags', 'r_cross' (A→B), 'r_own_a', 'r_own_b',
    all in bps per sign-event.
    """
    lags = np.arange(1, max_lag + 1)
    r_cross = np.zeros(lags.size)
    r_own_a = np.zeros(lags.size)
    r_own_b = np.zeros(lags.size)
    cnt_cross = np.zeros(lags.size)
    cnt_own_a = np.zeros(lags.size)
    cnt_own_b = np.zeros(lags.size)
    n = N_SESSION_MINUTES

    for eps, log_a, log_b in _aligned_pairs(days_a, days_b):
        pre_a = np.concatenate([log_a[:1], log_a[:-1]])
        pre_b = np.concatenate([log_b[:1], log_b[:-1]])
        for j, tau in enumerate(lags):
            mask = eps[: n - tau] != 0.0
            if mask.sum() == 0:
                continue
            e = eps[: n - tau][mask]
            r_cross[j] += np.dot(e, (log_b[tau:] - pre_b[: n - tau])[mask])
            cnt_cross[j] += int(mask.sum())
            # own response of A (sign of A × mid change of A)
            r_own_a[j] += np.dot(e, (log_a[tau:] - pre_a[: n - tau])[mask])
            cnt_own_a[j] += int(mask.sum())
            # own response of B (sign of B × mid change of B) — use B's own signs
            eps_b = np.where(np.asarray(days_b[0].minute_signed) != 0,
                             np.sign(np.asarray(days_b[0].minute_signed)), 0.0)
            # per-day: recompute B's signs
            # (handled below in the loop for correctness)

    # B's own response needs B's signs, recompute properly
    idx_a = _date_index(days_a)
    for day_b in days_b:
        if day_b.date not in idx_a or day_b.minute_signed is None:
            continue
        day_a = days_a[idx_a[day_b.date]]
        if day_a.minute_signed is None or day_b.minute_mid is None:
            continue
        eps_b = np.where(day_b.minute_signed != 0,
                         np.sign(day_b.minute_signed), 0.0)
        log_b = np.log(np.maximum(day_b.minute_mid, 1e-12))
        pre_b = np.concatenate([log_b[:1], log_b[:-1]])
        for j, tau in enumerate(lags):
            mask = eps_b[: n - tau] != 0.0
            if mask.sum() == 0:
                continue
            r_own_b[j] += np.dot(eps_b[: n - tau][mask],
                                 (log_b[tau:] - pre_b[: n - tau])[mask])
            cnt_own_b[j] += int(mask.sum())

    scale = 1e4  # log return → bps
    return {
        "lags": lags.astype(float),
        "r_cross": r_cross / np.maximum(cnt_cross, 1.0) * scale,
        "r_own_a": r_own_a / np.maximum(cnt_own_a, 1.0) * scale,
        "r_own_b": r_own_b / np.maximum(cnt_own_b, 1.0) * scale,
    }


def spillover_ratio(result: Dict[str, np.ndarray], n_short: int = 5) -> float:
    """Mean |R_AB| / mean |R_AA| over the first n_short lags.

    Short lags carry the impact signal; long lags are noise-dominated, so
    using the max over all lags overstates the spillover in finite samples.
    """
    own = float(np.abs(result["r_own_a"][:n_short]).mean())
    cross = float(np.abs(result["r_cross"][:n_short]).mean())
    if own < 1e-12:
        return 0.0
    return cross / own


def spillover_label(ratio: float) -> str:
    if ratio > 0.3:
        return "强溢出"
    if ratio > 0.1:
        return "中等溢出"
    if ratio > 0.05:
        return "弱溢出"
    return "可忽略"
