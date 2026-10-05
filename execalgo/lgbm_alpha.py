"""LGBM direction-model adapter for alpha-aware execution (v2.0).

Loads the Model-LightGBM minute-scale direction booster (h300s = 5-min
forward return, 50 OHLCV features), computes its kline features from a
TradingDay's per-minute OHLCV, and returns a per-minute alpha vector in
solver cost units ready for `ExecParams(alpha=...)`.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .realdata import N_SESSION_MINUTES

# Model-LightGBM repo root (for importing its feature pipeline)
_LGBM_ROOT = Path("/home/hsyc/code/Model-LightGBM")
if str(_LGBM_ROOT) not in sys.path:
    sys.path.insert(0, str(_LGBM_ROOT))

DEFAULT_MODEL = (
    _LGBM_ROOT / "checkpoints" / "lgbm_vwap_minute" / "h300s" / "lgbm_baseline.txt"
)

# 25 kline factor names (must match src/features/kline_factors.py)
_FACTOR_NAMES = [
    "mom_1", "mom_3", "mom_5", "mom_10", "mom_20", "mom_60",
    "vol_ret_5", "vol_ret_10", "vol_ret_20",
    "range_5", "range_10", "range_20",
    "vol_mean_5", "vol_mean_20", "vol_ratio_5_20",
    "close_pos", "body_ratio",
    "up_ratio_10", "up_ratio_20", "ret_skew_20", "max_dd_20",
    "gap_1", "hl_spread_last", "trend_slope_20", "rsi_14",
]


def _load_booster(model_path):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(model_path))


def _compute_factors(open_, high, low, close, volume, lookback):
    """Compute the 25 kline factors from a single-scale OHLCV window.

    v2.2 P0-3: rewritten to exactly mirror src/features/kline_factors.py —
    6 factor definitions were wrong (vol_ret was volume ratio not price
    return std; max_dd sign flipped; RSI averaged subsets not clipped all;
    vol_ratio had an off-by-one; short-window fallbacks returned partial
    computations where training returns zeros). The function now matches
    features_from_windows on identical input.
    """
    n = len(close)
    rets = np.diff(close) / np.maximum(close[:-1], 1e-12) if n > 1 else np.array([])

    def mom(k):
        if n <= k:
            return 0.0
        return (close[-1] - close[-k - 1]) / max(abs(close[-k - 1]), 1e-12)

    def ret_std(k):
        if len(rets) < k:
            return 0.0
        return float(np.std(rets[-k:], ddof=1))

    def rolling_range(k):
        if n < k:
            return 0.0
        return (np.max(high[-k:]) - np.min(low[-k:])) / max(abs(close[-1]), 1e-12)

    vol5 = float(np.mean(volume[-5:])) if n >= 5 else float(np.mean(volume))
    vol20 = float(np.mean(volume[-20:])) if n >= 20 else float(np.mean(volume))
    hl = high[-1] - low[-1]

    def up_ratio(k):
        if len(rets) < k:
            return 0.0
        return float(np.mean(rets[-k:] > 0))

    def ret_skew(k):
        if len(rets) < k:
            return 0.0
        chunk = rets[-k:]
        m, s = np.mean(chunk), np.std(chunk)
        if s < 1e-12:
            return 0.0
        return float(np.mean(((chunk - m) / s) ** 3))

    def max_dd(k):
        if n < k:
            return 0.0
        path = close[-k:]
        peak = np.maximum.accumulate(path)
        dd = (path - peak) / np.maximum(peak, 1e-12)
        return float(np.min(dd))

    def trend_slope(k):
        if n < k:
            return 0.0
        y = close[-k:]
        x = np.arange(k, dtype=float)
        x = x - x.mean()
        denom = np.sum(x * x)
        if denom <= 0:
            return 0.0
        slope = np.sum((y - np.mean(y)) * x) / denom
        return slope / max(abs(y[-1]), 1e-12)

    def rsi(period=14):
        if n < period + 1:
            return 50.0
        diff = np.diff(close[-(period + 1):])
        gain = np.maximum(diff, 0.0)
        loss = np.maximum(-diff, 0.0)
        avg_gain = float(np.mean(gain))
        avg_loss = float(np.mean(loss))
        if avg_loss < 1e-12:
            return 50.0  # training fill=1.0 for rs → rsi=50 when avg_loss≈0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    f = {
        "mom_1": mom(1), "mom_3": mom(3), "mom_5": mom(5),
        "mom_10": mom(10), "mom_20": mom(20), "mom_60": mom(60),
        "vol_ret_5": ret_std(5), "vol_ret_10": ret_std(10), "vol_ret_20": ret_std(20),
        "range_5": rolling_range(5), "range_10": rolling_range(10),
        "range_20": rolling_range(20),
        "vol_mean_5": vol5, "vol_mean_20": vol20,
        "vol_ratio_5_20": vol5 / vol20 if abs(vol20) > 1e-12 else 1.0,
        "close_pos": (close[-1] - low[-1]) / hl if abs(hl) > 1e-12 else 0.5,
        "body_ratio": abs(close[-1] - open_[-1]) / hl if abs(hl) > 1e-12 else 0.0,
        "up_ratio_10": up_ratio(10), "up_ratio_20": up_ratio(20),
        "ret_skew_20": ret_skew(20),
        "max_dd_20": max_dd(20),
        "gap_1": (open_[-1] - close[-2]) / max(abs(close[-2]), 1e-12) if n > 1 else 0.0,
        "hl_spread_last": hl / max(abs(close[-1]), 1e-12),
        "trend_slope_20": trend_slope(20),
        "rsi_14": rsi(14),
    }
    return [f[name] for name in _FACTOR_NAMES]


def compute_lgbm_alpha(day, model_path=None,
                       lookback_fast=30, lookback_slow=12,
                       lambda_scale=1.0, clip=0.3) -> np.ndarray:
    """Per-minute LGBM alpha in solver units from TradingDay OHLCV.

    Builds 1-min (30-bar) and 5-min (12-bar) windows, computes the 25
    kline factors at both scales, runs booster.predict() in a single
    batched call, and scales the predicted 5-min forward return to solver
    cost units. Minutes before the warm-up window get zero alpha.
    """
    booster = _load_booster(model_path or DEFAULT_MODEL)

    o = np.asarray(day.minute_open, dtype=float)
    h = np.asarray(day.minute_high, dtype=float)
    l = np.asarray(day.minute_low, dtype=float)
    c = np.asarray(day.minute_close, dtype=float)
    v = np.asarray(day.minute_volume, dtype=float)

    warm = max(lookback_fast, lookback_slow * 5)
    if warm >= N_SESSION_MINUTES:
        return np.zeros(N_SESSION_MINUTES)

    # build all feature vectors, then predict in one batched call
    # v2.2: fast window ends at t-1 (removing the one-bar look-ahead —
    # minute t's close isn't known at the START of minute t's execution)
    rows = []
    for t in range(warm + 1, N_SESSION_MINUTES):
        fast_end = t - 1  # data through t-1, decision at t
        of = _normalize_window(o[fast_end - lookback_fast + 1:fast_end + 1], c, fast_end, lookback_fast)
        hf = _normalize_window(h[fast_end - lookback_fast + 1:fast_end + 1], c, fast_end, lookback_fast)
        lf = _normalize_window(l[fast_end - lookback_fast + 1:fast_end + 1], c, fast_end, lookback_fast)
        cf = _normalize_window(c[fast_end - lookback_fast + 1:fast_end + 1], c, fast_end, lookback_fast)
        vf = _normalize_volume(v[fast_end - lookback_fast + 1:fast_end + 1])
        fast = _compute_factors(of, hf, lf, cf, vf, lookback_fast)

        # v2.2: clock-aligned 5-min bars from session start through t-1
        os_ = _resample_5m(o, fast_end, agg="first")
        hs_ = _resample_5m(h, fast_end, agg="max")
        ls_ = _resample_5m(l, fast_end, agg="min")
        cs_ = _resample_5m(c, fast_end, agg="last")
        vs_ = _resample_5m(v, fast_end, agg="sum")
        slow_n = len(cs_)
        os_n = _normalize_window(os_, cs_, slow_n - 1, slow_n)
        hs_n = _normalize_window(hs_, cs_, slow_n - 1, slow_n)
        ls_n = _normalize_window(ls_, cs_, slow_n - 1, slow_n)
        cs_n = _normalize_window(cs_, cs_, slow_n - 1, slow_n)
        vs_n = _normalize_volume(vs_)
        slow = _compute_factors(os_n, hs_n, ls_n, cs_n, vs_n, lookback_slow)
        rows.append(fast + slow)

    x = np.array(rows, dtype=np.float64)
    preds = booster.predict(x, num_iteration=-1)

    alpha = np.zeros(N_SESSION_MINUTES)
    alpha[warm + 1:] = np.clip(preds * lambda_scale, -clip, clip)
    return alpha


def _normalize_window(prices, close_ref, end_idx, window):
    """Divide by the FIRST bar's close in the window (training convention).

    v2.2: anchored to close_ref[end_idx - window] (the bar BEFORE the first
    window bar, i.e., training's `out[0, 3]`), fixing an off-by-one that
    used the second bar's close.
    """
    anchor = end_idx - window
    if anchor < 0:
        anchor = 0
    base = close_ref[anchor]
    if abs(base) < 1e-12:
        return np.ones(len(prices))
    return prices / base


def _normalize_volume(vol):
    """log1p + z-score within the window (LGBM convention)."""
    lv = np.log1p(np.maximum(vol, 0.0))
    m, s = np.mean(lv), np.std(lv)
    return (lv - m) / s if s > 1e-12 else np.zeros_like(lv)


def _resample_5m(arr, end, agg="first"):
    """Clock-aligned 5-min resample over [0, end], matching training bars.

    Training uses pandas right-edge 5-min bars aligned to 09:00 (slot 0),
    which also never straddles the 10:15 or 11:30 breaks (each trading block
    is a multiple of 5). v2.2: aligned to the session grid instead of a
    rolling window, and the window is [0, end-1] (excludes end for the slow
    scale, matching training's right-edge convention where bar features at
    time t use data through bar t-1 on the slow scale).
    """
    bar_ends = np.arange(5, end + 1, 5)  # right edges: 5, 10, 15, ...
    n = len(bar_ends)
    out = np.zeros(n)
    for i, be in enumerate(bar_ends):
        chunk = arr[max(0, be - 5):be]
        if len(chunk) == 0:
            out[i] = 0.0
        elif agg == "first":
            out[i] = chunk[0]
        elif agg == "max":
            out[i] = np.max(chunk)
        elif agg == "min":
            out[i] = np.min(chunk)
        elif agg == "last":
            out[i] = chunk[-1]
        elif agg == "sum":
            out[i] = np.sum(chunk)
    return out
