"""v2.2 fixes: factor alignment + alpha solver + quality gate."""

import numpy as np
import pytest

from execalgo.lgbm_alpha import _FACTOR_NAMES, _compute_factors


def _training_factors(open_, high, low, close, volume):
    """Direct call to the training pipeline for comparison."""
    import sys
    sys.path.insert(0, "/home/hsyc/code/Model-LightGBM")
    from src.features.kline_factors import features_from_windows
    window = np.column_stack([open_, high, low, close, volume])
    return features_from_windows(window[np.newaxis, ...])[0]


def _random_ohlcv(n=30, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n) * 0.002))
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n) * 0.002))
    volume = rng.lognormal(5, 0.5, n)
    return open_, high, low, close, volume


def test_factors_match_training_pipeline():
    """v2.2 P0-3: all 25 factors identical to kline_factors.features_from_windows."""
    for seed in (7, 11, 42):
        o, h, l, c, v = _random_ohlcv(30, seed)
        ours = np.array(_compute_factors(o, h, l, c, v, 30))
        training = _training_factors(o, h, l, c, v)
        np.testing.assert_allclose(ours, training, rtol=1e-5, atol=1e-8,
                                   err_msg=f"seed={seed}")


def test_factors_match_training_short_window():
    """Slow-scale windows (12 bars) must also match (training zeros when < 20)."""
    o, h, l, c, v = _random_ohlcv(12, 3)
    ours = np.array(_compute_factors(o, h, l, c, v, 12))
    training = _training_factors(o, h, l, c, v)
    np.testing.assert_allclose(ours, training, rtol=1e-5, atol=1e-8)


def test_vol_ret_is_return_std_not_volume_ratio():
    """vol_ret_k must be the rolling std of PRICE returns (ddof=1)."""
    o, h, l, c, v = _random_ohlcv(30, 5)
    ours = np.array(_compute_factors(o, h, l, c, v, 30))
    idx = _FACTOR_NAMES.index("vol_ret_5")
    rets = np.diff(c) / c[:-1]
    expected = np.std(rets[-5:], ddof=1)
    assert ours[idx] == pytest.approx(expected, rel=1e-10)


def test_max_dd_is_negative():
    """max_dd_20 must be <= 0 (deepest drawdown, training convention)."""
    c = np.linspace(1.0, 0.8, 20)  # monotone decline
    o = np.roll(c, 1); o[0] = c[0]
    ours = np.array(_compute_factors(o, c, c, c, np.ones(20), 20))
    assert ours[_FACTOR_NAMES.index("max_dd_20")] <= 0


def test_vol_ratio_no_minus_one():
    """vol_ratio_5_20 must be vol5/vol20 (no -1, fill=1.0)."""
    o, h, l, c, v = _random_ohlcv(25, 9)
    ours = np.array(_compute_factors(o, h, l, c, v, 25))
    vr = ours[_FACTOR_NAMES.index("vol_ratio_5_20")]
    vol5 = np.mean(v[-5:]); vol20 = np.mean(v[-20:])
    assert vr == pytest.approx(vol5 / vol20, rel=1e-10)


def test_rsi_fill_50_on_zero_loss():
    """RSI fills 50 (not 100) when there are no losses."""
    c = np.linspace(1.0, 1.1, 20)  # monotone rise, no losses
    o = np.roll(c, 1); o[0] = c[0]
    ours = np.array(_compute_factors(o, c, c, c, np.ones(20), 20))
    assert ours[_FACTOR_NAMES.index("rsi_14")] == pytest.approx(50.0)


def test_quality_gate_rejects_degenerate():
    """The quality gate catches cost > 2.5x baseline on extreme alpha.

    v2.3: the old try/except-pass could never fail — now we assert that
    extreme non-smooth alpha either raises (gate or solver rejection) or
    produces a non-degenerate schedule (rate.max() < 3x the alpha-free max).
    """
    from execalgo import ExecParams, ISAlgorithm
    from execalgo.model import fit_square_root

    kernel, _ = fit_square_root()
    base = ISAlgorithm(ExecParams(share=1.0, eta=0.2055, eps=0.5,
                                  grid_size=151, max_order=20)
                       ).solve(method="continuation")
    base_max = float(base.rate.max())

    extreme = np.full(151, 0.5)
    extreme[::10] = -0.5
    params = ExecParams(share=1.0, eta=0.2055, eps=0.5,
                        grid_size=151, max_order=20, alpha=extreme)
    try:
        sched = ISAlgorithm(params).solve(method="continuation")
        # if it converges, the schedule must not be degenerate
        assert sched.rate.max() < 3.0 * base_max, (
            f"degenerate: max {sched.rate.max():.1f} vs base {base_max:.1f}"
        )
    except (ValueError, RuntimeError):
        pass  # correctly rejected by quality gate or solver


def test_alpha_solver_converges_and_no_degenerate_spikes():
    """Smooth alpha at moderate amplitude: converges, no spike schedule."""
    from execalgo import ExecParams, ISAlgorithm
    from execalgo.model import fit_square_root

    kernel, _ = fit_square_root()
    t = np.linspace(0, 1, 151)
    smooth = 0.01 * np.sin(2 * np.pi * t)  # smooth, moderate

    base = ISAlgorithm(ExecParams(share=1.0, eta=0.2055, eps=0.5,
                                  grid_size=151, max_order=20)
                       ).solve(method="continuation")
    aware = ISAlgorithm(ExecParams(share=1.0, eta=0.2055, eps=0.5,
                                   grid_size=151, max_order=20, alpha=smooth)
                        ).solve(method="continuation")
    assert aware.rate.max() < 2.0 * base.rate.max()  # not degenerate
    assert aware.rate.min() >= -1e-10
    assert aware.grid.weights @ aware.rate == pytest.approx(1.0, abs=1e-8)


def test_alpha_sign_direction():
    """Positive alpha (reward early) should front-load more than negative."""
    from execalgo import ExecParams, ISAlgorithm
    from execalgo.model import fit_square_root

    kernel, _ = fit_square_root()
    grid_size = 151
    t = np.linspace(0, 1, grid_size)
    pos_alpha = 0.005 * (1 - t)   # decreasing = front-load reward
    neg_alpha = 0.005 * t          # increasing = back-load reward

    base = ISAlgorithm(ExecParams(share=1.0, eta=0.2055, grid_size=grid_size,
                                  max_order=20)).solve(method="continuation")
    pos = ISAlgorithm(ExecParams(share=1.0, eta=0.2055, grid_size=grid_size,
                                 max_order=20, alpha=pos_alpha)).solve(method="continuation")
    neg = ISAlgorithm(ExecParams(share=1.0, eta=0.2055, grid_size=grid_size,
                                 max_order=20, alpha=neg_alpha)).solve(method="continuation")
    bf, pf, nf = base.front_load_fraction(), pos.front_load_fraction(), neg.front_load_fraction()
    assert pf > nf, f"positive alpha should front-load more: {pf} vs {nf}"
