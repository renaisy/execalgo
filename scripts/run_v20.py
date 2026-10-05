"""v2.0 LGBM 方向模型接入：alpha 感知执行 + 真实预测信号（T33-T35）。

流程: 加载 CU 测试期（含 OHLC）→ 逐日 LGBM 推理 → IC 检验(T33) →
  alpha-aware vs blind 调度(T34 时机) → 回放成本差异(T35)。

用法: python3 scripts/run_v20.py [--product CU] [--days 60]
输出: scripts/out/fig_v20_{产品}.png、lgbm_alpha_v20.json、控制台表
"""

import argparse
import json
import pathlib
import sys
from datetime import date

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import ExecParams, ISAlgorithm
from execalgo.lgbm_alpha import compute_lgbm_alpha
from execalgo.model import fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    replay_day,
    window_stats,
)
from execalgo.signal import momentum_signal, signal_ic
from replay_common import DATA_DIR, MINUTE_TIMES, TEST_RANGE, TRAIN_RANGE, paired_t


def run(symbol, n_days):
    print(f"\n===== {symbol}: v2.0 LGBM alpha 感知执行 =====")
    train = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
    test = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))[:n_days]
    kernel, _ = fit_square_root()
    stats = window_stats(train)
    calib = calibrate(stats, kernel)
    print(f"训练 {len(train)} 天, 测试 {len(test)} 天")

    base_params = ExecParams(share=1.0, eta=calib["eta_solver"],
                             grid_size=301, max_order=20)
    base_sched = ISAlgorithm(base_params).solve(method="continuation")
    base_frac = base_sched.slices(N_SESSION_MINUTES)[1]
    base_front = float(base_frac[: N_SESSION_MINUTES // 2].sum())

    # T33: 逐日 LGBM alpha + IC
    lgbm_ics, momentum_ics = [], []
    rows_blind, rows_aware = [], []
    timing_shifts, lgbm_means = [], []

    solve_failures = 0
    RAMP_AMPLITUDE = 0.005  # same safe amplitude as v1.3
    for day in test:
        mean_price = float(np.mean(day.minute_close))
        raw = compute_lgbm_alpha(day, lambda_scale=mean_price / calib["lambda"])

        # IC uses the RAW signal (true signal quality)
        forward = np.zeros(N_SESSION_MINUTES)
        c = day.minute_close
        forward[:-5] = c[5:] / np.maximum(c[:-5], 1e-12) - 1
        valid = raw != 0
        if valid.sum() > 10:
            ic = np.corrcoef(raw[valid], forward[valid])[0, 1]
            lgbm_ics.append(ic)
        m_ic = signal_ic(day, lookback_minutes=15, forward_minutes=15)
        if np.isfinite(m_ic):
            momentum_ics.append(m_ic)

        # v2.1 fix: solver now handles time-varying alpha natively (Armijo
        # line search + long-only box). Use the smoothed signal directly
        # (trailing 15-min) at its natural magnitude — no day-level ramp needed
        kernel_sm = np.ones(15) / 15
        alpha = np.convolve(raw, kernel_sm, mode="full")[:N_SESSION_MINUTES]
        alpha[:15] = raw[:15].cumsum() / np.arange(1, 16)
        lgbm_means.append(float(np.mean(alpha[30:])))

        # alpha-aware solve: v2.1 solver handles per-minute alpha natively
        alpha_grid = np.interp(base_params.make_grid().nodes, MINUTE_TIMES, alpha)
        aware_params = ExecParams(share=1.0, eta=calib["eta_solver"],
                                  grid_size=301, max_order=20, alpha=alpha_grid)
        try:
            aware = ISAlgorithm(aware_params).solve(method="continuation")
            aware_frac = aware.slices(N_SESSION_MINUTES)[1]
            timing_shifts.append(float(aware_frac[:N_SESSION_MINUTES // 2].sum()) - base_front)
        except (ValueError, RuntimeError):
            solve_failures += 1
            aware_frac = base_frac
            timing_shifts.append(0.0)

        rows_blind.append(replay_day(base_frac * calib["share"], day,
                                     calib["lambda"], calib["eta_p"], kernel))
        rows_aware.append(replay_day(aware_frac * calib["share"], day,
                                     calib["lambda"], calib["eta_p"], kernel))

    # T33: 信号质量
    lgbm_ic = float(np.mean(lgbm_ics)) if lgbm_ics else float("nan")
    mom_ic = float(np.mean(momentum_ics)) if momentum_ics else float("nan")
    print(f"[T33] LGBM IC: {lgbm_ic:+.4f} (正向) vs 动量 IC: {mom_ic:+.4f} (反转)")
    # a positive IC means the signal correctly predicts direction —
    # this is what matters for alpha generation (a negative IC requires
    # signal inversion)
    assert lgbm_ic > 0, f"T33: LGBM 信号应为正向 IC, got {lgbm_ic:.4f}"
    print("      LGBM 方向正确 ✓（IC > 0 即可直接用于 alpha 奖励）")

    # T34: 时机响应——信号与偏移的相关性（比二元分割更稳健）
    print(f"求解失败: {solve_failures}/{len(test)} 天")
    shifts = np.array(timing_shifts)
    sigs = np.array(lgbm_means)
    if np.std(sigs) > 1e-12 and np.std(shifts) > 1e-12:
        corr = float(np.corrcoef(sigs, shifts)[0, 1])
    else:
        corr = 0.0
    pos_days = sigs > np.median(sigs)
    shift_pos = float(shifts[pos_days].mean())
    shift_neg = float(shifts[~pos_days].mean())
    print(f"[T34] 信号-偏移相关: {corr:+.3f} "
          f"(高信号日 {shift_pos:+.4f}, 低信号日 {shift_neg:+.4f})")
    direction_ok = corr > 0 if lgbm_ic > 0 else corr < 0
    print(f"      方向{'正确' if direction_ok else '不正确（信号弱或噪声主导）'}")

    # T35: 成本差异
    diffs = [b["is_bps"] - a["is_bps"] for b, a in zip(rows_blind, rows_aware)]
    mean_diff, t_stat = paired_t(diffs)
    blind_mean = float(np.mean([r["is_bps"] for r in rows_blind]))
    aware_mean = float(np.mean([r["is_bps"] for r in rows_aware]))
    print(f"[T35] IS 口径: blind={blind_mean:.2f} aware={aware_mean:.2f} "
          f"配对差={mean_diff:+.3f}bp (t={t_stat:.1f})")
    print(f"      {'aware 更省' if mean_diff > 0 else 'aware 更贵'} "
          f"({'显著' if abs(t_stat) >= 2 else '边际' if abs(t_stat) >= 1 else '不显著'})")

    figure(symbol, lgbm_ics, shifts, sigs, lgbm_ic)
    return {
        "t33": {"lgbm_ic": lgbm_ic, "momentum_ic": mom_ic},
        "t34": {"shift_positive": shift_pos, "shift_negative": shift_neg},
        "t35": {"blind": blind_mean, "aware": aware_mean,
                "paired_diff": (mean_diff, t_stat)},
    }


def figure(symbol, ics, shifts, sigs, ic_mean):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.hist(ics, bins=20, color="C0", alpha=0.7, edgecolor="white")
    ax.axvline(ic_mean, color="C3", ls="--", lw=1.5,
               label=f"mean IC = {ic_mean:.3f}")
    ax.set_xlabel("daily LGBM IC")
    ax.set_ylabel("days")
    ax.set_title(f"{symbol}: LGBM signal quality (T33)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.scatter(sigs, shifts, s=12, alpha=0.5, color="C0")
    z = np.polyfit(sigs, shifts, 1)
    xs = np.linspace(sigs.min(), sigs.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="C3", lw=1.5, label=f"slope={z[0]:.3f}")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xlabel("day mean LGBM alpha")
    ax.set_ylabel("front-load shift (aware − blind)")
    ax.set_title(f"{symbol}: timing shift vs signal (T34)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / f"fig_v20_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default="CU")
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()
    summary = run(args.product, args.days)
    out = ROOT / "scripts" / "out" / "lgbm_alpha_v20.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nJSON 已写入 {out}")


if __name__ == "__main__":
    main()
