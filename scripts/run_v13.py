"""v1.3 alpha 感知执行：动量信号注入 IS 目标，验证调度时机偏移与成本差异（T24-T26）。

流程: 加载训练/测试期 → 计算每日动量信号（因果）→ 信号 IC（T24）→
  alpha-aware vs alpha-blind 调度对比（T25 时机偏移）→ 回放成本差异（T26）。

用法: python3 scripts/run_v13.py [--product CU] [--lookback 15]
输出: scripts/out/fig_v13_{产品}.png、alpha_execution_v13.json、控制台表
"""

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import ExecParams, ISAlgorithm
from execalgo.model import fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    replay_day,
    window_stats,
)
from execalgo.signal import momentum_signal, signal_ic, solver_alpha
from replay_common import DATA_DIR, MINUTE_TIMES, TEST_RANGE, TRAIN_RANGE, paired_t


def run(symbol, lookback):
    print(f"\n===== {symbol}: v1.3 alpha 感知执行 =====")
    train = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
    test = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))
    kernel, _ = fit_square_root()
    profile = build_volume_profile(train)
    stats = window_stats(train)
    calib = calibrate(stats, kernel)
    print(f"训练 {len(train)} 天, 测试 {len(test)} 天; "
          f"Λ={calib['lambda']:.1f}, η={calib['eta_solver']:.4f}")

    # T24: 信号 IC 从训练期计算（v1.6 P1-8: 原从测试期算导致样本内配置）
    ics_train = [signal_ic(d, lookback_minutes=lookback, forward_minutes=lookback)
                 for d in train]
    ic_mean = float(np.mean([x for x in ics_train if np.isfinite(x)]))
    print(f"[T24] 训练期 IC: {ic_mean:+.4f} → tilt_sign={'+' if ic_mean >= 0 else '-'}")

    # 测试期 IC（仅报告，不用于配置——v2.2 P1-5: 不覆盖训练 IC）
    ics = [signal_ic(d, lookback_minutes=lookback, forward_minutes=lookback)
           for d in test]
    ics_valid = [x for x in ics if np.isfinite(x)]
    ic_test = float(np.mean(ics_valid))
    ic_pos_rate = float(np.mean([x > 0 for x in ics_valid]))
    signal_type = "反转" if ic_test < 0 else "动量"
    print(f"[T24] 测试期 IC: mean={ic_test:+.4f} ({signal_type}), "
          f"正IC日占比={ic_pos_rate:.1%} "
          f"({len(ics_valid)}/{len(test)} 天有效)")
    # IC 可计算且非退化即可（方向由数据决定，不强制正）
    assert len(ics_valid) >= len(test) * 0.9, "T24: IC 应在 ≥90% 的天有效"
    assert abs(ic_mean) > 0.01, "T24: IC 应非退化（|mean|>0.01）"

    # 求解 alpha-blind 基准调度（一次）
    base_params = ExecParams(share=1.0, eta=calib["eta_solver"],
                             grid_size=301, max_order=20)
    base_sched = ISAlgorithm(base_params).solve()
    base_frac = base_sched.slices(N_SESSION_MINUTES)[1]
    base_front = float(base_frac[: N_SESSION_MINUTES // 2].sum())
    print(f"alpha-blind: 前半段占比 = {base_front:.4f}")

    # 信号→倾斜: 延拓对时间变化 RHS 极敏感（噪声 alpha 导致 Newton 发散），
    # 用日级信号方向构造光滑斜坡（信号方向决定前载/后载倾斜）
    RAMP_AMPLITUDE = 0.005
    print(f"信号→倾斜: 日级动量方向 × 幅度 {RAMP_AMPLITUDE} (|IC|={abs(ic_mean):.3f})")

    rows = {"blind": [], "aware": []}
    timing_shifts = []
    solve_failures = 0
    # v1.6 P0-2 fix: use the FIRST-HALF mean (causal — available at the 09:00
    # decision) instead of the full-day mean which leaked afternoon info
    CAUSAL_CUTOFF = N_SESSION_MINUTES // 2
    for day in test:
        sig = momentum_signal(day, lookback_minutes=lookback)
        day_signal = float(np.mean(sig[:CAUSAL_CUTOFF]))

        # 反转(IC<0): 高动量日 → 预期未来回报为负 → 延后买入 → alpha 递增(后载)
        # 动量(IC>0): 高动量日 → 预期未来回报为正 → 立即买入 → alpha 递减(前载)
        tilt_sign = 1.0 if ic_mean < 0 else -1.0
        ramp = np.linspace(-1.0, 1.0, N_SESSION_MINUTES)
        alpha_profile = tilt_sign * RAMP_AMPLITUDE * ramp * np.sign(day_signal)
        alpha_on_grid = np.interp(base_params.make_grid().nodes, MINUTE_TIMES,
                                  alpha_profile)
        aware_params = ExecParams(share=1.0, eta=calib["eta_solver"],
                                  grid_size=301, max_order=20,
                                  alpha=alpha_on_grid)
        try:
            # alpha ~O(1) trips the asymptotic guard; use continuation
            aware_sched = ISAlgorithm(aware_params).solve(method="continuation")
            aware_frac = aware_sched.slices(N_SESSION_MINUTES)[1]
            aware_front = float(aware_frac[: N_SESSION_MINUTES // 2].sum())
            timing_shifts.append(aware_front - base_front)
        except (ValueError, RuntimeError):
            solve_failures += 1
            aware_frac = base_frac
            timing_shifts.append(0.0)

        rows["blind"].append(
            replay_day(base_frac * calib["share"], day,
                       calib["lambda"], calib["eta_p"], kernel))
        rows["aware"].append(
            replay_day(aware_frac * calib["share"], day,
                       calib["lambda"], calib["eta_p"], kernel))

    # T25: 时机偏移方向——正信号日 vs 负信号日的偏移差应与 IC 同号
    print(f"求解失败: {solve_failures}/{len(test)} 天 (回退到 alpha-blind)")
    shifts = np.array(timing_shifts)
    sig_means = [float(np.mean(momentum_signal(d, lookback_minutes=lookback)))
                 for d in test]
    sig_arr = np.array(sig_means)
    pos_days = sig_arr > np.median(sig_arr)
    shift_pos = float(shifts[pos_days].mean())
    shift_neg = float(shifts[~pos_days].mean())
    print(f"[T25] 时机偏移: 高信号日均 {shift_pos:+.4f}, 低信号日均 {shift_neg:+.4f}")
    # 反转信号(IC<0): 高动量日预期未来回报为负→alpha-aware 应后载(shift<0)
    # 动量信号(IC>0): 高动量日预期未来回报为正→alpha-aware 应前载(shift>0)
    expected_sign = 1 if ic_mean > 0 else -1
    if expected_sign > 0:
        assert shift_pos > shift_neg, "T25: 动量信号下高信号日应更前载"
    else:
        assert shift_pos < shift_neg, "T25: 反转信号下高信号日应更后载"
    print(f"      方向正确: {'反转(后载)' if expected_sign < 0 else '动量(前载)'} ✓")

    # T26: 成本差异（如实报告——信号弱则 alpha 感知不产生正收益）
    diffs = [b["is_bps"] - a["is_bps"] for b, a in zip(rows["blind"], rows["aware"])]
    mean_diff, t_stat = paired_t(diffs)
    blind_mean = float(np.mean([r["is_bps"] for r in rows["blind"]]))
    aware_mean = float(np.mean([r["is_bps"] for r in rows["aware"]]))
    blind_std = float(np.std([r["is_bps"] for r in rows["blind"]]))
    aware_std = float(np.std([r["is_bps"] for r in rows["aware"]]))
    print(f"[T26] IS 口径: blind={blind_mean:.2f}±{blind_std:.1f} "
          f"aware={aware_mean:.2f}±{aware_std:.1f} "
          f"配对差={mean_diff:+.3f}bp (t={t_stat:.1f})")
    verdict = "aware 更省" if mean_diff > 0 else "aware 更贵"
    print(f"      {verdict} ({significance_label(t_stat)}) — "
          f"{'弱信号不足以覆盖执行成本' if mean_diff < 0 else '信号方向正确且有净收益'}")

    figure(symbol, ics_valid, shifts, sig_arr, blind_mean, aware_mean)
    return {
        "t24": {"ic_mean": ic_mean, "ic_positive_rate": ic_pos_rate},
        "t25": {"shift_positive_signal": shift_pos, "shift_negative_signal": shift_neg},
        "t26": {"blind": (blind_mean, blind_std), "aware": (aware_mean, aware_std),
                "paired_diff": (mean_diff, t_stat)},
    }


def significance_label(t):
    if abs(t) >= 2.0:
        return "显著"
    if abs(t) >= 1.0:
        return "边际"
    return "不显著"


def figure(symbol, ics, shifts, sig_arr, blind_mean, aware_mean):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.hist(ics, bins=25, color="C0", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="C3", ls="--", lw=1)
    ax.set_xlabel("signal IC (15-min momentum vs 15-min forward)")
    ax.set_ylabel("days")
    ax.set_title(f"{symbol}: signal quality (T24)")

    ax = axes[1]
    ax.scatter(sig_arr, shifts, s=12, alpha=0.5, color="C0")
    z = np.polyfit(sig_arr, shifts, 1)
    xs = np.linspace(sig_arr.min(), sig_arr.max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="C3", lw=1.5,
            label=f"slope={z[0]:.3f}")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.set_xlabel("day mean signal (return)")
    ax.set_ylabel("front-load shift (aware − blind)")
    ax.set_title(f"{symbol}: timing shift vs signal (T25)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / f"fig_v13_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default="CU")
    parser.add_argument("--lookback", type=int, default=15)
    args = parser.parse_args()
    summary = run(args.product, args.lookback)
    out = ROOT / "scripts" / "out" / "alpha_execution_v13.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nJSON 已写入 {out}")


if __name__ == "__main__":
    main()
