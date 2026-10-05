"""真实期货数据验证：训练期量型/校准，测试期策略回放回测。

数据: /data/ticker_data 的 L5 快照（Model-LightGBM 项目同源），
      文件时间 = 北京时间 − 5h，volume 为日内累计（夜盘计入次日）。
窗口: 日盘 09:00–15:00 去除小节休息 → 225 个交易分钟，映射到 [0,1]。
流程: 训练期(2024H2)建平均量型 + 校准冲击/滑移尺度 → 规范求解四策略
      → 测试期(2025H1)逐日按真实分钟 VWAP 成交 + 模拟自身冲击回放。

用法: python3 scripts/validate_real.py [--products CU BU] [--quick]
输出: scripts/out/fig_real_{产品}.png、validate_real_summary.json、控制台表
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

from replay_common import paired_t

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm
from execalgo.model import fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    day_volume_density,
    iter_day_sessions,
    replay_day,
    window_stats,
)
from execalgo.replanner import RollingPlanner
from execalgo.vwap import from_samples

DATA_DIR = pathlib.Path("/data/ticker_data")
TRAIN_RANGE = (date(2024, 7, 1), date(2024, 12, 31))
TEST_RANGE = (date(2025, 1, 1), date(2025, 6, 30))
MINUTE_TIMES = (np.arange(N_SESSION_MINUTES) + 0.5) / N_SESSION_MINUTES


def load_product(symbol, rng):
    path = DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_{rng[0].year}.feather"
    if rng[0].year != rng[1].year:
        raise ValueError("range must stay within one calendar year/file")
    return list(iter_day_sessions(path, rng[0], rng[1]))


def run_product(symbol, kernel, participation, quick, risk_grid=None, rolling=False,
                forecast=False):
    print(f"\n===== {symbol}: 加载真实数据 =====")
    train_days = load_product(symbol, TRAIN_RANGE)
    test_days = load_product(symbol, TEST_RANGE)
    if quick:
        train_days, test_days = train_days[:40], test_days[:20]
    print(f"训练 {len(train_days)} 天 ({train_days[0].date}~{train_days[-1].date}), "
          f"测试 {len(test_days)} 天 ({test_days[0].date}~{test_days[-1].date})")

    profile = build_volume_profile(train_days)
    stats = window_stats(train_days)
    calib = calibrate(stats, kernel, participation=participation)
    print(f"训练期: 日均量 {stats['mean_volume']:.0f} 手, 均价 {stats['mean_price']:.0f}, "
          f"窗口波动 σ_w {stats['window_sigma']:.1f}, 半价差 {stats['mean_half_spread']:.1f}")
    print(f"校准: X={calib['share']:.0f} 手 ({participation:.0%} 参与率), "
          f"总冲击成本 {calib['total_bps']:.1f}bp, 滑移占比 {calib['slippage_share']:.2f}, "
          f"Λ={calib['lambda']:.3f}, η_solver={calib['eta_solver']:.4f}")

    # 规范形状问题: share=1, 速率 O(1), 扰动级数良态
    params = ExecParams(share=1.0, eta=calib["eta_solver"], eps=0.5)
    grid_profile = from_samples(params.make_grid(), MINUTE_TIMES, profile)

    schedules = {
        "VWAP-opt": VWAPAlgorithm(params, grid_profile).solve().slices(N_SESSION_MINUTES)[1],
        "IS-opt": ISAlgorithm(params).solve().slices(N_SESSION_MINUTES)[1],
        "TWAP": np.full(N_SESSION_MINUTES, 1.0 / N_SESSION_MINUTES),
        "naive-VWAP": profile,
    }
    front = {
        name: float(frac[: N_SESSION_MINUTES // 2].sum())
        for name, frac in schedules.items()
    }
    print("前半段占比: " + ", ".join(f"{k}={v:.3f}" for k, v in front.items()))

    # oracle: 完美量型信息的朴素 VWAP(逐日按当日真实量型成交)——文献标准的上界基准
    rows = {name: [] for name in list(schedules) + ["oracle-VWAP"]}
    for day in test_days:
        for name, frac in schedules.items():
            rows[name].append(
                replay_day(frac * calib["share"], day, calib["lambda"], calib["eta_p"], kernel)
            )
        rows["oracle-VWAP"].append(
            replay_day(day_volume_density(day) * calib["share"], day,
                       calib["lambda"], calib["eta_p"], kernel)
        )

    def summarize(name):
        is_bps = np.array([r["is_bps"] for r in rows[name]])
        vwap_bps = np.array([r["vwap_bps"] for r in rows[name]])
        return {
            "is_mean": float(is_bps.mean()), "is_std": float(is_bps.std()),
            "vwap_mean": float(vwap_bps.mean()), "vwap_std": float(vwap_bps.std()),
        }

    table = {name: summarize(name) for name in rows}
    pairs = {
        "naive-VWAP minus VWAP-opt (vwap_bps)": paired_t(
            [a["vwap_bps"] - b["vwap_bps"] for a, b in zip(rows["naive-VWAP"], rows["VWAP-opt"])]
        ),
        "TWAP minus IS-opt (is_bps)": paired_t(
            [a["is_bps"] - b["is_bps"] for a, b in zip(rows["TWAP"], rows["IS-opt"])]
        ),
        "TWAP minus VWAP-opt (vwap_bps)": paired_t(
            [a["vwap_bps"] - b["vwap_bps"] for a, b in zip(rows["TWAP"], rows["VWAP-opt"])]
        ),
    }

    print(f"\n{'策略':<12}{'IS口径bp':>18}{'VWAP口径bp':>18}")
    for name, row in table.items():
        print(f"{name:<12}{row['is_mean']:>10.2f}±{row['is_std']:<6.1f}"
              f"{row['vwap_mean']:>10.2f}±{row['vwap_std']:<6.1f}")
    print("配对差(正=优化算法更省):")
    for name, (mean, t) in pairs.items():
        print(f"  {name}: {mean:+.2f}bp (t={t:.1f})")

    figure(symbol, profile, schedules, table, pairs, len(test_days))
    summary = {
        "n_train": len(train_days), "n_test": len(test_days),
        "stats": stats, "calibration": calib, "front_load": front,
        "table": table,
        "pairs": {k: {"mean": v[0], "t": v[1]} for k, v in pairs.items()},
    }
    if risk_grid:
        summary["risk_frontier"] = run_risk_frontier_real(
            symbol, kernel, params, grid_profile, profile, test_days, calib,
            table, risk_grid,
        )
    if rolling:
        summary["rolling"] = run_rolling_comparison(
            symbol, kernel, profile, stats, calib, test_days, schedules,
        )
    if forecast:
        summary["forecast"] = run_forecast_validation(
            symbol, kernel, profile, stats, calib, train_days, test_days, schedules,
        )
    return summary


def run_forecast_validation(symbol, kernel, profile, stats, calib, train_days,
                            test_days, schedules):
    """v0.6 量型预测验证：T16 真实 TV 表 + T17 端到端配对 + 信息价值分解。"""
    from execalgo.profile_forecast import (
        AverageForecaster, OracleForecaster, PrefixForecaster, tail_distance,
    )

    print(f"\n-- {symbol}: 量型预测（T16 真实 TV / T17 端到端 / oracle 信息价值）--")
    train_volumes = np.array([d.minute_volume for d in train_days])
    train_dates = [d.date for d in train_days]
    train_nights = np.array([d.night_volume for d in train_days])
    test_vols = [d.minute_volume for d in test_days]

    fc = PrefixForecaster(train_volumes, train_dates=train_dates,
                          train_night_volumes=train_nights)
    avg = AverageForecaster(train_volumes)
    print(f"{'观测点':>8}{'平均基线TV':>12}{'前缀核TV':>12}{'改善':>8}{'两日间TV(参照)':>16}")
    t16 = {}
    for m in (15, 45, 105):
        err_avg = np.mean([tail_distance(avg.tail(v, m), v[m:]) for v in test_vols])
        err_fc = np.mean([tail_distance(fc.tail(v, m, today=d.date), v[m:])
                          for v, d in zip(test_vols, test_days)])
        noise = np.mean([tail_distance(test_vols[i][m:], test_vols[(i + 7) % len(test_vols)][m:])
                         for i in range(len(test_vols))])
        t16[m] = {"average": float(err_avg), "kernel": float(err_fc),
                  "noise_reference": float(noise)}
        print(f"{m:>6}min{err_avg:>12.4f}{err_fc:>12.4f}{1-err_fc/err_avg:>+8.1%}{noise:>16.4f}")

    # T17 + 信息价值: 平均尾部 / 前缀核尾部 / oracle 尾部 三种滚动
    planners = {
        "roll-avg": RollingPlanner(calib["eta_solver"], kernel, profile,
                                   train_mean_volume=stats["mean_volume"], benchmark="vwap"),
        "roll-fcst": RollingPlanner(
            calib["eta_solver"], kernel, profile,
            train_mean_volume=stats["mean_volume"], benchmark="vwap",
            tail_forecaster=PrefixForecaster(train_volumes, train_dates=train_dates,
                                             train_night_volumes=train_nights)),
        "roll-oracle": RollingPlanner(
            calib["eta_solver"], kernel, profile,
            train_mean_volume=stats["mean_volume"], benchmark="vwap",
            tail_forecaster=OracleForecaster()),
    }
    rows = {name: [] for name in planners}
    for day in test_days:
        scale = float(np.sqrt(stats["mean_volume"] / day.minute_volume.sum()))
        for name, planner in planners.items():
            if name == "roll-oracle":
                planner.forecaster.prime(day_volume=day.minute_volume)
            frac = planner.execute(day.minute_volume, night_volume=day.night_volume,
                                   today=day.date)
            rows[name].append(replay_day(frac * calib["share"], day, calib["lambda"],
                                         calib["eta_p"], kernel, volume_scale=scale))

    print(f"{'策略':<12}{'IS口径bp':>18}{'VWAP口径bp':>18}")
    table = {}
    for name in planners:
        is_bps = np.array([r["is_bps"] for r in rows[name]])
        vwap_bps = np.array([r["vwap_bps"] for r in rows[name]])
        table[name] = {"is_mean": float(is_bps.mean()), "is_std": float(is_bps.std()),
                       "vwap_mean": float(vwap_bps.mean()), "vwap_std": float(vwap_bps.std())}
        print(f"{name:<12}{table[name]['is_mean']:>10.2f}±{table[name]['is_std']:<6.1f}"
              f"{table[name]['vwap_mean']:>10.2f}±{table[name]['vwap_std']:<6.1f}")

    pairs = {}
    for a, b in (("roll-avg", "roll-fcst"), ("roll-avg", "roll-oracle")):
        diffs = [x["vwap_bps"] - y["vwap_bps"] for x, y in zip(rows[a], rows[b])]
        mean, t = paired_t(diffs)
        pairs[f"{a} minus {b}"] = {"mean": mean, "t": t}
        print(f"  {a} − {b}: {mean:+.3f}bp (t={t:.1f})")

    figure_forecast(symbol, t16, table, pairs, len(test_days))
    return {"t16": t16, "t17": {"table": table, "pairs": pairs}}


def figure_forecast(symbol, t16, table, pairs, n_days):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    horizons = sorted(t16)
    x = np.arange(len(horizons))
    ax.bar(x - 0.22, [t16[m]["average"] for m in horizons], 0.44, color="0.6",
           label="train average")
    ax.bar(x + 0.22, [t16[m]["kernel"] for m in horizons], 0.44, color="C0",
           label="prefix kernel")
    ax.plot(x, [t16[m]["noise_reference"] for m in horizons], "k^--", ms=6,
            label="day-vs-day TV (no-info reference)")
    ax.set_xticks(x, [f"{m} min" for m in horizons])
    ax.set_ylabel("tail TV distance")
    ax.set_title(f"{symbol}: tail-shape predictability (T16)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    names = list(table)
    x = np.arange(len(names))
    means = [table[n]["vwap_mean"] for n in names]
    ax.bar(x, means, 0.55, color=["0.6", "C0", "C4"])
    ax.errorbar(x, means, yerr=[table[n]["vwap_std"] / np.sqrt(n_days) for n in names],
                fmt="none", ecolor="0.3", capsize=3)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("mean slippage vs VWAP (bp)")
    gain = pairs["roll-avg minus roll-oracle"]
    ax.set_title(f"{symbol}: value of shape information (oracle = {gain['mean']:+.2f}bp, "
                 f"t={gain['t']:.1f})")
    fig.tight_layout()
    path = out / f"fig_real_{symbol}_forecast.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def run_rolling_comparison(symbol, kernel, profile, stats, calib, test_days, schedules):
    """滚动 vs 静态（T12）：日自适应冲击定价下同口径配对比较。

    该模式对回放启用日自适应冲击尺度（平方根律：量大的日子冲击便宜），
    滚动规划器的 s_vol 信念与之一致；静态策略同样在此口径下回放，
    保证比较公平。绝对数值与默认（窗口级校准）口径不可直接对比。
    """
    print(f"\n-- {symbol}: 滚动重规划 vs 静态（日自适应冲击口径, {len(test_days)} 天）--")
    planners = {
        "roll-VWAP": RollingPlanner(
            calib["eta_solver"], kernel, profile,
            train_mean_volume=stats["mean_volume"], benchmark="vwap",
        ),
        "roll-IS": RollingPlanner(
            calib["eta_solver"], kernel, profile,
            train_mean_volume=stats["mean_volume"], benchmark="is",
        ),
    }
    static_of = {"roll-VWAP": "VWAP-opt", "roll-IS": "IS-opt"}
    metric_of = {"roll-VWAP": "vwap_bps", "roll-IS": "is_bps"}

    rows = {name: [] for name in planners}
    rows.update({static_of[name]: [] for name in planners})
    roll_fractions = {name: [] for name in planners}
    for day in test_days:
        scale = float(np.sqrt(stats["mean_volume"] / day.minute_volume.sum()))
        for name, planner in planners.items():
            frac = planner.execute(day.minute_volume)
            roll_fractions[name].append(frac)
            rows[name].append(
                replay_day(frac * calib["share"], day, calib["lambda"], calib["eta_p"],
                           kernel, volume_scale=scale)
            )
            static_frac = schedules[static_of[name]]
            rows[static_of[name]].append(
                replay_day(static_frac * calib["share"], day, calib["lambda"],
                           calib["eta_p"], kernel, volume_scale=scale)
            )

    print(f"{'策略':<12}{'IS口径bp':>18}{'VWAP口径bp':>18}")
    summary = {}
    for name in ("VWAP-opt", "roll-VWAP", "IS-opt", "roll-IS"):
        is_bps = np.array([r["is_bps"] for r in rows[name]])
        vwap_bps = np.array([r["vwap_bps"] for r in rows[name]])
        summary[name] = {
            "is_mean": float(is_bps.mean()), "is_std": float(is_bps.std()),
            "vwap_mean": float(vwap_bps.mean()), "vwap_std": float(vwap_bps.std()),
        }
        print(f"{name:<12}{summary[name]['is_mean']:>10.2f}±{summary[name]['is_std']:<6.1f}"
              f"{summary[name]['vwap_mean']:>10.2f}±{summary[name]['vwap_std']:<6.1f}")
    print(f"回退次数: roll-VWAP={planners['roll-VWAP'].fallbacks}, "
          f"roll-IS={planners['roll-IS'].fallbacks}; "
          f"重规划总数={planners['roll-VWAP'].replans}")

    pairs = {}
    for roll_name in planners:
        static_name = static_of[roll_name]
        metric = metric_of[roll_name]
        diffs = [a[metric] - b[metric] for a, b in zip(rows[static_name], rows[roll_name])]
        mean, t = paired_t(diffs)
        std_static = summary[static_name][
            "is_std" if metric == "is_bps" else "vwap_std"]
        std_roll = summary[roll_name]["is_std" if metric == "is_bps" else "vwap_std"]
        pairs[f"{static_name} minus {roll_name} ({metric})"] = {"mean": mean, "t": t}
        print(f"  {static_name} − {roll_name} ({metric}): {mean:+.3f}bp (t={t:.1f}), "
              f"std {std_static:.1f} → {std_roll:.1f}bp")

    figure_rolling(symbol, roll_fractions, schedules, summary, pairs)
    return {"table": summary, "pairs": pairs,
            "fallbacks": {k: p.fallbacks for k, p in planners.items()}}


def figure_rolling(symbol, roll_fractions, schedules, summary, pairs):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    minutes = np.arange(1, N_SESSION_MINUTES + 1)
    ax.plot(minutes, schedules["VWAP-opt"] * N_SESSION_MINUTES, color="C3", lw=1.6,
            label="static VWAP-opt")
    ax.plot(minutes, np.mean(roll_fractions["roll-VWAP"], axis=0) * N_SESSION_MINUTES,
            color="C3", lw=1.6, ls="--", label="rolling VWAP-opt (day avg)")
    ax.plot(minutes, schedules["IS-opt"] * N_SESSION_MINUTES, color="C0", lw=1.6,
            label="static IS-opt")
    ax.plot(minutes, np.mean(roll_fractions["roll-IS"], axis=0) * N_SESSION_MINUTES,
            color="C0", lw=1.6, ls="--", label="rolling IS-opt (day avg)")
    ax.axhline(1.0, color="0.85", lw=0.7)
    ax.set_xlabel("trading minute")
    ax.set_ylabel("execution rate × 225")
    ax.set_title(f"{symbol}: static vs rolling schedules")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    names = list(summary)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [summary[n]["vwap_mean"] for n in names], 0.4, color="C3",
           label="vs VWAP")
    ax.bar(x + 0.2, [summary[n]["is_mean"] for n in names], 0.4, color="C0",
           label="vs arrival")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("mean slippage (bp)")
    gain = pairs["VWAP-opt minus roll-VWAP (vwap_bps)"]
    ax.set_title(f"{symbol}: rolling gains {gain['mean']:+.2f}bp (t={gain['t']:.1f}) "
                 "on its benchmark")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / f"fig_real_{symbol}_rolling.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def inventory_time_minutes(fraction):
    """∫(1−C)²dt on the 225-minute timeline (λ-independent exposure)."""
    cumulative = np.cumsum(fraction) - fraction / 2.0
    inventory = 1.0 - cumulative
    return float(np.sum(inventory**2) / N_SESSION_MINUTES)


def run_risk_frontier_real(symbol, kernel, base_params, grid_profile, profile,
                           test_days, calib, table, risk_grid):
    """风险厌恶前沿：真实数据回放下的 (均值, 波动率) 权衡（T7）。"""
    print(f"\n-- {symbol}: 风险前沿（λ 为求解器单位, 基准 λ=0 为上方 VWAP-opt/IS-opt）--")
    families = {"IS-opt": [], "VWAP-opt": []}
    frontier = {name: [] for name in families}
    for risk in risk_grid:
        params = ExecParams(**{**base_params.__dict__, "risk": risk})
        try:
            solved = {
                "IS-opt": ISAlgorithm(params).solve(),
                "VWAP-opt": VWAPAlgorithm(params, grid_profile).solve(),
            }
        except ValueError as error:  # 正基线守卫: 渐近路径越界
            print(f"  λ={risk:g}: {error}")
            continue
        for name, schedule in solved.items():
            frac = schedule.slices(N_SESSION_MINUTES)[1]
            replays = [
                replay_day(frac * calib["share"], day, calib["lambda"], calib["eta_p"], kernel)
                for day in test_days
            ]
            is_bps = np.array([r["is_bps"] for r in replays])
            vwap_bps = np.array([r["vwap_bps"] for r in replays])
            frontier[name].append({
                "risk": risk,
                "is_mean": float(is_bps.mean()), "is_std": float(is_bps.std()),
                "vwap_mean": float(vwap_bps.mean()), "vwap_std": float(vwap_bps.std()),
                "inventory": inventory_time_minutes(frac),
            })

    print(f"{'族':<10}{'λ':>6}{'库存暴露':>10}{'IS均值±std':>18}{'VWAP均值±std':>20}")
    for name, points in frontier.items():
        base = {
            "risk": 0.0,
            "is_mean": table[name]["is_mean"], "is_std": table[name]["is_std"],
            "vwap_mean": table[name]["vwap_mean"], "vwap_std": table[name]["vwap_std"],
            "inventory": inventory_time_minutes(
                _base_fraction(name, base_params, grid_profile, profile)
            ),
        }
        for point in [base] + points:
            print(f"{name:<10}{point['risk']:>6.2f}{point['inventory']:>10.4f}"
                  f"{point['is_mean']:>10.2f}±{point['is_std']:<6.1f}"
                  f"{point['vwap_mean']:>10.2f}±{point['vwap_std']:<8.1f}")
        # T7 检验: 库存暴露与 IS 口径波动率随 λ 单调下降
        inventories = [p["inventory"] for p in points]
        is_stds = [p["is_std"] for p in points]
        if len(points) >= 2:
            assert all(np.diff(inventories) < 1e-12), f"{name}: 库存暴露未随 λ 下降"
            assert all(np.diff(is_stds) < 1e-9), f"{name}: IS 波动率未随 λ 下降"
            print(f"  [T7] {name}: 库存暴露与 IS 口径 std 单调下降 ✓ "
                  f"(IS std {is_stds[0]:.1f}→{is_stds[-1]:.1f}bp)")

    figure_risk(symbol, frontier, table, len(test_days))
    return frontier


def _base_fraction(name, base_params, grid_profile, profile):
    schedule = (
        ISAlgorithm(base_params).solve()
        if name == "IS-opt"
        else VWAPAlgorithm(base_params, grid_profile).solve()
    )
    return schedule.slices(N_SESSION_MINUTES)[1]


def figure_risk(symbol, frontier, table, n_days):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (metric, label) in zip(
        axes, (("is", "vs arrival (bp)"), ("vwap", "vs VWAP (bp)"))
    ):
        for name, color, marker in (("IS-opt", "C0", "o"), ("VWAP-opt", "C3", "s")):
            points = frontier[name]
            if not points:
                continue
            xs = [table[name][f"{metric}_mean"]] + [p[f"{metric}_mean"] for p in points]
            ys = [table[name][f"{metric}_std"]] + [p[f"{metric}_std"] for p in points]
            ax.plot(xs, ys, marker + "-", color=color, lw=1.5, ms=5, label=name)
            for x, y, point in zip(xs[1:], ys[1:], points):
                ax.annotate(f"λ={point['risk']:g}", (x, y), fontsize=7,
                            textcoords="offset points", xytext=(4, 3))
        for ref, color in (("TWAP", "0.4"), ("naive-VWAP", "C2")):
            ax.scatter([table[ref][f"{metric}_mean"]], [table[ref][f"{metric}_std"]],
                       marker="^", color=color, zorder=5, label=ref)
        ax.set_xlabel(f"mean slippage {label}")
        ax.set_ylabel(f"std of daily slippage {label}")
        ax.set_title(f"{symbol}: risk-aversion frontier ({n_days} days)")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / f"fig_real_{symbol}_risk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def figure(symbol, profile, schedules, table, pairs, n_days):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    minutes = np.arange(1, N_SESSION_MINUTES + 1)
    ax.plot(minutes, profile * N_SESSION_MINUTES, color="C2", lw=1.4)
    ax.axhline(1.0, color="0.75", lw=0.8, ls="--")
    ax.set_xlabel("trading minute (09:00 → 15:00, breaks removed)")
    ax.set_ylabel("volume density × 225")
    ax.set_title(f"{symbol}: train-period volume profile")
    for boundary in (75, 135):
        ax.axvline(boundary, color="0.9", lw=0.7)

    ax = axes[1]
    styles = {"VWAP-opt": ("C3", "-"), "IS-opt": ("C0", "-"), "TWAP": ("0.6", "--"),
              "naive-VWAP": ("C2", ":")}
    for name, (color, ls) in styles.items():
        ax.plot(minutes, schedules[name] * N_SESSION_MINUTES, color=color, ls=ls, lw=1.6,
                label=name)
    ax.axhline(1.0, color="0.85", lw=0.7)
    ax.set_xlabel("trading minute")
    ax.set_ylabel("execution rate × 225")
    ax.set_title(f"{symbol}: schedules (share-normalized)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    names = list(table)
    x = np.arange(len(names))
    means = [table[n]["vwap_mean"] for n in names]
    stds = [table[n]["vwap_std"] / np.sqrt(n_days) for n in names]  # standard errors
    ax.bar(x, means, 0.55, yerr=stds, capsize=3,
           color=["C3", "C0", "0.6", "C2", "C4"])
    ax.set_xticks(x, names, fontsize=8, rotation=15)
    ax.set_ylabel("slippage vs day VWAP (bp)")
    gain = pairs["naive-VWAP minus VWAP-opt (vwap_bps)"]
    ax.set_title(f"{symbol}: {n_days} test days "
                 f"(naive−opt = {gain[0]:+.2f}bp, t={gain[1]:.1f})")
    fig.tight_layout()
    path = out / f"fig_real_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="+", default=["CU", "BU"])
    parser.add_argument("--participation", type=float, default=0.05)
    parser.add_argument("--quick", action="store_true", help="short ranges for smoke tests")
    parser.add_argument(
        "--risk-grid", nargs="+", type=float, default=None,
        help="solver-unit risk-aversion values for the frontier replay, e.g. 0.02 0.1 0.5",
    )
    parser.add_argument("--rolling", action="store_true",
                        help="rolling re-planning comparison (day-adaptive impact replay)")
    parser.add_argument("--forecast", action="store_true",
                        help="v0.6 volume-profile forecasting validation (T16/T17 + oracle bound)")
    args = parser.parse_args()

    kernel, _ = fit_square_root()
    summary = {}
    for symbol in args.products:
        summary[symbol] = run_product(
            symbol, kernel, args.participation, args.quick,
            risk_grid=args.risk_grid, rolling=args.rolling, forecast=args.forecast,
        )

    out = ROOT / "scripts" / "out" / "validate_real_summary.json"
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print(f"\nJSON 汇总已写入 {out}")


if __name__ == "__main__":
    main()
