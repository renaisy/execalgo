"""v0.5 真实数据核校准：响应估计 → 去卷积 → 多指数拟合 → 对执行算法的影响。

用法: python3 scripts/calibrate_kernel.py [--products CU BU]
输出: scripts/out/fig_kernel_{产品}.png、kernel_calibration_summary.json、控制台表

内容:
1. 训练期(2024H2)符号响应 R(τ) 与符号自相关 C(l)（1 分钟分辨率, 预交易基准）;
2. 传播核去卷积（前向混合 + 松弛项, 参数族尾部闭合）与三指数拟合（保完全单调）;
3. 与论文核(三指数拟合 t^{-1/2})对比: 半衰期、持续性 ∫G、前载方向(T15);
4. 回放稳健性: 测试期(2025H1)四策略在拟合核定价下的配对结论是否保持。
"""

import argparse
import json
import pathlib
import sys
from datetime import date

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm, profile_schedule, twap_schedule
from execalgo.kernel_calibration import (
    CalibrationConfig,
    bootstrap_response_bands,
    estimate_kernel,
    kernel_persistence,
)
from execalgo.model import MultiExponential, fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    replay_day,
    window_stats,
)
from execalgo.vwap import from_samples

DATA_DIR = pathlib.Path("/data/ticker_data")
TRAIN_RANGE = (date(2024, 7, 1), date(2024, 12, 31))
TEST_RANGE = (date(2025, 1, 1), date(2025, 6, 30))
MINUTE_TIMES = (np.arange(N_SESSION_MINUTES) + 0.5) / N_SESSION_MINUTES
CONFIG = CalibrationConfig(max_lag=120)


def normalized(kernel: MultiExponential) -> MultiExponential:
    """分钟尺度核 → 求解器归一化时域（lag×225）。"""
    return MultiExponential(kernel.weights, kernel.rates * N_SESSION_MINUTES)


def run_product(symbol, paper):
    print(f"\n===== {symbol}: 核校准 =====")
    train = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
    test = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))
    print(f"训练 {len(train)} 天, 测试 {len(test)} 天")

    kernel, diagnostics, estimate, g = estimate_kernel(train, CONFIG)
    bands = bootstrap_response_bands(train, CONFIG, draws=120)

    lags = estimate["lags"]
    half_lives = np.log(2) / kernel.rates
    paper_minutes = MultiExponential(paper.weights, paper.rates / N_SESSION_MINUTES)
    print(f"C(1)={diagnostics['autocorr_at_one']:.3f}  "
          f"R(1)={diagnostics['response_at_one']:.4g} (每符号事件, 价格单位)")
    print("拟合核(分钟尺度): 权重=" + np.array2string(kernel.weights, precision=3) +
          " 衰减率=" + np.array2string(kernel.rates, precision=4))
    print(f"  半衰期(分钟)={np.array2string(half_lives, precision=1)}  "
          f"持续性∫G(120min)={diagnostics['kernel_persistence_minutes']:.3f}")
    print(f"  拟合最大相对偏差={diagnostics['max_fit_deviation']:.3f}  "
          f"恢复段正性={diagnostics['kernel_positive']}")
    print(f"论文核参照: 半衰期={np.array2string(np.log(2) / paper_minutes.rates, precision=1)}"
          f"  持续性={kernel_persistence(paper_minutes, 120.0):.3f}")

    # T15: 拟合核 vs 论文核下的最优前载（η 用校准值, 与回放定价一致）
    profile_raw = build_volume_profile(train)
    stats = window_stats(train)
    calib = calibrate(stats, paper)
    front = {}
    for name, kern in (("fitted", normalized(kernel)), ("paper", paper)):
        params = ExecParams(share=1.0, eta=calib["eta_solver"], kernel=kern)
        grid_profile = from_samples(params.make_grid(), MINUTE_TIMES, profile_raw)
        front[name] = {
            "IS": ISAlgorithm(params).solve().front_load_fraction(),
            "VWAP": VWAPAlgorithm(params, grid_profile).solve().front_load_fraction(),
        }
    print(f"前半段占比(T15): 拟合核 IS={front['fitted']['IS']:.3f} VWAP={front['fitted']['VWAP']:.3f} | "
          f"论文核 IS={front['paper']['IS']:.3f} VWAP={front['paper']['VWAP']:.3f}")

    # 回放稳健性: 测试期四策略在两种核定价下的配对结论
    robust = replay_robustness(symbol, test, train, kernel, paper, calib, stats)
    figure(symbol, kernel, estimate, g, bands, front, paper)
    return {
        "diagnostics": diagnostics,
        "weights": kernel.weights.tolist(), "rates_minutes": kernel.rates.tolist(),
        "half_lives_minutes": half_lives.tolist(),
        "paper_persistence": kernel_persistence(paper_minutes, 120.0),
        "front_load": front,
        "robustness": robust,
    }


def replay_robustness(symbol, test, train, kernel, paper, calib, stats):
    profile = build_volume_profile(train)
    shares = calib["share"]
    kernels = {"fitted": normalized(kernel), "paper": paper}
    schedules = {}
    for kern_name, kern in kernels.items():
        params = ExecParams(share=1.0, eta=calib["eta_solver"], kernel=kern)
        grid_profile = from_samples(params.make_grid(), MINUTE_TIMES, profile)
        schedules[kern_name] = {
            "VWAP-opt": VWAPAlgorithm(params, grid_profile).solve().slices(N_SESSION_MINUTES)[1],
            "IS-opt": ISAlgorithm(params).solve().slices(N_SESSION_MINUTES)[1],
            "TWAP": np.full(N_SESSION_MINUTES, 1.0 / N_SESSION_MINUTES),
            "naive-VWAP": profile,
        }
    rows = {k: {s: [] for s in schedules[k]} for k in schedules}
    for day in test:
        scale = float(np.sqrt(stats["mean_volume"] / day.minute_volume.sum()))
        for kern_name, kern in kernels.items():
            for name, frac in schedules[kern_name].items():
                rows[kern_name][name].append(
                    replay_day(frac * shares, day, calib["lambda"], calib["eta_p"],
                               kern, volume_scale=scale)
                )
    print("回放稳健性(测试期, 各核自洽定价):")
    result = {}
    for kern_name, sched in rows.items():
        vwap_opt = np.array([r["vwap_bps"] for r in sched["VWAP-opt"]])
        naive = np.array([r["vwap_bps"] for r in sched["naive-VWAP"]])
        is_opt = np.array([r["is_bps"] for r in sched["IS-opt"]])
        twap = np.array([r["is_bps"] for r in sched["TWAP"]])
        diffs = naive - vwap_opt
        se = diffs.std(ddof=1) / np.sqrt(len(diffs))
        result[kern_name] = {
            "naive_minus_vwapopt": (float(diffs.mean()), float(diffs.mean() / se)),
            "twap_minus_isopt": (float((twap - is_opt).mean()),
                                 float((twap - is_opt).mean() /
                                       ((twap - is_opt).std(ddof=1) / np.sqrt(len(twap))))),
        }
        r1, t1 = result[kern_name]["naive_minus_vwapopt"]
        r2, t2 = result[kern_name]["twap_minus_isopt"]
        print(f"  {kern_name:>6}核: naive−VWAPopt={r1:+.2f}bp(t={t1:.1f})  "
              f"TWAP−ISopt={r2:+.2f}bp(t={t2:.1f})")
    return {
        k: {metric: {"mean": values[0], "t": values[1]}
            for metric, values in pairs.items()}
        for k, pairs in result.items()
    }


def figure(symbol, kernel, estimate, g, bands, front, paper):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    lags = estimate["lags"].astype(float)
    response = estimate["response"]
    paper_minutes = MultiExponential(paper.weights, paper.rates / N_SESSION_MINUTES)
    ax.plot(lags, response / response[0], color="0.5", lw=1.5, label="response R(τ)/R(1)")
    ax.fill_between(lags, bands["low"], bands["high"], color="0.8", alpha=0.4,
                    label="bootstrap 90% band")
    fit_length = min(g.size, len(lags))
    ax.plot(lags[:fit_length], g[:fit_length] / g[0], color="C0", lw=1.8,
            label="deconvolved G")
    ax.plot(lags, kernel(lags) / kernel(1.0), color="C3", lw=1.5, ls="--",
            label="fitted kernel")
    ax.plot(lags, paper_minutes(lags) / paper_minutes(1.0), color="C2", lw=1.5, ls=":",
            label="paper kernel")
    ax.set_xlabel("lag τ (minutes)")
    ax.set_ylabel("normalized impact")
    ax.set_title(f"{symbol}: response vs kernels")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    names = ["IS", "VWAP"]
    x = np.arange(2)
    ax.bar(x - 0.18, [front["paper"][n] for n in names], 0.36, color="C2", label="paper kernel")
    ax.bar(x + 0.18, [front["fitted"][n] for n in names], 0.36, color="C3", label="fitted kernel")
    ax.axhline(0.5, color="0.8", lw=0.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("front-half fraction")
    ax.set_title(f"{symbol}: optimal front-loading (T15)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / f"fig_kernel_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="+", default=["CU", "BU"])
    args = parser.parse_args()

    paper, _ = fit_square_root()
    summary = {}
    for symbol in args.products:
        summary[symbol] = run_product(symbol, paper)
    out = ROOT / "scripts" / "out" / "kernel_calibration_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nJSON 汇总已写入 {out}")


if __name__ == "__main__":
    main()
