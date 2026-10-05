"""v1.0 生产模式：参与率约束 + 盘口 maker/taker 执行（CU 真实数据演示）。

对比维度:
1. 参与率上限 (每分钟 ≤ 8% 预期市场量) 前后的最优轨迹与成本;
2. 旧执行模型 (线性滑移 η·v) vs 盘口模型 (被动队列捕获 + 吃单付价差);
3. T19 检验: 被动成交率与价差捕获处于经验带。

用法: python3 scripts/run_production.py [--product CU]
输出: scripts/out/fig_production_{产品}.png、production_summary.json、控制台表
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

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm, participation_upper
from execalgo.execution_model import MakerPolicy, replay_day_book
from execalgo.model import fit_square_root
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
PART_CAP = 0.08


def apply_cap(frac, cap_frac):
    """把分钟分数截到参与率上限并守恒重分配（供 TWAP/naive 公平对照）。

    `cap_frac` 已经是每分钟分数单位。
    """
    out = np.array(frac, dtype=float)
    for _ in range(200):
        over = out > cap_frac
        if not over.any():
            break
        excess = float((out[over] - cap_frac[over]).sum())
        out[over] = cap_frac[over]
        room = ~over
        if not room.any():
            break
        out[room] += excess * out[room] / out[room].sum()
    return out


def run_product(symbol):
    print(f"\n===== {symbol}: 生产模式 (参与率上限 {PART_CAP:.0%} + 盘口执行) =====")
    train = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
    test = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))
    kernel, _ = fit_square_root()
    profile = build_volume_profile(train)
    stats = window_stats(train)
    calib = calibrate(stats, kernel)
    print(f"训练 {len(train)} 天, 测试 {len(test)} 天; X={calib['share']:.0f} 手")

    grid = ExecParams(share=1.0, eta=calib["eta_solver"]).make_grid()
    upper = participation_upper(PART_CAP, profile, stats["mean_volume"], grid,
                                calib["share"])
    grid_profile = from_samples(grid, MINUTE_TIMES, profile)
    cap_minute_frac = np.interp(MINUTE_TIMES, grid.nodes, upper) / N_SESSION_MINUTES

    # 无约束解 (对照) 与约束解
    plain_params = ExecParams(share=1.0, eta=calib["eta_solver"], grid_size=grid.size,
                              max_order=20)
    capped_params = ExecParams(share=1.0, eta=calib["eta_solver"], grid_size=grid.size,
                               max_order=20, upper_rate=upper)
    schedules = {}
    schedules["VWAP-opt-cap"] = VWAPAlgorithm(
        capped_params, grid_profile).solve(method="continuation").slices(N_SESSION_MINUTES)[1]
    schedules["IS-opt-cap"] = ISAlgorithm(
        capped_params).solve(method="continuation").slices(N_SESSION_MINUTES)[1]
    schedules["VWAP-opt"] = VWAPAlgorithm(
        plain_params, grid_profile).solve(method="continuation").slices(N_SESSION_MINUTES)[1]
    schedules["IS-opt"] = ISAlgorithm(
        plain_params).solve(method="continuation").slices(N_SESSION_MINUTES)[1]
    schedules["TWAP"] = apply_cap(np.full(N_SESSION_MINUTES, 1.0 / N_SESSION_MINUTES),
                                  cap_minute_frac)
    schedules["naive-VWAP"] = apply_cap(profile, cap_minute_frac)

    from execalgo.constrained import solve_box_linear
    box = solve_box_linear(grid.cost_operator(calib["eta_solver"]), grid.weights,
                           np.zeros(grid.size), 1.0, upper)
    print(f"T18: 约束基线 KKT 残差 = {box.kkt_residual:.2e}, "
          f"上界激活节点 = {int(box.active_upper.sum())}/{grid.size}")
    for name in ("VWAP-opt-cap", "IS-opt-cap", "VWAP-opt", "IS-opt"):
        frac = schedules[name]
        binding = float((frac / np.interp(MINUTE_TIMES, grid.nodes,
                                          upper / N_SESSION_MINUTES) >= 0.999).mean())
        print(f"  {name:>12}: 前半段 {frac[:N_SESSION_MINUTES//2].sum():.3f}, "
              f"分钟峰值触顶比 {frac.max()/cap_minute_frac.max():.2f}×, "
              f"触顶分钟占比 {binding:.1%}")

    policy = MakerPolicy()
    legacy = {name: [] for name in schedules}
    book = {name: [] for name in schedules}
    for day in test:
        for name, frac in schedules.items():
            legacy[name].append(
                replay_day(frac * calib["share"], day, calib["lambda"], calib["eta_p"], kernel))
            book[name].append(
                replay_day_book(frac * calib["share"], day, calib["lambda"], kernel, policy))

    def summarize(rows, metric):
        values = np.array([r[metric] for r in rows])
        return float(values.mean()), float(values.std())

    print(f"\n{'策略':<14}{'旧模型 IS/VWAP bp':>22}{'盘口模型 IS/VWAP bp':>24}"
          f"{'被动占比':>9}{'价差捕获bp':>11}")
    table = {}
    for name in schedules:
        li, ls = summarize(legacy[name], "is_bps")
        lv, vs = summarize(legacy[name], "vwap_bps")
        bi, bs = summarize(book[name], "is_bps")
        bv, ws = summarize(book[name], "vwap_bps")
        passive = float(np.mean([r["passive_fraction"] for r in book[name]]))
        capture = float(np.mean([r["spread_capture_bps"] for r in book[name]]))
        table[name] = {"legacy": {"is": (li, ls), "vwap": (lv, vs)},
                       "book": {"is": (bi, bs), "vwap": (bv, ws)},
                       "passive": passive, "capture": capture}
        print(f"{name:<14}{li:>10.2f}±{ls:<5.1f}{lv:>6.2f}"
              f"{bi:>11.2f}±{bs:<5.1f}{bv:>6.2f}{passive:>9.1%}{capture:>11.2f}")

    # T19 检验: 被动占比经验带
    passive_values = [np.mean([r["passive_fraction"] for r in book[n]])
                      for n in ("VWAP-opt-cap", "IS-opt-cap", "TWAP")]
    ok = all(0.05 <= p <= 0.5 for p in passive_values)
    print(f"[T19] 被动成交率 {['%.1f%%' % (p*100) for p in passive_values]} "
          f"∈ 经验带 (5%, 50%]: {'✓' if ok else '✗'}")

    figure(symbol, schedules, table, cap_minute_frac)
    return {"t18_kkt": box.kkt_residual, "table": table,
            "passive_band_ok": ok, "participation_cap": PART_CAP}


def figure(symbol, schedules, table, cap_frac):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    minutes = np.arange(1, N_SESSION_MINUTES + 1)

    ax = axes[0]
    ax.plot(minutes, schedules["VWAP-opt"] * N_SESSION_MINUTES, color="C3", lw=1.4,
            ls="--", label="VWAP-opt (uncapped)")
    ax.plot(minutes, schedules["VWAP-opt-cap"] * N_SESSION_MINUTES, color="C3", lw=1.8,
            label=f"VWAP-opt (cap {PART_CAP:.0%})")
    ax.plot(minutes, schedules["IS-opt-cap"] * N_SESSION_MINUTES, color="C0", lw=1.8,
            label=f"IS-opt (cap {PART_CAP:.0%})")
    ax.plot(minutes, cap_frac * N_SESSION_MINUTES, color="0.4", lw=1.0, ls=":",
            label="participation ceiling ×225")
    ax.axhline(1.0, color="0.85", lw=0.7)
    ax.set_xlabel("trading minute")
    ax.set_ylabel("execution rate × 225")
    ax.set_title(f"{symbol}: capped schedules")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    names = ["VWAP-opt-cap", "IS-opt-cap", "TWAP", "naive-VWAP"]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [table[n]["legacy"]["vwap"][0] for n in names], 0.4, color="C2",
           label="legacy (η·v slippage)")
    ax.bar(x + 0.2, [table[n]["book"]["vwap"][0] for n in names], 0.4, color="C1",
           label="book maker/taker")
    ax.set_xticks(x, names, fontsize=8, rotation=12)
    ax.set_ylabel("mean slippage vs VWAP (bp)")
    ax.set_title(f"{symbol}: execution models compared")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out / f"fig_production_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default="CU")
    args = parser.parse_args()
    summary = run_product(args.product)
    out = ROOT / "scripts" / "out" / "production_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nJSON 已写入 {out}")


if __name__ == "__main__":
    main()
