"""四策略对比：IS / VWAP / TWAP / 朴素VWAP-profile，期望成本 + 蒙特卡洛。

用法: python3 scripts/compare_algorithms.py
输出: scripts/out/fig_compare.png 及控制台对比表
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import (
    ExecParams,
    ISAlgorithm,
    VWAPAlgorithm,
    profile_schedule,
    twap_schedule,
)
from execalgo.diagnostics import front_load_split
from execalgo.simulator import MarketSim
from execalgo.vwap import u_shaped


def main():
    out_dir = ROOT / "scripts" / "out"
    out_dir.mkdir(exist_ok=True)

    params = ExecParams()
    profile = u_shaped(params.make_grid(), sharpness=8.0)

    schedules = [
        ISAlgorithm(params).solve(),
        VWAPAlgorithm(params, profile).solve(),
        twap_schedule(params),
        profile_schedule(params, profile),
    ]
    sim = MarketSim(profile, participation=20.0, s0=100.0, sigma=0.30)

    print("== 策略对比（U 型市场分布，参与率 M/X = 20，σ = 0.30）==")
    header = (f"{'策略':<14}{'前半段占比':>10}{'期望IS':>10}{'期望VWAP滑点':>14}"
              f"{'MC-IS均值±std':>18}{'MC-VWAP均值±std':>20}")
    print(header)
    runs = {}
    for s in schedules:
        run = sim.run(s, n_slices=60, n_paths=20000)
        runs[s.label] = run
        split = front_load_split(s.grid, s.rate)
        print(f"{s.label:<14}{split['front_fraction']:>10.4f}{run['expected_is']:>10.5f}"
              f"{run['expected_vwap']:>14.5f}"
              f"{run['is_paths'].mean():>10.5f}±{run['is_paths'].std():<7.4f}"
              f"{run['vwap_paths'].mean():>10.5f}±{run['vwap_paths'].std():<9.4f}")

    print("\n观察: 各基准的最优算法在自身基准口径下期望滑点最低；")
    print("      朴素 VWAP-profile 与市场同步成交, 滑点波动率最小（对冲性质），")
    print("      但期望滑点最高；优化算法以少量基准风险换取期望成本下降。")

    figure(schedules, runs, out_dir / "fig_compare.png")
    print(f"\n图已写入 {out_dir / 'fig_compare.png'}")


def figure(schedules, runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    labels = [s.label for s in schedules]
    x = np.arange(len(labels))
    is_values = [runs[l]["expected_is"] for l in labels]
    vwap_values = [runs[l]["expected_vwap"] for l in labels]
    ax.bar(x - 0.2, is_values, 0.4, color="C0", label="vs arrival (IS)")
    ax.bar(x + 0.2, vwap_values, 0.4, color="C3", label="vs VWAP")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("expected slippage per share")
    ax.set_title("Deterministic expected costs")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    bins = np.linspace(-0.05, 0.25, 90)
    colors = ["C0", "C3", "C2", "C4"]
    for color, s in zip(colors, schedules):
        ax.hist(runs[s.label]["vwap_paths"], bins=bins, histtype="step", lw=1.6,
                color=color, label=s.label, density=True)
    ax.set_xlabel("realized slippage vs VWAP (per share)")
    ax.set_ylabel("density")
    ax.set_title("Monte Carlo: 20 000 paths, σ = 0.30")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
