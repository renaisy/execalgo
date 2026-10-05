"""VWAP 算法演示：U 型市场分布下的最优轨迹 vs IS 轨迹。

用法: python3 scripts/run_vwap.py
输出: scripts/out/fig_vwap.png 及控制台对比表
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm
from execalgo.diagnostics import front_load_split
from execalgo.vwap import u_shaped


def main():
    out_dir = ROOT / "scripts" / "out"
    out_dir.mkdir(exist_ok=True)

    params = ExecParams()
    grid = params.make_grid()
    profile = u_shaped(grid, sharpness=8.0)

    s_is = ISAlgorithm(params).solve(method="continuation")
    s_vwap = VWAPAlgorithm(params, profile).solve(method="continuation")

    print("== VWAP（VWAP 基准）vs IS（到达价基准）==")
    print(f"{'轨迹':<22}{'前半段占比':>10}{'期望IS成本':>12}{'期望VWAP滑点':>14}")
    for s in (s_is, s_vwap):
        costs = s.expected_costs(profile)
        split = front_load_split(s.grid, s.rate)
        print(f"{s.label + ' (' + s.method + ')':<22}{split['front_fraction']:>10.4f}"
              f"{costs['is']:>12.6f}{costs['vwap']:>14.6f}")
    print("\n两基准的最优性方程只差一个右端向量 q = X·Qμ；")
    print("VWAP 基准把早执行的冲击部分“返还”（它同样抬高基准），故更前载。")

    figure(grid, profile, s_is, s_vwap, out_dir / "fig_vwap.png")
    print(f"\n图已写入 {out_dir / 'fig_vwap.png'}")


def figure(grid, profile, s_is, s_vwap, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.fill_between(grid.nodes, profile, color="C2", alpha=0.25, label="market volume p(t)")
    ax.axhline(1.0 / grid.nodes[-1], color="0.75", lw=1, ls="--", label="flat (TWAP) density")
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.set_title("U-shaped market volume profile")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(grid.nodes, s_is.rate, color="C0", lw=2, label="IS optimal (arrival)")
    ax.plot(grid.nodes, s_vwap.rate, color="C3", lw=2, label="VWAP optimal (benchmark)")
    ax.axvline(grid.nodes[-1] / 2, color="0.85", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("trading rate v(t)")
    ax.set_title("Optimal trajectories under the two benchmarks")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
