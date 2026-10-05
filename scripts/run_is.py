"""IS 算法演示：扰动级数求解 + 论文数值诊断 + 轨迹图。

用法: python3 scripts/run_is.py
输出: scripts/out/fig_is.png 及控制台诊断表
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import ISAlgorithm
from execalgo.branch import continue_branch
from execalgo.diagnostics import front_load_split, radius_estimate, truncation_study
from execalgo.discretize import ConstrainedSolver
from execalgo.series import coefficients, optimal_truncation_order, partial_sum


def main():
    out_dir = ROOT / "scripts" / "out"
    out_dir.mkdir(exist_ok=True)

    algo = ISAlgorithm()
    params = algo._engine.params
    grid = algo._engine.grid
    eps = params.eps

    series = coefficients(grid, algo._engine.solver, params.share, params.max_order)
    optimizer, _, _ = continue_branch(grid, params.eta, params.share, eps)

    study = truncation_study(grid, series, optimizer, eps)
    radius = radius_estimate(grid, series)
    asymptotic = partial_sum(series, eps, optimal_truncation_order(grid, series, eps))

    print("== IS（到达价基准）扰动求解诊断 ==")
    from execalgo.model import fit_square_root

    _, deviation = fit_square_root()
    print(f"核: 三指数拟合 t^(-1/2), 最大相对偏差 = {deviation:.4f}（论文 ~1.7%）")
    print(f"最优截断阶 m*          = {study['best_order']}（论文 ~9）")
    print(f"基线误差 -> 最优误差    = {study['baseline_error']:.6f} -> {study['best_error']:.3e}"
          f"（改善 {study['improvement']:.1f}x，论文 ~43x）")
    print(f"系数增长因子 / 收敛半径 = {radius['growth']:.4f} / {radius['radius']:.4f}"
          f"（论文 2.24 / 0.446）")
    for name, rate in (("线性基线 eps=0", series[0]), ("渐近截断 eps=0.5", asymptotic)):
        split = front_load_split(grid, rate)
        print(f"{name}: 前半段占比 = {split['front_fraction']:.4f}")

    figure(grid, series, optimizer, asymptotic, study, eps, out_dir / "fig_is.png")
    print(f"\n图已写入 {out_dir / 'fig_is.png'}")


def figure(grid, series, optimizer, asymptotic, study, eps, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(grid.nodes, series[0], "--", color="0.55", label=r"linear baseline $\varepsilon=0$")
    ax.plot(grid.nodes, optimizer, color="C0", lw=2, label=r"continuation $\varepsilon=0.5$")
    ax.plot(grid.nodes, asymptotic, color="C3", lw=1.4, ls=":",
            label=r"asymptotic truncation $\varepsilon=0.5$")
    ax.axvline(grid.nodes[-1] / 2, color="0.85", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("trading rate v(t)")
    ax.set_title("IS optimal trajectory: front-loading")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    orders = np.arange(len(study["errors"]))
    ax.semilogy(orders, study["errors"], "o-", ms=3, color="C0")
    ax.axvline(study["best_order"], color="C3", ls="--", lw=1,
               label=f"optimal order m* = {study['best_order']}")
    ax.set_xlabel("truncation order m")
    ax.set_ylabel(r"$\|v^{[m]} - v^*\|_{L^2}$")
    ax.set_title("Asymptotic truncation error (radius ~0.45 < 0.5)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
