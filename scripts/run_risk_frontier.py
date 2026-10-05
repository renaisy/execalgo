"""风险厌恶前沿（合成参数）：λ 扫描下的期望成本-库存暴露权衡。

用法: python3 scripts/run_risk_frontier.py
输出: scripts/out/fig_risk_frontier.png 及控制台表
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm, profile_schedule, twap_schedule
from execalgo.vwap import u_shaped

IS_RISK_GRID = (0.0, 0.005, 0.02, 0.1, 0.5, 2.0)
# The VWAP family's asymptotic path needs a positive linear baseline; with a
# front-loaded profile the benchmark give-back pushes the tail negative as
# risk grows, so its scan stops inside the valid regime (v0.3's constrained
# solver will lift this boundary).
VWAP_RISK_GRID = (0.0, 0.005, 0.02, 0.05, 0.1)
PROFILE_SHARPNESS = 2.0


def inventory_time(schedule):
    """∫(1 − C_u)²dt: λ 无关的库存暴露指标（持货时间加权）。"""
    rate = schedule.rate
    weights = schedule.grid.weights
    cumulative = np.cumsum(weights * rate) - weights * rate / 2.0
    inventory = 1.0 - cumulative / schedule.params.share
    return float(weights @ (inventory**2))


def solve_family(base: ExecParams, profile):
    rows = []
    for risk in IS_RISK_GRID:
        params = ExecParams(**{**base.__dict__, "risk": risk})
        is_sched = ISAlgorithm(params).solve()
        row = {
            "risk": risk,
            "is_cost": is_sched.expected_costs()["is"],
            "vwap_cost": None,
            "inventory": inventory_time(is_sched),
            "vwap_inventory": None,
            "front": is_sched.front_load_fraction(),
            "is_rate": is_sched.rate,
        }
        if risk in VWAP_RISK_GRID:
            vwap_sched = VWAPAlgorithm(params, profile).solve()
            row["vwap_cost"] = vwap_sched.expected_costs(profile)["vwap"]
            row["vwap_inventory"] = inventory_time(vwap_sched)
        rows.append(row)
    return rows


def main():
    out_dir = ROOT / "scripts" / "out"
    out_dir.mkdir(exist_ok=True)

    base = ExecParams(share=1.0, eta=0.2055, eps=0.5)
    grid = base.make_grid()
    profile = u_shaped(grid, sharpness=PROFILE_SHARPNESS)
    nodes = grid.nodes

    rows = solve_family(base, profile)
    twap = twap_schedule(base)
    naive = profile_schedule(base, profile)

    print("== 风险厌恶前沿（规范单位, share=1, η=0.2055, ε=0.5, "
          f"U 型市场分布 sharpness={PROFILE_SHARPNESS:g}）==")
    print(f"{'λ':>8}{'前半段占比':>10}{'库存暴露∫I²':>12}{'期望IS成本':>12}{'期望VWAP滑点':>14}")
    for row in rows:
        vwap_cell = f"{row['vwap_cost']:>14.6f}" if row["vwap_cost"] is not None else f"{'(越界)':>14}"
        print(f"{row['risk']:>8.3f}{row['front']:>10.4f}{row['inventory']:>12.4f}"
              f"{row['is_cost']:>12.4f}{vwap_cell}")
    for name, sched in (("TWAP", twap), ("naive-VWAP", naive)):
        costs = sched.expected_costs(profile)
        print(f"{name:>8}{sched.front_load_fraction():>10.4f}{inventory_time(sched):>12.4f}"
              f"{costs['is']:>12.4f}{costs['vwap']:>14.6f}")

    inventory = [row["inventory"] for row in rows]
    assert all(np.diff(inventory) < 1e-12), "库存暴露应随 λ 单调下降"
    assert rows[-1]["front"] > rows[0]["front"], "λ 增大应更前载"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for row, color in zip(rows, plt.cm.viridis(np.linspace(0.15, 0.9, len(rows)))):
        ax.plot(nodes, row["is_rate"], color=color, lw=1.5, label=f"λ={row['risk']:g}")
    ax.plot(nodes, np.full(nodes.size, 1.0), color="0.7", ls="--", lw=1, label="TWAP")
    ax.set_xlabel("t")
    ax.set_ylabel("execution rate")
    ax.set_title("IS trajectories across risk aversion")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.plot([r["is_cost"] for r in rows], inventory, "o-", color="C0", label="IS family")
    vwap_rows = [r for r in rows if r["vwap_cost"] is not None]
    ax.plot([r["vwap_cost"] for r in vwap_rows],
            [r["vwap_inventory"] for r in vwap_rows],
            "s-", color="C3", label="VWAP family")
    costs_t = twap.expected_costs(profile)
    costs_n = naive.expected_costs(profile)
    ax.scatter([costs_t["is"]], [inventory_time(twap)], marker="^", color="0.4", zorder=5, label="TWAP")
    ax.scatter([costs_n["is"]], [inventory_time(naive)], marker="v", color="C2", zorder=5, label="naive-VWAP")
    ax.set_xlabel("expected cost (impact + slippage)")
    ax.set_ylabel(r"inventory exposure $\int(1-C_u)^2dt$")
    ax.set_title("Mean-cost vs inventory-risk frontier")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig_risk_frontier.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n图已写入 {path}")


if __name__ == "__main__":
    main()
