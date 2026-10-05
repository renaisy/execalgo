"""v0.4 敏感性与稳健性扫描：全部核心结论在参数扰动下的存活表（T13）。

轴: 参与率 {1%, 5%, 10%} × c_imp {0.5, 1, 2} × 核衰减 {×1.5, ×1.0, ×0.75}
× 品种 {CU, BU, MA, HC}；测试期 2025 全年（H2 为纯留出）。
结论: naive−VWAPopt、TWAP−ISopt、TWAP−VWAPopt（配对差 bp）、前载排序
0.5 < IS-opt < VWAP-opt。存活率 ≥80% 判稳健（T13）。

用法: python3 scripts/sensitivity_scan.py [--products CU BU MA HC]
输出: scripts/out/sensitivity_scan.json、fig_sensitivity.png、控制台存活表
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
from execalgo.model import MultiExponential, fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    replay_day,
    window_stats,
)
from execalgo.sensitivity import robust_verdict, survival_table
from execalgo.vwap import from_samples

DATA_DIR = pathlib.Path("/data/ticker_data")
TRAIN_RANGE = (date(2024, 7, 1), date(2024, 12, 31))
TEST_RANGE = (date(2025, 1, 1), date(2025, 12, 31))
MINUTE_TIMES = (np.arange(N_SESSION_MINUTES) + 0.5) / N_SESSION_MINUTES

PARTICIPATIONS = (0.01, 0.05, 0.10)
C_IMPS = (0.5, 1.0, 2.0)
KERNEL_SCALES = (1.5, 1.0, 0.75)   # 快 / 基准 / 慢（慢核 ×0.75 为正基线安全界）


def run_cell(train, test, stats, profile, kernel, participation, c_imp, kernel_scale):
    kern = MultiExponential(kernel.weights, kernel.rates * kernel_scale)
    calib = calibrate(stats, kern, participation=participation, c_imp=c_imp)
    params = ExecParams(share=1.0, eta=calib["eta_solver"], eps=0.5, kernel=kern)
    grid_profile = from_samples(params.make_grid(), MINUTE_TIMES, profile)
    twap = np.full(N_SESSION_MINUTES, 1.0 / N_SESSION_MINUTES)

    try:
        vwap_frac = VWAPAlgorithm(params, grid_profile).solve().slices(N_SESSION_MINUTES)[1]
    except ValueError:
        vwap_frac = None
    is_frac = ISAlgorithm(params).solve().slices(N_SESSION_MINUTES)[1]

    schedules = {"IS-opt": is_frac, "TWAP": twap, "naive-VWAP": profile}
    if vwap_frac is not None:
        schedules["VWAP-opt"] = vwap_frac

    rows = {name: [] for name in schedules}
    for day in test:
        for name, frac in schedules.items():
            rows[name].append(
                replay_day(frac * calib["share"], day, calib["lambda"], calib["eta_p"], kern)
            )

    pairs = {
        "twap_minus_isopt": paired_t(
            [a["is_bps"] - b["is_bps"] for a, b in zip(rows["TWAP"], rows["IS-opt"])]
        ),
    }
    if vwap_frac is not None:
        pairs["naive_minus_vwapopt"] = paired_t(
            [a["vwap_bps"] - b["vwap_bps"]
             for a, b in zip(rows["naive-VWAP"], rows["VWAP-opt"])]
        )
        pairs["twap_minus_vwapopt"] = paired_t(
            [a["vwap_bps"] - b["vwap_bps"]
             for a, b in zip(rows["TWAP"], rows["VWAP-opt"])]
        )
    front = {"IS-opt": float(is_frac[: N_SESSION_MINUTES // 2].sum())}
    if vwap_frac is not None:
        front["VWAP-opt"] = float(vwap_frac[: N_SESSION_MINUTES // 2].sum())
    return {
        "pairs": pairs, "front": front, "boundary": vwap_frac is None,
        "calibration": {k: calib[k] for k in ("eta_solver", "lambda", "share")},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--products", nargs="+", default=["CU", "BU", "MA", "HC"])
    args = parser.parse_args()

    paper, _ = fit_square_root()
    cells = []
    by_product = {}
    for symbol in args.products:
        print(f"\n===== {symbol} =====")
        train = list(iter_day_sessions(
            DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
        test = list(iter_day_sessions(
            DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))
        profile = build_volume_profile(train)
        stats = window_stats(train)
        print(f"训练 {len(train)} 天, 测试 {len(test)} 天 (2025 全年, H2 留出)")
        by_product[symbol] = {"train": train, "test": test, "stats": stats,
                              "profile": profile}
        for participation in PARTICIPATIONS:
            for c_imp in C_IMPS:
                for kernel_scale in KERNEL_SCALES:
                    cell = run_cell(train, test, stats, profile, paper,
                                    participation, c_imp, kernel_scale)
                    cell.update(product=symbol, participation=participation,
                                c_imp=c_imp, kernel_scale=kernel_scale)
                    cells.append(cell)
        done = sum(1 for c in cells if c["product"] == symbol)
        boundary = sum(1 for c in cells if c["product"] == symbol and c["boundary"])
        print(f"完成 {done} 格 (越界 {boundary})")

    table = survival_table(cells)
    verdicts = robust_verdict(table)
    print("\n== 存活表 (T13, 阈值 80%) ==")
    print(f"{'结论':<24}{'存活/总数':>10}{'存活率':>8}{'中位t':>8}{'中位bp':>9}  判定")
    for conclusion, row in table.items():
        if conclusion == "front_ordering":
            print(f"{conclusion:<24}{row['n_surviving']:>4}/{row['n_cells']:<5}"
                  f"{row['survival_rate']:>8.1%}{'—':>8}{'—':>9}  {verdicts[conclusion]}")
            continue
        print(f"{conclusion:<24}{row['n_surviving']:>4}/{row['n_cells']:<5}"
              f"{row['survival_rate']:>8.1%}{row['median_t']:>8.1f}"
              f"{row['median_bp']:>9.2f}  {verdicts[conclusion]}")
        w = row["weakest"]
        if w:
            print(f"    最弱格: {w['axes']} mean={w['mean']:+.2f}bp t={w['t']:.1f}")

    figure(cells)
    out = ROOT / "scripts" / "out" / "sensitivity_scan.json"
    out.write_text(json.dumps(
        {"table": table, "verdicts": verdicts,
         "cells": cells},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nJSON 已写入 {out}")


def figure(cells):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    products = sorted({c["product"] for c in cells})
    conclusions = ("naive_minus_vwapopt", "twap_minus_isopt")
    titles = ("naive − VWAP-opt (bp)", "TWAP − IS-opt (bp)")
    fig, axes = plt.subplots(len(conclusions), len(products),
                             figsize=(3.1 * len(products), 6.2), squeeze=False)
    for row, (conclusion, title) in enumerate(zip(conclusions, titles)):
        for col, symbol in enumerate(products):
            ax = axes[row][col]
            grid = np.full((len(PARTICIPATIONS), len(C_IMPS)), np.nan)
            for c in cells:
                if c["product"] == symbol and c["kernel_scale"] == 1.0:
                    pair = c["pairs"].get(conclusion)
                    if pair is not None:
                        grid[PARTICIPATIONS.index(c["participation"]),
                             C_IMPS.index(c["c_imp"])] = pair[0]
            vmax = np.nanmax(np.abs(grid))
            im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                                fontsize=8)
            ax.set_xticks(range(len(C_IMPS)), [f"c={c:g}" for c in C_IMPS], fontsize=8)
            ax.set_yticks(range(len(PARTICIPATIONS)),
                          [f"p={p:.0%}" for p in PARTICIPATIONS], fontsize=8)
            if row == 0:
                ax.set_title(symbol, fontsize=10)
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
    fig.suptitle("Sensitivity: paired-mean diffs (paper kernel slice), green = conclusion holds",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / "fig_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


if __name__ == "__main__":
    main()
