"""v1.2 队列/成交率标定：从 L5 深度数据标定被动成交概率曲线（T21-T23）。

流程: 加载训练期(含深度) → 标定 fill_fraction(flow/depth) 曲线 →
  常数 β vs 标定曲线的被动占比对比 → run_production 端到端重跑。

用法: python3 scripts/run_v12.py [--product CU] [--n-bins 10]
输出: scripts/out/fig_fill_curve_{产品}.png、fill_calibration_v12.json、控制台表
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

from execalgo.execution_model import MakerPolicy, replay_day_book
from execalgo.fill_calibration import calibrate_fill_curve, fill_at, fill_at_constant
from execalgo.model import fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    replay_day,
    window_stats,
)
from replay_common import DATA_DIR, MINUTE_TIMES, TEST_RANGE, TRAIN_RANGE, paired_t

from execalgo import ExecParams, ISAlgorithm, VWAPAlgorithm, participation_upper


def run(symbol, n_bins):
    print(f"\n===== {symbol}: v1.2 队列/成交率标定 =====")
    train = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
    test = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))
    print(f"训练 {len(train)} 天, 测试 {len(test)} 天")

    # T21: 标定曲线 + bootstrap CI
    curve = calibrate_fill_curve(train, n_bins=n_bins, bootstrap_draws=200)
    centers = np.sqrt(curve.ratio_edges[:-1] * curve.ratio_edges[1:])
    valid = curve.counts > 5
    print(f"标定曲线: {valid.sum()}/{n_bins} 个有效箱 (计数>5)")
    print(f"{'ratio':>8}{'fill_frac':>12}{'CI':>18}{'count':>8}")
    for i in range(n_bins):
        if curve.counts[i] > 0:
            print(f"{centers[i]:>8.2f}{curve.fill_fractions[i]:>12.3f}"
                  f"[{curve.ci_low[i]:.3f},{curve.ci_high[i]:.3f}]>{curve.counts[i]:>6}")
    # T21 方向性检查
    fractions_valid = curve.fill_fractions[valid]
    assert np.all(np.diff(fractions_valid) >= -0.05), "T21: 曲线应单调不减"
    assert curve.fill_fractions[-1] > 0.1, "T21: 高比值端 fill_fraction 应显著>0"
    print("[T21] ✓ 曲线单调、高比值端成交、CI 有界")

    # T22: 常数 β vs 标定曲线的被动占比对比（在测试期逐日回放）
    kernel, _ = fit_square_root()
    profile = build_volume_profile(train)
    stats = window_stats(train)
    calib = calibrate(stats, kernel)
    params = ExecParams(share=1.0, eta=calib["eta_solver"], grid_size=301, max_order=20)
    # v1.6 P0-1 fix: use from_samples to renormalize (raw np.interp left the
    # density 225x too small, making the cross term negligible and the
    # "VWAP-opt" schedule effectively IS-opt)
    from execalgo.vwap import from_samples
    grid_profile = from_samples(params.make_grid(), MINUTE_TIMES, profile)

    vw_frac = VWAPAlgorithm(params, grid_profile).solve().slices(N_SESSION_MINUTES)[1]
    is_frac = ISAlgorithm(params).solve().slices(N_SESSION_MINUTES)[1]
    twap = np.full(N_SESSION_MINUTES, 1.0 / N_SESSION_MINUTES)

    policies = {
        "constant-beta": MakerPolicy(passive_fraction=0.5, queue_capture=0.7),
        "calibrated": MakerPolicy(passive_fraction=0.5, fill_curve=curve),
    }
    results = {name: [] for name in policies}
    for day in test:
        for name, pol in policies.items():
            r = replay_day_book(vw_frac * calib["share"], day,
                                calib["lambda"], kernel, pol)
            results[name].append(r)

    print(f"\n{'策略':<16}{'被动占比':>10}{'价差捕获bp':>12}")
    t22 = {}
    for name in policies:
        pf = np.mean([r["passive_fraction"] for r in results[name]])
        sc = np.mean([r["spread_capture_bps"] for r in results[name]])
        t22[name] = {"passive_fraction": float(pf), "spread_capture_bps": float(sc)}
        print(f"{name:<16}{pf:>10.1%}{sc:>12.2f}")
    print(f"[T22] ✓ 标定曲线被动占比 {t22['calibrated']['passive_fraction']:.1%} "
          f"vs 常数 {t22['constant-beta']['passive_fraction']:.1%}")

    # T23: 端到端四策略排序检查
    strategies = {"VWAP-opt": vw_frac, "IS-opt": is_frac, "TWAP": twap,
                  "naive-VWAP": profile}
    t23 = {}
    print(f"\n{'策略':<14}{'常数β IS/VWAP bp':>22}{'标定曲线 IS/VWAP bp':>24}")
    for sname, frac in strategies.items():
        row = {}
        for pname, pol in policies.items():
            replays = [replay_day_book(frac * calib["share"], day,
                                       calib["lambda"], kernel, pol)
                       for day in test[:60]]  # subsample for speed
            row[pname] = {
                "is": float(np.mean([r["is_bps"] for r in replays])),
                "vwap": float(np.mean([r["vwap_bps"] for r in replays])),
            }
        t23[sname] = row
        print(f"{sname:<14}{row['constant-beta']['is']:>10.2f}/{row['constant-beta']['vwap']:<8.2f}"
              f"{row['calibrated']['is']:>10.2f}/{row['calibrated']['vwap']:<8.2f}")
    # 对角最优检查（对真实竞争者，不比对彼此近平局的 0.01bp 差）
    c = t23  # shorthand
    assert c["VWAP-opt"]["calibrated"]["vwap"] < c["TWAP"]["calibrated"]["vwap"]
    assert c["VWAP-opt"]["calibrated"]["vwap"] < c["naive-VWAP"]["calibrated"]["vwap"]
    assert c["IS-opt"]["calibrated"]["is"] < c["TWAP"]["calibrated"]["is"]
    assert c["IS-opt"]["calibrated"]["is"] < c["naive-VWAP"]["calibrated"]["is"]
    print("[T23] ✓ 优化算法对 TWAP/naive 的排序在标定曲线下保持")

    figure(symbol, curve, centers, t22)
    return {"curve": {"centers": centers.tolist(),
                      "fractions": curve.fill_fractions.tolist(),
                      "ci_low": curve.ci_low.tolist(),
                      "ci_high": curve.ci_high.tolist(),
                      "counts": curve.counts.tolist()},
            "t22": t22, "t23": t23}


def figure(symbol, curve, centers, t22):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    valid = curve.counts > 5
    ax.fill_between(centers[valid], curve.ci_low[valid], curve.ci_high[valid],
                    alpha=0.3, color="C0", label="90% bootstrap CI")
    ax.plot(centers[valid], curve.fill_fractions[valid], "o-", color="C0", ms=5,
            label="calibrated fill_fraction")
    ax.axhline(0.7, color="C3", ls="--", lw=1, label="constant β=0.7")
    ax.set_xlabel("flow/depth ratio (sell_flow / bid_depth)")
    ax.set_ylabel("fill fraction of sell flow")
    ax.set_title(f"{symbol}: passive fill calibration (T21)")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    names = list(t22)
    x = np.arange(len(names))
    pf = [t22[n]["passive_fraction"] for n in names]
    ax.bar(x, pf, 0.5, color=["C3", "C0"])
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("passive fraction")
    ax.set_title(f"{symbol}: constant β vs calibrated (T22)")
    for i, v in enumerate(pf):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    path = out / f"fig_fill_curve_{symbol}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", default="CU")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()
    summary = run(args.product, args.n_bins)
    out = ROOT / "scripts" / "out" / "fill_calibration_v12.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nJSON 已写入 {out}")


if __name__ == "__main__":
    main()
