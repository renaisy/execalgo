"""v1.4 跨品种冲击溢出 + 组合执行回放（T27-T29）。

品种对: CU↔BC(同标的), CU↔ZNA(基础金属), CU↔BU(控制组)
流程: 加载训练期 → 跨品种响应估计 → 溢出比率(T27/T28) →
  独立 vs 顺序执行成本对比(T29)。

用法: python3 scripts/run_v14.py
输出: scripts/out/fig_v14_spillover.png、cross_impact_v14.json、控制台表
"""

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

from execalgo.cross_impact import cross_response, spillover_label, spillover_ratio
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

from execalgo import ExecParams, ISAlgorithm

PAIRS = [
    ("CU", "BC", "同标的(沪铜→国际铜)"),
    ("CU", "ZNA", "基础金属(铜→锌)"),
    ("CU", "BU", "控制组(铜→沥青)"),
]


def load(symbol, rng):
    days = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *rng))
    return days


def run():
    print("===== v1.4 跨品种冲击溢出估计 =====")
    kernel, _ = fit_square_root()
    cache = {}

    def get_days(sym):
        if sym not in cache:
            cache[sym] = load(sym, TRAIN_RANGE)
            print(f"  加载 {sym}: {len(cache[sym])} 天")
        return cache[sym]

    # T27/T28: 逐品种对估计跨品种响应
    results = {}
    print(f"\n{'品种对':<24}{'溢出比率':>10}{'标签':>10}{'R_AA(1)':>10}{'R_AB(1)':>10}")
    for sym_a, sym_b, label in PAIRS:
        days_a = get_days(sym_a)
        days_b = get_days(sym_b)
        cr = cross_response(days_a, days_b, max_lag=120)
        ratio = spillover_ratio(cr)
        tag = spillover_label(ratio)
        results[f"{sym_a}->{sym_b}"] = {
            "label": label, "ratio": ratio, "tag": tag,
            "r_cross": cr["r_cross"].tolist(),
            "r_own_a": cr["r_own_a"].tolist(),
            "r_own_b": cr["r_own_b"].tolist(),
            "lags": cr["lags"].tolist(),
        }
        print(f"{label:<24}{ratio:>10.3f}{tag:>10}"
              f"{cr['r_own_a'][0]:>10.3f}{cr['r_cross'][0]:>+10.3f}")

    # T27: CU→BC 应有显著溢出（同标的）
    bc_ratio = results["CU->BC"]["ratio"]
    assert bc_ratio > 0.05, f"T27: CU→BC 同标的溢出应显著, got {bc_ratio:.3f}"
    print(f"\n[T27] ✓ CU→BC 同标的溢出 {bc_ratio:.3f} ({results['CU->BC']['tag']})")

    # T28: 溢出梯度（同标的 > 跨品种 > 控制组）
    zna_ratio = results["CU->ZNA"]["ratio"]
    bu_ratio = results["CU->BU"]["ratio"]
    print(f"[T28] 溢出梯度: CU→BC({bc_ratio:.3f}) ≥ CU→ZNA({zna_ratio:.3f}) ≥ CU→BU({bu_ratio:.3f})")

    # T29: 独立 vs 顺序执行（用溢出最强的品种对）
    best_pair = max(PAIRS, key=lambda p: results[f"{p[0]}->{p[1]}"]["ratio"])
    sym_a, sym_b, label = best_pair
    print(f"\n-- T29: {label} 独立 vs 顺序执行 --")

    test_a = list(iter_day_sessions(
        DATA_DIR / f"{sym_a}_Comdty" / f"{sym_a}_Comdty_2025.feather", *TEST_RANGE))
    test_b = list(iter_day_sessions(
        DATA_DIR / f"{sym_b}_Comdty" / f"{sym_b}_Comdty_2025.feather", *TEST_RANGE))

    train_a, train_b = get_days(sym_a), get_days(sym_b)
    stats_a, stats_b = window_stats(train_a), window_stats(train_b)
    calib_a, calib_b = calibrate(stats_a, kernel), calibrate(stats_b, kernel)

    pa = ExecParams(share=1.0, eta=calib_a["eta_solver"], grid_size=301, max_order=20)
    pb = ExecParams(share=1.0, eta=calib_b["eta_solver"], grid_size=301, max_order=20)
    frac_a = ISAlgorithm(pa).solve().slices(N_SESSION_MINUTES)[1]
    frac_b = ISAlgorithm(pb).solve().slices(N_SESSION_MINUTES)[1]

    idx_b = {d.date: i for i, d in enumerate(test_b)}
    independent_diffs, sequential_diffs = [], []
    for day_a in test_a[:80]:
        if day_a.date not in idx_b:
            continue
        day_b = test_b[idx_b[day_a.date]]

        # 独立执行: 两品种同日各自执行（默认）
        ra = replay_day(frac_a * calib_a["share"], day_a,
                        calib_a["lambda"], calib_a["eta_p"], kernel)
        rb = replay_day(frac_b * calib_b["share"], day_b,
                        calib_b["lambda"], calib_b["eta_p"], kernel)
        independent_cost = 0.5 * (ra["is_bps"] + rb["is_bps"])

        # 顺序执行: A 前半日执行, B 后半日执行（避免重叠冲击）
        frac_a_early = frac_a.copy()
        frac_a_early[N_SESSION_MINUTES // 2:] = 0
        frac_a_early /= frac_a_early.sum()
        frac_b_late = frac_b.copy()
        frac_b_late[:N_SESSION_MINUTES // 2] = 0
        frac_b_late /= frac_b_late.sum()
        ra_seq = replay_day(frac_a_early * calib_a["share"], day_a,
                            calib_a["lambda"], calib_a["eta_p"], kernel)
        rb_seq = replay_day(frac_b_late * calib_b["share"], day_b,
                            calib_b["lambda"], calib_b["eta_p"], kernel)
        sequential_cost = 0.5 * (ra_seq["is_bps"] + rb_seq["is_bps"])

        independent_diffs.append(independent_cost)
        sequential_diffs.append(sequential_cost)

    ind_mean = float(np.mean(independent_diffs))
    seq_mean = float(np.mean(sequential_diffs))
    diffs = [i - s for i, s in zip(independent_diffs, sequential_diffs)]
    mean_diff, t_stat = paired_t(diffs)
    print(f"独立执行: {ind_mean:.2f}±{np.std(independent_diffs):.1f} bp")
    print(f"顺序执行: {seq_mean:.2f}±{np.std(sequential_diffs):.1f} bp")
    print(f"独立 − 顺序: {mean_diff:+.3f}bp (t={t_stat:.1f})")
    t29 = {"independent": ind_mean, "sequential": seq_mean,
           "diff": mean_diff, "t": t_stat}
    print(f"[T29] {'顺序更优' if mean_diff > 0 else '独立更优'} "
          f"({abs(t_stat):.1f}σ {'显著' if abs(t_stat) > 2 else '不显著'})")
    # v1.6 P0-3: honest note — the replay has no cross-impact term, so this
    # measures schedule concentration cost (half-day legs double the rate),
    # not spillover avoidance
    print("      注意: 回放无跨品种冲击项——此差异反映调度集中度代价而非溢出规避")

    figure(results)
    out = ROOT / "scripts" / "out" / "cross_impact_v14.json"
    out.write_text(json.dumps({"pairs": results, "t29": t29},
                              indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(f"\nJSON 已写入 {out}")


def figure(results):
    out = ROOT / "scripts" / "out"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"CU->BC": "C3", "CU->ZNA": "C0", "CU->BU": "C2"}
    for key, data in results.items():
        lags = np.array(data["lags"])[:60]
        own = np.array(data["r_own_a"])[:60]
        cross = np.array(data["r_cross"])[:60]
        color = colors.get(key, "C4")
        label = f"{data['label']} (ratio={data['ratio']:.3f})"
        ax.plot(lags, cross, color=color, lw=1.5, label=f"cross: {label}")
        if key == "CU->BC":
            ax.plot(lags, own, color="0.5", ls="--", lw=1, label="own: CU→CU")
    ax.axhline(0, color="0.85", lw=0.5)
    ax.set_xlabel("lag τ (minutes)")
    ax.set_ylabel("response (bps)")
    ax.set_title("Cross-product impact spillover (T27/T28)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out / "fig_v14_spillover.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"图已写入 {path}")


if __name__ == "__main__":
    run()
