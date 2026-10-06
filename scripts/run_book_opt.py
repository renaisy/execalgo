"""v2.4 盘口优化验证：分钟级快赢 + tick 级策略对比（117 测试日）。

Stage 1（分钟级）: π=1.0 + L1 队列基线 + isotonic 曲线 + 分钟级价差，
  在 TWAP 日程上用 replay_day_book 重放，对照 v1.2 基线（π=0.5、L5、
  日级价差）。
Stage 2（tick 级）: 同一日程过 tick 模拟器，策略对比
  taker / passive_fallback / ladder{30,60} / +gate / +guard，
  主指标为相对 taker 的逐日配对成本差（bp）与被动占比，
  诊断指标为成交后 5 分钟 drift（逆向选择）。

用法: python3 scripts/run_book_opt.py [--symbol CU] [--max-days N]
输出: scripts/out/book_opt_v24.json、fig_book_opt_{symbol}.png、控制台表
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import date

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execalgo.bookdata import load_day_ticks
from execalgo.execution_model import MakerPolicy, replay_day_book
from execalgo.fill_calibration import calibrate_fill_curve
from execalgo.model import fit_square_root
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    window_stats,
)
from execalgo.tick_sim import LadderPolicy, simulate_day
from replay_common import DATA_DIR, TEST_RANGE, TRAIN_RANGE, paired_t

OUT = ROOT / "scripts" / "out"

MINUTE_POLICIES = {
    "v1.2 baseline (pi=.5, L5, day spread)":
        dict(passive_fraction=0.5, queue_basis="l5", spread_mode="day"),
    "auto pi, L5 curve":
        dict(passive_fraction="auto", queue_basis="l5", spread_mode="day"),
    "auto pi, L1 curve":
        dict(passive_fraction="auto", queue_basis="l1", spread_mode="day"),
    "auto pi, L1 curve, minute spread":
        dict(passive_fraction="auto", queue_basis="l1", spread_mode="minute"),
}

TICK_POLICIES = {
    "taker": LadderPolicy(mode="taker"),
    "passive_fallback": LadderPolicy(mode="passive_fallback"),
    "fallback+guard": LadderPolicy(mode="passive_fallback", cancel_guard=True),
    "ladder30": LadderPolicy(mode="ladder", passive_timeout=30.0),
    "ladder30+gate": LadderPolicy(mode="ladder", passive_timeout=30.0,
                                  gate="microprice"),
    "ladder30+gate+guard": LadderPolicy(mode="ladder", passive_timeout=30.0,
                                        gate="microprice", cancel_guard=True),
}


def stage1_minute(symbol, train, test, calib, kernel):
    print(f"\n----- Stage 1: 分钟级（TWAP, X={calib['share']:.0f} 手） -----")
    curves = {
        "l5": calibrate_fill_curve(train, queue_basis="l5"),
        "l1": calibrate_fill_curve(train, queue_basis="l1"),
    }
    twap = np.full(N_SESSION_MINUTES, calib["share"] / N_SESSION_MINUTES)

    rows = []
    for name, spec in MINUTE_POLICIES.items():
        policy = MakerPolicy(passive_fraction=spec["passive_fraction"],
                             fill_curve=curves[spec["queue_basis"]],
                             spread_mode=spec["spread_mode"])
        replays = [replay_day_book(twap, d, calib["lambda"], kernel,
                                   policy=policy)
                   for d in test]
        rows.append({
            "policy": name,
            "passive_fraction": float(np.mean([r["passive_fraction"] for r in replays])),
            "is_bps": float(np.mean([r["is_bps"] for r in replays])),
            "vwap_bps": float(np.mean([r["vwap_bps"] for r in replays])),
            "spread_capture_bps": float(np.mean([r["spread_capture_bps"] for r in replays])),
        })
    base = rows[0]["is_bps"]
    print(f"{'policy':>38}{'passive':>9}{'is_bps':>8}{'save':>8}{'capture':>9}")
    for r in rows:
        print(f"{r['policy']:>38}{r['passive_fraction']:>9.1%}"
              f"{r['is_bps']:>8.2f}{base - r['is_bps']:>8.2f}"
              f"{r['spread_capture_bps']:>9.2f}")
    return rows


def stage2_tick(symbol, test, calib, max_days):
    days = test if max_days is None else test[:max_days]
    print(f"\n----- Stage 2: tick 级（{len(days)} 天, 同一 TWAP 日程） -----")
    path = DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather"
    twap = np.full(N_SESSION_MINUTES, calib["share"] / N_SESSION_MINUTES)

    per_policy = {name: [] for name in TICK_POLICIES}
    t0 = time.time()
    for i, day in enumerate(days):
        try:
            ticks = load_day_ticks(path, day.date)
        except ValueError as err:
            print(f"  skip {day.date}: {err}")
            continue
        arrival = float(ticks.mid[0])
        if not np.isfinite(arrival):
            print(f"  skip {day.date}: no opening quote")
            continue
        tape_vwap = float((ticks.last * ticks.vol_inc).sum() / ticks.vol_inc.sum())
        for name, pol in TICK_POLICIES.items():
            r = simulate_day(ticks, twap, pol)
            per_policy[name].append({
                "passive_fraction": r.passive_fraction,
                "is_bps": (r.avg_price - arrival) / arrival * 1e4,
                "vwap_bps": (r.avg_price - tape_vwap) / tape_vwap * 1e4,
                "drift_passive_bps": r.adverse_drift_bps(ticks, "passive"),
                "drift_taker_bps": r.adverse_drift_bps(ticks, "taker"),
                "shortfall": r.shortfall,
                "escalations": r.escalations,
            })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(days)} 天 ({time.time() - t0:.0f}s)")

    taker_is = np.array([d["is_bps"] for d in per_policy["taker"]])
    rows = []
    print(f"\n{'policy':>22}{'passive':>9}{'is_bps':>8}{'save':>8}{'t':>6}"
          f"{'drift_p':>9}{'drift_t':>9}{'esc/day':>8}")
    for name, days_data in per_policy.items():
        if not days_data:
            continue
        is_bps = np.array([d["is_bps"] for d in days_data])
        passive = np.array([d["passive_fraction"] for d in days_data])
        drift_p = np.array([d["drift_passive_bps"] for d in days_data])
        drift_t = np.array([d["drift_taker_bps"] for d in days_data])
        esc = np.array([d["escalations"] for d in days_data])
        save, tstat = paired_t(taker_is - is_bps)
        rows.append({
            "policy": name, "days": len(days_data),
            "passive_fraction": float(passive.mean()),
            "is_bps": float(is_bps.mean()),
            "save_vs_taker_bps": save, "t_stat": tstat,
            "drift_passive_bps": float(np.nanmean(drift_p)),
            "drift_taker_bps": float(np.nanmean(drift_t)),
            "escalations_per_day": float(esc.mean()),
        })
        print(f"{name:>22}{passive.mean():>9.1%}{is_bps.mean():>8.2f}"
              f"{save:>8.3f}{tstat:>6.1f}{np.nanmean(drift_p):>9.3f}"
              f"{np.nanmean(drift_t):>9.3f}{esc.mean():>8.0f}")
    return rows


def make_figure(symbol, minute_rows, tick_rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    names = [r["policy"] for r in tick_rows]
    saves = [r["save_vs_taker_bps"] for r in tick_rows]
    passive = [r["passive_fraction"] * 100 for r in tick_rows]
    colors = ["#888"] * len(names)
    for i, n in enumerate(names):
        if "gate" in n:
            colors[i] = "#d62728"
        elif "ladder" in n:
            colors[i] = "#1f77b4"
        elif "fallback" in n:
            colors[i] = "#2ca02c"
    y = np.arange(len(names))
    axes[0].barh(y, saves, color=colors)
    axes[0].set_yticks(y, names, fontsize=9)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("saving vs taker (bp/day, paired mean)")
    axes[0].set_title(f"{symbol} tick-level policies vs taker")
    axes[1].barh(y, passive, color=colors)
    axes[1].set_yticks(y, names, fontsize=9)
    axes[1].set_xlabel("passive fill fraction (%)")
    axes[1].set_title("passive share (v1.2 minute model: 17.9%)")
    fig.suptitle(f"v2.4 order-book optimization — {symbol} 2025H1", y=1.02)
    fig.tight_layout()
    out = OUT / f"fig_book_opt_{symbol}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"图: {out}")


def run(symbol, max_days):
    print(f"===== {symbol}: v2.4 盘口优化验证 =====")
    train = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
    test = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))
    print(f"训练 {len(train)} 天, 测试 {len(test)} 天")

    kernel, _ = fit_square_root()
    stats = window_stats(train)
    calib = calibrate(stats, kernel)
    print(f"X = {calib['share']:.0f} 手 (5% 参与率), "
          f"半价差 {calib['half_spread_bps']:.2f}bp, "
          f"总成本目标 {calib['total_bps']:.1f}bp")

    minute_rows = stage1_minute(symbol, train, test, calib, kernel)
    tick_rows = stage2_tick(symbol, test, calib, max_days)

    best = max(tick_rows, key=lambda r: r["save_vs_taker_bps"])
    acceptance = {
        "passive_ge_40pct": best["passive_fraction"] >= 0.40,
        "save_ge_0_3bp": best["save_vs_taker_bps"] >= 0.30,
        "best_policy": best["policy"],
        "v1_2_minute_passive": 0.179,
    }
    print("\n验收: 被动占比 >=40%: {}, 净节省 >=0.3bp: {} (最优: {})".format(
        acceptance["passive_ge_40pct"], acceptance["save_ge_0_3bp"],
        best["policy"]))

    OUT.mkdir(exist_ok=True)
    payload = {
        "symbol": symbol,
        "share": calib["share"],
        "half_spread_bps": calib["half_spread_bps"],
        "total_bps": calib["total_bps"],
        "minute_stage": minute_rows,
        "tick_stage": tick_rows,
        "acceptance": acceptance,
    }
    out = OUT / "book_opt_v24.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"结果: {out}")

    try:
        make_figure(symbol, minute_rows, tick_rows)
    except Exception as err:  # figure is cosmetic; never fail the run
        print(f"(figure skipped: {err})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="CU")
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    run(args.symbol, args.max_days)
