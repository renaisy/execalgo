"""v2.6 生产路径端到端：滚动重规划轨迹 + tick 级盘口执行。

分解"轨迹层"与"微观层"的成本贡献（同一 X、同一测试日、tick 级成交）:
  A planner+taker     — 滚动重规划轨迹（IS 基准），纯吃单执行
  B planner+fallback  — 同一轨迹，整分钟挂单 + 分钟末强制补全
  C twap+fallback     — TWAP 轨迹 + 同一微观策略
  D twap+taker        — 双层基线

配对差: B−A = 微观层贡献; C−B = 轨迹层贡献（同微观）; C−D = 合计。
冲击项不在 tick 层定价（两臂共模），配对差只反映微观执行差异。

用法: python3 scripts/run_v26.py [--symbol CU] [--days 20]
输出: scripts/out/book_exec_v26_{symbol}.json + 控制台表
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

from execalgo.bookdata import load_day_ticks
from execalgo.model import fit_square_root
from execalgo.planner_book import execute_with_book
from execalgo.realdata import (
    N_SESSION_MINUTES,
    build_volume_profile,
    calibrate,
    iter_day_sessions,
    window_stats,
)
from execalgo.replanner import RollingPlanner
from execalgo.tick_sim import LadderPolicy, simulate_day
from replay_common import DATA_DIR, TEST_RANGE, TRAIN_RANGE, paired_t

OUT = ROOT / "scripts" / "out"


def run(symbol, n_days):
    print(f"===== {symbol}: v2.6 生产路径端到端（planner + tick 执行） =====")
    train = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2024.feather", *TRAIN_RANGE))
    test = list(iter_day_sessions(
        DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather", *TEST_RANGE))[:n_days]
    kernel, _ = fit_square_root()
    profile = build_volume_profile(train)
    stats = window_stats(train)
    calib = calibrate(stats, kernel)
    share = calib["share"]
    print(f"训练 {len(train)} 天, 执行 {len(test)} 天; X={share:.0f} 手")

    twap = np.full(N_SESSION_MINUTES, share / N_SESSION_MINUTES)
    arms = {a: [] for a in ("planner+taker", "planner+fallback",
                            "twap+fallback", "twap+taker")}
    t0 = time.time()
    for i, day in enumerate(test):
        try:
            ticks = load_day_ticks(
                DATA_DIR / f"{symbol}_Comdty" / f"{symbol}_Comdty_2025.feather",
                day.date)
        except ValueError as err:
            print(f"  skip {day.date}: {err}")
            continue
        if not np.isfinite(ticks.mid[0]):
            continue
        planner = RollingPlanner(calib["eta_solver"], kernel, profile,
                                 train_mean_volume=stats["mean_volume"],
                                 benchmark="is")
        exec_b = execute_with_book(planner, ticks, share,
                                   LadderPolicy(mode="passive_fallback"))
        frac = exec_b.fractions
        r_a = simulate_day(ticks, frac * share, LadderPolicy(mode="taker"))
        r_c = simulate_day(ticks, twap, LadderPolicy(mode="passive_fallback"))
        r_d = simulate_day(ticks, twap, LadderPolicy(mode="taker"))
        for name, r in (("planner+taker", r_a), ("planner+fallback", exec_b),
                        ("twap+fallback", r_c), ("twap+taker", r_d)):
            arms[name].append({
                "is_bps": r.is_bps(ticks) if hasattr(r, "is_bps")
                          else (r.avg_price - ticks.mid[0]) / ticks.mid[0] * 1e4,
                "passive_fraction": r.passive_fraction,
            })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test)} 天 ({time.time() - t0:.0f}s)")

    def col(name, key):
        return np.array([d[key] for d in arms[name]])

    print(f"\n{'arm':>18}{'passive':>9}{'is_bps':>8}")
    summary = {}
    for name in arms:
        summary[name] = {
            "passive_fraction": float(col(name, "passive_fraction").mean()),
            "is_bps": float(col(name, "is_bps").mean()),
            "days": len(arms[name]),
        }
        print(f"{name:>18}{summary[name]['passive_fraction']:>9.1%}"
              f"{summary[name]['is_bps']:>8.2f}")

    pairs = {
        "micro (B-A): planner fallback vs taker":
            col("planner+taker", "is_bps") - col("planner+fallback", "is_bps"),
        "trajectory (B-C): planner vs TWAP, same micro":
            col("twap+fallback", "is_bps") - col("planner+fallback", "is_bps"),
        "combined (D-C): twap fallback vs taker":
            col("twap+taker", "is_bps") - col("twap+fallback", "is_bps"),
    }
    print(f"\n{'配对差（正 = 第二个更省）':>44}{'save_bps':>10}{'t':>7}")
    pair_stats = {}
    for name, diffs in pairs.items():
        save, tstat = paired_t(diffs)
        pair_stats[name] = {"save_bps": save, "t_stat": tstat}
        print(f"{name:>44}{save:>10.3f}{tstat:>7.1f}")

    OUT.mkdir(exist_ok=True)
    out = OUT / f"book_exec_v26_{symbol}.json"
    out.write_text(json.dumps({
        "symbol": symbol, "share": share,
        "arms": summary, "paired": pair_stats,
    }, indent=2, ensure_ascii=False))
    print(f"\n结果: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="CU")
    parser.add_argument("--days", type=int, default=20)
    args = parser.parse_args()
    run(args.symbol, args.days)
