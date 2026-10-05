"""Performance benchmarks: solver, planner, LGBM inference, data loading."""
import sys, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from datetime import date
from execalgo import ExecParams, ISAlgorithm
from execalgo.model import fit_square_root
from execalgo.discretize import ConstrainedSolver
from execalgo.series import coefficients
from execalgo.replanner import N_MINUTES, ReplanConfig, RollingPlanner
from execalgo.realdata import iter_day_sessions

kernel, _ = fit_square_root()
R = {}

# Series (asymptotic, production path)
for size in (301, 601):
    p = ExecParams(share=0.1, grid_size=size)
    g = p.make_grid()
    sv = ConstrainedSolver(g, 0.65)
    t0 = time.perf_counter()
    for _ in range(5):
        coefficients(g, sv, 0.1, 44)
    R[f"series_n{size}"] = (time.perf_counter() - t0) / 5 * 1000

# Continuation (certificate path)
for size in (151, 301):
    p = ExecParams(share=1.0, eta=0.2055, grid_size=size, max_order=20)
    algo = ISAlgorithm(p)
    t0 = time.perf_counter()
    algo.solve(method="continuation")
    R[f"continuation_n{size}"] = (time.perf_counter() - t0) * 1000

# Rolling planner (one day, 15 replans)
t_arr = (np.arange(N_MINUTES) + 0.5) / N_MINUTES
d = np.exp(-4 * t_arr) + 0.6 * np.exp(-4 * (1 - t_arr)) + 0.4
profile = d / d.sum()
planner = RollingPlanner(0.2055, kernel, profile, config=ReplanConfig(interval_minutes=15))
day_vol = profile * 1000
t0 = time.perf_counter()
planner.execute(day_vol)
R["rolling_1day"] = (time.perf_counter() - t0) * 1000

# LGBM inference (one day)
from execalgo.lgbm_alpha import compute_lgbm_alpha
rng = np.random.default_rng(7)
c = 70000 * np.exp(np.cumsum(rng.standard_normal(225) * 1e-3))
td = type("D", (), {
    "minute_open": c.copy(), "minute_high": c * 1.001, "minute_low": c * 0.999,
    "minute_close": c, "minute_volume": rng.lognormal(5, 0.5, 225)
})()
t0 = time.perf_counter()
compute_lgbm_alpha(td, lambda_scale=1228)
R["lgbm_1day"] = (time.perf_counter() - t0) * 1000

# Data loading (CU 2025 H1)
path = "/data/ticker_data/CU_Comdty/CU_Comdty_2025.feather"
t0 = time.perf_counter()
days = list(iter_day_sessions(path, date(2025, 1, 2), date(2025, 6, 30)))
R["loader_CU_6mo"] = (time.perf_counter() - t0) * 1000
R["loader_days"] = len(days)

print("=== Performance Benchmarks ===")
for k in sorted(R):
    v = R[k]
    if isinstance(v, int):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: {v:.1f} ms")
