"""v0.6 volume-profile forecasting: conditioning must beat the average (T16)."""

from datetime import date, timedelta

import numpy as np
import pytest

from execalgo.model import fit_square_root
from execalgo.profile_forecast import (
    AverageForecaster,
    ForecastConfig,
    OracleForecaster,
    PrefixForecaster,
    tail_distance,
)
from execalgo.realdata import N_SESSION_MINUTES
from execalgo.replanner import ReplanConfig, RollingPlanner

N = N_SESSION_MINUTES


def _regime_shape(kind):
    """Two intraday regimes: a fast open burst and a close-heavy day.

    The open burst decays on a ~19-minute scale so the regime is legible
    from the first-15-minute prefix (real open-heaviness acts similarly).
    """
    t = (np.arange(N) + 0.5) / N
    if kind == "open":
        density = np.exp(-12.0 * t) + 0.3
    else:
        density = np.exp(-5.0 * (1.0 - t)) + 0.3
    return density / density.sum()


def _synthetic_days(seed=3, n_train=60, n_test=40):
    rng = np.random.default_rng(seed)
    days, kinds, dates = [], [], []
    base = date(2025, 3, 3)
    for i in range(n_train + n_test):
        kind = "open" if rng.random() < 0.5 else "close"
        shape = _regime_shape(kind)
        volume = rng.poisson(shape * 5000).astype(float) + 1.0
        days.append(volume)
        kinds.append(kind)
        dates.append(base + timedelta(days=i))
    return (np.array(days[:n_train]), days[n_train:], kinds[n_train:],
            dates[:n_train], dates[n_train:])


def test_t16_conditioning_beats_average_on_bimodal_regimes():
    train, test, kinds, _, _ = _synthetic_days()
    forecaster = PrefixForecaster(train, config=ForecastConfig(shape_bandwidth=0.03))
    average = AverageForecaster(train)

    horizons = (15, 45, 105)
    err_cond, err_avg, err_floor = {}, {}, {}
    for m in horizons:
        err_cond[m] = [tail_distance(forecaster.tail(day, m), day[m:]) for day in test]
        err_avg[m] = [tail_distance(average.tail(day, m), day[m:]) for day in test]
        # the irreducible error: TV between the realized day and its own
        # expected shape (Poisson noise floor)
        err_floor[m] = [
            tail_distance(_regime_shape(kind)[m:], day[m:]) for day, kind in zip(test, kinds)
        ]

    for m in horizons:
        mean_cond = float(np.mean(err_cond[m]))
        mean_avg = float(np.mean(err_avg[m]))
        mean_floor = float(np.mean(err_floor[m]))
        # conditioning cuts the average's error by >= 25% ...
        assert mean_cond < 0.75 * mean_avg, (m, mean_cond, mean_avg)
        # ... and sits near the noise floor (regime identification works)
        assert mean_cond < 1.5 * mean_floor, (m, mean_cond, mean_floor)


def test_tail_mass_semantics():
    """The tail restriction must sum to the predicted remaining fraction."""
    train, test, *_ = _synthetic_days()
    forecaster = PrefixForecaster(train)
    day = test[0]
    for m in (15, 100):
        tail = forecaster.tail(day, m)
        assert tail.size == N - m
        # same mass as the (unknown) realized tail fraction up to forecast error
        realized_fraction = day[m:].sum() / day.sum()
        assert abs(tail.sum() - realized_fraction) < 0.15


def test_weekday_and_night_features_run():
    train, test, _, train_dates, _ = _synthetic_days()
    nights = np.full(len(train), 2000.0)
    forecaster = PrefixForecaster(train, train_dates=train_dates,
                                  train_night_volumes=nights)
    tail = forecaster.tail(test[0], 45, today=date(2025, 6, 4))
    assert tail.size == N - 45
    assert (tail >= 0).all()


def test_planner_accepts_forecaster(kernel):
    train, test, *_ = _synthetic_days()
    profile = train.mean(axis=0)
    profile = profile / profile.sum()
    plain = RollingPlanner(0.2055, kernel, profile, config=ReplanConfig(interval_minutes=45))
    smart = RollingPlanner(
        0.2055, kernel, profile,
        config=ReplanConfig(interval_minutes=45),
        tail_forecaster=PrefixForecaster(train),
    )
    day = test[0]
    base_plan = plain.execute(day)
    smart_plan = smart.execute(day, today=date(2025, 6, 4))
    assert base_plan.sum() == pytest.approx(1.0)
    assert smart_plan.sum() == pytest.approx(1.0)
    # the first segment uses no prefix information: both fall back to an
    # unconditional average (shape-mean vs volume-mean estimators differ slightly)
    assert np.allclose(base_plan[:45], smart_plan[:45], atol=3e-3)
    # afterwards conditioning must move the plan on regime days
    assert not np.allclose(base_plan, smart_plan)


def test_oracle_forecaster_zero_distance():
    train, test, *_ = _synthetic_days()
    oracle = OracleForecaster()
    day = test[0]
    oracle.prime(day_volume=day)
    assert tail_distance(oracle.tail(day, 60), day[60:]) == pytest.approx(0.0)


@pytest.fixture(scope="module")
def kernel():
    return fit_square_root()[0]
