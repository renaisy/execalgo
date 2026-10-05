"""v0.4 survival-table helpers (T13)."""

import pytest

from execalgo.sensitivity import robust_verdict, survival_table


def _cell(product, participation, c_imp, kernel_scale, pairs, front=None, boundary=False):
    return {
        "product": product, "participation": participation, "c_imp": c_imp,
        "kernel_scale": kernel_scale, "pairs": pairs, "front": front,
        "boundary": boundary,
    }


def test_survival_rates_and_weakest_cell():
    cells = [
        _cell("CU", 0.05, 1.0, 1.0, {"naive_minus_vwapopt": (1.2, 1.8),
                                     "twap_minus_isopt": (0.8, 1.1),
                                     "twap_minus_vwapopt": (1.4, 1.9)},
              {"IS-opt": 0.65, "VWAP-opt": 0.70}),
        _cell("BU", 0.10, 2.0, 0.75, {"naive_minus_vwapopt": (-0.3, -0.4),
                                      "twap_minus_isopt": (0.5, 0.6),
                                      "twap_minus_vwapopt": (0.9, 1.0)},
              {"IS-opt": 0.63, "VWAP-opt": 0.69}),
        _cell("MA", 0.01, 0.5, 1.5, {"naive_minus_vwapopt": (0.9, 1.5),
                                     "twap_minus_isopt": (0.7, 0.9),
                                     "twap_minus_vwapopt": (1.1, 1.4)},
              {"IS-opt": 0.60, "VWAP-opt": 0.67}),
        _cell("HC", 0.05, 1.0, 1.0, {"naive_minus_vwapopt": None},
              boundary=True),
    ]
    table = survival_table(cells)

    assert table["naive_minus_vwapopt"]["n_cells"] == 3     # boundary excluded
    assert table["naive_minus_vwapopt"]["n_surviving"] == 2
    assert table["naive_minus_vwapopt"]["survival_rate"] == pytest.approx(2 / 3)
    assert table["naive_minus_vwapopt"]["weakest"]["axes"]["product"] == "BU"
    assert table["twap_minus_isopt"]["n_surviving"] == 3
    assert table["front_ordering"]["survival_rate"] == 1.0

    verdicts = robust_verdict(table, threshold=0.8)
    assert verdicts["naive_minus_vwapopt"] == "fragile"
    assert verdicts["twap_minus_isopt"] == "robust"
    assert verdicts["front_ordering"] == "robust"


def test_empty_table_is_safe():
    table = survival_table([])
    assert table["naive_minus_vwapopt"]["survival_rate"] is None
    assert robust_verdict(table)["naive_minus_vwapopt"] is None
