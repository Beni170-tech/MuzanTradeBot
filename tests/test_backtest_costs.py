"""Tests for explicit backtest fee and slippage assumptions."""

import pytest

from backtest.costs import (
    TradingCostModel,
    ZERO_COSTS,
    calculate_trade_costs,
)


def test_zero_cost_default_preserves_gross_pnl():
    result = calculate_trade_costs("LONG", 100.0, 110.0, 2.0)

    assert ZERO_COSTS.fee_rate == 0.0
    assert ZERO_COSTS.slippage_rate == 0.0
    assert result.gross_pnl == 20.0
    assert result.net_pnl == 20.0
    assert result.entry_fill_price == 100.0
    assert result.exit_fill_price == 110.0
    assert result.total_fees == 0.0
    assert result.slippage_cost == 0.0
    assert result.total_costs == 0.0


def test_long_fees_are_charged_on_entry_and_exit():
    result = calculate_trade_costs(
        "LONG",
        100.0,
        110.0,
        2.0,
        TradingCostModel(fee_rate=0.001),
    )

    assert result.gross_pnl == 20.0
    assert result.entry_fee == pytest.approx(0.2)
    assert result.exit_fee == pytest.approx(0.22)
    assert result.total_fees == pytest.approx(0.42)
    assert result.net_pnl == pytest.approx(19.58)
    assert result.total_costs == pytest.approx(0.42)


def test_short_fees_are_charged_on_entry_and_exit():
    result = calculate_trade_costs(
        "SHORT",
        110.0,
        100.0,
        2.0,
        TradingCostModel(fee_rate=0.001),
    )

    assert result.gross_pnl == 20.0
    assert result.entry_fee == pytest.approx(0.22)
    assert result.exit_fee == pytest.approx(0.2)
    assert result.total_fees == pytest.approx(0.42)
    assert result.net_pnl == pytest.approx(19.58)


def test_long_entry_and_exit_slippage_are_adverse():
    result = calculate_trade_costs(
        "LONG",
        100.0,
        110.0,
        2.0,
        TradingCostModel(slippage_rate=0.01),
    )

    assert result.entry_fill_price == pytest.approx(101.0)
    assert result.exit_fill_price == pytest.approx(108.9)
    assert result.gross_pnl == 20.0
    assert result.gross_pnl_after_slippage == pytest.approx(15.8)
    assert result.slippage_cost == pytest.approx(4.2)
    assert result.net_pnl == pytest.approx(15.8)


def test_short_entry_and_exit_slippage_are_adverse():
    result = calculate_trade_costs(
        "SHORT",
        110.0,
        100.0,
        2.0,
        TradingCostModel(slippage_rate=0.01),
    )

    assert result.entry_fill_price == pytest.approx(108.9)
    assert result.exit_fill_price == pytest.approx(101.0)
    assert result.gross_pnl == 20.0
    assert result.gross_pnl_after_slippage == pytest.approx(15.8)
    assert result.slippage_cost == pytest.approx(4.2)
    assert result.net_pnl == pytest.approx(15.8)


def test_fees_and_slippage_combine_in_long_net_pnl():
    result = calculate_trade_costs(
        "LONG",
        100.0,
        110.0,
        10.0,
        TradingCostModel(fee_rate=0.001, slippage_rate=0.01),
    )

    assert result.gross_pnl == pytest.approx(100.0)
    assert result.gross_pnl_after_slippage == pytest.approx(79.0)
    assert result.total_fees == pytest.approx(2.099)
    assert result.slippage_cost == pytest.approx(21.0)
    assert result.total_costs == pytest.approx(23.099)
    assert result.net_pnl == pytest.approx(76.901)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fee_rate", -0.01),
        ("fee_rate", 1.0),
        ("fee_rate", 1.5),
        ("fee_rate", float("nan")),
        ("fee_rate", float("inf")),
        ("fee_rate", True),
        ("fee_rate", "0.01"),
        ("slippage_rate", -0.01),
        ("slippage_rate", 1.0),
        ("slippage_rate", float("nan")),
        ("slippage_rate", float("-inf")),
        ("slippage_rate", False),
        ("slippage_rate", "0.01"),
    ],
)
def test_invalid_cost_configurations_are_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        TradingCostModel(**{field: value})


@pytest.mark.parametrize(
    ("side", "entry", "exit", "quantity", "cost_model"),
    [
        ("SIDEWAYS", 100.0, 110.0, 1.0, ZERO_COSTS),
        (None, 100.0, 110.0, 1.0, ZERO_COSTS),
        ("LONG", 0.0, 110.0, 1.0, ZERO_COSTS),
        ("LONG", 100.0, -1.0, 1.0, ZERO_COSTS),
        ("LONG", 100.0, float("nan"), 1.0, ZERO_COSTS),
        ("LONG", 100.0, 110.0, 0.0, ZERO_COSTS),
        ("LONG", 100.0, 110.0, True, ZERO_COSTS),
        ("LONG", 100.0, 110.0, 1.0, object()),
    ],
)
def test_invalid_cost_inputs_are_rejected(
    side, entry, exit, quantity, cost_model
):
    with pytest.raises((TypeError, ValueError)):
        calculate_trade_costs(side, entry, exit, quantity, cost_model)


def test_cost_overflow_is_rejected_safely():
    with pytest.raises(ValueError, match="not finite"):
        calculate_trade_costs(
            "LONG",
            1e308,
            1e308,
            10.0,
            TradingCostModel(fee_rate=0.5),
        )
