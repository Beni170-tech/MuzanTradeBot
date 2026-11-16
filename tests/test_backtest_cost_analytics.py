"""Tests for gross/net analytics with explicit backtest costs."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from backtest.analytics import compute_backtest_metrics
from backtest.costs import TradingCostModel, ZERO_COSTS, calculate_trade_costs
from backtest.trades import HistoricalTrade
from trader.paper.order import PaperOrderStatus

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
INITIAL_BALANCE = 1000.0


def _closed_trade(
    *,
    side="LONG",
    entry_price=100.0,
    exit_price=110.0,
    quantity=1.0,
    exit_at=None,
):
    if exit_at is None:
        exit_at = START + timedelta(hours=2)
    if side.upper() == "LONG":
        stop_loss = 90.0
        take_profit = 120.0
    else:
        stop_loss = 110.0
        take_profit = 80.0
    entry_at = START + timedelta(hours=1)
    return HistoricalTrade(
        symbol="BTC",
        signal_timestamp=START,
        entry_timestamp=entry_at,
        entry_price=entry_price,
        side=side,
        quantity=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trade_timestamp=entry_at.strftime("%Y-%m-%d %H:%M:%S"),
        exit_timestamp=exit_at,
        exit_price=exit_price,
        exit_status=PaperOrderStatus.TAKEN_PROFIT,
    )


def test_long_metrics_report_net_pnl_and_expose_assumptions():
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    trade = _closed_trade(quantity=10.0)

    result = compute_backtest_metrics([trade], INITIAL_BALANCE, model)

    assert result.cost_model == model
    assert result.gross_profit == pytest.approx(100.0)
    assert result.total_gross_pnl == pytest.approx(100.0)
    assert result.net_profit == pytest.approx(76.901)
    assert result.net_loss == 0.0
    assert result.total_fees == pytest.approx(2.099)
    assert result.total_slippage_cost == pytest.approx(21.0)
    assert result.total_costs == pytest.approx(23.099)
    assert result.total_realized_pnl == pytest.approx(76.901)
    assert result.ending_equity == pytest.approx(1076.901)
    assert result.wins == 1
    assert result.max_drawdown == 0.0


def test_short_metrics_report_net_pnl_after_costs():
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    trade = _closed_trade(
        side="SHORT",
        entry_price=110.0,
        exit_price=100.0,
        quantity=10.0,
    )

    result = compute_backtest_metrics([trade], INITIAL_BALANCE, model)

    assert result.gross_profit == pytest.approx(100.0)
    assert result.total_slippage_cost == pytest.approx(21.0)
    assert result.total_fees == pytest.approx(2.099)
    assert result.total_costs == pytest.approx(23.099)
    assert result.total_realized_pnl == pytest.approx(76.901)
    assert result.ending_equity == pytest.approx(1076.901)


def test_equity_and_drawdown_use_net_pnl_after_costs():
    model = TradingCostModel(fee_rate=0.01)
    winner = _closed_trade(
        entry_price=100.0,
        exit_price=110.0,
        quantity=10.0,
        exit_at=START + timedelta(hours=1),
    )
    loser = _closed_trade(
        entry_price=100.0,
        exit_price=95.0,
        quantity=10.0,
        exit_at=START + timedelta(hours=2),
    )

    result = compute_backtest_metrics(
        [winner, loser],
        INITIAL_BALANCE,
        model,
    )

    # Winner: gross +100, fees 10 + 11 -> net +79.
    # Loser: gross -50, fees 10 + 9.5 -> net -69.5.
    assert result.gross_profit == pytest.approx(100.0)
    assert result.gross_loss == pytest.approx(50.0)
    assert result.net_profit == pytest.approx(79.0)
    assert result.net_loss == pytest.approx(69.5)
    assert result.total_fees == pytest.approx(40.5)
    assert result.total_realized_pnl == pytest.approx(9.5)
    assert result.ending_equity == pytest.approx(1009.5)
    assert result.max_drawdown == pytest.approx(69.5)
    assert result.max_drawdown_percent == pytest.approx(
        69.5 / 1079.0 * 100.0
    )


def test_costs_can_turn_a_gross_winner_into_a_net_loser():
    model = TradingCostModel(fee_rate=0.01)
    trade = _closed_trade(
        entry_price=100.0,
        exit_price=100.5,
        quantity=10.0,
    )

    result = compute_backtest_metrics([trade], INITIAL_BALANCE, model)

    assert result.gross_profit == pytest.approx(5.0)
    assert result.total_fees == pytest.approx(20.05)
    assert result.total_realized_pnl == pytest.approx(-15.05)
    assert result.wins == 0
    assert result.losses == 1
    assert result.win_rate == 0.0
    assert result.profit_factor is None
    assert result.net_profit_factor == 0.0


def test_zero_costs_are_backward_compatible():
    trade = _closed_trade(quantity=2.0)

    implicit = compute_backtest_metrics([trade], INITIAL_BALANCE)
    explicit = compute_backtest_metrics(
        [trade],
        INITIAL_BALANCE,
        ZERO_COSTS,
    )

    assert implicit == explicit
    assert implicit.cost_model == ZERO_COSTS
    assert implicit.total_realized_pnl == 20.0
    assert implicit.ending_equity == 1020.0
    assert implicit.total_costs == 0.0


def test_metrics_infer_cost_model_from_completed_trade_breakdowns():
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    trade = _closed_trade(quantity=10.0)
    breakdown = calculate_trade_costs(
        trade.side,
        trade.entry_price,
        trade.exit_price,
        trade.quantity,
        model,
    )
    trade = replace(trade, cost_breakdown=breakdown)

    result = compute_backtest_metrics([trade], INITIAL_BALANCE)

    assert result.cost_model == model
    assert result.total_realized_pnl == pytest.approx(breakdown.net_pnl)
    assert result.total_costs == pytest.approx(breakdown.total_costs)


def test_mixed_cost_breakdown_presence_is_rejected():
    model = TradingCostModel(fee_rate=0.001)
    first = _closed_trade()
    second = _closed_trade(exit_at=START + timedelta(hours=3))
    second = replace(
        second,
        cost_breakdown=calculate_trade_costs(
            second.side,
            second.entry_price,
            second.exit_price,
            second.quantity,
            model,
        ),
    )

    with pytest.raises(ValueError, match="cost breakdowns"):
        compute_backtest_metrics([first, second], INITIAL_BALANCE)


def test_inconsistent_trade_cost_models_are_rejected():
    first_model = TradingCostModel(fee_rate=0.001)
    second_model = TradingCostModel(fee_rate=0.002)
    first = _closed_trade()
    second = _closed_trade(exit_at=START + timedelta(hours=3))
    first = replace(
        first,
        cost_breakdown=calculate_trade_costs(
            first.side,
            first.entry_price,
            first.exit_price,
            first.quantity,
            first_model,
        ),
    )
    second = replace(
        second,
        cost_breakdown=calculate_trade_costs(
            second.side,
            second.entry_price,
            second.exit_price,
            second.quantity,
            second_model,
        ),
    )

    with pytest.raises(ValueError, match="single cost model"):
        compute_backtest_metrics([first, second], INITIAL_BALANCE)


def test_invalid_analytics_cost_model_is_rejected():
    with pytest.raises(TypeError):
        compute_backtest_metrics(
            [_closed_trade()],
            INITIAL_BALANCE,
            object(),
        )
