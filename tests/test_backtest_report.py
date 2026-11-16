"""Tests for the pure backtest reporting layer."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from backtest.costs import TradingCostModel, ZERO_COSTS
from backtest.report import (
    BacktestReport,
    BacktestTradeSummary,
    build_backtest_report,
    format_backtest_report,
)
from backtest.trades import HistoricalTrade
from trader.paper.order import PaperOrderStatus

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
INITIAL_BALANCE = 1000.0


def _closed_trade(
    *,
    symbol="BTC",
    side="LONG",
    entry_price=100.0,
    exit_price=110.0,
    quantity=1.0,
    exit_at=None,
    status=PaperOrderStatus.TAKEN_PROFIT,
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
        symbol=symbol,
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
        exit_status=status,
    )


def _open_trade(**overrides):
    trade = _closed_trade(**overrides)
    return replace(
        trade,
        exit_timestamp=None,
        exit_price=None,
        exit_status=None,
    )


def test_empty_backtest_report_has_explicit_warnings():
    report = build_backtest_report([], INITIAL_BALANCE)

    assert isinstance(report, BacktestReport)
    assert report.initial_equity == INITIAL_BALANCE
    assert report.ending_equity == INITIAL_BALANCE
    assert report.total_trades == 0
    assert report.open_trades == 0
    assert report.net_pnl == 0.0
    assert report.gross_pnl == 0.0
    assert report.trades == ()
    assert report.cost_model == ZERO_COSTS
    assert any("No trades were generated" in line for line in report.warnings)
    assert any("Zero fees and zero slippage" in line for line in report.warnings)

    text = format_backtest_report(report)

    assert "BACKTEST REPORT" in text
    assert "No trade summaries to display." in text
    assert "Zero fees and zero slippage" in text
    assert text.endswith("\n")


def test_winning_and_losing_trades_are_summarized():
    winner = _closed_trade(
        entry_price=100.0,
        exit_price=110.0,
        quantity=2.0,
        exit_at=START + timedelta(hours=1),
    )
    loser = _closed_trade(
        side="SHORT",
        entry_price=110.0,
        exit_price=115.0,
        quantity=2.0,
        exit_at=START + timedelta(hours=2),
    )

    report = build_backtest_report([winner, loser], INITIAL_BALANCE)

    assert report.total_trades == 2
    assert report.open_trades == 0
    assert report.wins == 1
    assert report.losses == 1
    assert report.win_rate == 50.0
    assert report.gross_pnl == 10.0
    assert report.net_pnl == 10.0
    assert report.ending_equity == 1010.0
    assert len(report.trades) == 2

    first, second = report.trades
    assert isinstance(first, BacktestTradeSummary)
    assert first.number == 1
    assert first.side == "LONG"
    assert first.net_pnl == 20.0
    assert not first.is_open
    assert second.number == 2
    assert second.side == "SHORT"
    assert second.net_pnl == -10.0

    text = format_backtest_report(report)
    assert "Wins:       1" in text
    assert "Losses:     1" in text
    assert "Win rate:   50.00%" in text
    assert "Trade 1: BTC LONG - TAKEN_PROFIT" in text
    assert "Trade 2: BTC SHORT - TAKEN_PROFIT" in text


def test_cost_aware_report_exposes_assumptions_and_net_results():
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    trade = _closed_trade(quantity=10.0)

    report = build_backtest_report([trade], INITIAL_BALANCE, model)

    assert report.cost_model == model
    assert report.net_pnl == pytest.approx(76.901)
    assert report.total_costs == pytest.approx(23.099)
    assert report.ending_equity == pytest.approx(1076.901)

    summary = report.trades[0]
    assert summary.entry_fill_price == pytest.approx(101.0)
    assert summary.exit_fill_price == pytest.approx(108.9)
    assert summary.gross_pnl == pytest.approx(100.0)
    assert summary.net_pnl == pytest.approx(76.901)
    assert summary.total_fees == pytest.approx(2.099)
    assert summary.slippage_cost == pytest.approx(21.0)
    assert summary.total_costs == pytest.approx(23.099)

    text = format_backtest_report(report)
    assert "Fees: 0.1000% of notional per side" in text
    assert "Slippage: 1.0000% adverse price move per side" in text
    assert "not exchange-reported rates or actual fills" in text
    assert "Fees and slippage are modeled assumptions" in text


def test_zero_cost_report_is_backward_compatible_and_explicit():
    trade = _closed_trade(quantity=2.0)

    implicit = build_backtest_report([trade], INITIAL_BALANCE)
    explicit = build_backtest_report([trade], INITIAL_BALANCE, ZERO_COSTS)

    assert implicit == explicit
    assert implicit.cost_model == ZERO_COSTS
    assert implicit.net_pnl == 20.0
    assert implicit.total_costs == 0.0
    assert implicit.trades[0].net_pnl == 20.0

    text = format_backtest_report(implicit)
    assert "Fees: 0.0000% of notional per side" in text
    assert "Slippage: 0.0000% adverse price move per side" in text
    assert "does not represent realistic exchange costs" in text


def test_drawdown_and_profit_factor_are_reported():
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

    report = build_backtest_report([winner, loser], INITIAL_BALANCE)

    assert report.metrics.gross_profit == 100.0
    assert report.metrics.gross_loss == 50.0
    assert report.profit_factor == 2.0
    assert report.metrics.net_profit_factor == 2.0
    assert report.max_drawdown == 50.0
    assert report.metrics.max_drawdown_percent == pytest.approx(
        50.0 / 1100.0 * 100.0
    )

    text = format_backtest_report(report)
    assert "Profit factor (gross): 2.00" in text
    assert "Profit factor (net):   2.00" in text
    assert "Max drawdown (net): $50.00 (4.55%)" in text


def test_open_trade_is_listed_but_not_included_in_realized_metrics():
    open_trade = _open_trade(entry_price=100.0, quantity=2.0)

    report = build_backtest_report([open_trade], INITIAL_BALANCE)

    assert report.total_trades == 1
    assert report.open_trades == 1
    assert report.net_pnl == 0.0
    assert report.ending_equity == INITIAL_BALANCE
    assert report.trades[0].is_open
    assert report.trades[0].gross_pnl is None
    assert report.trades[0].net_pnl is None
    assert any("not marked to market" in line for line in report.warnings)

    text = format_backtest_report(report)
    assert "Trade 1: BTC LONG - OPEN" in text
    assert "Gross PnL: not realized" in text
    assert "Net PnL:   not realized" in text
    assert "All trades are open" in text


def test_formatting_is_deterministic_ascii_and_supports_truncation():
    trades = [
        _closed_trade(
            symbol="BTC",
            entry_price=100.0,
            exit_price=110.0,
            exit_at=START + timedelta(hours=1),
        ),
        _closed_trade(
            symbol="ETH",
            entry_price=100.0,
            exit_price=105.0,
            exit_at=START + timedelta(hours=2),
        ),
        _closed_trade(
            symbol="SOL",
            entry_price=100.0,
            exit_price=99.0,
            exit_at=START + timedelta(hours=3),
        ),
    ]
    report = build_backtest_report(trades, INITIAL_BALANCE)

    full = format_backtest_report(report)
    limited = format_backtest_report(report, max_trades=2)

    assert full == format_backtest_report(report)
    assert full.isascii()
    assert limited.isascii()
    assert "Trade 1: BTC LONG" in limited
    assert "Trade 2: ETH LONG" in limited
    assert "Trade 3: SOL LONG" not in limited
    assert "... 1 more trade(s) not shown." in limited


@pytest.mark.parametrize("max_trades", [True, False, "2", 1.5])
def test_non_integer_max_trades_is_rejected(max_trades):
    report = build_backtest_report([], INITIAL_BALANCE)

    with pytest.raises(TypeError):
        format_backtest_report(report, max_trades=max_trades)


def test_negative_max_trades_is_rejected():
    report = build_backtest_report([], INITIAL_BALANCE)

    with pytest.raises(ValueError):
        format_backtest_report(report, max_trades=-1)


def test_invalid_report_type_is_rejected():
    with pytest.raises(TypeError):
        format_backtest_report(object())


@pytest.mark.parametrize(
    ("trades", "balance", "cost_model"),
    [
        (None, INITIAL_BALANCE, None),
        ([object()], INITIAL_BALANCE, None),
        ([], 0.0, None),
        ([], INITIAL_BALANCE, object()),
    ],
)
def test_invalid_report_inputs_are_rejected(trades, balance, cost_model):
    with pytest.raises((TypeError, ValueError)):
        build_backtest_report(trades, balance, cost_model)
