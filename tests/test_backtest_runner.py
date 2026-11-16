"""Tests for the pure historical backtest runner."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.runner as runner_module
import backtest.trades as trades_module
from backtest.costs import TradingCostModel, ZERO_COSTS
from backtest.replay import HistoricalBar, SignalRecord
from backtest.runner import (
    BacktestRunConfig,
    BacktestRunResult,
    run_backtest,
)
from backtest.trades import HistoricalTrade
from market.indicators import MIN_CANDLES
from risk.risk_profile import RiskProfile
from trader.paper.order import PaperOrderStatus

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROFILE = RiskProfile(
    balance=1000.0,
    risk_percent=2.0,
    stop_loss_percent=2.0,
    take_profit_percent=4.0,
)
CONFIG = BacktestRunConfig(symbol="btc", risk_profile=PROFILE)


def _bars(closes):
    return [
        HistoricalBar(
            timestamp=START + timedelta(hours=index),
            close=close,
            volume=1_000.0,
        )
        for index, close in enumerate(closes)
    ]


def _record(bar, decision):
    if decision == "BUY":
        signal = "BUY"
        trend = "Bullish"
    elif decision == "SELL":
        signal = "SELL"
        trend = "Bearish"
    else:
        signal = "WAIT"
        trend = "Bullish"
    return SignalRecord(
        timestamp=bar.timestamp,
        signal=signal,
        trend=trend,
        confidence=80,
        decision=decision,
    )


def _closed_trade(
    *,
    symbol="BTC",
    side="LONG",
    entry_price=100.0,
    exit_price=110.0,
    quantity=10.0,
    status=PaperOrderStatus.TAKEN_PROFIT,
):
    entry_at = START + timedelta(hours=1)
    if side.upper() == "LONG":
        stop_loss = 90.0
        take_profit = 120.0
    else:
        stop_loss = 110.0
        take_profit = 80.0
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
        exit_timestamp=START + timedelta(hours=2),
        exit_price=exit_price,
        exit_status=status,
    )


def _open_trade():
    trade = _closed_trade()
    return replace(
        trade,
        exit_timestamp=None,
        exit_price=None,
        exit_status=None,
    )


def _patch_simulation(monkeypatch, trades):
    calls = []

    def fake_simulate(symbol, bars, profile, cost_model=None):
        calls.append((symbol, bars, profile, cost_model))
        return list(trades)

    monkeypatch.setattr(runner_module, "simulate_trades", fake_simulate)
    return calls


def test_successful_run_connects_config_trades_and_report(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    trade = _closed_trade()
    calls = _patch_simulation(monkeypatch, [trade])

    result = run_backtest(bars, CONFIG)

    assert isinstance(result, BacktestRunResult)
    assert len(calls) == 1
    symbol, received_bars, profile, cost_model = calls[0]
    assert symbol == "BTC"
    assert tuple(received_bars) == tuple(bars)
    assert profile is PROFILE
    assert cost_model == ZERO_COSTS

    assert result.report.total_trades == 1
    assert result.report.wins == 1
    assert result.report.net_pnl == 100.0
    assert result.trades == (trade,)
    assert isinstance(result.trades, tuple)
    assert result.config is CONFIG
    assert result.bar_count == 3
    assert result.first_bar_timestamp == START
    assert result.last_bar_timestamp == START + timedelta(hours=2)
    assert result.minimum_history_bars == MIN_CANDLES
    assert not result.history_sufficient
    assert result.completed_trades == 1
    assert result.open_trades == 0


def test_successful_run_uses_actual_replay_and_execution_pipeline(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    records = [
        _record(bars[0], "BUY"),
        _record(bars[1], "WAIT"),
        _record(bars[2], "WAIT"),
    ]
    monkeypatch.setattr(
        trades_module,
        "replay_signals",
        lambda received_bars: records,
    )
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    config = BacktestRunConfig("BTC", PROFILE, model)

    result = run_backtest(bars, config)

    assert result.report.total_trades == 1
    trade = result.trades[0]
    assert trade.entry_price == 110.0
    assert trade.exit_price == pytest.approx(114.4)
    assert trade.exit_status is PaperOrderStatus.TAKEN_PROFIT
    assert trade.cost_breakdown is not None
    assert trade.cost_breakdown.cost_model == model
    assert result.report.cost_model == model
    assert result.report.net_pnl == pytest.approx(
        trade.cost_breakdown.net_pnl
    )
    assert result.completed_trades == 1


def test_explicit_cost_model_controls_report_and_simulation(monkeypatch):
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    config = BacktestRunConfig("BTC", PROFILE, model)
    calls = _patch_simulation(monkeypatch, [_closed_trade()])

    result = run_backtest(_bars([100.0, 110.0, 115.0]), config)

    assert calls[0][3] is model
    assert result.report.cost_model is model
    assert result.report.net_pnl == pytest.approx(76.901)
    assert result.report.total_costs == pytest.approx(23.099)
    assert result.report.ending_equity == pytest.approx(1076.901)


@pytest.mark.parametrize("bar_count", [0, MIN_CANDLES - 1])
def test_insufficient_history_returns_safe_empty_report(bar_count):
    bars = _bars([100.0] * bar_count)

    result = run_backtest(bars, CONFIG)

    assert result.report.total_trades == 0
    assert result.report.ending_equity == PROFILE.balance
    assert result.trades == ()
    assert result.bar_count == bar_count
    assert not result.history_sufficient
    if bar_count == 0:
        assert result.first_bar_timestamp is None
        assert result.last_bar_timestamp is None
    else:
        assert result.first_bar_timestamp == START
        assert result.last_bar_timestamp == START + timedelta(hours=bar_count - 1)
    assert any(
        "No trades were generated" in warning
        for warning in result.report.warnings
    )


def test_zero_trade_run_is_safe_and_has_sufficient_history():
    result = run_backtest(_bars([100.0] * 120), CONFIG)

    assert result.report.total_trades == 0
    assert result.report.open_trades == 0
    assert result.report.net_pnl == 0.0
    assert result.report.ending_equity == PROFILE.balance
    assert result.trades == ()
    assert result.bar_count == 120
    assert result.history_sufficient
    assert any(
        "No trades were generated" in warning
        for warning in result.report.warnings
    )


def test_zero_completed_trades_with_open_position_are_reported(monkeypatch):
    _patch_simulation(monkeypatch, [_open_trade()])

    result = run_backtest(_bars([100.0, 110.0, 111.0]), CONFIG)

    assert result.report.total_trades == 1
    assert result.report.open_trades == 1
    assert result.report.net_pnl == 0.0
    assert result.completed_trades == 0
    assert result.open_trades == 1
    assert any(
        "All trades are open" in warning
        for warning in result.report.warnings
    )


def test_repeated_runs_are_deterministic(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    records = [
        _record(bars[0], "BUY"),
        _record(bars[1], "WAIT"),
        _record(bars[2], "WAIT"),
    ]
    monkeypatch.setattr(
        trades_module,
        "replay_signals",
        lambda received_bars: records,
    )
    model = TradingCostModel(fee_rate=0.001)
    config = BacktestRunConfig("BTC", PROFILE, model)

    first = run_backtest(bars, config)
    second = run_backtest(bars, config)

    assert first == second
    assert first.report.trades == second.report.trades
    assert first.trades == second.trades


def test_input_bars_are_not_modified(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    original = list(bars)
    _patch_simulation(monkeypatch, [_closed_trade()])

    run_backtest(bars, CONFIG)

    assert bars == original


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"symbol": "  "}, ValueError),
        ({"symbol": None}, ValueError),
        ({"risk_profile": object()}, TypeError),
        ({"cost_model": object()}, TypeError),
        ({"risk_profile": RiskProfile(balance=0.0)}, ValueError),
        ({"risk_profile": RiskProfile(balance=-1.0)}, ValueError),
        ({"risk_profile": RiskProfile(balance=float("nan"))}, ValueError),
        ({"risk_profile": RiskProfile(balance=float("inf"))}, ValueError),
        ({"risk_profile": RiskProfile(risk_percent=0.0)}, ValueError),
        ({"risk_profile": RiskProfile(stop_loss_percent=-1.0)}, ValueError),
        ({"risk_profile": RiskProfile(take_profit_percent=0.0)}, ValueError),
    ],
)
def test_invalid_config_is_rejected(overrides, error):
    values = {"symbol": "BTC", "risk_profile": PROFILE}
    values.update(overrides)

    with pytest.raises(error):
        BacktestRunConfig(**values)


@pytest.mark.parametrize("bars", [None, "not-a-sequence", {"close": []}, 42])
def test_non_sequence_bars_are_rejected(bars):
    with pytest.raises(TypeError):
        run_backtest(bars, CONFIG)


@pytest.mark.parametrize(
    ("bars", "error"),
    [
        ([object()], TypeError),
        ([HistoricalBar(START, 0.0, 1_000.0)], ValueError),
        ([HistoricalBar(START, 100.0, -1.0)], ValueError),
        ([HistoricalBar(START, float("nan"), 1_000.0)], ValueError),
        (
            [
                HistoricalBar(START, 100.0, 1_000.0),
                HistoricalBar(START, 101.0, 1_000.0),
            ],
            ValueError,
        ),
    ],
)
def test_invalid_bars_are_rejected(bars, error):
    with pytest.raises(error):
        run_backtest(bars, CONFIG)


def test_invalid_config_type_is_rejected():
    with pytest.raises(TypeError):
        run_backtest([], object())


def test_config_normalizes_symbol():
    assert BacktestRunConfig(" btc ", PROFILE).symbol == "BTC"
