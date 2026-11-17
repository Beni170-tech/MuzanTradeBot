"""Tests for historical signal-to-trade orchestration."""

from datetime import datetime, timedelta, timezone

import pytest

import backtest.trades as trades_module
from backtest.replay import HistoricalBar, SignalRecord
from backtest.trades import HistoricalTrade, simulate_trades
from market.indicators import MIN_CANDLES
from risk.risk_profile import RiskProfile
from strategy.parameters import DEFAULT_STRATEGY_PARAMETERS, StrategyParameters
from trader.paper.order import PaperOrderStatus

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROFILE = RiskProfile(
    balance=1000.0,
    risk_percent=2.0,
    stop_loss_percent=2.0,
    take_profit_percent=4.0,
)


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
        confidence = 80
    elif decision == "SELL":
        signal = "SELL"
        trend = "Bearish"
        confidence = 80
    else:
        signal = "WAIT"
        trend = "Bullish"
        confidence = 45
    return SignalRecord(
        timestamp=bar.timestamp,
        signal=signal,
        trend=trend,
        confidence=confidence,
        decision=decision,
    )


def _patch_replay(monkeypatch, bars, decisions):
    records = [
        _record(bar, decision)
        for bar, decision in zip(bars, decisions)
    ]
    calls = []

    def fake_replay(received_bars, *, parameters=None):
        calls.append(received_bars)
        return records

    monkeypatch.setattr(trades_module, "replay_signals", fake_replay)
    return calls


def test_wait_creates_no_trade(monkeypatch):
    bars = _bars([100.0, 101.0, 102.0])
    calls = _patch_replay(monkeypatch, bars, ["WAIT", "WAIT", "WAIT"])

    result = simulate_trades("BTC", bars, PROFILE)

    assert result == []
    assert len(calls) == 1


def test_buy_uses_next_bar_close_as_entry(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    _patch_replay(monkeypatch, bars, ["BUY", "WAIT", "WAIT"])

    result = simulate_trades("btc", bars, PROFILE)

    assert len(result) == 1
    trade = result[0]
    assert trade.symbol == "BTC"
    assert trade.side == "LONG"
    assert trade.entry_price == 110.0
    assert trade.entry_timestamp == START + timedelta(hours=1)
    assert trade.quantity > 0
    assert trade.stop_loss < trade.entry_price < trade.take_profit


def test_sell_uses_next_bar_close_as_entry(monkeypatch):
    bars = _bars([100.0, 110.0, 100.0])
    _patch_replay(monkeypatch, bars, ["SELL", "WAIT", "WAIT"])

    result = simulate_trades("BTC", bars, PROFILE)

    assert len(result) == 1
    trade = result[0]
    assert trade.side == "SHORT"
    assert trade.entry_price == 110.0
    assert trade.take_profit < trade.entry_price < trade.stop_loss


def test_signal_on_final_bar_creates_no_trade(monkeypatch):
    bars = _bars([100.0, 101.0, 102.0])
    _patch_replay(monkeypatch, bars, ["WAIT", "WAIT", "BUY"])

    assert simulate_trades("BTC", bars, PROFILE) == []


def test_blocked_risk_verdict_creates_no_trade(monkeypatch):
    bars = _bars([100.0, 110.0, 120.0])
    _patch_replay(monkeypatch, bars, ["BUY", "WAIT", "WAIT"])
    blocked_profile = RiskProfile(balance=0.0)

    assert simulate_trades("BTC", bars, blocked_profile) == []


def test_historical_timestamps_replace_wall_clock_timestamps(monkeypatch):
    bars = _bars([100.0, 110.0, 111.0])
    _patch_replay(monkeypatch, bars, ["BUY", "WAIT", "WAIT"])

    trade = simulate_trades("BTC", bars, PROFILE)[0]

    expected_entry = START + timedelta(hours=1)
    assert trade.signal_timestamp == START
    assert trade.entry_timestamp == expected_entry
    assert trade.trade_timestamp == expected_entry.strftime("%Y-%m-%d %H:%M:%S")
    assert not trade.trade_timestamp.startswith("2026")


def test_no_overlapping_positions(monkeypatch):
    bars = _bars([100.0, 110.0, 111.0, 111.0, 111.0])
    _patch_replay(
        monkeypatch,
        bars,
        ["BUY", "BUY", "BUY", "BUY", "WAIT"],
    )

    result = simulate_trades("BTC", bars, PROFILE)

    assert len(result) == 1
    assert result[0].signal_timestamp == START
    assert result[0].entry_timestamp == START + timedelta(hours=1)


def test_no_new_entry_on_exit_bar(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0, 115.0, 120.0])
    _patch_replay(
        monkeypatch,
        bars,
        ["BUY", "WAIT", "BUY", "BUY", "WAIT"],
    )

    result = simulate_trades("BTC", bars, PROFILE)

    assert len(result) == 2
    first, second = result
    assert first.exit_timestamp == START + timedelta(hours=2)
    assert first.exit_status is PaperOrderStatus.TAKEN_PROFIT
    assert second.signal_timestamp == START + timedelta(hours=3)
    assert second.entry_timestamp == START + timedelta(hours=4)
    assert second.entry_price == 120.0


def test_unclosed_position_prevents_later_entries(monkeypatch):
    bars = _bars([100.0, 110.0, 111.0, 111.0, 111.0])
    _patch_replay(
        monkeypatch,
        bars,
        ["BUY", "WAIT", "BUY", "BUY", "BUY"],
    )

    result = simulate_trades("BTC", bars, PROFILE)

    assert len(result) == 1
    assert result[0].exit_timestamp is None
    assert result[0].exit_price is None
    assert result[0].exit_status is None


def test_repeated_runs_produce_identical_results(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0, 115.0, 120.0])
    _patch_replay(
        monkeypatch,
        bars,
        ["BUY", "WAIT", "BUY", "BUY", "WAIT"],
    )

    first = simulate_trades("BTC", bars, PROFILE)
    second = simulate_trades("BTC", bars, PROFILE)

    assert all(isinstance(trade, HistoricalTrade) for trade in first)
    assert first == second


def test_replay_records_map_to_original_bar_indices(monkeypatch):
    bars = _bars([100.0 + index for index in range(MIN_CANDLES + 1)])
    replay_bars = bars[MIN_CANDLES - 1:]
    records = [_record(replay_bars[0], "BUY")]
    records.extend(_record(bar, "WAIT") for bar in replay_bars[1:])
    monkeypatch.setattr(
        trades_module,
        "replay_signals",
        lambda received_bars, *, parameters=None: records,
    )

    trade = simulate_trades("BTC", bars, PROFILE)[0]

    assert trade.signal_timestamp == bars[MIN_CANDLES - 1].timestamp
    assert trade.entry_timestamp == bars[MIN_CANDLES].timestamp
    assert trade.entry_price == bars[MIN_CANDLES].close


def test_invalid_symbol_and_profile_are_rejected(monkeypatch):
    bars = _bars([100.0])
    _patch_replay(monkeypatch, bars, ["WAIT"])

    with pytest.raises(ValueError):
        simulate_trades("  ", bars, PROFILE)
    with pytest.raises(TypeError):
        simulate_trades("BTC", bars, object())


def test_parameters_are_forwarded_to_replay(monkeypatch):
    bars = _bars([100.0, 101.0])
    captured = []

    def fake_replay(received_bars, *, parameters=None):
        captured.append(parameters)
        return []

    monkeypatch.setattr(trades_module, "replay_signals", fake_replay)
    custom = StrategyParameters(
        fast_ema=5,
        slow_ema=10,
        rsi_period=5,
        macd_fast=5,
        macd_slow=10,
        macd_signal=3,
        volume_period=5,
        min_candles=20,
    )

    assert simulate_trades("BTC", bars, PROFILE, parameters=custom) == []
    assert captured == [custom]


def test_omitted_parameters_use_default_instance(monkeypatch):
    bars = _bars([100.0, 101.0])
    captured = []

    def fake_replay(received_bars, *, parameters=None):
        captured.append(parameters)
        return []

    monkeypatch.setattr(trades_module, "replay_signals", fake_replay)

    simulate_trades("BTC", bars, PROFILE)

    assert captured == [DEFAULT_STRATEGY_PARAMETERS]


def test_invalid_parameters_type_is_rejected(monkeypatch):
    bars = _bars([100.0])
    _patch_replay(monkeypatch, bars, ["WAIT"])

    with pytest.raises(TypeError, match="parameters"):
        simulate_trades("BTC", bars, PROFILE, parameters=object())
