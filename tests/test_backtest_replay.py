"""Tests for the offline chronological signal replay core."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.replay as replay_module
from backtest.replay import HistoricalBar, SignalRecord, replay_signals
from market.indicators import MIN_CANDLES
from strategy.parameters import DEFAULT_STRATEGY_PARAMETERS, StrategyParameters

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _bars(closes, volumes=None):
    """Build ordered hourly bars from close prices and optional volumes."""
    if volumes is None:
        volumes = [1_000.0] * len(closes)
    return [
        HistoricalBar(
            timestamp=START + timedelta(hours=index),
            close=close,
            volume=volume,
        )
        for index, (close, volume) in enumerate(zip(closes, volumes))
    ]


def _rising(n=250, growth=0.001, start=100.0):
    closes = [start]
    for _ in range(1, n):
        closes.append(closes[-1] * (1 + growth))
    return closes


def _falling(n=220, curvature=0.0015, start=100.0):
    return [start - curvature * index * index for index in range(n)]


def test_insufficient_history_returns_no_records():
    assert replay_signals([]) == []
    assert replay_signals(_bars(_rising(n=MIN_CANDLES - 1))) == []


def test_first_eligible_bar_requires_min_candles():
    records = replay_signals(_bars(_rising(n=MIN_CANDLES)))

    assert len(records) == 1
    assert records[0].timestamp == START + timedelta(hours=MIN_CANDLES - 1)


def test_replay_is_chronological_and_covers_every_eligible_bar():
    bars = _bars(_rising(n=80))

    records = replay_signals(bars)

    assert len(records) == len(bars) - MIN_CANDLES + 1
    assert all(isinstance(record, SignalRecord) for record in records)
    assert [record.timestamp for record in records] == [
        bar.timestamp for bar in bars[MIN_CANDLES - 1:]
    ]


def test_rising_series_produces_buy_decisions():
    records = replay_signals(_bars(_rising()))

    assert records[-1].trend == "Bullish"
    assert records[-1].signal == "BUY"
    assert records[-1].decision == "BUY"


def test_falling_series_produces_sell_decisions():
    records = replay_signals(_bars(_falling()))

    assert records[-1].trend == "Bearish"
    assert records[-1].signal == "SELL"
    assert records[-1].decision == "SELL"


def test_flat_series_produces_wait_decisions():
    records = replay_signals(_bars([100.0] * 120))

    assert records
    assert all(record.signal == "WAIT" for record in records)
    assert all(record.decision == "WAIT" for record in records)


def test_future_bars_cannot_change_earlier_decisions():
    bars = _bars(_rising())
    cutoff = MIN_CANDLES + 10
    baseline = replay_signals(bars)
    records_through_cutoff = cutoff - (MIN_CANDLES - 1) + 1

    changed_future = [
        replace(bar, close=bar.close * 0.1, volume=bar.volume * 100)
        for bar in bars[cutoff + 1:]
    ]
    modified = replay_signals(bars[: cutoff + 1] + changed_future)

    assert modified[:records_through_cutoff] == baseline[:records_through_cutoff]
    assert modified[records_through_cutoff:] != baseline[records_through_cutoff:]


@pytest.mark.parametrize("bars", [None, "not-a-list", {"open": []}, 42])
def test_non_sequence_input_is_rejected(bars):
    with pytest.raises(TypeError):
        replay_signals(bars)


def test_non_bar_item_is_rejected():
    with pytest.raises(TypeError):
        replay_signals([object()])


@pytest.mark.parametrize(
    "bar",
    [
        HistoricalBar(START, True, 100.0),
        HistoricalBar(START, 0, 100.0),
        HistoricalBar(START, -1.0, 100.0),
        HistoricalBar(START, float("nan"), 100.0),
        HistoricalBar(START, float("inf"), 100.0),
        HistoricalBar(START, "100", 100.0),
        HistoricalBar(START, 100.0, -1.0),
        HistoricalBar(START, 100.0, float("nan")),
        HistoricalBar(START, 100.0, "100"),
        HistoricalBar("2024-01-01", 100.0, 100.0),
    ],
)
def test_invalid_bar_values_are_rejected(bar):
    with pytest.raises((TypeError, ValueError)):
        replay_signals([bar])


def test_decreasing_timestamps_are_rejected_without_sorting():
    bars = _bars([100.0, 101.0, 102.0])
    reversed_bars = list(reversed(bars))

    with pytest.raises(ValueError, match="strictly increasing"):
        replay_signals(reversed_bars)


def test_duplicate_timestamps_are_rejected():
    bars = _bars([100.0, 101.0, 102.0])
    bars[1] = replace(bars[1], timestamp=bars[0].timestamp)

    with pytest.raises(ValueError, match="strictly increasing"):
        replay_signals(bars)


def test_incomparable_timestamps_are_rejected():
    bars = _bars([100.0, 101.0])
    bars[0] = replace(bars[0], timestamp=START)
    bars[1] = replace(
        bars[1],
        timestamp=datetime(2024, 1, 1, 1),  # naive datetime
    )

    with pytest.raises(ValueError, match="mutually comparable"):
        replay_signals(bars)


def test_default_parameters_match_omitted_behavior():
    bars = _bars(_rising(n=80))

    assert replay_signals(bars) == replay_signals(
        bars,
        parameters=DEFAULT_STRATEGY_PARAMETERS,
    )


def test_custom_min_candles_controls_first_eligible_bar():
    custom = StrategyParameters(
        fast_ema=5,
        slow_ema=10,
        rsi_period=5,
        macd_fast=5,
        macd_slow=10,
        macd_signal=3,
        volume_period=5,
        min_candles=12,
    )
    bars = _bars(_rising(n=15))

    records = replay_signals(bars, parameters=custom)

    assert len(records) == 4
    assert records[0].timestamp == bars[11].timestamp


def test_parameters_are_forwarded_to_analysis_and_decision(monkeypatch):
    custom = StrategyParameters(
        fast_ema=5,
        slow_ema=10,
        rsi_period=5,
        macd_fast=5,
        macd_slow=10,
        macd_signal=3,
        volume_period=5,
        min_candles=12,
    )
    calls = []

    def fake_compute(closes, volumes=None, *, parameters=None):
        calls.append(parameters)
        return {
            "trend": "Bullish",
            "signal": "BUY",
            "confidence": 80,
            "indicators": {},
        }

    def fake_decision(analysis, *, parameters=None):
        calls.append(parameters)
        return "WAIT"

    monkeypatch.setattr(replay_module, "compute_analysis", fake_compute)
    monkeypatch.setattr(replay_module, "make_decision", fake_decision)

    records = replay_signals(
        _bars(_rising(n=13)),
        parameters=custom,
    )

    assert len(records) == 2
    assert all(parameters is custom for parameters in calls)


def test_invalid_parameters_type_is_rejected():
    with pytest.raises(TypeError, match="parameters"):
        replay_signals(_bars([100.0]), parameters=object())
