"""Tests for pure strategy validation across market conditions."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.trades as trades_module
import backtest.validation as validation_module
from backtest.costs import TradingCostModel
from backtest.replay import HistoricalBar, SignalRecord
from backtest.runner import BacktestRunConfig
from backtest.validation import (
    MIN_WINDOW_BARS,
    MarketCondition,
    StrategyValidationConfig,
    classify_market_condition,
    validate_strategy,
)
from risk.risk_profile import RiskProfile

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROFILE = RiskProfile(
    balance=1000.0,
    risk_percent=2.0,
    stop_loss_percent=2.0,
    take_profit_percent=4.0,
)
CONFIG = BacktestRunConfig("BTC", PROFILE)
WINDOW_CONFIG = StrategyValidationConfig(window_bars=MIN_WINDOW_BARS)

REALISTIC_BULLISH_CLOSES = [
    100.0,
    98.8,
    100.6,
    99.9,
    101.7,
    100.9,
    102.8,
    102.0,
    103.6,
    102.8,
    104.9,
    104.1,
    106.2,
]
REALISTIC_BEARISH_CLOSES = [
    110.0,
    111.2,
    109.4,
    110.3,
    108.2,
    109.0,
    107.1,
    107.9,
    105.8,
    106.7,
    104.6,
    105.5,
    103.3,
]
REALISTIC_SIDEWAYS_CLOSES = [
    100.0,
    102.4,
    99.3,
    101.6,
    98.9,
    101.2,
    99.1,
    100.8,
    98.7,
    100.4,
    99.5,
]


def _bars(closes):
    return [
        HistoricalBar(
            timestamp=START + timedelta(hours=index),
            close=close,
            volume=1_000.0,
        )
        for index, close in enumerate(closes)
    ]


def _flat_bars(count, price=100.0):
    return _bars([price] * count)


def _condition_bars(condition, count, start_index=0):
    if condition == "bull":
        closes = [100.0 + index * 0.5 for index in range(count)]
    elif condition == "bear":
        closes = [200.0 - index * 0.5 for index in range(count)]
    else:
        closes = [100.0] * count
    return [
        HistoricalBar(
            timestamp=START + timedelta(hours=start_index + index),
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


def _patch_replay(monkeypatch):
    def fake_replay(received_bars):
        records = [_record(received_bars[0], "BUY")]
        records.extend(
            _record(bar, "WAIT") for bar in received_bars[1:]
        )
        return records

    monkeypatch.setattr(trades_module, "replay_signals", fake_replay)


def test_classify_bullish_series():
    assert classify_market_condition(_bars([100.0, 105.0, 110.0])) is (
        MarketCondition.BULLISH
    )


def test_classify_bearish_series():
    assert classify_market_condition(_bars([110.0, 105.0, 100.0])) is (
        MarketCondition.BEARISH
    )


def test_classify_flat_and_choppy_series_as_sideways():
    assert classify_market_condition(_flat_bars(10)) is (
        MarketCondition.SIDEWAYS
    )
    assert classify_market_condition(
        _bars([100.0, 130.0, 100.0, 130.0, 100.0, 125.0])
    ) is MarketCondition.SIDEWAYS


@pytest.mark.parametrize(
    ("closes", "expected"),
    [
        (REALISTIC_BULLISH_CLOSES, MarketCondition.BULLISH),
        (REALISTIC_BEARISH_CLOSES, MarketCondition.BEARISH),
        (REALISTIC_SIDEWAYS_CLOSES, MarketCondition.SIDEWAYS),
    ],
)
def test_realistic_fixtures_use_default_classification(closes, expected):
    assert classify_market_condition(_bars(closes)) is expected


def test_extreme_finite_prices_keep_directional_classification():
    assert classify_market_condition(
        _bars([1e300, 1.05e300, 1.1e300])
    ) is MarketCondition.BULLISH
    assert classify_market_condition(
        _bars([1e300, 0.95e300, 0.9e300])
    ) is MarketCondition.BEARISH


def test_extreme_flat_prices_avoid_division_errors():
    max_float = 1.7976931348623157e308

    assert classify_market_condition(
        _bars([max_float, max_float, max_float])
    ) is MarketCondition.SIDEWAYS
    assert classify_market_condition(
        _bars([5e-324, 5e-324, 5e-324])
    ) is MarketCondition.SIDEWAYS


def test_non_finite_extreme_return_raises_controlled_value_error():
    with pytest.raises(ValueError, match="not finite"):
        classify_market_condition(_bars([5e-324, 1e308]))


def test_extreme_path_overflow_raises_controlled_value_error():
    with pytest.raises(ValueError, match="overflowed"):
        classify_market_condition(_bars([1e308, 5e-324, 1e308]))


def test_classification_threshold_boundaries():
    assert classify_market_condition(
        _bars([100.0, 150.0]),
        trend_threshold_percent=50.0,
    ) is MarketCondition.BULLISH
    assert classify_market_condition(
        _bars([100.0, 150.0]),
        trend_threshold_percent=50.000001,
    ) is MarketCondition.SIDEWAYS
    assert classify_market_condition(
        _bars([100.0, 50.0]),
        trend_threshold_percent=50.0,
    ) is MarketCondition.BEARISH


def test_classification_efficiency_boundary():
    bars = _bars([100.0, 120.0, 90.0, 120.0])

    assert classify_market_condition(
        bars,
        min_directional_efficiency=0.25,
    ) is MarketCondition.BULLISH
    assert classify_market_condition(
        bars,
        min_directional_efficiency=0.250001,
    ) is MarketCondition.SIDEWAYS


@pytest.mark.parametrize(
    ("threshold", "error"),
    [
        (True, TypeError),
        ("2", TypeError),
        (None, TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_invalid_classification_threshold_is_rejected(threshold, error):
    with pytest.raises(error):
        classify_market_condition(
            _bars([100.0, 110.0]),
            trend_threshold_percent=threshold,
        )


@pytest.mark.parametrize(
    ("efficiency", "error"),
    [
        (True, TypeError),
        ("0.25", TypeError),
        (None, TypeError),
        (-0.01, ValueError),
        (1.01, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_invalid_classification_efficiency_is_rejected(efficiency, error):
    with pytest.raises(error):
        classify_market_condition(
            _bars([100.0, 110.0]),
            min_directional_efficiency=efficiency,
        )


@pytest.mark.parametrize("bars", [[], _bars([100.0])])
def test_classification_requires_two_bars(bars):
    with pytest.raises(ValueError, match="two bars"):
        classify_market_condition(bars)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"window_bars": True}, TypeError),
        ({"window_bars": 62.0}, TypeError),
        ({"window_bars": "62"}, TypeError),
        ({"window_bars": MIN_WINDOW_BARS - 1}, ValueError),
        ({"trend_threshold_percent": True}, TypeError),
        ({"trend_threshold_percent": "2"}, TypeError),
        ({"trend_threshold_percent": 0.0}, ValueError),
        ({"trend_threshold_percent": -1.0}, ValueError),
        ({"trend_threshold_percent": float("nan")}, ValueError),
        ({"trend_threshold_percent": float("inf")}, ValueError),
        ({"min_directional_efficiency": True}, TypeError),
        ({"min_directional_efficiency": "0.25"}, TypeError),
        ({"min_directional_efficiency": -0.01}, ValueError),
        ({"min_directional_efficiency": 1.01}, ValueError),
        ({"min_directional_efficiency": float("nan")}, ValueError),
        ({"min_directional_efficiency": float("inf")}, ValueError),
    ],
)
def test_invalid_validation_config_is_rejected(overrides, error):
    with pytest.raises(error):
        StrategyValidationConfig(**overrides)


def test_minimum_window_size_is_accepted():
    config = StrategyValidationConfig(window_bars=MIN_WINDOW_BARS)

    assert config.window_bars == MIN_WINDOW_BARS
    assert isinstance(config.trend_threshold_percent, float)
    assert isinstance(config.min_directional_efficiency, float)


def test_invalid_argument_types_are_rejected():
    with pytest.raises(TypeError, match="config"):
        validate_strategy([], object())
    with pytest.raises(TypeError, match="validation_config"):
        validate_strategy([], CONFIG, object())
    with pytest.raises(TypeError, match="bars"):
        validate_strategy(object(), CONFIG)


@pytest.mark.parametrize("bars", [None, "bars", {"close": []}, 42])
def test_non_sequence_validation_bars_are_rejected(bars):
    with pytest.raises(TypeError, match="bars"):
        validate_strategy(bars, CONFIG)


@pytest.mark.parametrize(
    ("bars", "error"),
    [
        ([object()], TypeError),
        ([HistoricalBar(START, True, 1_000.0)], ValueError),
        ([HistoricalBar(START, 0.0, 1_000.0)], ValueError),
        ([HistoricalBar(START, -1.0, 1_000.0)], ValueError),
        ([HistoricalBar(START, float("nan"), 1_000.0)], ValueError),
        ([HistoricalBar(START, 100.0, -1.0)], ValueError),
        ([HistoricalBar(START, 100.0, float("inf"))], ValueError),
        ([HistoricalBar("2024-01-01", 100.0, 1_000.0)], TypeError),
    ],
)
def test_invalid_validation_bars_are_rejected(bars, error):
    with pytest.raises(error):
        validate_strategy(bars, CONFIG)


def test_duplicate_and_unordered_timestamps_are_rejected():
    bars = _flat_bars(2)

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_strategy(
            [bars[0], replace(bars[0], close=101.0)],
            CONFIG,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_strategy(list(reversed(bars)), CONFIG)


def test_invalid_bar_in_unused_tail_is_rejected():
    bars = _flat_bars(MIN_WINDOW_BARS) + [object()]

    with pytest.raises(TypeError, match="HistoricalBar"):
        validate_strategy(bars, CONFIG, WINDOW_CONFIG)


@pytest.mark.parametrize("bar_count", [0, MIN_WINDOW_BARS - 1])
def test_insufficient_history_returns_safe_empty_result(bar_count):
    bars = _flat_bars(bar_count)

    result = validate_strategy(bars, CONFIG, WINDOW_CONFIG)

    assert result.bar_count == bar_count
    assert result.unused_bars == bar_count
    assert result.window_count == 0
    assert result.windows == ()
    assert result.conditions == ()
    assert result.warnings
    if bar_count == 0:
        assert "No historical bars" in result.warnings[0]
    else:
        assert "Insufficient history" in result.warnings[0]


def test_windows_are_contiguous_non_overlapping_and_track_remainder():
    bars = _flat_bars(3 * MIN_WINDOW_BARS + 5)

    result = validate_strategy(bars, CONFIG, WINDOW_CONFIG)

    assert result.bar_count == len(bars)
    assert result.window_count == 3
    assert result.unused_bars == 5
    assert [window.index for window in result.windows] == [0, 1, 2]
    assert all(
        window.bar_count == MIN_WINDOW_BARS for window in result.windows
    )

    first, second, third = result.windows
    assert first.first_timestamp == bars[0].timestamp
    assert first.last_timestamp == bars[MIN_WINDOW_BARS - 1].timestamp
    assert second.first_timestamp == bars[MIN_WINDOW_BARS].timestamp
    assert second.last_timestamp == bars[2 * MIN_WINDOW_BARS - 1].timestamp
    assert third.first_timestamp == bars[2 * MIN_WINDOW_BARS].timestamp
    assert third.last_timestamp == bars[3 * MIN_WINDOW_BARS - 1].timestamp
    assert any("5 trailing bars" in warning for warning in result.warnings)
    assert all(
        window.condition is MarketCondition.SIDEWAYS
        for window in result.windows
    )


def test_real_runner_validation_across_conditions(monkeypatch):
    bars = (
        _condition_bars("bull", MIN_WINDOW_BARS, start_index=0)
        + _condition_bars("bear", MIN_WINDOW_BARS, start_index=MIN_WINDOW_BARS)
        + _condition_bars(
            "side",
            MIN_WINDOW_BARS,
            start_index=2 * MIN_WINDOW_BARS,
        )
    )
    _patch_replay(monkeypatch)
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    config = BacktestRunConfig("BTC", PROFILE, model)

    result = validate_strategy(bars, config, WINDOW_CONFIG)

    assert result.bar_count == 3 * MIN_WINDOW_BARS
    assert result.unused_bars == 0
    assert result.window_count == 3
    assert [window.condition for window in result.windows] == [
        MarketCondition.BULLISH,
        MarketCondition.BEARISH,
        MarketCondition.SIDEWAYS,
    ]
    assert [summary.condition for summary in result.conditions] == [
        MarketCondition.BULLISH,
        MarketCondition.BEARISH,
        MarketCondition.SIDEWAYS,
    ]
    assert result.warnings == ()

    for window in result.windows:
        assert window.backtest.config is config
        assert window.backtest.report.cost_model is model
        assert window.backtest.trades

    bull, bear, side = result.conditions
    assert bull.window_count == 1
    assert bull.completed_trades == 1
    assert bull.net_pnl > 0.0
    assert bull.total_costs > 0.0
    assert bear.completed_trades == 1
    assert bear.net_pnl < 0.0
    assert bear.total_costs > 0.0
    assert side.open_trades == 1
    assert side.completed_trades == 0
    assert side.net_pnl == 0.0
    assert side.net_profit_factor is None
    assert side.worst_max_drawdown == 0.0


def test_condition_summary_sums_independent_windows(monkeypatch):
    bars = (
        _condition_bars("bear", MIN_WINDOW_BARS, start_index=0)
        + _condition_bars(
            "bear",
            MIN_WINDOW_BARS,
            start_index=MIN_WINDOW_BARS,
        )
    )
    _patch_replay(monkeypatch)
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    config = BacktestRunConfig("BTC", PROFILE, model)

    result = validate_strategy(bars, config, WINDOW_CONFIG)

    assert result.window_count == 2
    assert result.unused_bars == 0
    assert result.warnings == ()
    summary = result.conditions[0]
    reports = [window.backtest.report for window in result.windows]

    assert summary.condition is MarketCondition.BEARISH
    assert summary.window_count == 2
    assert summary.bar_count == 2 * MIN_WINDOW_BARS
    assert summary.total_trades == 2
    assert summary.completed_trades == 2
    assert summary.wins == 0
    assert summary.losses == 2
    assert summary.net_pnl == pytest.approx(
        sum(report.net_pnl for report in reports)
    )
    assert summary.gross_pnl == pytest.approx(
        sum(report.gross_pnl for report in reports)
    )
    assert summary.total_costs == pytest.approx(
        sum(report.total_costs for report in reports)
    )
    assert summary.net_profit_factor == 0.0
    assert summary.worst_max_drawdown == max(
        report.max_drawdown for report in reports
    )
    assert summary.best_window_net_pnl == max(
        report.net_pnl for report in reports
    )
    assert summary.worst_window_net_pnl == min(
        report.net_pnl for report in reports
    )


def test_zero_trade_window_is_safe():
    result = validate_strategy(
        _flat_bars(MIN_WINDOW_BARS),
        CONFIG,
        WINDOW_CONFIG,
    )

    summary = result.conditions[0]
    assert summary.condition is MarketCondition.SIDEWAYS
    assert summary.total_trades == 0
    assert summary.completed_trades == 0
    assert summary.win_rate == 0.0
    assert summary.net_pnl == 0.0
    assert summary.gross_profit_factor is None
    assert summary.net_profit_factor is None
    assert summary.worst_max_drawdown == 0.0
    assert summary.best_window_net_pnl == 0.0
    assert summary.worst_window_net_pnl == 0.0


def test_validation_is_deterministic(monkeypatch):
    bars = (
        _condition_bars("bull", MIN_WINDOW_BARS, start_index=0)
        + _condition_bars(
            "bear",
            MIN_WINDOW_BARS,
            start_index=MIN_WINDOW_BARS,
        )
    )
    _patch_replay(monkeypatch)
    model = TradingCostModel(fee_rate=0.001)
    config = BacktestRunConfig("BTC", PROFILE, model)

    first = validate_strategy(bars, config, WINDOW_CONFIG)
    second = validate_strategy(bars, config, WINDOW_CONFIG)

    assert first == second
    assert first.windows == second.windows
    assert first.conditions == second.conditions
    assert first.warnings == second.warnings
