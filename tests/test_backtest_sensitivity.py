"""Tests for descriptive one-at-a-time parameter sensitivity."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.sensitivity as sensitivity_module
import backtest.trades as trades_module
from backtest.costs import TradingCostModel
from backtest.replay import HistoricalBar, SignalRecord
from backtest.runner import BacktestRunConfig
from backtest.sensitivity import (
    ParameterSweep,
    SensitivityResult,
    SensitivityVariantResult,
    run_parameter_sensitivity,
)
from backtest.validation import (
    MIN_WINDOW_BARS,
    MarketCondition,
    StrategyValidationConfig,
    StrategyValidationResult,
)
from risk.risk_profile import RiskProfile
from strategy.parameters import DEFAULT_STRATEGY_PARAMETERS, StrategyParameters

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROFILE = RiskProfile(
    balance=1000.0,
    risk_percent=2.0,
    stop_loss_percent=2.0,
    take_profit_percent=4.0,
)
CONFIG = BacktestRunConfig("BTC", PROFILE)
WINDOW_CONFIG = StrategyValidationConfig(window_bars=MIN_WINDOW_BARS)


def _bars(closes, volumes=None):
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


def _bull_then_bear(count=MIN_WINDOW_BARS):
    first = [100.0 + index * 0.5 for index in range(count)]
    second = [first[-1] - index * 0.5 for index in range(count)]
    return _bars(first + second)


def _flat(count=2 * MIN_WINDOW_BARS):
    return _bars([100.0] * count)


def _fail_if_called(*args, **kwargs):
    raise AssertionError("validation ran before preflight validation completed")


# --- ParameterSweep -------------------------------------------------


def test_parameter_sweep_normalizes_field_and_values():
    sweep = ParameterSweep("  fast_ema ", [10, 30])

    assert sweep.field == "fast_ema"
    assert sweep.values == (10, 30)
    assert isinstance(sweep.values, tuple)


@pytest.mark.parametrize(
    "field",
    ["", "   ", "unknown", "strategy", "__dict__"],
)
def test_parameter_sweep_rejects_unknown_fields(field):
    with pytest.raises(ValueError):
        ParameterSweep(field, (10,))


@pytest.mark.parametrize("field", [None, 1, True])
def test_parameter_sweep_rejects_non_string_fields(field):
    with pytest.raises(ValueError):
        ParameterSweep(field, (10,))


@pytest.mark.parametrize("values", [(), [], None, "10", {"value": 10}])
def test_parameter_sweep_rejects_empty_or_non_sequence_values(values):
    with pytest.raises((TypeError, ValueError)):
        ParameterSweep("fast_ema", values)


@pytest.mark.parametrize("values", [(10, 10), (10, 10.0), (0.5, 0.5)])
def test_parameter_sweep_rejects_duplicate_values(values):
    with pytest.raises(ValueError, match="duplicate"):
        ParameterSweep("fast_ema", values)


@pytest.mark.parametrize(
    "values",
    [(True,), (10, True), ("10",), (None,), (float("nan"),), (10**400,)],
)
def test_parameter_sweep_rejects_invalid_values(values):
    with pytest.raises((TypeError, ValueError)):
        ParameterSweep("fast_ema", values)


# --- Baseline and variants ------------------------------------------


def test_baseline_and_all_variants_run_with_one_bar_sequence(monkeypatch):
    bars = _bull_then_bear()
    calls = []
    original = sensitivity_module.validation_module.validate_strategy

    def spy(received_bars, config, validation_config=None):
        calls.append((received_bars, config, validation_config))
        return original(received_bars, config, validation_config)

    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        spy,
    )
    sweeps = (
        ParameterSweep("fast_ema", (10, 30)),
        ParameterSweep("min_trade_confidence", (60, 80)),
    )

    result = run_parameter_sensitivity(
        bars,
        CONFIG,
        sweeps,
        WINDOW_CONFIG,
    )

    assert isinstance(result, SensitivityResult)
    assert result.variant_count == 4
    assert len(calls) == 5
    assert all(call[0] is bars for call in calls)
    assert calls[0][1] is CONFIG
    assert all(call[2] is WINDOW_CONFIG for call in calls)

    assert result.baseline.label == "baseline"
    assert result.baseline.parameter_name is None
    assert result.baseline.parameter_value is None
    assert result.baseline.parameters is DEFAULT_STRATEGY_PARAMETERS
    assert [variant.label for variant in result.variants] == [
        "fast_ema=10",
        "fast_ema=30",
        "min_trade_confidence=60",
        "min_trade_confidence=80",
    ]
    assert [variant.parameter_name for variant in result.variants] == [
        "fast_ema",
        "fast_ema",
        "min_trade_confidence",
        "min_trade_confidence",
    ]
    assert [variant.parameter_value for variant in result.variants] == [
        10,
        30,
        60,
        80,
    ]


def test_variants_change_only_one_parameter(monkeypatch):
    bars = _bull_then_bear()
    original = sensitivity_module.validation_module.validate_strategy
    configs = []

    def spy(received_bars, config, validation_config=None):
        configs.append(config)
        return original(received_bars, config, validation_config)

    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        spy,
    )

    result = run_parameter_sensitivity(
        bars,
        CONFIG,
        (ParameterSweep("fast_ema", (10, 30)),),
        WINDOW_CONFIG,
    )

    for variant, config in zip(result.variants, configs[1:]):
        changed = [
            field
            for field in (
                "fast_ema",
                "slow_ema",
                "rsi_period",
                "rsi_overbought",
                "rsi_oversold",
                "rsi_neutral",
                "macd_fast",
                "macd_slow",
                "macd_signal",
                "volume_period",
                "volume_confirm_ratio",
                "volume_confirm_boost",
                "confidence_strong",
                "confidence_mid",
                "confidence_wait",
                "max_confidence",
                "min_trade_confidence",
                "min_candles",
                "min_directional_votes",
            )
            if getattr(variant.parameters, field)
            != getattr(DEFAULT_STRATEGY_PARAMETERS, field)
        ]
        assert changed == ["fast_ema"]
        assert variant.parameters.fast_ema == variant.parameter_value
        assert config.strategy is variant.parameters


def test_risk_cost_and_validation_invariants_are_preserved():
    bars = _bull_then_bear()
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.002)
    config = BacktestRunConfig("BTC", PROFILE, model)

    result = run_parameter_sensitivity(
        bars,
        config,
        (ParameterSweep("volume_confirm_ratio", (1.0, 2.0)),),
        WINDOW_CONFIG,
    )

    for variant in (result.baseline, *result.variants):
        assert variant.validation.config.risk_profile is PROFILE
        assert variant.validation.config.cost_model is model
        assert variant.validation.validation_config is WINDOW_CONFIG


def test_window_boundaries_and_conditions_are_identical():
    bars = _bull_then_bear()

    result = run_parameter_sensitivity(
        bars,
        CONFIG,
        (
            ParameterSweep("fast_ema", (10, 30)),
            ParameterSweep("min_trade_confidence", (60, 80)),
        ),
        WINDOW_CONFIG,
    )

    baseline = result.baseline.validation
    baseline_signature = [
        (
            window.index,
            window.first_timestamp,
            window.last_timestamp,
            window.bar_count,
            window.condition,
        )
        for window in baseline.windows
    ]
    assert result.bar_count == baseline.bar_count
    assert result.unused_bars == baseline.unused_bars
    assert result.window_conditions == tuple(
        window.condition for window in baseline.windows
    )

    for variant in result.variants:
        validation = variant.validation
        assert validation.bar_count == baseline.bar_count
        assert validation.unused_bars == baseline.unused_bars
        assert [
            (
                window.index,
                window.first_timestamp,
                window.last_timestamp,
                window.bar_count,
                window.condition,
            )
            for window in validation.windows
        ] == baseline_signature
        assert variant.conditions is validation.conditions


def test_inconsistent_variant_windows_are_rejected(monkeypatch):
    bars = _bull_then_bear()
    original = sensitivity_module.validation_module.validate_strategy

    def inconsistent(received_bars, config, validation_config=None):
        result = original(received_bars, config, validation_config)
        if config.strategy.fast_ema == 20:
            return result
        changed_window = replace(
            result.windows[0],
            condition=MarketCondition.SIDEWAYS,
        )
        return replace(
            result,
            windows=(changed_window, *result.windows[1:]),
        )

    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        inconsistent,
    )

    with pytest.raises(ValueError, match="inconsistent window"):
        run_parameter_sensitivity(
            bars,
            CONFIG,
            (ParameterSweep("fast_ema", (10,)),),
            WINDOW_CONFIG,
        )


def test_sensitivity_is_deterministic():
    bars = _bull_then_bear()
    sweeps = (ParameterSweep("fast_ema", (10, 30)),)

    first = run_parameter_sensitivity(bars, CONFIG, sweeps, WINDOW_CONFIG)
    second = run_parameter_sensitivity(bars, CONFIG, sweeps, WINDOW_CONFIG)

    assert first == second
    assert first.baseline == second.baseline
    assert first.variants == second.variants
    assert first.warnings == second.warnings


# --- Preflight rejection --------------------------------------------


@pytest.mark.parametrize(
    "sweeps, error",
    [
        ((), ValueError),
        ([], ValueError),
        ("fast_ema", TypeError),
        ((object(),), TypeError),
        (
            (
                ParameterSweep("fast_ema", (10,)),
                ParameterSweep("fast_ema", (20,)),
            ),
            ValueError,
        ),
    ],
)
def test_invalid_sweep_sets_are_rejected_before_backtests(
    monkeypatch, sweeps, error
):
    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        _fail_if_called,
    )

    with pytest.raises(error):
        run_parameter_sensitivity(
            _bull_then_bear(),
            CONFIG,
            sweeps,
            WINDOW_CONFIG,
        )


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("fast_ema", 20, ValueError),
        ("fast_ema", 50, ValueError),
        ("fast_ema", True, TypeError),
        ("min_trade_confidence", 150, ValueError),
        ("min_candles", 70, ValueError),
    ],
)
def test_invalid_variant_values_are_rejected_before_backtests(
    monkeypatch, field, value, error
):
    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        _fail_if_called,
    )

    with pytest.raises(error):
        run_parameter_sensitivity(
            _bull_then_bear(),
            CONFIG,
            (ParameterSweep(field, (value,)),),
            WINDOW_CONFIG,
        )


@pytest.mark.parametrize("max_variants", [True, 1.0, "2", None])
def test_invalid_max_variants_types_are_rejected(monkeypatch, max_variants):
    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        _fail_if_called,
    )

    with pytest.raises(TypeError):
        run_parameter_sensitivity(
            _bull_then_bear(),
            CONFIG,
            (ParameterSweep("fast_ema", (10,)),),
            WINDOW_CONFIG,
            max_variants=max_variants,
        )


@pytest.mark.parametrize("max_variants", [0, -1])
def test_invalid_max_variants_values_are_rejected(monkeypatch, max_variants):
    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        _fail_if_called,
    )

    with pytest.raises(ValueError):
        run_parameter_sensitivity(
            _bull_then_bear(),
            CONFIG,
            (ParameterSweep("fast_ema", (10,)),),
            WINDOW_CONFIG,
            max_variants=max_variants,
        )


def test_variant_cap_is_enforced_before_backtests(monkeypatch):
    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        _fail_if_called,
    )

    with pytest.raises(ValueError, match="exceeding max_variants"):
        run_parameter_sensitivity(
            _bull_then_bear(),
            CONFIG,
            (ParameterSweep("fast_ema", (10, 30, 40)),),
            WINDOW_CONFIG,
            max_variants=2,
        )


@pytest.mark.parametrize("bars", [None, "bars", {"close": []}, 42])
def test_invalid_bars_are_rejected(monkeypatch, bars):
    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        _fail_if_called,
    )

    with pytest.raises(TypeError):
        run_parameter_sensitivity(
            bars,
            CONFIG,
            (ParameterSweep("fast_ema", (10,)),),
            WINDOW_CONFIG,
        )


def test_invalid_base_config_is_rejected():
    with pytest.raises(TypeError):
        run_parameter_sensitivity(
            _bull_then_bear(),
            object(),
            (ParameterSweep("fast_ema", (10,)),),
            WINDOW_CONFIG,
        )


def test_invalid_validation_config_is_rejected():
    with pytest.raises(TypeError):
        run_parameter_sensitivity(
            _bull_then_bear(),
            CONFIG,
            (ParameterSweep("fast_ema", (10,)),),
            object(),
        )


def test_window_must_fit_largest_configured_warmup(monkeypatch):
    monkeypatch.setattr(
        sensitivity_module.validation_module,
        "validate_strategy",
        _fail_if_called,
    )

    with pytest.raises(ValueError, match="window_bars"):
        run_parameter_sensitivity(
            _bull_then_bear(),
            CONFIG,
            (ParameterSweep("min_candles", (70,)),),
            WINDOW_CONFIG,
        )


def test_window_large_enough_for_custom_warmup_is_accepted():
    bars = _bull_then_bear(72)

    result = run_parameter_sensitivity(
        bars,
        CONFIG,
        (ParameterSweep("min_candles", (70,)),),
        StrategyValidationConfig(window_bars=72),
    )

    assert result.variant_count == 1
    assert result.variants[0].parameters.min_candles == 70


# --- Descriptive results --------------------------------------------


def test_zero_trade_sensitivity_preserves_none_profit_factor():
    result = run_parameter_sensitivity(
        _flat(),
        CONFIG,
        (ParameterSweep("fast_ema", (10, 30)),),
        WINDOW_CONFIG,
    )

    for variant in (result.baseline, *result.variants):
        assert variant.total_trades == 0
        assert variant.completed_trades == 0
        assert variant.open_trades == 0
        assert variant.win_rate == 0.0
        assert variant.gross_pnl == 0.0
        assert variant.net_pnl == 0.0
        assert variant.total_costs == 0.0
        assert variant.net_profit_factor is None
        assert variant.worst_max_drawdown == 0.0
        assert variant.conditions


def test_open_trades_remain_visible_and_excluded_from_realized_metrics(
    monkeypatch,
):
    def fake_replay(received_bars, *, parameters=None):
        records = []
        for index, bar in enumerate(received_bars):
            decision = "BUY" if index == len(received_bars) - 2 else "WAIT"
            records.append(
                SignalRecord(
                    timestamp=bar.timestamp,
                    signal="BUY" if decision == "BUY" else "WAIT",
                    trend="Bullish",
                    confidence=80,
                    decision=decision,
                )
            )
        return records

    monkeypatch.setattr(trades_module, "replay_signals", fake_replay)
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.001)
    config = BacktestRunConfig("BTC", PROFILE, model)

    result = run_parameter_sensitivity(
        _flat(),
        config,
        (ParameterSweep("fast_ema", (10, 30)),),
        WINDOW_CONFIG,
    )

    for variant in (result.baseline, *result.variants):
        assert variant.total_trades == 2
        assert variant.completed_trades == 0
        assert variant.open_trades == 2
        assert variant.wins == 0
        assert variant.losses == 0
        assert variant.win_rate == 0.0
        assert variant.net_pnl == 0.0
        assert variant.total_costs == 0.0
        assert variant.net_profit_factor is None
        assert variant.worst_max_drawdown == 0.0


def test_result_warnings_document_interpretation_limits():
    result = run_parameter_sensitivity(
        _bull_then_bear(),
        CONFIG,
        (ParameterSweep("fast_ema", (10,)),),
        WINDOW_CONFIG,
    )

    text = " ".join(result.warnings).lower()
    assert "in-sample" in text
    assert "interactions" in text
    assert "independently backtested" in text
    assert "never automatically" in text


def test_variant_exposes_task9_condition_and_metric_fields():
    mask = TradingCostModel(fee_rate=0.001, slippage_rate=0.001)
    config = BacktestRunConfig("BTC", PROFILE, mask)

    result = run_parameter_sensitivity(
        _bull_then_bear(),
        config,
        (ParameterSweep("fast_ema", (10,)),),
        WINDOW_CONFIG,
    )

    variant = result.variants[0]
    assert isinstance(variant, SensitivityVariantResult)
    assert isinstance(variant.validation, StrategyValidationResult)
    assert variant.total_trades == sum(
        window.backtest.report.total_trades
        for window in variant.validation.windows
    )
    assert variant.completed_trades == (
        variant.total_trades - variant.open_trades
    )
    assert variant.total_costs >= 0.0
    assert variant.worst_max_drawdown >= 0.0
    assert variant.conditions == variant.validation.conditions
    assert all(
        summary.condition in (
            MarketCondition.BULLISH,
            MarketCondition.BEARISH,
            MarketCondition.SIDEWAYS,
        )
        for summary in variant.conditions
    )
