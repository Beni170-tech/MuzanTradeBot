"""Tests for deterministic chronological out-of-sample validation."""

from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone

import pytest

import backtest.out_of_sample as out_of_sample_module
import backtest.service as service_module
import backtest.trades as trades_module
from backtest.out_of_sample import (
    PERIOD_ROLES,
    ChronologicalSplitConfig,
    OutOfSampleResult,
    PeriodRole,
    run_out_of_sample_validation,
    split_chronological_bars,
)
from backtest.replay import HistoricalBar, SignalRecord
from backtest.runner import BacktestRunConfig
from backtest.validation import StrategyValidationConfig
from risk.risk_profile import RiskProfile
from strategy.parameters import StrategyParameters

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROFILE = RiskProfile(
    balance=1000.0,
    risk_percent=2.0,
    stop_loss_percent=2.0,
    take_profit_percent=4.0,
)
COMPACT_STRATEGY = StrategyParameters(
    fast_ema=3,
    slow_ema=5,
    rsi_period=2,
    macd_fast=2,
    macd_slow=3,
    macd_signal=2,
    volume_period=2,
    min_candles=5,
)
CONFIG = BacktestRunConfig(
    "BTC",
    PROFILE,
    strategy=COMPACT_STRATEGY,
)
SPLIT = ChronologicalSplitConfig(
    development_bars=7,
    validation_bars=7,
    test_bars=7,
    embargo_bars=2,
)
TOTAL_BARS = SPLIT.required_bars
REAL_SPLIT = ChronologicalSplitConfig(
    development_bars=62,
    validation_bars=62,
    test_bars=62,
    embargo_bars=3,
)
REAL_TOTAL_BARS = REAL_SPLIT.required_bars


def _bar(index, close, volume=1_000.0):
    return HistoricalBar(
        timestamp=START + timedelta(hours=index),
        close=close,
        volume=volume,
    )


def _bars(count, *, price=100.0, step=0.5):
    return [
        _bar(index, price + index * step)
        for index in range(count)
    ]


def _spiking_bars():
    """Bars where every period ends with a take-profit spike."""
    period_starts = (0, 9, 18)
    closes = {}
    for start in period_starts:
        for offset in range(6):
            closes[start + offset] = 100.0 + offset
        closes[start + 6] = 115.5
    closes[7] = closes[8] = 100.0
    closes[16] = closes[17] = 100.0
    return [_bar(index, closes[index]) for index in range(TOTAL_BARS)]


def _record(bar, decision):
    if decision == "BUY":
        signal = "BUY"
        trend = "Bullish"
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


def _patch_period_replay(monkeypatch):
    """Emit one BUY near the end of whatever bars a period receives."""

    def fake_replay(received_bars, *, parameters=None):
        window = received_bars[-3:]
        return [
            _record(window[0], "BUY"),
            _record(window[1], "WAIT"),
            _record(window[2], "WAIT"),
        ]

    monkeypatch.setattr(trades_module, "replay_signals", fake_replay)


def _mutate(bars, indices, factor=2.0):
    updated = list(bars)
    for index in indices:
        updated[index] = replace(
            bars[index],
            close=bars[index].close * factor,
        )
    return updated


def test_split_produces_chronological_contiguous_periods():
    bars = _bars(TOTAL_BARS)

    split = split_chronological_bars(bars, SPLIT)

    assert split.bar_count == TOTAL_BARS
    assert split.unused_leading_bars == 0
    assert [boundary.role for boundary in split.boundaries] == list(
        PERIOD_ROLES
    )
    development, validation, test = split.boundaries
    assert (development.start_index, development.end_index) == (0, 7)
    assert (validation.start_index, validation.end_index) == (9, 16)
    assert (test.start_index, test.end_index) == (18, 25)
    assert development.start_timestamp == bars[0].timestamp
    assert development.end_timestamp == bars[6].timestamp
    assert test.end_timestamp == bars[24].timestamp
    assert test.end_index == len(bars)
    assert split.evaluated_bar_count == 21


def test_period_boundaries_never_overlap_and_test_is_last():
    bars = _bars(TOTAL_BARS)
    split = split_chronological_bars(bars, SPLIT)

    for earlier, later in zip(split.boundaries, split.boundaries[1:]):
        assert earlier.end_index <= later.start_index
    assert split.boundaries[-1].role is PeriodRole.TEST
    assert split.boundaries[-1].end_index == len(bars)


def test_embargo_spans_are_reported_and_belong_to_no_period():
    split = split_chronological_bars(_bars(TOTAL_BARS), SPLIT)

    first, second = split.embargo_spans
    assert (first.start_index, first.end_index) == (7, 9)
    assert (second.start_index, second.end_index) == (16, 18)
    assert first.preceding_role is PeriodRole.DEVELOPMENT
    assert first.following_role is PeriodRole.VALIDATION
    assert second.following_role is PeriodRole.TEST
    assert split.embargo_bar_count == 4
    assert split.embargo_bar_indices == (7, 8, 16, 17)

    assigned = set()
    for boundary in split.boundaries:
        assigned.update(range(boundary.start_index, boundary.end_index))
    assert assigned.isdisjoint(set(split.embargo_bar_indices))


def test_zero_embargo_makes_periods_adjacent():
    split = split_chronological_bars(
        _bars(21),
        ChronologicalSplitConfig(7, 7, 7),
    )

    assert split.embargo_spans == ()
    assert split.embargo_bar_indices == ()
    assert [
        (boundary.start_index, boundary.end_index)
        for boundary in split.boundaries
    ] == [(0, 7), (7, 14), (14, 21)]


def test_split_is_anchored_to_the_end_of_history():
    bars = _bars(TOTAL_BARS + 5)

    split = split_chronological_bars(bars, SPLIT)

    assert split.unused_leading_bars == 5
    assert split.boundaries[0].start_index == 5
    assert split.boundaries[-1].end_index == len(bars)
    assert split.boundaries[-1].end_timestamp == bars[-1].timestamp
    assert [
        (boundary.start_index, boundary.end_index)
        for boundary in split.boundaries
    ] == [(5, 12), (14, 21), (23, 30)]
    assert split.embargo_bar_indices == (12, 13, 21, 22)
    assert split.evaluated_bar_count == 21


def test_leading_bars_are_never_evaluated(monkeypatch):
    calls = []
    real_run = out_of_sample_module.run_backtest

    def spy(period_bars, config):
        calls.append(tuple(period_bars))
        return real_run(period_bars, config)

    monkeypatch.setattr(out_of_sample_module, "run_backtest", spy)
    bars = _bars(TOTAL_BARS + 5)

    result = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    assert [len(received) for received in calls] == [7, 7, 7]
    for boundary, received in zip(result.boundaries, calls):
        assert received == tuple(
            bars[boundary.start_index:boundary.end_index]
        )
    evaluated = {bar.timestamp for received in calls for bar in received}
    for index in range(5):
        assert bars[index].timestamp not in evaluated
    assert result.boundaries[-1].end_index == len(bars)


def test_leading_bars_do_not_change_period_evaluation(monkeypatch):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS + 5)

    padded = run_out_of_sample_validation(bars, CONFIG, SPLIT)
    exact = run_out_of_sample_validation(bars[5:], CONFIG, SPLIT)

    for padded_period, exact_period in zip(
        padded.periods,
        exact.periods,
    ):
        assert padded_period.backtest == exact_period.backtest
        assert padded_period.condition is exact_period.condition
        assert padded_period.return_percent == exact_period.return_percent
        assert padded_period.directional_efficiency == (
            exact_period.directional_efficiency
        )


def test_mutating_leading_bars_cannot_change_period_results(monkeypatch):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS + 5)
    baseline = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    mutated = _mutate(bars, range(0, 5), factor=4.0)
    changed = run_out_of_sample_validation(mutated, CONFIG, SPLIT)

    assert changed.boundaries == baseline.boundaries
    assert changed.periods == baseline.periods
    assert changed.unused_leading_bars == 5


def test_split_is_deterministic():
    bars = _bars(TOTAL_BARS)

    assert split_chronological_bars(bars, SPLIT) == (
        split_chronological_bars(bars, SPLIT)
    )


@pytest.mark.parametrize("bar_count", [0, 1, TOTAL_BARS - 1])
def test_insufficient_history_is_rejected(bar_count):
    with pytest.raises(ValueError, match="insufficient history"):
        split_chronological_bars(_bars(bar_count), SPLIT)


@pytest.mark.parametrize(
    "overrides",
    [
        {"development_bars": 0},
        {"validation_bars": -1},
        {"test_bars": 0},
        {"embargo_bars": -1},
    ],
)
def test_invalid_split_values_are_rejected(overrides):
    values = {
        "development_bars": 7,
        "validation_bars": 7,
        "test_bars": 7,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        ChronologicalSplitConfig(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"development_bars": True},
        {"development_bars": 7.0},
        {"validation_bars": "7"},
        {"test_bars": None},
        {"embargo_bars": True},
        {"embargo_bars": 1.5},
    ],
)
def test_invalid_split_types_are_rejected(overrides):
    values = {
        "development_bars": 1,
        "validation_bars": 1,
        "test_bars": 1,
    }
    values.update(overrides)

    with pytest.raises(TypeError):
        ChronologicalSplitConfig(**values)


def test_required_bars_counts_embargo():
    assert SPLIT.period_bar_count == 21
    assert SPLIT.embargo_bar_count == 4
    assert SPLIT.required_bars == 25


def test_timezone_naive_timestamps_are_rejected():
    naive = replace(_bar(0, 100.0), timestamp=datetime(2024, 1, 1))

    with pytest.raises(ValueError, match="timezone-aware"):
        split_chronological_bars([naive], SPLIT)


def test_unsorted_and_duplicate_timestamps_are_rejected():
    bars = _bars(3)

    with pytest.raises(ValueError, match="strictly increasing"):
        split_chronological_bars(list(reversed(bars)), SPLIT)
    with pytest.raises(ValueError, match="strictly increasing"):
        split_chronological_bars(
            [bars[0], replace(bars[0], close=101.0)],
            SPLIT,
        )


@pytest.mark.parametrize(
    ("bars", "error"),
    [
        ([object()], TypeError),
        ([_bar(0, 0.0)], ValueError),
        ([_bar(0, -1.0)], ValueError),
        ([_bar(0, float("nan"))], ValueError),
        ([_bar(0, 100.0, volume=-1.0)], ValueError),
        ([_bar(0, 100.0, volume=float("inf"))], ValueError),
        ([_bar(0, True)], ValueError),
    ],
)
def test_invalid_bars_are_rejected(bars, error):
    with pytest.raises(error):
        split_chronological_bars(bars, SPLIT)


@pytest.mark.parametrize("bars", [None, "bars", {"close": []}, 42])
def test_non_sequence_bars_are_rejected(bars):
    with pytest.raises(TypeError, match="bars"):
        split_chronological_bars(bars, SPLIT)


def test_invalid_argument_types_are_rejected():
    with pytest.raises(TypeError, match="split_config"):
        split_chronological_bars(_bars(TOTAL_BARS), object())
    with pytest.raises(TypeError, match="config"):
        run_out_of_sample_validation(_bars(TOTAL_BARS), object(), SPLIT)
    with pytest.raises(TypeError, match="split_config"):
        run_out_of_sample_validation(
            _bars(TOTAL_BARS),
            CONFIG,
            object(),
        )
    with pytest.raises(TypeError, match="validation_config"):
        run_out_of_sample_validation(
            _bars(TOTAL_BARS),
            CONFIG,
            SPLIT,
            object(),
        )


def test_input_bars_are_never_sorted_or_modified():
    bars = _bars(TOTAL_BARS)
    original = list(bars)

    split_chronological_bars(bars, SPLIT)

    assert bars == original

    unordered = list(reversed(bars))
    with pytest.raises(ValueError):
        split_chronological_bars(unordered, SPLIT)
    assert unordered == list(reversed(bars))


def test_each_period_receives_only_its_own_bars(monkeypatch):
    calls = []
    real_run = out_of_sample_module.run_backtest

    def spy(period_bars, config):
        calls.append(tuple(period_bars))
        return real_run(period_bars, config)

    monkeypatch.setattr(out_of_sample_module, "run_backtest", spy)
    bars = _bars(TOTAL_BARS)

    result = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    assert len(calls) == 3
    assert [len(received) for received in calls] == [7, 7, 7]
    for boundary, received in zip(result.boundaries, calls):
        assert received == tuple(
            bars[boundary.start_index:boundary.end_index]
        )

    embargo = set(split_chronological_bars(bars, SPLIT).embargo_bar_indices)
    received_indices = set()
    for received in calls:
        for bar in received:
            index = int(
                (bar.timestamp - START) / timedelta(hours=1)
            )
            received_indices.add(index)
    assert received_indices.isdisjoint(embargo)


def test_no_data_loader_is_called_inside_validation(monkeypatch):
    assert not hasattr(out_of_sample_module, "load_historical_bars")

    def fail(*args, **kwargs):
        raise AssertionError("historical data loader must not be called")

    monkeypatch.setattr(service_module, "load_historical_bars", fail)

    result = run_out_of_sample_validation(_bars(TOTAL_BARS), CONFIG, SPLIT)

    assert result.bar_count == TOTAL_BARS


def test_every_period_reuses_the_same_config(monkeypatch):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS)
    model = CONFIG.cost_model

    result = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    for period in result.periods:
        assert period.backtest.config is CONFIG
        assert period.report.cost_model is model
    assert result.config is CONFIG


def test_mutating_test_bars_does_not_change_earlier_periods(monkeypatch):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS)
    baseline = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    mutated = _mutate(bars, range(18, 25), factor=1.75)
    changed = run_out_of_sample_validation(mutated, CONFIG, SPLIT)

    assert changed.development_period == baseline.development_period
    assert changed.validation_period == baseline.validation_period
    assert changed.test_period != baseline.test_period


def test_mutating_validation_and_test_bars_does_not_change_development(
    monkeypatch,
):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS)
    baseline = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    indices = list(range(9, 25))
    mutated = _mutate(bars, indices, factor=0.5)
    changed = run_out_of_sample_validation(mutated, CONFIG, SPLIT)

    assert changed.development_period == baseline.development_period
    assert changed.development_period.role is PeriodRole.DEVELOPMENT


def test_mutating_test_bars_does_not_change_validation(monkeypatch):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS)
    baseline = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    mutated = _mutate(bars, range(18, 25), factor=3.0)
    changed = run_out_of_sample_validation(mutated, CONFIG, SPLIT)

    assert changed.validation_period == baseline.validation_period


def test_period_results_respond_to_their_own_bars(monkeypatch):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS)
    baseline = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    # Index 5 is the development period's entry bar.
    mutated = _mutate(bars, [5], factor=1.5)
    changed = run_out_of_sample_validation(mutated, CONFIG, SPLIT)

    assert changed.development_period != baseline.development_period
    assert changed.validation_period == baseline.validation_period
    assert changed.test_period == baseline.test_period


def test_repeated_runs_are_deterministic(monkeypatch):
    _patch_period_replay(monkeypatch)
    bars = _bars(TOTAL_BARS)

    first = run_out_of_sample_validation(bars, CONFIG, SPLIT)
    second = run_out_of_sample_validation(bars, CONFIG, SPLIT)

    assert first == second
    assert first.periods == second.periods
    assert first.warnings == second.warnings


def test_result_exposes_no_selection_or_ranking_api(monkeypatch):
    _patch_period_replay(monkeypatch)

    result = run_out_of_sample_validation(_bars(TOTAL_BARS), CONFIG, SPLIT)

    names = {field.name for field in fields(OutOfSampleResult)}
    period_names = {field.name for field in fields(type(result.periods[0]))}
    forbidden = ("select", "rank", "best", "winner", "optimi")
    for name in names | period_names:
        assert not any(term in name for term in forbidden)
    for attribute in ("selected_config", "best_config", "ranking"):
        assert not hasattr(result, attribute)


def test_period_minimum_bars_are_enforced():
    too_small = ChronologicalSplitConfig(
        development_bars=7,
        validation_bars=7,
        test_bars=6,
    )

    with pytest.raises(ValueError, match="test period has 6 bars"):
        run_out_of_sample_validation(_bars(20), CONFIG, too_small)


def test_embargo_can_exhaust_available_history():
    wide_embargo = ChronologicalSplitConfig(
        development_bars=7,
        validation_bars=7,
        test_bars=7,
        embargo_bars=20,
    )

    with pytest.raises(ValueError, match="insufficient history"):
        run_out_of_sample_validation(
            _bars(TOTAL_BARS),
            CONFIG,
            wide_embargo,
        )


def test_no_trade_periods_are_reported(monkeypatch):
    monkeypatch.setattr(
        trades_module,
        "replay_signals",
        lambda received_bars, *, parameters=None: [],
    )

    result = run_out_of_sample_validation(_bars(TOTAL_BARS), CONFIG, SPLIT)

    for period in result.periods:
        assert period.total_trades == 0
        assert period.completed_trades == 0
        assert period.net_pnl == 0.0
    assert any(
        "Development period generated no trades" in warning
        for warning in result.warnings
    )


def test_all_open_trade_periods_are_reported(monkeypatch):
    _patch_period_replay(monkeypatch)

    result = run_out_of_sample_validation(_bars(TOTAL_BARS), CONFIG, SPLIT)

    for period in result.periods:
        assert period.total_trades == 1
        assert period.completed_trades == 0
        assert period.open_trades == 1
        assert period.net_pnl == 0.0
    assert any(
        "All trades in the test period are open" in warning
        for warning in result.warnings
    )


def test_completed_trades_are_reported_per_period(monkeypatch):
    _patch_period_replay(monkeypatch)

    result = run_out_of_sample_validation(
        _spiking_bars(),
        CONFIG,
        SPLIT,
    )

    for period in result.periods:
        assert period.total_trades == 1
        assert period.completed_trades == 1
        assert period.open_trades == 0
        assert period.net_pnl > 0.0
        assert period.win_rate == 100.0
    assert not any(
        "are open" in warning for warning in result.warnings
    )


def test_embargo_and_leading_warnings_are_explicit(monkeypatch):
    _patch_period_replay(monkeypatch)

    result = run_out_of_sample_validation(
        _bars(TOTAL_BARS + 4),
        CONFIG,
        SPLIT,
    )

    assert any("4 embargo bars" in warning for warning in result.warnings)
    assert any("4 leading bars" in warning for warning in result.warnings)
    assert result.unused_leading_bars == 4
    assert result.test_period.boundary.end_index == TOTAL_BARS + 4


def test_standard_limitation_warnings_are_always_present(monkeypatch):
    _patch_period_replay(monkeypatch)

    result = run_out_of_sample_validation(_bars(TOTAL_BARS), CONFIG, SPLIT)

    assert any(
        "independent backtests" in warning for warning in result.warnings
    )
    assert any(
        "cannot be proven by software" in warning
        for warning in result.warnings
    )
    assert any(
        "freeze the strategy configuration" in warning
        for warning in result.warnings
    )


def test_classification_thresholds_are_applied(monkeypatch):
    _patch_period_replay(monkeypatch)
    strict = StrategyValidationConfig(trend_threshold_percent=1_000_000.0)

    result = run_out_of_sample_validation(
        _bars(TOTAL_BARS, step=1.0),
        CONFIG,
        SPLIT,
        strict,
    )

    assert result.validation_config is strict
    assert all(
        period.condition.value == "sideways"
        for period in result.periods
    )


def test_unused_window_bars_are_flagged(monkeypatch):
    _patch_period_replay(monkeypatch)
    custom = StrategyValidationConfig(window_bars=90)

    result = run_out_of_sample_validation(
        _bars(TOTAL_BARS),
        CONFIG,
        SPLIT,
        custom,
    )

    assert any(
        "window_bars is not used" in warning
        for warning in result.warnings
    )


def test_period_results_slice_boundaries_not_windows(monkeypatch):
    _patch_period_replay(monkeypatch)

    result = run_out_of_sample_validation(_bars(TOTAL_BARS), CONFIG, SPLIT)

    assert [period.boundary.bar_count for period in result.periods] == [
        7,
        7,
        7,
    ]
    for period in result.periods:
        report = period.report
        assert period.backtest.bar_count == period.boundary.bar_count
        assert period.backtest.first_bar_timestamp == (
            period.boundary.start_timestamp
        )
        assert period.backtest.last_bar_timestamp == (
            period.boundary.end_timestamp
        )
        assert report.initial_equity == PROFILE.balance


def test_real_pipeline_runs_across_three_periods():
    bars = _bars(REAL_TOTAL_BARS, step=0.6)

    first = run_out_of_sample_validation(bars, BacktestRunConfig(
        "BTC",
        PROFILE,
    ), REAL_SPLIT)
    second = run_out_of_sample_validation(bars, BacktestRunConfig(
        "BTC",
        PROFILE,
    ), REAL_SPLIT)

    assert first == second
    assert first.unused_leading_bars == 0
    assert [period.boundary.bar_count for period in first.periods] == [
        62,
        62,
        62,
    ]
    assert first.test_period.boundary.end_index == REAL_TOTAL_BARS
    assert first.embargo_bar_indices == (62, 63, 64, 127, 128, 129)
    assert first.development_period.role is PeriodRole.DEVELOPMENT
    assert first.validation_period.role is PeriodRole.VALIDATION
    assert first.test_period.role is PeriodRole.TEST
