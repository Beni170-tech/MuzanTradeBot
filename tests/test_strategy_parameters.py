"""Tests for the immutable strategy parameter model."""

from dataclasses import FrozenInstanceError

import pytest

import market.indicators as indicators
from strategy import DEFAULT_STRATEGY_PARAMETERS, StrategyParameters


def test_defaults_match_existing_indicator_constants():
    parameters = StrategyParameters()

    assert parameters == DEFAULT_STRATEGY_PARAMETERS
    assert parameters.fast_ema == indicators.FAST_EMA == 20
    assert parameters.slow_ema == indicators.SLOW_EMA == 50
    assert parameters.rsi_period == indicators.RSI_PERIOD == 14
    assert parameters.macd_fast == indicators.MACD_FAST == 12
    assert parameters.macd_slow == indicators.MACD_SLOW == 26
    assert parameters.macd_signal == indicators.MACD_SIGNAL == 9
    assert parameters.volume_period == indicators.VOLUME_PERIOD == 20
    assert parameters.volume_confirm_ratio == indicators.VOLUME_CONFIRM_RATIO
    assert parameters.volume_confirm_boost == indicators.VOLUME_CONFIRM_BOOST
    assert parameters.confidence_strong == indicators.CONFIDENCE_STRONG
    assert parameters.confidence_mid == indicators.CONFIDENCE_MID
    assert parameters.confidence_wait == indicators.CONFIDENCE_WAIT
    assert parameters.min_candles == indicators.MIN_CANDLES == 60


def test_defaults_match_previous_decision_and_rsi_literals():
    parameters = StrategyParameters()

    assert parameters.rsi_overbought == 70
    assert parameters.rsi_oversold == 30
    assert parameters.rsi_neutral == 50
    assert parameters.max_confidence == 95
    assert parameters.min_trade_confidence == 70
    assert parameters.min_directional_votes == 2


def test_parameters_are_frozen_and_slotted():
    parameters = StrategyParameters()

    assert not hasattr(parameters, "__dict__")
    with pytest.raises(FrozenInstanceError):
        parameters.fast_ema = 10


@pytest.mark.parametrize(
    "field_name",
    [
        "fast_ema",
        "slow_ema",
        "rsi_period",
        "macd_fast",
        "macd_slow",
        "macd_signal",
        "volume_period",
        "min_candles",
        "min_directional_votes",
    ],
)
@pytest.mark.parametrize("value", [True, False, 20.0, "20", None])
def test_integer_fields_reject_non_integers(field_name, value):
    with pytest.raises(TypeError, match=field_name):
        StrategyParameters(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    [
        "fast_ema",
        "slow_ema",
        "rsi_period",
        "macd_fast",
        "macd_slow",
        "macd_signal",
        "volume_period",
        "min_candles",
    ],
)
def test_period_fields_must_be_positive(field_name):
    with pytest.raises(ValueError, match=field_name):
        StrategyParameters(**{field_name: 0})

    with pytest.raises(ValueError, match=field_name):
        StrategyParameters(**{field_name: -1})


@pytest.mark.parametrize(
    "field_name",
    [
        "rsi_overbought",
        "rsi_oversold",
        "rsi_neutral",
        "volume_confirm_ratio",
        "volume_confirm_boost",
        "confidence_strong",
        "confidence_mid",
        "confidence_wait",
        "max_confidence",
        "min_trade_confidence",
    ],
)
@pytest.mark.parametrize("value", [True, False, "20", None])
def test_numeric_fields_reject_non_numbers(field_name, value):
    with pytest.raises(TypeError, match=field_name):
        StrategyParameters(**{field_name: value})


@pytest.mark.parametrize(
    "field_name",
    [
        "rsi_overbought",
        "rsi_oversold",
        "rsi_neutral",
        "volume_confirm_ratio",
        "volume_confirm_boost",
        "confidence_strong",
        "confidence_mid",
        "confidence_wait",
        "max_confidence",
        "min_trade_confidence",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**400])
def test_numeric_fields_reject_non_finite_values(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        StrategyParameters(**{field_name: value})


def test_ema_periods_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="fast_ema"):
        StrategyParameters(fast_ema=50, slow_ema=50)
    with pytest.raises(ValueError, match="fast_ema"):
        StrategyParameters(fast_ema=60, slow_ema=50)


def test_macd_periods_must_be_strictly_increasing():
    with pytest.raises(ValueError, match="macd_fast"):
        StrategyParameters(macd_fast=26, macd_slow=26)
    with pytest.raises(ValueError, match="macd_fast"):
        StrategyParameters(macd_fast=30, macd_slow=26)


@pytest.mark.parametrize(
    "overrides",
    [
        {"rsi_oversold": 0},
        {"rsi_oversold": 70, "rsi_neutral": 50},
        {"rsi_neutral": 80, "rsi_overbought": 70},
        {"rsi_overbought": 101},
        {"rsi_oversold": -1},
    ],
)
def test_rsi_threshold_relationships_are_enforced(overrides):
    with pytest.raises(ValueError, match="RSI"):
        StrategyParameters(**overrides)


def test_volume_parameters_must_be_non_negative():
    with pytest.raises(ValueError, match="volume_confirm_ratio"):
        StrategyParameters(volume_confirm_ratio=-0.1)
    with pytest.raises(ValueError, match="volume_confirm_boost"):
        StrategyParameters(volume_confirm_boost=-0.1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"confidence_wait": -1},
        {"confidence_wait": 80, "confidence_mid": 74},
        {"confidence_mid": 90, "confidence_strong": 88},
        {"confidence_strong": 96, "max_confidence": 95},
        {"max_confidence": 101},
        {"min_trade_confidence": -1},
        {"min_trade_confidence": 101},
    ],
)
def test_confidence_ranges_and_order_are_enforced(overrides):
    with pytest.raises(ValueError, match="confidence"):
        StrategyParameters(**overrides)


@pytest.mark.parametrize("votes", [0, 4, -1])
def test_minimum_directional_votes_must_be_between_one_and_three(votes):
    with pytest.raises(ValueError, match="min_directional_votes"):
        StrategyParameters(min_directional_votes=votes)


def test_min_candles_must_cover_indicator_warmup():
    with pytest.raises(ValueError, match="warmup"):
        StrategyParameters(min_candles=49)

    assert StrategyParameters(min_candles=50).min_candles == 50


def test_min_candles_uses_configured_periods_not_just_slow_ema():
    with pytest.raises(ValueError, match="warmup"):
        StrategyParameters(
            fast_ema=10,
            slow_ema=20,
            rsi_period=30,
            macd_fast=8,
            macd_slow=15,
            macd_signal=5,
            volume_period=10,
            min_candles=25,
        )


def test_valid_custom_parameters_are_accepted():
    parameters = StrategyParameters(
        fast_ema=10,
        slow_ema=30,
        rsi_period=10,
        rsi_overbought=80,
        rsi_oversold=20,
        rsi_neutral=45,
        macd_fast=8,
        macd_slow=20,
        macd_signal=5,
        volume_period=10,
        volume_confirm_ratio=2.0,
        volume_confirm_boost=4,
        confidence_strong=90,
        confidence_mid=75,
        confidence_wait=40,
        max_confidence=96,
        min_trade_confidence=65,
        min_candles=40,
        min_directional_votes=3,
    )

    assert parameters.fast_ema == 10
    assert parameters.min_candles == 40
    assert parameters.min_directional_votes == 3


def test_invalid_values_are_not_silently_clamped():
    with pytest.raises(ValueError):
        StrategyParameters(min_trade_confidence=150)

    parameters = StrategyParameters(min_trade_confidence=100)
    assert parameters.min_trade_confidence == 100
