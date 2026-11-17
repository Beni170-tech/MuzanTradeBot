"""Tests for market-analysis decision making."""

from __future__ import annotations

import pytest

from brain.decision_engine import make_decision
from strategy.parameters import DEFAULT_STRATEGY_PARAMETERS, StrategyParameters


@pytest.mark.parametrize(
    "confidence",
    [
        True,
        False,
        float("nan"),
        float("inf"),
        float("-inf"),
        -1,
        -0.1,
        100.1,
        101,
        10**400,
        "80",
        None,
        [80],
        {"value": 80},
        complex(80, 0),
    ],
)
def test_invalid_confidence_returns_wait(confidence) -> None:
    analysis = {"signal": "BUY", "trend": "Bullish", "confidence": confidence}

    assert make_decision(analysis) == "WAIT"


@pytest.mark.parametrize("confidence", [70, 70.0, 100])
def test_bullish_buy_at_or_above_threshold_returns_buy(confidence) -> None:
    analysis = {
        "signal": "BUY",
        "trend": "Bullish",
        "confidence": confidence,
    }

    assert make_decision(analysis) == "BUY"


@pytest.mark.parametrize("confidence", [70, 70.0, 100])
def test_bearish_sell_at_or_above_threshold_returns_sell(confidence) -> None:
    analysis = {
        "signal": "SELL",
        "trend": "Bearish",
        "confidence": confidence,
    }

    assert make_decision(analysis) == "SELL"


@pytest.mark.parametrize(
    ("signal", "trend", "confidence"),
    [
        ("BUY", "Bullish", 69.999),
        ("SELL", "Bearish", 69.999),
        ("BUY", "Bearish", 80),
        ("SELL", "Bullish", 80),
        ("HOLD", "Bullish", 80),
    ],
)
def test_non_matching_or_weak_signals_return_wait(
    signal, trend, confidence
) -> None:
    analysis = {
        "signal": signal,
        "trend": trend,
        "confidence": confidence,
    }

    assert make_decision(analysis) == "WAIT"


@pytest.mark.parametrize(
    ("signal", "trend"),
    [(None, "Bullish"), ("BUY", None), (1, "Bullish"), ("BUY", 1)],
)
def test_non_string_signal_or_trend_returns_wait(signal, trend) -> None:
    analysis = {
        "signal": signal,
        "trend": trend,
        "confidence": 80,
    }

    assert make_decision(analysis) == "WAIT"


def test_default_parameters_match_omitted_behavior() -> None:
    analysis = {
        "signal": "BUY",
        "trend": "Bullish",
        "confidence": 70,
    }

    assert make_decision(analysis) == make_decision(
        analysis,
        parameters=DEFAULT_STRATEGY_PARAMETERS,
    )


@pytest.mark.parametrize(
    ("confidence", "threshold", "expected"),
    [
        (60, 60, "BUY"),
        (59.999, 60, "WAIT"),
        (80, 80, "SELL"),
        (79.999, 80, "WAIT"),
    ],
)
def test_custom_minimum_trade_confidence(
    confidence, threshold, expected
) -> None:
    parameters = StrategyParameters(min_trade_confidence=threshold)
    analysis = {
        "signal": "BUY" if expected == "BUY" else "SELL",
        "trend": "Bullish" if expected == "BUY" else "Bearish",
        "confidence": confidence,
    }

    assert make_decision(analysis, parameters=parameters) == expected


def test_invalid_parameters_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="parameters"):
        make_decision(
            {"signal": "BUY", "trend": "Bullish", "confidence": 80},
            parameters=object(),
        )
