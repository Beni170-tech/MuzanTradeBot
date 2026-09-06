"""Tests for risk.risk_manager.validate_trade."""

import pytest

from risk.risk_manager import RiskVerdict, validate_trade
from risk.risk_profile import RiskProfile

DEFAULT_PROFILE = RiskProfile()

VALID_ANALYSIS = {
    "symbol": "BTC",
    "price": 65000.0,
    "trend": "Bullish",
    "signal": "BUY",
    "confidence": 85,
}


def test_valid_buy_is_approved():
    verdict = validate_trade("BUY", VALID_ANALYSIS, DEFAULT_PROFILE)
    assert verdict.approved is True
    assert verdict.decision == "BUY"
    assert verdict.reasons


def test_valid_sell_is_approved():
    analysis = {**VALID_ANALYSIS, "trend": "Bearish", "signal": "SELL"}
    verdict = validate_trade("SELL", analysis, DEFAULT_PROFILE)
    assert verdict.approved is True
    assert verdict.decision == "SELL"


def test_wait_decision_is_blocked():
    verdict = validate_trade("WAIT", VALID_ANALYSIS, DEFAULT_PROFILE)
    assert verdict.approved is False
    assert verdict.decision == "WAIT"
    assert verdict.reasons


@pytest.mark.parametrize("confidence", [None, "high", 101, -1, -20])
def test_out_of_range_confidence_is_blocked(confidence):
    analysis = {**VALID_ANALYSIS, "confidence": confidence}
    verdict = validate_trade("BUY", analysis, DEFAULT_PROFILE)
    assert verdict.approved is False
    assert any("Confidence" in reason for reason in verdict.reasons)


@pytest.mark.parametrize("price", ["Unknown", 0, -10, None])
def test_missing_or_invalid_price_is_blocked(price):
    analysis = {**VALID_ANALYSIS, "price": price}
    verdict = validate_trade("BUY", analysis, DEFAULT_PROFILE)
    assert verdict.approved is False
    assert any("price" in reason.lower() for reason in verdict.reasons)


@pytest.mark.parametrize(
    "decision,trend",
    [("BUY", "Bearish"), ("SELL", "Bullish")],
)
def test_trend_conflicting_with_direction_is_blocked(decision, trend):
    analysis = {**VALID_ANALYSIS, "trend": trend, "signal": decision}
    verdict = validate_trade(decision, analysis, DEFAULT_PROFILE)
    assert verdict.approved is False
    assert any("conflict" in reason.lower() for reason in verdict.reasons)


@pytest.mark.parametrize("trend", [None, "Sideways", "Unknown"])
def test_unknown_trend_is_blocked(trend):
    analysis = {**VALID_ANALYSIS, "trend": trend}
    verdict = validate_trade("BUY", analysis, DEFAULT_PROFILE)
    assert verdict.approved is False
    assert any("condition" in reason.lower() for reason in verdict.reasons)


@pytest.mark.parametrize(
    "profile",
    [
        RiskProfile(balance=0),
        RiskProfile(risk_percent=0),
        RiskProfile(stop_loss_percent=0),
        RiskProfile(take_profit_percent=0),
        RiskProfile(balance=-500),
    ],
)
def test_bad_risk_profile_is_blocked(profile):
    verdict = validate_trade("BUY", VALID_ANALYSIS, profile)
    assert verdict.approved is False
    assert verdict.decision == "WAIT"


def test_non_dict_analysis_is_blocked():
    verdict = validate_trade("BUY", None, DEFAULT_PROFILE)
    assert verdict.approved is False


def test_profile_defaults_to_standard():
    verdict = validate_trade("BUY", VALID_ANALYSIS)
    assert verdict.approved is True


def test_verdict_is_dataclass_with_reasons():
    verdict = validate_trade("WAIT", VALID_ANALYSIS, DEFAULT_PROFILE)
    assert isinstance(verdict, RiskVerdict)
    assert isinstance(verdict.reasons, list)
