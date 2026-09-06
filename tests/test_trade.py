"""Tests for trader.trade.build_trade."""

import pytest

from risk.risk_manager import RiskVerdict
from risk.risk_profile import RiskProfile
from trader.trade import Trade, build_trade

PROFILE = RiskProfile(balance=1000, risk_percent=2, stop_loss_percent=2, take_profit_percent=4)

APPROVED_BUY = RiskVerdict(decision="BUY", approved=True, reasons=["Risk checks passed."])
APPROVED_SELL = RiskVerdict(decision="SELL", approved=True, reasons=["Risk checks passed."])
BLOCKED = RiskVerdict(
    decision="WAIT",
    approved=False,
    reasons=["No valid market price available; cannot size or protect a position."],
)

VALID_ANALYSIS = {
    "symbol": "BTC",
    "price": 100.0,
    "trend": "Bullish",
    "signal": "BUY",
    "confidence": 85,
}


def test_approved_buy_populates_full_trade():
    trade = build_trade(VALID_ANALYSIS, APPROVED_BUY, PROFILE)
    assert isinstance(trade, Trade)
    assert trade.decision == "BUY"
    assert trade.risk_verdict == "APPROVED"
    assert trade.approved is True
    assert trade.side == "LONG"
    assert trade.position_size == pytest.approx(10.0)
    assert trade.stop_loss == 98.0
    assert trade.take_profit == 104.0


def test_approved_sell_is_short_sided():
    analysis = {**VALID_ANALYSIS, "trend": "Bearish", "signal": "SELL"}
    trade = build_trade(analysis, APPROVED_SELL, PROFILE)
    assert trade.side == "SHORT"
    assert trade.decision == "SELL"
    assert trade.stop_loss == 102.0
    assert trade.take_profit == 96.0


def test_blocked_trade_keeps_zeros_and_reasons():
    trade = build_trade(VALID_ANALYSIS, BLOCKED, PROFILE)
    assert trade.decision == "WAIT"
    assert trade.risk_verdict == "BLOCKED"
    assert trade.approved is False
    assert trade.position_size == 0
    assert trade.stop_loss == 0
    assert trade.take_profit == 0
    assert trade.risk_reasons == BLOCKED.reasons


def test_blocked_trade_carries_reasons_for_display():
    trade = build_trade(VALID_ANALYSIS, BLOCKED, PROFILE)
    assert any("price" in reason.lower() for reason in trade.risk_reasons)


def test_unknown_price_approved_verdict_still_sizes_zero():
    analysis = {**VALID_ANALYSIS, "price": "Unknown"}
    # Risk manager would never approve this, but build_trade must be
    # defensive regardless: no numeric price means no sizing.
    trade = build_trade(analysis, APPROVED_BUY, PROFILE)
    assert trade.position_size == 0
    assert trade.stop_loss == 0
    assert trade.take_profit == 0


def test_profile_defaults_to_standard():
    trade = build_trade(VALID_ANALYSIS, APPROVED_BUY)
    assert trade.position_size > 0


def test_reasons_are_copied_not_shared():
    trade_a = build_trade(VALID_ANALYSIS, BLOCKED, PROFILE)
    trade_b = build_trade(VALID_ANALYSIS, BLOCKED, PROFILE)
    trade_a.risk_reasons.append("extra")
    assert len(trade_b.risk_reasons) == len(BLOCKED.reasons)
