"""Tests for trader.paper.rules validation."""

import pytest

from trader.paper.rules import PaperRuleResult, validate_paper_trade
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_trade(
    symbol: str = "BTC",
    side: str = "LONG",
    position_size: float = 10.0,
    price: float = 100.0,
) -> Trade:
    """Build a minimal valid Trade for validation tests."""
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=position_size,
        stop_loss=98.0,
        take_profit=104.0,
        side=side,
        approved=True,
    )


# ---------------------------------------------------------------------------
# validate_paper_trade — happy path
# ---------------------------------------------------------------------------

def test_valid_long_trade_passes() -> None:
    result = validate_paper_trade(_valid_trade())
    assert result.ok is True
    assert result.reason == ""


def test_valid_short_trade_passes() -> None:
    result = validate_paper_trade(_valid_trade(side="SHORT"))
    assert result.ok is True
    assert result.reason == ""


def test_valid_buy_alias_passes() -> None:
    result = validate_paper_trade(_valid_trade(side="BUY"))
    assert result.ok is True
    assert result.reason == ""


def test_valid_sell_alias_passes() -> None:
    result = validate_paper_trade(_valid_trade(side="SELL"))
    assert result.ok is True
    assert result.reason == ""


def test_valid_trade_with_integer_fields_passes() -> None:
    trade = _valid_trade(position_size=10, price=100)
    result = validate_paper_trade(trade)
    assert result.ok is True


# ---------------------------------------------------------------------------
# validate_paper_trade — None / missing
# ---------------------------------------------------------------------------

def test_none_trade_fails() -> None:
    result = validate_paper_trade(None)  # type: ignore[arg-type]
    assert result.ok is False
    assert result.reason == "trade must be provided"


def test_empty_symbol_fails() -> None:
    trade = _valid_trade(symbol="")
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "symbol" in result.reason


def test_whitespace_only_symbol_fails() -> None:
    trade = _valid_trade(symbol="   ")
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "symbol" in result.reason


def test_non_string_symbol_fails() -> None:
    trade = _valid_trade(symbol=123)  # type: ignore[arg-type]
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "symbol" in result.reason


# ---------------------------------------------------------------------------
# validate_paper_trade — side validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_side", ["", "L", "SH", "BU", "SEL", "HOLD", "WAIT", "NEUTRAL"])
def test_invalid_side_fails(bad_side: str) -> None:
    trade = _valid_trade(side=bad_side)
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "side" in result.reason


def test_non_string_side_fails() -> None:
    trade = _valid_trade(side=123)  # type: ignore[arg-type]
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "side" in result.reason


# ---------------------------------------------------------------------------
# validate_paper_trade — position_size validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_size", [0, -1, -0.01, 0.0])
def test_non_positive_position_size_fails(bad_size: float) -> None:
    trade = _valid_trade(position_size=bad_size)
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "position_size" in result.reason


def test_zero_position_size_fails() -> None:
    trade = _valid_trade(position_size=0)
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "position_size" in result.reason


def test_negative_position_size_fails() -> None:
    trade = _valid_trade(position_size=-5)
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "position_size" in result.reason


def test_non_numeric_position_size_fails() -> None:
    trade = _valid_trade(position_size="10")  # type: ignore[arg-type]
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "position_size" in result.reason


# ---------------------------------------------------------------------------
# validate_paper_trade — price validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_price", [0, -1, -0.01, 0.0])
def test_non_positive_price_fails(bad_price: float) -> None:
    trade = _valid_trade(price=bad_price)
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "price" in result.reason


def test_zero_price_fails() -> None:
    trade = _valid_trade(price=0)
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "price" in result.reason


def test_negative_price_fails() -> None:
    trade = _valid_trade(price=-50)
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "price" in result.reason


def test_non_numeric_price_fails() -> None:
    trade = _valid_trade(price="100")  # type: ignore[arg-type]
    result = validate_paper_trade(trade)
    assert result.ok is False
    assert "price" in result.reason


# ---------------------------------------------------------------------------
# PaperRuleResult
# ---------------------------------------------------------------------------

def test_paper_rule_result_ok_has_empty_reason() -> None:
    result = PaperRuleResult(ok=True, reason="")
    assert result.ok is True
    assert result.reason == ""


def test_paper_rule_result_not_ok_has_reason() -> None:
    result = PaperRuleResult(ok=False, reason="something wrong")
    assert result.ok is False
    assert result.reason == "something wrong"


def test_paper_rule_result_is_frozen() -> None:
    result = PaperRuleResult(ok=True, reason="")
    with pytest.raises(AttributeError):
        result.ok = False


def test_paper_rule_result_is_slots_based() -> None:
    result = PaperRuleResult(ok=True, reason="")
    # Slotted dataclasses don't have __dict__
    assert not hasattr(result, "__dict__")
