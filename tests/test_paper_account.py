"""Tests for trader.paper.account.PaperAccount."""

import pytest

from trader.paper.account import PaperAccount
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_order(
    order_id: str = "ORD-1",
    symbol: str = "BTC",
    side: PaperOrderSide = PaperOrderSide.LONG,
    quantity: float = 10.0,
    entry_price: float = 100.0,
    status: PaperOrderStatus = PaperOrderStatus.OPEN,
) -> PaperOrder:
    """Build a minimal open PaperOrder."""
    return PaperOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=98.0,
        take_profit=104.0,
        status=status,
    )


def _account(
    initial_balance: float = 1000.0,
    current_balance: float = 1000.0,
) -> PaperAccount:
    """Build a fresh PaperAccount."""
    return PaperAccount(
        initial_balance=initial_balance,
        current_balance=current_balance,
    )


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

def test_account_has_default_empty_lists() -> None:
    acc = _account()
    assert acc.open_positions == []
    assert acc.closed_positions == []


def test_account_initial_balance_is_set() -> None:
    acc = _account(initial_balance=500.0)
    assert acc.initial_balance == 500.0


def test_account_current_balance_is_independent_of_initial() -> None:
    acc = _account(initial_balance=1000.0, current_balance=950.0)
    assert acc.initial_balance == 1000.0
    assert acc.current_balance == 950.0


# ---------------------------------------------------------------------------
# Open position tracking
# ---------------------------------------------------------------------------

def test_add_open_position_increases_open_count() -> None:
    acc = _account()
    order = _open_order()
    acc.open_position(order)
    assert acc.open_count == 1


def test_add_multiple_open_positions() -> None:
    acc = _account()
    for i in range(3):
        acc.open_position(_open_order(order_id=f"ORD-{i}"))
    assert acc.open_count == 3


def test_duplicate_order_id_raises_value_error() -> None:
    acc = _account()
    order = _open_order(order_id="DUP")
    acc.open_position(order)
    with pytest.raises(ValueError, match="Open position already exists"):
        acc.open_position(_open_order(order_id="DUP"))


def test_add_position_with_same_symbol_different_id_allowed() -> None:
    acc = _account()
    acc.open_position(_open_order(order_id="A", symbol="BTC"))
    acc.open_position(_open_order(order_id="B", symbol="BTC"))
    assert acc.open_count == 2


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

def test_close_position_moves_to_closed() -> None:
    acc = _account()
    order = _open_order(order_id="CLOSE-ME")
    acc.open_position(order)
    closed = acc.close_position("CLOSE-ME")
    assert acc.open_count == 0
    assert acc.closed_count == 1
    assert closed is order
    assert order in acc.closed_positions


def test_close_position_returns_the_closed_order() -> None:
    acc = _account()
    order = _open_order(order_id="RET")
    acc.open_position(order)
    returned = acc.close_position("RET")
    assert returned.order_id == "RET"


def test_close_nonexistent_order_raises_key_error() -> None:
    acc = _account()
    with pytest.raises(KeyError, match="No open position found"):
        acc.close_position("NONEXISTENT")


def test_close_position_does_not_change_balance() -> None:
    acc = _account(current_balance=1000.0)
    order = _open_order(order_id="NO-PNL")
    acc.open_position(order)
    acc.close_position("NO-PNL")
    assert acc.current_balance == 1000.0


# ---------------------------------------------------------------------------
# get_open_position
# ---------------------------------------------------------------------------

def test_get_open_position_returns_order() -> None:
    acc = _account()
    order = _open_order(order_id="GET-ME")
    acc.open_position(order)
    result = acc.get_open_position("GET-ME")
    assert result is order


def test_get_open_position_returns_none_for_missing() -> None:
    acc = _account()
    assert acc.get_open_position("MISSING") is None


def test_get_open_position_returns_none_for_closed() -> None:
    acc = _account()
    order = _open_order(order_id="CLOSED")
    acc.open_position(order)
    acc.close_position("CLOSED")
    assert acc.get_open_position("CLOSED") is None


# ---------------------------------------------------------------------------
# has_open_position
# ---------------------------------------------------------------------------

def test_has_open_position_true_when_open_exists() -> None:
    acc = _account()
    acc.open_position(_open_order(symbol="ETH"))
    assert acc.has_open_position("ETH") is True


def test_has_open_position_false_when_none_open() -> None:
    acc = _account()
    assert acc.has_open_position("BTC") is False


def test_has_open_position_false_when_only_closed() -> None:
    acc = _account()
    order = _open_order(order_id="ONLY-CLOSED", symbol="BTC")
    acc.open_position(order)
    acc.close_position("ONLY-CLOSED")
    assert acc.has_open_position("BTC") is False


def test_has_open_position_case_sensitive_symbol() -> None:
    acc = _account()
    acc.open_position(_open_order(symbol="BTC"))
    assert acc.has_open_position("BTC") is True
    assert acc.has_open_position("btc") is False


# ---------------------------------------------------------------------------
# open_count / closed_count
# ---------------------------------------------------------------------------

def test_counts_start_at_zero() -> None:
    acc = _account()
    assert acc.open_count == 0
    assert acc.closed_count == 0


def test_counts_reflect_added_and_closed() -> None:
    acc = _account()
    acc.open_position(_open_order(order_id="1"))
    acc.open_position(_open_order(order_id="2"))
    acc.close_position("1")
    assert acc.open_count == 1
    assert acc.closed_count == 1


# ---------------------------------------------------------------------------
# equity
# ---------------------------------------------------------------------------

def test_equity_equals_current_balance_when_no_open_positions() -> None:
    acc = _account(current_balance=1000.0)
    assert acc.equity() == 1000.0


def test_equity_at_entry_price_is_balance() -> None:
    """At entry price, unrealized PnL is 0, so equity == balance."""
    acc = _account(current_balance=1000.0)
    order = _open_order(quantity=10.0, entry_price=100.0)
    acc.open_position(order)
    assert acc.equity() == pytest.approx(1000.0)


def test_equity_ignores_closed_positions() -> None:
    acc = _account(current_balance=1000.0)
    order = _open_order(order_id="X")
    acc.open_position(order)
    acc.close_position("X")
    assert acc.equity() == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# add_realized_pnl
# ---------------------------------------------------------------------------

def test_add_positive_realized_pnl_increases_balance() -> None:
    acc = _account(current_balance=1000.0)
    acc.add_realized_pnl(50.0)
    assert acc.current_balance == 1050.0


def test_add_negative_realized_pnl_decreases_balance() -> None:
    acc = _account(current_balance=1000.0)
    acc.add_realized_pnl(-30.0)
    assert acc.current_balance == 970.0


def test_add_realized_pnl_is_cumulative() -> None:
    acc = _account(current_balance=1000.0)
    acc.add_realized_pnl(20.0)
    acc.add_realized_pnl(30.0)
    acc.add_realized_pnl(-10.0)
    assert acc.current_balance == 1040.0


def test_add_zero_realized_pnl_leaves_balance_unchanged() -> None:
    acc = _account(current_balance=1000.0)
    acc.add_realized_pnl(0.0)
    assert acc.current_balance == 1000.0


# ---------------------------------------------------------------------------
# open_position_for_symbol
# ---------------------------------------------------------------------------

def test_open_position_for_symbol_returns_match() -> None:
    acc = _account()
    acc.open_position(_open_order(order_id="S1", symbol="BTC"))
    result = acc.open_position_for_symbol("BTC")
    assert result is not None
    assert result.order_id == "S1"


def test_open_position_for_symbol_returns_none_when_none_open() -> None:
    acc = _account()
    assert acc.open_position_for_symbol("BTC") is None


def test_open_position_for_symbol_returns_none_when_only_closed() -> None:
    acc = _account()
    order = _open_order(order_id="C", symbol="BTC")
    acc.open_position(order)
    acc.close_position("C")
    assert acc.open_position_for_symbol("BTC") is None


# ---------------------------------------------------------------------------
# closed_positions_for_symbol
# ---------------------------------------------------------------------------

def test_closed_positions_for_symbol_returns_only_that_symbol() -> None:
    acc = _account()
    acc.open_position(_open_order(order_id="A", symbol="BTC"))
    acc.open_position(_open_order(order_id="B", symbol="ETH"))
    acc.close_position("A")
    result = acc.closed_positions_for_symbol("BTC")
    assert len(result) == 1
    assert result[0].order_id == "A"


def test_closed_positions_for_symbol_returns_empty_when_none() -> None:
    acc = _account()
    assert acc.closed_positions_for_symbol("BTC") == []
