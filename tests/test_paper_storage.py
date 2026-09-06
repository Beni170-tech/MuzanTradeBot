"""Tests for trader.paper.storage persistence layer (Sprint 5F)."""

import json
import tempfile
from pathlib import Path

import pytest

from trader.paper.account import PaperAccount
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.paper.storage import load_account, load_orders, save_account, save_orders
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(
    order_id: str = "ORD-1",
    symbol: str = "BTC",
    side: PaperOrderSide = PaperOrderSide.LONG,
    quantity: float = 10.0,
    entry_price: float = 100.0,
    status: PaperOrderStatus = PaperOrderStatus.OPEN,
    realized_pnl: float = 0.0,
    exit_price: float | None = None,
    exit_at: str | None = None,
    trade: Trade | None = None,
) -> PaperOrder:
    return PaperOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=98.0,
        take_profit=104.0,
        status=status,
        trade=trade,
        created_at="2025-01-01 12:00:00",
        filled_at="2025-01-01 12:05:00",
        exit_at=exit_at,
        exit_price=exit_price,
        realized_pnl=realized_pnl,
        unrealized_pnl=50.0,
        paper=True,
        notes=["test note"],
    )


def _make_trade() -> Trade:
    return Trade(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=10.0,
        stop_loss=98.0,
        take_profit=104.0,
        side="LONG",
        approved=True,
        risk_reasons=["test reason"],
        timestamp="2025-01-01 12:00:00",
    )


def _make_account(
    initial_balance: float = 1000.0,
    current_balance: float = 1000.0,
    open_orders: list[PaperOrder] | None = None,
    closed_orders: list[PaperOrder] | None = None,
) -> PaperAccount:
    return PaperAccount(
        initial_balance=initial_balance,
        current_balance=current_balance,
        open_positions=open_orders or [],
        closed_positions=closed_orders or [],
    )


# ---------------------------------------------------------------------------
# save_account / load_account � empty account
# ---------------------------------------------------------------------------

def test_save_and_load_empty_account(tmp_path: Path) -> None:
    acc = _make_account()
    path = tmp_path / "empty_account.json"
    save_account(acc, path)
    loaded = load_account(path)
    assert loaded.initial_balance == 1000.0
    assert loaded.current_balance == 1000.0
    assert loaded.open_count == 0
    assert loaded.closed_count == 0


def test_save_and_load_account_with_different_balances(tmp_path: Path) -> None:
    acc = _make_account(initial_balance=5000.0, current_balance=4500.0)
    path = tmp_path / "balances.json"
    save_account(acc, path)
    loaded = load_account(path)
    assert loaded.initial_balance == 5000.0
    assert loaded.current_balance == 4500.0


# ---------------------------------------------------------------------------
# save_account / load_account � with open positions
# ---------------------------------------------------------------------------

def test_save_and_load_account_with_open_position(tmp_path: Path) -> None:
    order = _make_order(order_id="OPEN-1", status=PaperOrderStatus.OPEN)
    acc = _make_account(open_orders=[order])
    path = tmp_path / "open_pos.json"
    save_account(acc, path)
    loaded = load_account(path)
    assert loaded.open_count == 1
    assert loaded.closed_count == 0
    loaded_order = loaded.open_positions[0]
    assert loaded_order.order_id == "OPEN-1"
    assert loaded_order.symbol == "BTC"
    assert loaded_order.side == PaperOrderSide.LONG
    assert loaded_order.quantity == 10.0
    assert loaded_order.entry_price == 100.0
    assert loaded_order.stop_loss == 98.0
    assert loaded_order.take_profit == 104.0
    assert loaded_order.status == PaperOrderStatus.OPEN
    assert loaded_order.realized_pnl == 0.0
    assert loaded_order.unrealized_pnl == 50.0
    assert loaded_order.created_at == "2025-01-01 12:00:00"
    assert loaded_order.filled_at == "2025-01-01 12:05:00"
    assert loaded_order.paper is True
    assert loaded_order.notes == ["test note"]


def test_save_and_load_account_with_multiple_open_positions(tmp_path: Path) -> None:
    o1 = _make_order(order_id="OPEN-1", symbol="BTC", quantity=10.0, entry_price=100.0)
    o2 = _make_order(order_id="OPEN-2", symbol="ETH", quantity=5.0, entry_price=2000.0,
                      side=PaperOrderSide.SHORT, status=PaperOrderStatus.OPEN)
    acc = _make_account(open_orders=[o1, o2])
    path = tmp_path / "multi_open.json"
    save_account(acc, path)
    loaded = load_account(path)
    assert loaded.open_count == 2
    assert {o.symbol for o in loaded.open_positions} == {"BTC", "ETH"}


# ---------------------------------------------------------------------------
# save_account / load_account � with closed positions
# ---------------------------------------------------------------------------

def test_save_and_load_account_with_closed_position(tmp_path: Path) -> None:
    closed_order = _make_order(
        order_id="CLOSED-1",
        status=PaperOrderStatus.TAKEN_PROFIT,
        realized_pnl=100.0,
        exit_price=110.0,
        exit_at="2025-01-02 15:00:00",
    )
    acc = _make_account(closed_orders=[closed_order])
    path = tmp_path / "closed_pos.json"
    save_account(acc, path)
    loaded = load_account(path)
    assert loaded.open_count == 0
    assert loaded.closed_count == 1
    loaded_order = loaded.closed_positions[0]
    assert loaded_order.order_id == "CLOSED-1"
    assert loaded_order.status == PaperOrderStatus.TAKEN_PROFIT
    assert loaded_order.realized_pnl == 100.0
    assert loaded_order.exit_price == 110.0
    assert loaded_order.exit_at == "2025-01-02 15:00:00"


def test_save_and_load_account_with_mixed_positions(tmp_path: Path) -> None:
    open_order = _make_order(order_id="OPEN-1", status=PaperOrderStatus.OPEN)
    closed_order = _make_order(
        order_id="CLOSED-1",
        status=PaperOrderStatus.STOPPED,
        realized_pnl=-50.0,
        exit_price=95.0,
        exit_at="2025-01-02 10:00:00",
    )
    acc = _make_account(open_orders=[open_order], closed_orders=[closed_order])
    path = tmp_path / "mixed.json"
    save_account(acc, path)
    loaded = load_account(path)
    assert loaded.open_count == 1
    assert loaded.closed_count == 1
    assert loaded.open_positions[0].order_id == "OPEN-1"
    assert loaded.closed_positions[0].order_id == "CLOSED-1"
    assert loaded.closed_positions[0].realized_pnl == -50.0


# ---------------------------------------------------------------------------
# save_account / load_account � with trade reference
# ---------------------------------------------------------------------------

def test_save_and_load_order_with_trade_reference(tmp_path: Path) -> None:
    trade = _make_trade()
    order = _make_order(order_id="TRADE-1", trade=trade)
    acc = _make_account(open_orders=[order])
    path = tmp_path / "with_trade.json"
    save_account(acc, path)
    loaded = load_account(path)
    loaded_order = loaded.open_positions[0]
    assert loaded_order.trade is not None
    assert loaded_order.trade.symbol == "BTC"
    assert loaded_order.trade.price == 100.0
    assert loaded_order.trade.confidence == 80
    assert loaded_order.trade.approved is True
    assert loaded_order.trade.risk_reasons == ["test reason"]


def test_save_and_load_order_without_trade_reference(tmp_path: Path) -> None:
    order = _make_order(order_id="NO-TRADE", trade=None)
    acc = _make_account(open_orders=[order])
    path = tmp_path / "no_trade.json"
    save_account(acc, path)
    loaded = load_account(path)
    assert loaded.open_positions[0].trade is None


# ---------------------------------------------------------------------------
# save_orders / load_orders
# ---------------------------------------------------------------------------

def test_save_and_load_empty_orders(tmp_path: Path) -> None:
    path = tmp_path / "empty_orders.json"
    save_orders([], path)
    loaded = load_orders(path)
    assert loaded == []


def test_save_and_load_orders(tmp_path: Path) -> None:
    o1 = _make_order(order_id="ORD-1", symbol="BTC", quantity=10.0, entry_price=100.0)
    o2 = _make_order(order_id="ORD-2", symbol="ETH", quantity=5.0, entry_price=2000.0,
                      side=PaperOrderSide.SHORT)
    orders = [o1, o2]
    path = tmp_path / "orders.json"
    save_orders(orders, path)
    loaded = load_orders(path)
    assert len(loaded) == 2
    assert loaded[0].order_id == "ORD-1"
    assert loaded[1].order_id == "ORD-2"
    assert loaded[0].symbol == "BTC"
    assert loaded[1].symbol == "ETH"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_load_nonexistent_account_raises_file_not_found(tmp_path: Path) -> None:
    path = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError, match="Account file not found"):
        load_account(path)


def test_load_nonexistent_orders_raises_file_not_found(tmp_path: Path) -> None:
    path = tmp_path / "nonexistent_orders.json"
    with pytest.raises(FileNotFoundError, match="Orders file not found"):
        load_orders(path)


def test_load_account_with_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("not valid json {")
    with pytest.raises(json.JSONDecodeError):
        load_account(path)


def test_load_orders_with_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "invalid_orders.json"
    path.write_text("[not valid")
    with pytest.raises(json.JSONDecodeError):
        load_orders(path)


def test_load_account_with_wrong_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong_type.json"
    path.write_text('"not an object"')
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_account(path)


def test_load_orders_with_wrong_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong_type_orders.json"
    path.write_text('"not an array"')
    with pytest.raises(ValueError, match="expected a JSON array"):
        load_orders(path)


def test_load_account_creates_parent_directories(tmp_path: Path) -> None:
    subdir = tmp_path / "subdir" / "nested"
    path = subdir / "account.json"
    acc = _make_account()
    save_account(acc, path)
    assert path.exists()
    loaded = load_account(path)
    assert loaded.initial_balance == 1000.0


def test_save_orders_creates_parent_directories(tmp_path: Path) -> None:
    subdir = tmp_path / "subdir" / "nested"
    path = subdir / "orders.json"
    orders = [_make_order(order_id="ORD-1")]
    save_orders(orders, path)
    assert path.exists()
    loaded = load_orders(path)
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Round-trip fidelity
# ---------------------------------------------------------------------------

def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    trade = _make_trade()
    order = _make_order(
        order_id="ROUND-1",
        symbol="BTC",
        side=PaperOrderSide.SHORT,
        quantity=5.5,
        entry_price=150.75,
        status=PaperOrderStatus.TAKEN_PROFIT,
        realized_pnl=25.50,
        exit_price=155.25,
        exit_at="2025-06-15 09:30:00",
        trade=trade,
    )
    order.notes.append("another note")
    acc = _make_account(
        initial_balance=2500.0,
        current_balance=2525.50,
        open_orders=[],
        closed_orders=[order],
    )
    path = tmp_path / "roundtrip.json"
    save_account(acc, path)

    # Read raw JSON to verify structure
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    assert raw["initial_balance"] == 2500.0
    assert raw["current_balance"] == 2525.50
    assert len(raw["closed_positions"]) == 1
    closed_data = raw["closed_positions"][0]
    assert closed_data["side"] == "SHORT"
    assert closed_data["quantity"] == 5.5
    assert closed_data["entry_price"] == 150.75
    assert closed_data["status"] == "TAKEN_PROFIT"
    assert closed_data["realized_pnl"] == 25.50
    assert closed_data["exit_price"] == 155.25
    assert closed_data["exit_at"] == "2025-06-15 09:30:00"
    assert closed_data["trade"]["symbol"] == "BTC"
    assert closed_data["trade"]["approved"] is True
    assert closed_data["trade"]["risk_reasons"] == ["test reason"]

    # Load back and verify object equivalence
    loaded = load_account(path)
    loaded_order = loaded.closed_positions[0]
    assert loaded_order.order_id == order.order_id
    assert loaded_order.side == order.side
    assert loaded_order.quantity == order.quantity
    assert loaded_order.entry_price == order.entry_price
    assert loaded_order.status == order.status
    assert loaded_order.realized_pnl == order.realized_pnl
    assert loaded_order.exit_price == order.exit_price
    assert loaded_order.exit_at == order.exit_at
    assert loaded_order.trade is not None
    assert loaded_order.trade.symbol == order.trade.symbol
    assert loaded_order.trade.confidence == order.trade.confidence
    assert loaded_order.notes == order.notes


def test_account_behavior_after_load(tmp_path: Path) -> None:
    """Verify loaded account behaves like a fresh account."""
    order = _make_order(order_id="BEHAVIOR-1", status=PaperOrderStatus.OPEN)
    acc = _make_account(open_orders=[order])
    path = tmp_path / "behavior.json"
    save_account(acc, path)
    loaded = load_account(path)

    # Verify methods work
    assert loaded.open_count == 1
    assert loaded.has_open_position("BTC") is True
    assert loaded.get_open_position("BEHAVIOR-1") is not None
    equity = loaded.equity()
    assert isinstance(equity, float)

    # Can add new position
    new_order = _make_order(order_id="NEW-1", symbol="ETH", status=PaperOrderStatus.OPEN)
    loaded.open_position(new_order)
    assert loaded.open_count == 2

    # Can close position
    closed = loaded.close_position("NEW-1")
    assert closed.order_id == "NEW-1"
    assert loaded.open_count == 1
    assert loaded.closed_count == 1


def test_orders_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    o1 = _make_order(order_id="ORD-1", symbol="BTC", quantity=10.0, entry_price=100.0,
                     status=PaperOrderStatus.OPEN)
    o2 = _make_order(
        order_id="ORD-2",
        symbol="ETH",
        side=PaperOrderSide.SHORT,
        quantity=2.5,
        entry_price=3000.0,
        status=PaperOrderStatus.STOPPED,
        realized_pnl=-100.0,
        exit_price=3100.0,
        exit_at="2025-03-15 14:00:00",
    )
    orders = [o1, o2]
    path = tmp_path / "orders_roundtrip.json"
    save_orders(orders, path)
    loaded = load_orders(path)

    assert len(loaded) == 2
    assert loaded[0].order_id == o1.order_id
    assert loaded[0].symbol == o1.symbol
    assert loaded[0].side == o1.side
    assert loaded[0].quantity == o1.quantity
    assert loaded[0].entry_price == o1.entry_price
    assert loaded[0].status == o1.status
    assert loaded[0].realized_pnl == o1.realized_pnl
    assert loaded[0].exit_price == o1.exit_price
    assert loaded[0].notes == o1.notes

    assert loaded[1].order_id == o2.order_id
    assert loaded[1].side == o2.side
    assert loaded[1].quantity == o2.quantity
    assert loaded[1].entry_price == o2.entry_price
    assert loaded[1].status == o2.status
    assert loaded[1].realized_pnl == o2.realized_pnl
    assert loaded[1].exit_price == o2.exit_price
    assert loaded[1].exit_at == o2.exit_at
