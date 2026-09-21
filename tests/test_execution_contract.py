"""Tests for the backend-agnostic execution contract (Sprint 10.1)."""

import pytest

from trader.execution.contract import ExecutionBackend
from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.paper.service import PaperTradingService
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_trade(
    symbol: str = "BTC",
    side: str = "LONG",
    position_size: float = 10.0,
    price: float = 100.0,
    stop_loss: float = 98.0,
    take_profit: float = 104.0,
) -> Trade:
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        side=side,
        approved=True,
    )


def _open_order(
    order_id: str = "ORD-1",
    symbol: str = "BTC",
    side: PaperOrderSide = PaperOrderSide.LONG,
    quantity: float = 10.0,
    entry_price: float = 100.0,
) -> PaperOrder:
    return PaperOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=98.0,
        take_profit=104.0,
        status=PaperOrderStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# Contract compliance
# ---------------------------------------------------------------------------

def test_paper_broker_satisfies_execution_contract() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    assert isinstance(broker, ExecutionBackend)


def test_paper_service_via_broker_satisfies_execution_contract() -> None:
    service = PaperTradingService.create_new_account()
    # PaperTradingService itself is a higher-level service, but the
    # underlying broker satisfies the contract.
    assert isinstance(service._broker, ExecutionBackend)


def test_execution_backend_is_runtime_checkable() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    # is_instance check for runtime_checkable protocols, not direct
    # instantiation.
    assert isinstance(broker, ExecutionBackend)


# ---------------------------------------------------------------------------
# Contract shape sanity checks
# ---------------------------------------------------------------------------

def test_contract_requires_place_order() -> None:
    # The protocol is defined by place_order, close_order, get_open_positions,
    # get_total_realized_pnl, and get_open_pnl. A valid paper broker
    # provides all of them.
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    trade = _valid_trade()
    order = broker.place_order(trade)
    assert order is not None
    assert order.status == PaperOrderStatus.OPEN


def test_contract_requires_close_order() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    trade = _valid_trade(price=100.0)
    order = broker.place_order(trade)
    assert order is not None
    closed = broker.close_order(order, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    assert closed.status == PaperOrderStatus.TAKEN_PROFIT
    assert closed.realized_pnl == pytest.approx(100.0)


def test_contract_get_open_positions_returns_snapshot() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    broker.place_order(_valid_trade(symbol="BTC"))
    broker.place_order(_valid_trade(symbol="ETH", price=2000.0))
    positions = broker.get_open_positions()
    assert len(list(positions)) == 2


def test_contract_get_total_realized_pnl_reflects_closed_trades() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    order = broker.place_order(_valid_trade(price=100.0, position_size=10.0))
    assert order is not None
    broker.close_order(order, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    assert broker.get_total_realized_pnl() == pytest.approx(100.0)


def test_contract_get_open_pnl_reflects_open_trades() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    order = broker.place_order(_valid_trade(price=100.0, position_size=10.0))
    assert order is not None
    # At entry price, unrealized PnL is 0
    assert broker.get_open_pnl() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Symbol-scoped contract compliance

def test_contract_get_open_positions_by_symbol_returns_snapshot() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    broker.place_order(_valid_trade(symbol="BTC"))
    broker.place_order(_valid_trade(symbol="ETH", price=2000.0))
    positions = broker.get_open_positions_by_symbol("BTC")
    assert len(list(positions)) == 1
    assert all(p.symbol == "BTC" for p in positions)


def test_contract_get_closed_positions_by_symbol_returns_snapshot() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    order = broker.place_order(_valid_trade(symbol="BTC", price=100.0, position_size=10.0))
    assert order is not None
    broker.close_order(order, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    positions = broker.get_closed_positions_by_symbol("BTC")
    assert len(list(positions)) == 1
    assert positions[0].symbol == "BTC"


def test_contract_get_total_realized_pnl_by_symbol_reflects_closed_trades() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    btc = broker.place_order(_valid_trade(symbol="BTC", price=100.0, position_size=10.0))
    eth = broker.place_order(_valid_trade(symbol="ETH", price=2000.0, position_size=1.0))
    assert btc is not None
    assert eth is not None
    broker.close_order(btc, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    broker.close_order(eth, exit_price=1900.0, status=PaperOrderStatus.STOPPED)
    assert broker.get_total_realized_pnl_by_symbol("BTC") == pytest.approx(100.0)
    assert broker.get_total_realized_pnl_by_symbol("ETH") == pytest.approx(-100.0)
    assert broker.get_total_realized_pnl_by_symbol("SOL") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Contract behavior: placement refusal returns None, not raise
# ---------------------------------------------------------------------------

def test_contract_returns_none_for_none_trade() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    result = broker.place_order(None)  # type: ignore[arg-type]
    assert result is None


def test_contract_returns_none_for_invalid_trade() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    trade = _valid_trade(symbol="")
    result = broker.place_order(trade)
    assert result is None


def test_contract_returns_none_for_duplicate_order() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    trade = _valid_trade()
    first = broker.place_order(trade)
    assert first is not None
    second = broker.place_order(trade)
    assert second is None


# ---------------------------------------------------------------------------
# Contract behavior: close raises only for non-open orders
# ---------------------------------------------------------------------------

def test_contract_close_open_order_succeeds() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    order = broker.place_order(_valid_trade(price=100.0, position_size=10.0))
    assert order is not None
    closed = broker.close_order(order, exit_price=90.0, status=PaperOrderStatus.STOPPED)
    assert closed.status == PaperOrderStatus.STOPPED


def test_contract_close_non_open_order_raises() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    trade = _valid_trade(price=100.0)
    order = broker.place_order(trade)
    assert order is not None
    broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)
    with pytest.raises(ValueError, match="is not open"):
        broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.TAKEN_PROFIT)


# ---------------------------------------------------------------------------
# Contract backward compatibility: ordinary paper behavior is unchanged
# ---------------------------------------------------------------------------

def test_contract_does_not_change_paper_order_id_determinism() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker_a = PaperBroker(account)
    broker_b = PaperBroker(PaperAccount(initial_balance=1000.0, current_balance=1000.0))
    trade = _valid_trade()
    assert broker_a.place_order(trade).order_id == broker_b.place_order(trade).order_id


def test_contract_does_not_change_paper_side_mapping() -> None:
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    broker = PaperBroker(account)
    long_order = broker.place_order(_valid_trade(side="BUY"))
    assert long_order is not None
    assert long_order.side == PaperOrderSide.LONG

    short_order = broker.place_order(_valid_trade(side="SELL", price=50.0, stop_loss=51.0, take_profit=49.0))
    assert short_order is not None
    assert short_order.side == PaperOrderSide.SHORT
