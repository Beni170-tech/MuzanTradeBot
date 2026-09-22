"""Tests for trader.paper.service.PaperTradingService (Sprint 5G)."""

import pytest

from trader.execution.contract import ExecutionBackend
from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.order import PaperOrderStatus
from trader.paper.service import PaperTradingService
from trader.paper.summary import PaperAccountSummary
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(initial_balance: float = 1000.0) -> PaperTradingService:
    """Create a fresh PaperTradingService for testing."""
    return PaperTradingService.create_new_account(initial_balance)


def _make_approved_trade(
    symbol: str = "BTC",
    side: str = "LONG",
    position_size: float = 10.0,
    price: float = 100.0,
    stop_loss: float = 98.0,
    take_profit: float = 104.0,
) -> Trade:
    """Build an approved Trade for placement tests."""
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


def _make_unapproved_trade(
    symbol: str = "BTC",
    side: str = "LONG",
    position_size: float = 10.0,
    price: float = 100.0,
) -> Trade:
    """Build an unapproved Trade for rejection tests."""
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="WAIT",
        risk_verdict="BLOCKED",
        position_size=position_size,
        stop_loss=0.0,
        take_profit=0.0,
        side=side,
        approved=False,
    )


# ---------------------------------------------------------------------------
# Factory methods
# ---------------------------------------------------------------------------

def test_create_new_account_default_balance() -> None:
    service = PaperTradingService.create_new_account()
    summary = service.summary()
    assert summary.equity == 1000.0
    assert summary.closed_count == 0
    assert summary.open_positions == []


def test_create_new_account_custom_balance() -> None:
    service = PaperTradingService.create_new_account(initial_balance=5000.0)
    summary = service.summary()
    assert summary.equity == 5000.0


def test_create_new_account_returns_service_with_broker() -> None:
    service = PaperTradingService.create_new_account()
    # Internal state should be wired correctly
    assert service.summary().equity == 1000.0


def test_load_account_from_file(tmp_path) -> None:
    # Create, place a trade, save, then load
    service1 = _make_service(2000.0)
    trade = _make_approved_trade(symbol="ETH", price=2000.0, position_size=1.0)
    service1.place_trade(trade)
    path = tmp_path / "test_account.json"
    service1.save_account(path)

    service2 = PaperTradingService.load_account(path)
    summary = service2.summary()
    assert summary.equity == 2000.0
    assert summary.open_positions[0].symbol == "ETH"


def test_load_account_nonexistent_raises(tmp_path) -> None:
    path = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        PaperTradingService.load_account(path)


# ---------------------------------------------------------------------------
# Backend construction seam (Sprint 10.3)
# ---------------------------------------------------------------------------

def _record_backend_construction(monkeypatch) -> list:
    """Spy on the seam and return the accounts it was called with."""
    accounts: list = []
    original = PaperTradingService._create_backend

    def spy(cls, account):
        accounts.append(account)
        return original(account)

    monkeypatch.setattr(PaperTradingService, "_create_backend", classmethod(spy))
    return accounts


def test_create_new_account_uses_backend_seam(monkeypatch) -> None:
    accounts = _record_backend_construction(monkeypatch)

    service = PaperTradingService.create_new_account(1500.0)

    assert len(accounts) == 1
    assert isinstance(service._broker, PaperBroker)
    assert service._broker.account is accounts[0]


def test_load_account_uses_backend_seam(tmp_path, monkeypatch) -> None:
    path = tmp_path / "seam_load.json"
    _make_service(2000.0).save_account(path)

    accounts = _record_backend_construction(monkeypatch)

    service = PaperTradingService.load_account(path)

    assert len(accounts) == 1
    assert isinstance(service._broker, PaperBroker)
    assert service._broker.account is accounts[0]


# ---------------------------------------------------------------------------
# place_trade
# ---------------------------------------------------------------------------

def test_place_approved_trade_returns_order() -> None:
    service = _make_service()
    trade = _make_approved_trade()
    order = service.place_trade(trade)
    assert order is not None
    assert order.symbol == "BTC"
    assert order.side.name == "LONG"


def test_place_approved_trade_stores_in_open_positions() -> None:
    service = _make_service()
    trade = _make_approved_trade(symbol="ETH", price=2000.0, position_size=2.0)
    service.place_trade(trade)
    assert len(service.open_positions()) == 1
    assert service.open_positions()[0].symbol == "ETH"


def test_place_unapproved_trade_returns_none() -> None:
    service = _make_service()
    trade = _make_unapproved_trade()
    result = service.place_trade(trade)
    assert result is None
    assert service.open_positions() == []


def test_place_none_trade_returns_none() -> None:
    service = _make_service()
    result = service.place_trade(None)  # type: ignore[arg-type]
    assert result is None


def test_place_invalid_trade_returns_none() -> None:
    service = _make_service()
    trade = _make_approved_trade(symbol="")  # invalid: empty symbol
    result = service.place_trade(trade)
    assert result is None


def test_place_duplicate_trade_returns_none() -> None:
    service = _make_service()
    trade = _make_approved_trade()
    o1 = service.place_trade(trade)
    assert o1 is not None
    o2 = service.place_trade(trade)
    assert o2 is None  # duplicate order id


def test_place_multiple_trades() -> None:
    service = _make_service()
    btc_trade = _make_approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    eth_trade = _make_approved_trade(symbol="ETH", price=2000.0, position_size=1.0,
                                     side="SHORT")
    o1 = service.place_trade(btc_trade)
    o2 = service.place_trade(eth_trade)
    assert o1 is not None
    assert o2 is not None
    assert len(service.open_positions()) == 2


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

def test_close_existing_position() -> None:
    service = _make_service()
    trade = _make_approved_trade(price=100.0, position_size=10.0)
    service.place_trade(trade)
    closed = service.close_position(
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert closed is not None
    assert closed.status == PaperOrderStatus.TAKEN_PROFIT
    assert closed.realized_pnl == pytest.approx(100.0)
    assert service.open_positions() == []
    assert len(service.closed_positions()) == 1


def test_close_position_updates_balance_via_summary() -> None:
    service = _make_service()
    trade = _make_approved_trade(price=100.0, position_size=10.0)
    service.place_trade(trade)
    service.close_position(
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    summary = service.summary()
    assert summary.total_realized_pnl == pytest.approx(100.0)
    assert summary.equity == pytest.approx(1100.0)


def test_close_nonexistent_position_returns_none() -> None:
    service = _make_service()
    result = service.close_position(
        symbol="NONEXISTENT",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    assert result is None


def test_close_position_case_insensitive_symbol() -> None:
    service = _make_service()
    trade = _make_approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    service.place_trade(trade)
    closed = service.close_position(
        symbol="btc",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    assert closed is not None
    assert closed.symbol == "BTC"


def test_close_position_short_profit() -> None:
    service = _make_service()
    trade = _make_approved_trade(
        symbol="BTC", side="SHORT", price=100.0, position_size=10.0,
        stop_loss=102.0, take_profit=96.0,
    )
    service.place_trade(trade)
    closed = service.close_position(
        symbol="BTC",
        exit_price=90.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert closed is not None
    assert closed.realized_pnl == pytest.approx(100.0)


def test_close_position_requires_explicit_status() -> None:
    service = _make_service()
    trade = _make_approved_trade(price=100.0, position_size=10.0)
    service.place_trade(trade)
    closed = service.close_position(
        symbol="BTC",
        exit_price=100.0,
        status=PaperOrderStatus.STOPPED,
    )
    assert closed is not None
    assert closed.status == PaperOrderStatus.STOPPED


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Execution backend access


def test_execution_backend_returns_underlying_broker() -> None:
    service = _make_service()
    backend = service.execution_backend()
    assert backend is service._broker


def test_execution_backend_satisfies_execution_backend_contract() -> None:
    service = _make_service()
    backend = service.execution_backend()
    assert isinstance(backend, ExecutionBackend)


def test_execution_backend_is_same_instance_after_placement() -> None:
    service = _make_service()
    backend_a = service.execution_backend()
    trade = _make_approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    service.place_trade(trade)
    backend_b = service.execution_backend()
    assert backend_a is backend_b
    assert isinstance(backend_b, ExecutionBackend)


# ---------------------------------------------------------------------------
# summary


def test_summary_empty_account() -> None:
    service = _make_service()
    summary = service.summary()
    assert isinstance(summary, PaperAccountSummary)
    assert summary.open_positions == []
    assert summary.closed_count == 0
    assert summary.total_realized_pnl == 0.0
    assert summary.open_pnl == 0.0
    assert summary.equity == 1000.0


def test_summary_after_opening_position() -> None:
    service = _make_service()
    trade = _make_approved_trade(price=100.0, position_size=10.0)
    service.place_trade(trade)
    summary = service.summary()
    assert len(summary.open_positions) == 1
    assert summary.open_positions[0].symbol == "BTC"
    assert summary.closed_count == 0
    assert summary.open_pnl == 0.0  # at entry price
    assert summary.equity == 1000.0


def test_summary_after_closing_position() -> None:
    service = _make_service()
    trade = _make_approved_trade(price=100.0, position_size=10.0)
    service.place_trade(trade)
    service.close_position(
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    summary = service.summary()
    assert summary.open_positions == []
    assert summary.closed_count == 1
    assert summary.total_realized_pnl == pytest.approx(100.0)
    assert summary.equity == pytest.approx(1100.0)


def test_summary_is_frozen() -> None:
    service = _make_service()
    summary = service.summary()
    with pytest.raises(AttributeError):
        summary.equity = 9999.0


def test_summary_multiple_positions() -> None:
    service = _make_service()
    btc_trade = _make_approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    eth_trade = _make_approved_trade(symbol="ETH", price=2000.0, position_size=1.0)
    service.place_trade(btc_trade)
    service.place_trade(eth_trade)

    summary = service.summary()
    assert len(summary.open_positions) == 2
    assert summary.closed_count == 0

    # Close one
    service.close_position(
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    summary = service.summary()
    assert summary.closed_count == 1
    assert summary.total_realized_pnl == pytest.approx(100.0)
    assert len(summary.open_positions) == 1
    assert summary.open_positions[0].symbol == "ETH"


# ---------------------------------------------------------------------------
# open_positions / closed_positions
# ---------------------------------------------------------------------------

def test_open_positions_returns_copy() -> None:
    service = _make_service()
    trade = _make_approved_trade(symbol="BTC")
    service.place_trade(trade)
    positions = service.open_positions()
    assert len(positions) == 1
    # Mutating the returned list should not affect the service
    positions.clear()
    assert len(service.open_positions()) == 1


def test_closed_positions_returns_copy() -> None:
    service = _make_service()
    trade = _make_approved_trade(price=100.0, position_size=10.0)
    service.place_trade(trade)
    service.close_position(
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    closed = service.closed_positions()
    assert len(closed) == 1
    closed.clear()
    assert len(service.closed_positions()) == 1


def test_open_positions_empty_when_none() -> None:
    service = _make_service()
    assert service.open_positions() == []


def test_closed_positions_empty_when_none() -> None:
    service = _make_service()
    assert service.closed_positions() == []


# ---------------------------------------------------------------------------
# get_position
# ---------------------------------------------------------------------------

def test_get_position_found() -> None:
    service = _make_service()
    trade = _make_approved_trade(symbol="BTC")
    service.place_trade(trade)
    position = service.get_position("BTC")
    assert position is not None
    assert position.symbol == "BTC"


def test_get_position_not_found() -> None:
    service = _make_service()
    result = service.get_position("NONEXISTENT")
    assert result is None


def test_get_position_case_insensitive() -> None:
    service = _make_service()
    trade = _make_approved_trade(symbol="BTC")
    service.place_trade(trade)
    position = service.get_position("btc")
    assert position is not None


def test_get_position_returns_none_after_close() -> None:
    service = _make_service()
    trade = _make_approved_trade(symbol="BTC")
    service.place_trade(trade)
    service.close_position(
        symbol="BTC",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    result = service.get_position("BTC")
    assert result is None


# ---------------------------------------------------------------------------
# Persistence: save and reload preserves full state
# ---------------------------------------------------------------------------

def test_save_and_reload_preserves_full_state(tmp_path) -> None:
    service = _make_service(initial_balance=3000.0)

    # Place multiple trades
    btc_trade = _make_approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    eth_trade = _make_approved_trade(symbol="ETH", price=2000.0, position_size=1.0,
                                     side="SHORT")
    service.place_trade(btc_trade)
    service.place_trade(eth_trade)

    # Close BTC
    service.close_position(
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )

    # Save
    path = tmp_path / "full_state.json"
    service.save_account(path)

    # Reload
    loaded_service = PaperTradingService.load_account(path)
    summary = loaded_service.summary()

    assert summary.equity == pytest.approx(3100.0)
    assert summary.closed_count == 1
    assert summary.total_realized_pnl == pytest.approx(100.0)
    assert len(summary.open_positions) == 1
    assert summary.open_positions[0].symbol == "ETH"

    # Verify closed positions
    closed = loaded_service.closed_positions()
    assert len(closed) == 1
    assert closed[0].symbol == "BTC"
    assert closed[0].realized_pnl == pytest.approx(100.0)


def test_persistence_with_loss(tmp_path) -> None:
    service = _make_service(initial_balance=1000.0)
    trade = _make_approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    service.place_trade(trade)
    service.close_position(
        symbol="BTC",
        exit_price=90.0,
        status=PaperOrderStatus.STOPPED,
    )
    path = tmp_path / "loss.json"
    service.save_account(path)

    loaded = PaperTradingService.load_account(path)
    summary = loaded.summary()
    assert summary.equity == pytest.approx(900.0)
    assert summary.total_realized_pnl == pytest.approx(-100.0)


def test_persistence_empty_account(tmp_path) -> None:
    service = _make_service(initial_balance=500.0)
    path = tmp_path / "empty.json"
    service.save_account(path)

    loaded = PaperTradingService.load_account(path)
    summary = loaded.summary()
    assert summary.equity == 500.0
    assert summary.closed_count == 0
    assert summary.open_positions == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_place_trade_does_not_affect_unapproved_trade_balance() -> None:
    service = _make_service()
    trade = _make_unapproved_trade()
    service.place_trade(trade)
    summary = service.summary()
    assert summary.equity == 1000.0  # unchanged


def test_multiple_services_are_independent() -> None:
    service1 = _make_service(initial_balance=1000.0)
    service2 = _make_service(initial_balance=2000.0)

    trade = _make_approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    service1.place_trade(trade)

    assert len(service1.open_positions()) == 1
    assert service2.open_positions() == []  # independent

    service1.close_position(
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert service1.summary().equity == pytest.approx(1100.0)
    assert service2.summary().equity == 2000.0  # unchanged


def test_service_preserves_trades_order_in_positions() -> None:
    service = _make_service()
    for i, sym in enumerate(["BTC", "ETH", "SOL"]):
        trade = _make_approved_trade(
            symbol=sym, price=100.0 + i * 100, position_size=10.0 + i,
        )
        service.place_trade(trade)

    positions = service.open_positions()
    assert len(positions) == 3
    symbols = [p.symbol for p in positions]
    assert "BTC" in symbols
    assert "ETH" in symbols
    assert "SOL" in symbols
