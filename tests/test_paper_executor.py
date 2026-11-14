"""Tests for trader.paper.executor helpers (Sprint 5E)."""

import pytest

from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.executor import (
    close_paper_position,
    get_open_position,
    get_paper_summary,
    place_paper_trade,
)
from trader.paper.order import PaperOrderStatus
from trader.paper.summary import PaperAccountSummary
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _account(
    initial_balance: float = 1000.0,
    current_balance: float = 1000.0,
) -> PaperAccount:
    return PaperAccount(
        initial_balance=initial_balance,
        current_balance=current_balance,
    )


def _broker(account: PaperAccount | None = None) -> PaperBroker:
    return PaperBroker(account or _account())


def _approved_trade(
    symbol: str = "BTC",
    side: str = "LONG",
    position_size: float = 10.0,
    price: float = 100.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> Trade:
    is_short = side.strip().upper() == "SHORT"
    if stop_loss is None:
        stop_loss = price * 1.02 if is_short else price * 0.98
    if take_profit is None:
        take_profit = price * 0.96 if is_short else price * 1.04
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


def _unapproved_trade(
    symbol: str = "BTC",
    side: str = "LONG",
    position_size: float = 10.0,
    price: float = 100.0,
) -> Trade:
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
# place_paper_trade
# ---------------------------------------------------------------------------

def test_place_paper_trade_with_approved_trade_returns_order() -> None:
    broker = _broker()
    trade = _approved_trade()
    order = place_paper_trade(broker, trade)
    assert order is not None
    assert order.symbol == "BTC"
    assert order.side.name == "LONG"
    assert order.quantity == 10.0


def test_place_paper_trade_stores_order_in_account() -> None:
    broker = _broker()
    trade = _approved_trade()
    place_paper_trade(broker, trade)
    assert broker.account.open_count == 1


def test_place_paper_trade_rejects_unapproved_trade() -> None:
    broker = _broker()
    trade = _unapproved_trade()
    result = place_paper_trade(broker, trade)
    assert result is None
    assert broker.account.open_count == 0


def test_place_paper_trade_rejects_none_trade() -> None:
    broker = _broker()
    result = place_paper_trade(broker, None)  # type: ignore[arg-type]
    assert result is None


def test_place_paper_trade_rejects_invalid_trade() -> None:
    broker = _broker()
    trade = _approved_trade(symbol="")  # invalid: empty symbol
    result = place_paper_trade(broker, trade)
    assert result is None
    assert broker.account.open_count == 0


def test_place_paper_trade_rejects_zero_position_size() -> None:
    broker = _broker()
    trade = _approved_trade(position_size=0)
    result = place_paper_trade(broker, trade)
    assert result is None


def test_place_paper_trade_with_short_side() -> None:
    broker = _broker()
    trade = _approved_trade(side="SHORT", price=50.0, stop_loss=51.0, take_profit=49.0)
    order = place_paper_trade(broker, trade)
    assert order is not None
    assert order.side.name == "SHORT"


def test_place_paper_trade_does_not_place_duplicate() -> None:
    broker = _broker()
    trade = _approved_trade()
    o1 = place_paper_trade(broker, trade)
    assert o1 is not None
    o2 = place_paper_trade(broker, trade)
    assert o2 is None  # duplicate order id


# ---------------------------------------------------------------------------
# close_paper_position
# ---------------------------------------------------------------------------

def test_close_paper_position_closes_existing_position() -> None:
    broker = _broker()
    trade = _approved_trade(price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    closed = close_paper_position(
        broker,
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert closed is not None
    assert closed.status == PaperOrderStatus.TAKEN_PROFIT
    assert closed.realized_pnl == pytest.approx(100.0)
    assert broker.account.open_count == 0
    assert broker.account.closed_count == 1


def test_close_paper_position_updates_balance() -> None:
    broker = _broker()
    trade = _approved_trade(price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    close_paper_position(
        broker,
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert broker.account.current_balance == pytest.approx(1100.0)


def test_close_paper_position_returns_none_when_no_open_position() -> None:
    broker = _broker()
    result = close_paper_position(
        broker,
        symbol="NONEXISTENT",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    assert result is None


def test_close_paper_position_requires_explicit_status() -> None:
    broker = _broker()
    trade = _approved_trade(price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    closed = close_paper_position(
        broker,
        symbol="BTC",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    assert closed is not None
    assert closed.status == PaperOrderStatus.CLOSED_MANUAL


def test_close_paper_position_case_insensitive_symbol() -> None:
    broker = _broker()
    trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    closed = close_paper_position(
        broker,
        symbol="btc",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    assert closed is not None


def test_close_paper_position_short_profit() -> None:
    broker = _broker()
    trade = _approved_trade(side="SHORT", price=100.0, position_size=10.0,
                            stop_loss=102.0, take_profit=96.0)
    place_paper_trade(broker, trade)
    closed = close_paper_position(
        broker,
        symbol="BTC",
        exit_price=90.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert closed is not None
    assert closed.realized_pnl == pytest.approx(100.0)


def test_close_paper_position_multiple_symbols_independent() -> None:
    broker = _broker()
    btc_trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    eth_trade = _approved_trade(symbol="ETH", price=2000.0, position_size=1.0)
    place_paper_trade(broker, btc_trade)
    place_paper_trade(broker, eth_trade)
    closed_btc = close_paper_position(
        broker,
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert closed_btc is not None
    assert broker.account.open_count == 1
    assert broker.account.closed_count == 1


# ---------------------------------------------------------------------------
# get_open_position
# ---------------------------------------------------------------------------

def test_get_open_position_returns_order() -> None:
    broker = _broker()
    trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    order = get_open_position(broker, "BTC")
    assert order is not None
    assert order.symbol == "BTC"


def test_get_open_position_returns_none_when_none_open() -> None:
    broker = _broker()
    result = get_open_position(broker, "BTC")
    assert result is None


def test_get_open_position_case_insensitive() -> None:
    broker = _broker()
    trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    order = get_open_position(broker, "btc")
    assert order is not None


def test_get_open_position_returns_none_after_close() -> None:
    broker = _broker()
    trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    close_paper_position(
        broker,
        symbol="BTC",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    result = get_open_position(broker, "BTC")
    assert result is None


# ---------------------------------------------------------------------------
# get_paper_summary
# ---------------------------------------------------------------------------

def test_get_paper_summary_empty_account() -> None:
    broker = _broker()
    summary = get_paper_summary(broker)
    assert isinstance(summary, PaperAccountSummary)
    assert summary.open_positions == []
    assert summary.closed_count == 0
    assert summary.total_realized_pnl == 0.0
    assert summary.open_pnl == 0.0
    assert summary.equity == 1000.0


def test_get_paper_summary_reflects_open_position() -> None:
    broker = _broker()
    trade = _approved_trade(price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    summary = get_paper_summary(broker)
    assert len(summary.open_positions) == 1
    assert summary.open_positions[0].symbol == "BTC"
    assert summary.closed_count == 0
    assert summary.open_pnl == 0.0
    assert summary.equity == 1000.0


def test_get_paper_summary_after_closed_trade() -> None:
    broker = _broker()
    trade = _approved_trade(price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    close_paper_position(
        broker,
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    summary = get_paper_summary(broker)
    assert summary.open_positions == []
    assert summary.closed_count == 1
    assert summary.total_realized_pnl == pytest.approx(100.0)
    assert summary.open_pnl == 0.0
    assert summary.equity == pytest.approx(1100.0)


def test_get_paper_summary_multiple_trades() -> None:
    broker = _broker()
    btc_trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    eth_trade = _approved_trade(symbol="ETH", price=2000.0, position_size=1.0)
    place_paper_trade(broker, btc_trade)
    place_paper_trade(broker, eth_trade)
    summary = get_paper_summary(broker)
    assert summary.closed_count == 0
    assert len(summary.open_positions) == 2
    close_paper_position(
        broker,
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    summary = get_paper_summary(broker)
    assert summary.closed_count == 1
    assert summary.total_realized_pnl == pytest.approx(100.0)
    assert len(summary.open_positions) == 1
    assert summary.open_positions[0].symbol == "ETH"


def test_get_paper_summary_summary_is_frozen() -> None:
    broker = _broker()
    summary = get_paper_summary(broker)
    with pytest.raises(AttributeError):
        summary.equity = 9999.0


def test_get_paper_summary_with_loss() -> None:
    broker = _broker()
    trade = _approved_trade(price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    close_paper_position(
        broker,
        symbol="BTC",
        exit_price=90.0,
        status=PaperOrderStatus.STOPPED,
    )
    summary = get_paper_summary(broker)
    assert summary.total_realized_pnl == pytest.approx(-100.0)
    assert summary.equity == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# End-to-end: full lifecycle via executor helpers
# ---------------------------------------------------------------------------

def test_full_lifecycle_via_executor_long_win() -> None:
    broker = _broker()
    trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    order = place_paper_trade(broker, trade)
    assert order is not None
    assert broker.account.open_count == 1
    closed = close_paper_position(
        broker,
        symbol="BTC",
        exit_price=110.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert closed is not None
    assert closed.realized_pnl == pytest.approx(100.0)
    summary = get_paper_summary(broker)
    assert summary.closed_count == 1
    assert summary.total_realized_pnl == pytest.approx(100.0)
    assert summary.equity == pytest.approx(1100.0)


def test_full_lifecycle_via_executor_long_loss() -> None:
    broker = _broker()
    trade = _approved_trade(symbol="BTC", price=100.0, position_size=10.0)
    place_paper_trade(broker, trade)
    close_paper_position(
        broker,
        symbol="BTC",
        exit_price=90.0,
        status=PaperOrderStatus.STOPPED,
    )
    summary = get_paper_summary(broker)
    assert summary.total_realized_pnl == pytest.approx(-100.0)
    assert summary.equity == pytest.approx(900.0)


def test_executor_does_not_place_unapproved_trade() -> None:
    broker = _broker()
    trade = _unapproved_trade()
    result = place_paper_trade(broker, trade)
    assert result is None
    assert broker.account.open_count == 0
    closed = close_paper_position(
        broker,
        symbol="BTC",
        exit_price=100.0,
        status=PaperOrderStatus.CLOSED_MANUAL,
    )
    assert closed is None
    summary = get_paper_summary(broker)
    assert summary.closed_count == 0
    assert summary.total_realized_pnl == 0.0


# ---------------------------------------------------------------------------
# Sprint 10.3: summary is built through the execution contract
# ---------------------------------------------------------------------------

def test_get_paper_summary_reads_state_through_contract() -> None:
    """A backend with no paper-specific ``account`` attribute summarises fine."""

    class Backend:
        def get_initial_balance(self) -> float:
            return 500.0

        def get_current_balance(self) -> float:
            return 640.0

        def get_closed_count(self) -> int:
            return 3

        def get_equity(self) -> float:
            return 630.0

        def get_open_positions(self) -> tuple:
            return ()

        def get_total_realized_pnl(self) -> float:
            return 140.0

        def get_open_pnl(self) -> float:
            return -10.0

    backend = Backend()
    assert not hasattr(backend, "account")

    summary = get_paper_summary(backend)

    assert summary.initial_balance == 500.0
    assert summary.current_balance == 640.0
    assert summary.closed_count == 3
    assert summary.equity == 630.0
    assert summary.total_realized_pnl == 140.0
    assert summary.open_pnl == -10.0
    assert summary.open_positions == []
