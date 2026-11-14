"""Tests for trader.paper.broker.PaperBroker."""

import pytest

from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
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


def _valid_trade(
    symbol: str = "BTC",
    side: str = "LONG",
    position_size: float = 10.0,
    price: float = 100.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> Trade:
    is_short = side.strip().upper() in {"SHORT", "SELL"}
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


def _invalid_protection_trade(
    *,
    stop_loss: object,
    take_profit: object,
    side: str = "LONG",
) -> Trade:
    return Trade(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=10.0,
        stop_loss=stop_loss,  # type: ignore[arg-type]
        take_profit=take_profit,  # type: ignore[arg-type]
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
# Construction
# ---------------------------------------------------------------------------

def test_broker_stores_account() -> None:
    acc = _account()
    broker = PaperBroker(acc)
    assert broker.account is acc


def test_broker_default_account_is_fresh() -> None:
    broker = _broker()
    assert broker.account.open_count == 0
    assert broker.account.closed_count == 0


# ---------------------------------------------------------------------------
# place_order - happy path
# ---------------------------------------------------------------------------

def test_place_valid_trade_returns_order() -> None:
    broker = _broker()
    trade = _valid_trade()
    order = broker.place_order(trade)
    assert order is not None
    assert order.symbol == "BTC"
    assert order.side == PaperOrderSide.LONG
    assert order.quantity == 10.0
    assert order.entry_price == 100.0


def test_place_order_stores_in_account() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade())
    assert broker.account.open_count == 1
    assert order in broker.account.open_positions


def test_place_order_sets_open_status() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade())
    assert order.status == PaperOrderStatus.OPEN


def test_place_order_generates_deterministic_id() -> None:
    broker = _broker()
    trade = _valid_trade()
    o1 = broker.place_order(trade)
    broker2 = _broker()
    o2 = broker2.place_order(trade)
    assert o1.order_id == o2.order_id


def test_place_order_with_buy_side_alias() -> None:
    broker = _broker()
    trade = _valid_trade(side="BUY")
    order = broker.place_order(trade)
    assert order is not None
    assert order.side == PaperOrderSide.LONG


def test_place_order_with_sell_side_alias() -> None:
    broker = _broker()
    trade = _valid_trade(side="SELL", price=50.0, stop_loss=51.0, take_profit=49.0)
    order = broker.place_order(trade)
    assert order is not None
    assert order.side == PaperOrderSide.SHORT


def test_place_short_order() -> None:
    broker = _broker()
    trade = _valid_trade(side="SHORT", price=50.0, stop_loss=51.0, take_profit=49.0)
    order = broker.place_order(trade)
    assert order is not None
    assert order.side == PaperOrderSide.SHORT


# ---------------------------------------------------------------------------
# place_order - rejection cases
# ---------------------------------------------------------------------------

def test_place_none_trade_returns_none() -> None:
    broker = _broker()
    assert broker.place_order(None) is None


def test_place_invalid_trade_returns_none() -> None:
    broker = _broker()
    trade = _valid_trade(symbol="")  # invalid: empty symbol
    assert broker.place_order(trade) is None


def test_place_order_with_zero_quantity_returns_none() -> None:
    broker = _broker()
    trade = _valid_trade(position_size=0)
    assert broker.place_order(trade) is None


def test_place_order_with_zero_price_returns_none() -> None:
    broker = _broker()
    trade = _valid_trade(price=0)
    assert broker.place_order(trade) is None


def test_place_duplicate_order_returns_none() -> None:
    broker = _broker()
    trade = _valid_trade()
    broker.place_order(trade)
    assert broker.place_order(trade) is None


def test_place_order_does_not_add_to_account_on_failure() -> None:
    broker = _broker()
    trade = _valid_trade(symbol="")
    before = broker.account.open_count
    broker.place_order(trade)
    assert broker.account.open_count == before


# ---------------------------------------------------------------------------
# get_open_positions
# ---------------------------------------------------------------------------

def test_get_open_positions_returns_empty_when_none() -> None:
    broker = _broker()
    assert broker.get_open_positions() == ()


def test_get_open_positions_returns_tuple_snapshot() -> None:
    broker = _broker()
    broker.place_order(_valid_trade())
    positions = broker.get_open_positions()
    assert isinstance(positions, tuple)
    positions_list = list(positions)
    positions_list.clear()
    assert broker.account.open_count == 1


def test_get_open_positions_excludes_closed() -> None:
    broker = _broker()
    trade = _valid_trade()
    order = broker.place_order(trade)
    broker.account.close_position(order.order_id)
    assert broker.get_open_positions() == ()


# ---------------------------------------------------------------------------
# place_order - protection validation at the execution boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "stop_loss, take_profit, side",
    [
        (None, 104.0, "LONG"),
        (98.0, None, "LONG"),
        ("98", 104.0, "LONG"),
        (98.0, "104", "LONG"),
        (True, 104.0, "LONG"),
        (98.0, False, "LONG"),
        (float("nan"), 104.0, "LONG"),
        (98.0, float("inf"), "LONG"),
        (float("-inf"), 104.0, "LONG"),
        (0.0, 104.0, "LONG"),
        (-1.0, 104.0, "LONG"),
        (98.0, 0.0, "LONG"),
        (98.0, -1.0, "LONG"),
        (100.0, 104.0, "LONG"),
        (101.0, 104.0, "LONG"),
        (98.0, 100.0, "LONG"),
        (98.0, 99.0, "LONG"),
        (100.0, 96.0, "SHORT"),
        (99.0, 96.0, "SHORT"),
        (102.0, 100.0, "SHORT"),
        (102.0, 101.0, "SHORT"),
    ],
)
def test_invalid_protection_cannot_enter_broker(
    stop_loss: object, take_profit: object, side: str
) -> None:
    broker = _broker()
    trade = _invalid_protection_trade(
        stop_loss=stop_loss, take_profit=take_profit, side=side
    )

    assert broker.place_order(trade) is None
    assert broker.account.open_count == 0
    assert broker.get_open_positions() == ()


def test_valid_short_protection_enters_broker() -> None:
    broker = _broker()
    trade = _valid_trade(
        side="SHORT", price=100.0, stop_loss=102.0, take_profit=96.0
    )

    order = broker.place_order(trade)

    assert order is not None
    assert order.stop_loss == 102.0
    assert order.take_profit == 96.0


def test_placement_records_initial_risk_distance() -> None:
    broker = _broker()
    order = broker.place_order(
        _valid_trade(price=100.0, stop_loss=90.0, take_profit=120.0)
    )

    assert order is not None
    assert order.initial_risk_distance == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# close_order - happy path
# ---------------------------------------------------------------------------

def test_close_order_computes_realized_pnl_long_profit() -> None:
    broker = _broker()
    trade = _valid_trade(price=100.0, stop_loss=98.0, take_profit=104.0)
    order = broker.place_order(trade)
    assert order is not None
    closed = broker.close_order(order, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    assert closed.realized_pnl == pytest.approx(100.0)
    assert closed.exit_price == 110.0
    assert closed.status == PaperOrderStatus.TAKEN_PROFIT


def test_close_order_computes_realized_pnl_long_loss() -> None:
    broker = _broker()
    trade = _valid_trade(price=100.0)
    order = broker.place_order(trade)
    assert order is not None
    closed = broker.close_order(order, exit_price=90.0, status=PaperOrderStatus.STOPPED)
    assert closed.realized_pnl == pytest.approx(-100.0)
    assert closed.status == PaperOrderStatus.STOPPED


def test_close_order_computes_realized_pnl_short_profit() -> None:
    broker = _broker()
    trade = _valid_trade(side="SHORT", price=100.0, stop_loss=102.0, take_profit=96.0)
    order = broker.place_order(trade)
    assert order is not None
    closed = broker.close_order(order, exit_price=90.0, status=PaperOrderStatus.TAKEN_PROFIT)
    assert closed.realized_pnl == pytest.approx(100.0)


def test_close_order_computes_realized_pnl_short_loss() -> None:
    broker = _broker()
    trade = _valid_trade(side="SHORT", price=100.0)
    order = broker.place_order(trade)
    assert order is not None
    closed = broker.close_order(order, exit_price=110.0, status=PaperOrderStatus.STOPPED)
    assert closed.realized_pnl == pytest.approx(-100.0)


def test_close_order_moves_position_to_closed() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade())
    assert broker.account.open_count == 1
    broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)
    assert broker.account.open_count == 0
    assert broker.account.closed_count == 1


def test_close_order_updates_account_balance() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade(price=100.0))
    assert broker.account.current_balance == 1000.0
    broker.close_order(order, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    assert broker.account.current_balance == pytest.approx(1100.0)


def test_close_order_records_exit_timestamp() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade())
    custom_exit_at = "2025-01-01 12:00:00"
    broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL, exit_at=custom_exit_at)
    assert order.exit_at == custom_exit_at


def test_close_order_uses_current_time_when_no_exit_at() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade())
    broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)
    assert order.exit_at is not None


def test_close_order_returns_the_closed_order() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade())
    returned = broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)
    assert returned is order


# ---------------------------------------------------------------------------
# close_order - error cases
# ---------------------------------------------------------------------------

def test_close_already_closed_order_raises() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade())
    broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)
    with pytest.raises(ValueError, match="is not open"):
        broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.TAKEN_PROFIT)


def test_close_pending_order_raises() -> None:
    broker = _broker()
    trade = _valid_trade()
    order = broker.place_order(trade)
    order.status = PaperOrderStatus.PENDING
    with pytest.raises(ValueError, match="is not open"):
        broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)


def test_close_ghost_order_raises() -> None:
    broker = _broker()
    order = _open_order(order_id="GHOST")
    # Ghost order has OPEN status but isn't in the account.
    # close_order checks is_open() (passes) then tries close_position
    # which raises KeyError because the order isn't tracked.
    with pytest.raises(KeyError, match="No open position found"):
        broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)


# ---------------------------------------------------------------------------
# Symbol-scoped lookups
# ---------------------------------------------------------------------------

def test_get_open_positions_by_symbol_returns_only_that_symbol() -> None:
    broker = _broker()
    broker.place_order(_valid_trade(symbol="BTC"))
    broker.place_order(_valid_trade(symbol="ETH", price=2000.0))
    result = broker.get_open_positions_by_symbol("BTC")
    assert len(result) == 1
    assert all(p.symbol == "BTC" for p in result)


def test_get_open_positions_by_symbol_case_insensitive() -> None:
    broker = _broker()
    broker.place_order(_valid_trade(symbol="BTC"))
    result = broker.get_open_positions_by_symbol("btc")
    assert len(result) == 1


def test_get_open_positions_by_symbol_returns_empty_when_none() -> None:
    broker = _broker()
    assert broker.get_open_positions_by_symbol("BTC") == ()


def test_get_closed_positions_by_symbol_returns_only_that_symbol() -> None:
    broker = _broker()
    btc_trade = _valid_trade(symbol="BTC")
    eth_trade = _valid_trade(symbol="ETH", price=2000.0)
    btc_order = broker.place_order(btc_trade)
    eth_order = broker.place_order(eth_trade)
    broker.close_order(btc_order, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    result = broker.get_closed_positions_by_symbol("BTC")
    assert len(result) == 1
    assert result[0].order_id == btc_order.order_id


def test_get_closed_positions_by_symbol_case_insensitive() -> None:
    broker = _broker()
    order = broker.place_order(_valid_trade(symbol="BTC"))
    broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)
    result = broker.get_closed_positions_by_symbol("btc")
    assert len(result) == 1


def test_get_closed_positions_by_symbol_returns_most_recent_first() -> None:
    broker = _broker()
    o1 = broker.place_order(_valid_trade(symbol="BTC", price=100.0))
    o2 = broker.place_order(_valid_trade(symbol="BTC", price=110.0, position_size=5.0))
    broker.close_order(o1, exit_price=105.0, status=PaperOrderStatus.CLOSED_MANUAL, exit_at="2025-01-01 10:00:00")
    broker.close_order(o2, exit_price=115.0, status=PaperOrderStatus.TAKEN_PROFIT, exit_at="2025-01-01 12:00:00")
    result = broker.get_closed_positions_by_symbol("BTC")
    assert len(result) == 2
    assert result[0].order_id == o2.order_id
    assert result[1].order_id == o1.order_id


def test_get_closed_positions_by_symbol_returns_empty_when_none() -> None:
    broker = _broker()
    assert broker.get_closed_positions_by_symbol("BTC") == ()


# ---------------------------------------------------------------------------
# PnL aggregation helpers
# ---------------------------------------------------------------------------

def test_get_total_realized_pnl_sums_closed_positions() -> None:
    broker = _broker()
    o1 = broker.place_order(_valid_trade(price=100.0))
    o2 = broker.place_order(_valid_trade(symbol="ETH", price=2000.0, position_size=1.0))
    broker.close_order(o1, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    broker.close_order(o2, exit_price=1900.0, status=PaperOrderStatus.STOPPED)
    # o1: (110-100)*10 = +100, o2: (1900-2000)*1 = -100, total = 0
    assert broker.get_total_realized_pnl() == pytest.approx(0.0)


def test_get_total_realized_pnl_returns_zero_when_none_closed() -> None:
    broker = _broker()
    assert broker.get_total_realized_pnl() == 0.0


def test_get_open_pnl_sums_unrealized() -> None:
    broker = _broker()
    broker.place_order(_valid_trade(price=100.0, position_size=10.0))
    assert broker.get_open_pnl() == pytest.approx(0.0)


def test_get_open_pnl_is_zero_when_none_open() -> None:
    broker = _broker()
    assert broker.get_open_pnl() == 0.0


# ---------------------------------------------------------------------------
# End-to-end: place, let price move, close
# ---------------------------------------------------------------------------

def test_full_lifecycle_long_win() -> None:
    acc = _account(current_balance=1000.0)
    broker = PaperBroker(acc)
    trade = _valid_trade(symbol="BTC", price=100.0, position_size=10.0)
    order = broker.place_order(trade)
    assert broker.account.current_balance == 1000.0
    broker.close_order(order, exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    assert broker.account.current_balance == pytest.approx(1100.0)
    assert broker.account.closed_count == 1


def test_full_lifecycle_long_loss() -> None:
    acc = _account(current_balance=1000.0)
    broker = PaperBroker(acc)
    trade = _valid_trade(symbol="BTC", price=100.0, position_size=10.0)
    order = broker.place_order(trade)
    broker.close_order(order, exit_price=90.0, status=PaperOrderStatus.STOPPED)
    assert broker.account.current_balance == pytest.approx(900.0)


def test_multiple_trades_accumulate_pnl() -> None:
    acc = _account(current_balance=1000.0)
    broker = PaperBroker(acc)
    trades = [
        _valid_trade(symbol="BTC", price=100.0, position_size=10.0),
        _valid_trade(symbol="ETH", price=2000.0, position_size=1.0, side="SHORT"),
    ]
    orders = [broker.place_order(t) for t in trades]
    broker.close_order(orders[0], exit_price=110.0, status=PaperOrderStatus.TAKEN_PROFIT)
    broker.close_order(orders[1], exit_price=1900.0, status=PaperOrderStatus.TAKEN_PROFIT)
    assert broker.account.current_balance == pytest.approx(1200.0)
    assert broker.get_total_realized_pnl() == pytest.approx(200.0)
