"""Paper trading execution helpers.

Sprint 5E: Convenience layer between the analysis pipeline and the
paper broker. These helpers validate, place, and close paper trades
without modifying any production trading logic.
"""

from __future__ import annotations

from trader.execution.contract import ExecutionBackend
from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.order import PaperOrder, PaperOrderStatus
from trader.paper.summary import PaperAccountSummary
from trader.paper.rules import validate_paper_trade
from trader.trade import Trade


def place_paper_trade(broker: PaperBroker, trade: Trade) -> PaperOrder | None:
    """Place an approved Trade as a paper order.

    The trade must be approved by the risk system (``trade.approved is
    True``) and must pass paper trading validation. If either condition
    is not met, the function returns ``None`` without placing an order.

    Args:
        broker: The PaperBroker to place the order with.
        trade: The Trade to place. Must have ``approved`` set to True.

    Returns:
        The placed PaperOrder on success, or None when the trade is not
        approved or fails validation.
    """
    if trade is None:
        return None

    if not trade.approved:
        return None

    validation = validate_paper_trade(trade)
    if not validation.ok:
        return None

    return broker.place_order(trade)


def close_paper_position(
    broker: PaperBroker,
    symbol: str,
    exit_price: float,
    status: PaperOrderStatus,
    exit_at: str | None = None,
) -> PaperOrder | None:
    """Close an open paper position for a symbol.

    Looks up the open position for the given symbol and closes it with
    the provided exit parameters. If no open position exists for the
    symbol, returns None.

    Args:
        broker: The PaperBroker that holds the position.
        symbol: The symbol to close (case-insensitive).
        exit_price: The price at which the position is closed.
        status: The terminal status for the closed order (e.g.
            PaperOrderStatus.TAKEN_PROFIT, PaperOrderStatus.STOPPED,
            PaperOrderStatus.CLOSED_MANUAL). Must be provided explicitly.
        exit_at: Optional timestamp for the exit. When None, the current
            time is used.

    Returns:
        The closed PaperOrder on success, or None when no open position
        exists for the symbol.
    """
    open_positions = broker.get_open_positions_by_symbol(symbol)
    if not open_positions:
        return None

    order = open_positions[0]
    return broker.close_order(order, exit_price, status, exit_at)


def get_open_position_by_id(
    broker: PaperBroker, order_id: str
) -> PaperOrder | None:
    """Return the open position with the exact order id, or None.

    Order ids are compared case-insensitively and must match exactly; no
    positional fallback is applied, so callers can never accidentally
    select a different position on the same symbol.

    Args:
        broker: The PaperBroker that holds the position.
        order_id: The exact order id to select.

    Returns:
        The matching open PaperOrder, or None when it is missing.
    """
    target = (order_id or "").strip().upper()
    if not target:
        return None
    for order in broker.get_open_positions():
        if str(order.order_id).strip().upper() == target:
            return order
    return None


def close_paper_position_by_id(
    broker: PaperBroker,
    order_id: str,
    exit_price: float,
    status: PaperOrderStatus,
    exit_at: str | None = None,
) -> PaperOrder | None:
    """Close exactly one open paper position, selected by order id.

    Unlike :func:`close_paper_position`, this never falls back to the
    first position on a symbol. When the id does not identify an open
    position, nothing is closed and None is returned.

    Args:
        broker: The PaperBroker that holds the position.
        order_id: The exact order id of the position to close.
        exit_price: The price at which the position is closed.
        status: The terminal status for the closed order.
        exit_at: Optional timestamp for the exit.

    Returns:
        The closed PaperOrder, or None when no open position with that
        order id exists.
    """
    order = get_open_position_by_id(broker, order_id)
    if order is None:
        return None
    return broker.close_order(order, exit_price, status, exit_at)


def get_open_position(broker: PaperBroker, symbol: str) -> PaperOrder | None:
    """Return the open position for a symbol, or None if none exists.

    Args:
        broker: The PaperBroker to query.
        symbol: The symbol to look up (case-insensitive).

    Returns:
        The open PaperOrder for the symbol, or None when no open position
        exists.
    """
    positions = broker.get_open_positions_by_symbol(symbol)
    if not positions:
        return None
    return positions[0]


def get_paper_summary(backend: ExecutionBackend) -> PaperAccountSummary:
    """Build a typed summary of the backend's account state.

    Sprint 10.3: every value is read through the ExecutionBackend
    contract, so the summary no longer needs to reach into a
    paper-specific ``broker.account`` attribute.

    Args:
        backend: The execution backend to summarize.

    Returns:
        A PaperAccountSummary with open positions, closed count, and
        realized/unrealized PnL totals.
    """
    return PaperAccountSummary(
        initial_balance=backend.get_initial_balance(),
        current_balance=backend.get_current_balance(),
        open_positions=list(backend.get_open_positions()),
        closed_count=backend.get_closed_count(),
        total_realized_pnl=backend.get_total_realized_pnl(),
        open_pnl=backend.get_open_pnl(),
        equity=backend.get_equity(),
    )
