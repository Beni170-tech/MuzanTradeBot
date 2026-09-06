"""Paper trading broker facade.

This module defines the PaperBroker class, which is the entrypoint for
paper trade lifecycle operations in MuzanTradeBot.

Sprint 5C scope:
    - PaperBroker is now wired to a PaperAccount.
    - place_order validates a Trade via
      validate_paper_trade, builds a PaperOrder if the
      trade is acceptable, and stores it in the account's open positions.
    - get_open_positions returns the account's current open positions.
    - Closing, persistence, market feed, and Telegram integration remain
      out of scope.

Sprint 5D scope:
    - close_order now computes realized PnL, settles it into the
      account balance, and moves the position to closed positions with
      the supplied terminal status.
    - get_open_positions_by_symbol and
      get_closed_positions_by_symbol helpers are added.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from trader.paper.account import PaperAccount
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.paper.rules import validate_paper_trade

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trader.trade import Trade


def _map_side(trade_side: str) -> PaperOrderSide:
    """Map a Trade.side string to a PaperOrderSide.

    Trade.side is documented as "LONG" / "SHORT", but some
    older callers may pass "BUY" / "SELL". This helper keeps the
    mapping local so the rest of the broker never has to care.
    """
    normalized = (trade_side or "").strip().upper()
    if normalized in {"LONG", "BUY"}:
        return PaperOrderSide.LONG
    if normalized in {"SHORT", "SELL"}:
        return PaperOrderSide.SHORT
    return PaperOrderSide.LONG


def _make_order_id(trade: Trade) -> str:
    """Build a deterministic order id for a paper order.

    The id is derived from the trade's identifying fields so that two calls
    with equivalent trades produce the same id. This lets the account's
    duplicate-order check reject re-placement of the same trade.

    The id is stable across calls but unique across different trades.
    """
    symbol = (trade.symbol or "").strip().upper()
    side = _map_side(trade.side).value
    qty = trade.position_size if isinstance(trade.position_size, (int, float)) else 0.0
    entry = _coerce_entry_price(trade)
    sl = float(trade.stop_loss) if isinstance(trade.stop_loss, (int, float)) else 0.0
    tp = float(trade.take_profit) if isinstance(trade.take_profit, (int, float)) else 0.0
    payload = f"{symbol}|{side}|{qty}|{entry}|{sl}|{tp}"
    digest = uuid.uuid5(uuid.NAMESPACE_DNS, payload).hex[:10].upper()
    return f"PAPE-{symbol}-{side}-{int(qty * 1000):08d}-{digest}"


def _coerce_entry_price(trade: Trade) -> float:
    """Coerce Trade.price to a float entry price.

    Trade.price is typed as float | str, so a plain float()
    cast would crash on non-numeric strings. We treat any unparseable value
    as 0.0 - the order will be created, but a zero entry price is an
    obvious signal that the input was bad.
    """
    price = trade.price
    if isinstance(price, (int, float)) and not isinstance(price, bool):
        return float(price)
    try:
        return float(price)
    except (TypeError, ValueError):
        return 0.0


class PaperBroker:
    """Paper broker facade for the paper trading engine.

    This class represents the paper trading broker aspect of the bot. It is
    the single place where paper orders are placed, closed, and inspected
    during development.

    Sprint 5C scope:
        - Accepts a PaperAccount and delegates open-position
          tracking to it.
        - place_order validates through validate_paper_trade,
          builds a PaperOrder, and stores it in the account.
        - Returns None (without raising) when placement can't proceed.

    Sprint 5D scope:
        - close_order computes realized PnL, settles it into the
          account balance, and closes the position with a terminal status.
        - Helper methods for symbol-scoped position lookups are provided.
    """

    def __init__(self, account: PaperAccount) -> None:
        """Initialize the paper broker with an account to track positions in.

        Args:
            account: The PaperAccount that holds open/closed
                positions and balance state.
        """
        self._account = account

    @property
    def account(self) -> PaperAccount:
        """The PaperAccount this broker writes positions into."""
        return self._account

    def place_order(self, trade: Trade) -> PaperOrder | None:
        """Validate a candidate trade and, if acceptable, place a paper order.

        The method is intentionally side-effect-free on failure: if the trade
        does not pass validate_paper_trade, or if the resulting order
        cannot be stored (for example because an open position with the same
        order id already exists), the method returns None and raises no
        exception.

        On success the trade is recorded as a PaperOrder inside the
        broker's PaperAccount, and that order is returned.

        Args:
            trade: The approved Trade to place as a paper order.

        Returns:
            The placed PaperOrder on success, or None when the
            trade is invalid or cannot be stored.
        """
        if trade is None:
            return None

        validation = validate_paper_trade(trade)
        if not validation.ok:
            return None

        try:
            order = PaperOrder(
                order_id=_make_order_id(trade),
                symbol=trade.symbol.strip().upper(),
                side=_map_side(trade.side),
                quantity=float(trade.position_size),
                entry_price=_coerce_entry_price(trade),
                stop_loss=float(trade.stop_loss),
                take_profit=float(trade.take_profit),
                trade=trade,
            )
        except (TypeError, ValueError):
            return None

        try:
            self._account.open_position(order)
        except ValueError:
            return None

        # Mark the order as open now that it has been placed.
        order.status = PaperOrderStatus.OPEN
        order.filled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return order

    def close_order(self, order: PaperOrder, exit_price: float, status: PaperOrderStatus, exit_at: str | None = None) -> PaperOrder:
        """Close an open paper order and settle the realized PnL.

        The position is moved from open to closed, realized PnL is computed
        from exit_price and added to the account balance, and the order
        is marked with the supplied terminal status.

        Args:
            order: The open PaperOrder to close.
            exit_price: The price at which the position is closed.
            status: The terminal status to record on the order (e.g.
                PaperOrderStatus.TAKEN_PROFIT,
                PaperOrderStatus.STOPPED, or
                PaperOrderStatus.CLOSED_MANUAL).
            exit_at: Optional timestamp for the exit. When None the
                current time is used.

        Returns:
            The closed PaperOrder with exit details populated.

        Raises:
            ValueError: If order is not currently open.
        """
        if not order.is_open():
            raise ValueError(
                f"Order {order.order_id!r} is not open (status={order.status.value})."
            )

        order.record_exit(exit_price=exit_price, status=status, exit_at=exit_at)
        self._account.add_realized_pnl(order.realized_pnl)
        self._account.close_position(order.order_id)

        return order

    def get_open_positions(self) -> Sequence[PaperOrder]:
        """Return the currently open paper positions.

        Returns a snapshot of the broker's account open positions so callers
        cannot mutate the account's internal list through the return value.

        Returns:
            A sequence of PaperOrder objects currently tracked as
            open. Empty when there are no open positions.
        """
        return tuple(self._account.open_positions)

    def get_open_positions_by_symbol(self, symbol: str) -> Sequence[PaperOrder]:
        """Return open positions for a given symbol.

        Args:
            symbol: The symbol to filter by (case-insensitive).

        Returns:
            A sequence of open PaperOrder objects for the symbol.
            Empty when there are no matching open positions.
        """
        symbol_upper = symbol.strip().upper()
        return tuple(
            p for p in self._account.open_positions
            if p.symbol == symbol_upper and p.is_open()
        )

    def get_closed_positions_by_symbol(self, symbol: str) -> Sequence[PaperOrder]:
        """Return closed positions for a given symbol, most recent first.

        Args:
            symbol: The symbol to filter by (case-insensitive).

        Returns:
            A sequence of closed PaperOrder objects for the symbol.
            Empty when there are no matching closed positions.
        """
        symbol_upper = symbol.strip().upper()
        closed = [p for p in self._account.closed_positions if p.symbol == symbol_upper]
        closed.sort(key=lambda p: p.exit_at or p.created_at, reverse=True)
        return tuple(closed)

    def get_total_realized_pnl(self) -> float:
        """Return the sum of realized PnL across all closed positions.

        Returns:
            Total realized profit (positive) or loss (negative) from all
            closed positions.
        """
        return sum(p.realized_pnl for p in self._account.closed_positions)

    def get_open_pnl(self) -> float:
        """Return the sum of unrealized PnL across all open positions.

        Returns:
            Total unrealized profit (positive) or loss (negative) across
            currently open positions.
        """
        return sum(p.unrealized_pnl_for(p.entry_price) for p in self._account.open_positions)

