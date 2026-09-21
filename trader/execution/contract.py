"""Execution contract for MuzanTradeBot.

Sprint 10.1: backend-agnostic contract for order placement, order closure,
and portfolio-state queries so future paper and live backends can be swapped
behind the same interface.

This module intentionally does not place, close, persist, or notify.
It only defines the contract and the result types that any backend
must satisfy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from trader.trade import Trade

from trader.paper.order import PaperOrder, PaperOrderStatus


# NOTE: This contract intentionally does not import or depend on any
# live execution module. When a live backend is introduced it must be
# added to config.settings._LIVE_EXECUTION_MODES explicitly, and the
# startup safety check in main.py must still reject any config that
# would enable live trading unintentionally.


@runtime_checkable
class ExecutionBackend(Protocol):
    """Backend-agnostic contract for placing and closing orders.

    A backend can be anything that fulfills this protocol: the existing
    paper broker, a future live broker, or a test double. The contract is
    deliberately small and does not prescribe persistence, alerting,
    market data, or account storage.

    Placement and closure are the core execution operations. The query
    methods below exist so portfolio and risk modules can inspect backend
    state through the same interface used for execution.
    """

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def place_order(self, trade: Trade) -> PaperOrder | None:
        """Place a candidate trade as an order.

        Args:
            trade: The analyzed trade to place. The caller is responsible
                for ensuring the trade has already passed all required
                risk checks before it is passed here.

        Returns:
            The placed order on success, or None when the backend refuses
            or cannot place the order. Backends must never raise for
            ordinary rejection reasons (invalid trade, duplicate order,
            limit exceeded, etc.); they return None instead.
        """
        ...

    # ------------------------------------------------------------------
    # Closure
    # ------------------------------------------------------------------

    def close_order(
        self,
        order: PaperOrder,
        exit_price: float,
        status: PaperOrderStatus,
        exit_at: str | None = None,
    ) -> PaperOrder:
        """Close an open order at a given exit price.

        Args:
            order: The open order to close.
            exit_price: The price at which the position is closed.
            status: The terminal status to record on the order.
            exit_at: Optional timestamp for the exit. When None the
                backend uses its own current-time source.

        Returns:
            The closed order with exit details populated.

        Raises:
            ValueError: When the order is not currently open.
        """
        ...

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_open_positions(self):
        """Return the backend's currently open orders.

        Returns:
            A sequence of open orders. The returned value must be a stable
            snapshot that callers cannot mutate to affect backend state.
        """
        ...

    def get_total_realized_pnl(self) -> float:
        """Return total realized PnL across closed orders.

        Returns:
            Sum of realized profit (positive) or loss (negative) across
            all closed orders.
        """
        ...

    def get_open_pnl(self) -> float:
        """Return total unrealized PnL across open orders.

        Returns:
            Sum of unrealized profit (positive) or loss (negative) across
            currently open orders.
        """
        ...

    # ------------------------------------------------------------------
    # Symbol-scoped queries
    # ------------------------------------------------------------------

    def get_open_positions_by_symbol(self, symbol: str):
        """Return the backend's currently open orders for one symbol.

        Args:
            symbol: The symbol to filter by (case-insensitive).

        Returns:
            A sequence of open orders for the symbol. The returned value
            must be a stable snapshot that callers cannot mutate to affect
            backend state. Empty when there are no matching open orders.
        """
        ...

    def get_closed_positions_by_symbol(self, symbol: str):
        """Return the backend's closed orders for one symbol.

        Args:
            symbol: The symbol to filter by (case-insensitive).

        Returns:
            A sequence of closed orders for the symbol, most recent first.
            The returned value must be a stable snapshot that callers
            cannot mutate to affect backend state. Empty when there are
            no matching closed orders.
        """
        ...

    def get_total_realized_pnl_by_symbol(self, symbol: str) -> float:
        """Return realized PnL for one symbol across closed orders.

        Args:
            symbol: The symbol to filter by (case-insensitive).

        Returns:
            Sum of realized profit (positive) or loss (negative) for the
            symbol across all closed orders.
        """
        ...
