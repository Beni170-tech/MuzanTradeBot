"""
MuzanTradeBot - Paper Account State (Sprint 5A/5D)

Dataclass that holds simulated paper account state.

Sprint 5A scope only:
- This module defines the shape of a paper account: initial and
  current balance, open positions, and closed positions. It does
  not update balances, move positions, trigger exits, persist
  state, or connect to any executor. All behavior is out of scope.

Sprint 5D scope:
- Account equity calculation.
- Balance update after realized PnL.
- Open/closed position queries by symbol and terminal state.
"""

from dataclasses import dataclass, field
from typing import List

from trader.paper.order import PaperOrder, PaperOrderStatus


@dataclass
class PaperAccount:
    """Simulated paper trading account state.

    Sprint 5A/5B scope:
        Holds state and supports adding/removing paper positions for
        testing and future execution layers. Does not update balances,
        compute PnL, persist state, or place orders.

    Sprint 5D scope:
        Equity calculation and balance updates from realized PnL are
        added so the broker can settle closed trades against the
        account. Open-position convenience queries are also added.
    """

    initial_balance: float
    current_balance: float
    open_positions: List[PaperOrder] = field(default_factory=list)
    closed_positions: List[PaperOrder] = field(default_factory=list)

    # --- Open position tracking ----------------------------------------

    def _order_ids(self) -> set:
        """Return the set of order ids across open positions for quick checks."""
        return {position.order_id for position in self.open_positions}

    def open_position(self, order: PaperOrder) -> None:
        """Add a paper order to open positions.

        Args:
            order: The PaperOrder to track as open.

        Raises:
            ValueError: If an open position with the same order id
                already exists.
        """
        if order.order_id in self._order_ids():
            raise ValueError(
                f"Open position already exists for order id {order.order_id!r}."
            )
        self.open_positions.append(order)

    def close_position(self, order_id: str) -> PaperOrder:
        """Move a paper order from open positions to closed positions.

        Does not change any balance or compute any PnL. Those behaviors
        are intentionally out of scope for Sprint 5B.

        Args:
            order_id: The order id of the open position to close.

        Returns:
            The :class:`PaperOrder` that was moved to closed positions.

        Raises:
            KeyError: If no open position with the given order id exists.
        """
        for idx, position in enumerate(self.open_positions):
            if position.order_id == order_id:
                closed = self.open_positions.pop(idx)
                self.closed_positions.append(closed)
                return closed
        raise KeyError(f"No open position found for order id {order_id!r}.")

    def get_open_position(self, order_id: str) -> PaperOrder | None:
        """Return the open PaperOrder with the given order id, or None."""
        for position in self.open_positions:
            if position.order_id == order_id:
                return position
        return None

    def has_open_position(self, symbol: str) -> bool:
        """Return True if any OPEN order uses the given symbol."""
        return any(
            position.symbol == symbol and position.status == PaperOrderStatus.OPEN
            for position in self.open_positions
        )

    @property
    def open_count(self) -> int:
        """Number of currently open positions."""
        return len(self.open_positions)

    @property
    def closed_count(self) -> int:
        """Number of closed positions."""
        return len(self.closed_positions)

    # --- Equity and balance -------------------------------------------

    def equity(self) -> float:
        """Compute current account equity.

        Equity = current_balance + sum of unrealized PnL across all
        open positions. Closed positions have already settled into
        current_balance via :meth:`add_realized_pnl`, so they are not
        counted again here.

        Returns:
            Total account value including floating unrealized gains/losses.
        """
        unrealized = sum(
            position.unrealized_pnl_for(position.entry_price)
            for position in self.open_positions
        )
        return self.current_balance + unrealized

    def add_realized_pnl(self, realized_pnl: float) -> None:
        """Add realized PnL from a closed position to the current balance.

        This is the settlement step that happens when a position closes.
        It updates ``current_balance`` but does not touch open positions
        or closed positions lists — those are managed by
        :meth:`close_position` and the broker.

        Args:
            realized_pnl: Realized profit (positive) or loss (negative)
                from a closed position.
        """
        self.current_balance += float(realized_pnl)

    # --- Convenience queries ------------------------------------------

    def open_position_for_symbol(self, symbol: str) -> PaperOrder | None:
        """Return the open position for a symbol, or None if none exists.

        When multiple open positions exist for the same symbol the first
        one encountered is returned. In practice the paper broker avoids
        duplicate symbols via its placement rules.
        """
        for position in self.open_positions:
            if position.symbol == symbol and position.is_open():
                return position
        return None

    def closed_positions_for_symbol(self, symbol: str) -> List[PaperOrder]:
        """Return closed positions for a symbol, most recent first."""
        return [
            p for p in self.closed_positions if p.symbol == symbol
        ]
