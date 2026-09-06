"""
MuzanTradeBot - Paper Order Model (Sprint 5A/5D)

Dataclass and status enums for a paper order.

Sprint 5A scope only:
- This module defines the data shape of a paper order and its
  lifecycle statuses. It does not place, fill, update, persist,
  or execute anything. All execution logic is out of scope.

Sprint 5D scope:
- Unrealized PnL calculation helper.
- Exit recording helper for the broker close path.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from trader.trade import Trade


class PaperOrderStatus(str, Enum):
    """Lifecycle status of a paper order/position."""

    PENDING = "PENDING"
    OPEN = "OPEN"
    STOPPED = "STOPPED"
    TAKEN_PROFIT = "TAKEN_PROFIT"
    CLOSED_MANUAL = "CLOSED_MANUAL"
    CLOSED_BY_MARKET = "CLOSED_BY_MARKET"


class PaperOrderSide(str, Enum):
    """Direction of a paper order."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class PaperOrder:
    """One paper order/position tracked by the paper account.

    PaperOrder is the execution-time model. It is distinct from
    Trade, which is the analysis-time recommendation. A PaperOrder
    is created when an approved Trade is submitted to the paper
    broker (out of scope for Sprint 5A).

    Sprint 5A scope:
        This dataclass only carries state. No methods that place,
        fill, close, persist, or trigger exits are included here.

    Sprint 5D scope:
        Unrealized PnL and exit recording helpers are added so the
        broker and account can compute results without importing
        extra modules.
    """

    order_id: str
    symbol: str
    side: PaperOrderSide
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    status: PaperOrderStatus = PaperOrderStatus.PENDING
    trade: Optional[Trade] = None
    created_at: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    filled_at: Optional[str] = None
    exit_at: Optional[str] = None
    exit_price: Optional[float] = None
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    paper: bool = True
    notes: list[str] = field(default_factory=list)

    # --- PnL helpers --------------------------------------------------

    def unrealized_pnl_for(self, current_price: float) -> float:
        """Compute unrealized PnL at a given market price.

        For LONG positions PnL rises with price; for SHORT positions
        PnL falls as price rises. The result is in the account's base
        currency (e.g. USD), not in the asset's quote currency.

        Args:
            current_price: The current market price of the symbol.

        Returns:
            Unrealized profit (positive) or loss (negative) for the
            open position at ``current_price``.
        """
        if self.quantity <= 0 or self.entry_price <= 0:
            return 0.0
        price_diff = current_price - self.entry_price
        if self.side == PaperOrderSide.SHORT:
            price_diff = -price_diff
        return price_diff * self.quantity

    def record_exit(
        self,
        exit_price: float,
        status: PaperOrderStatus,
        exit_at: str | None = None,
    ) -> None:
        """Record an exit for this order and compute realized PnL.

        Side effects:
            - Sets ``exit_price``, ``exit_at``, ``realized_pnl``, and
              ``status`` on this order.

        Args:
            exit_price: The price at which the position was closed.
            status: The terminal status for this order (e.g.
                ``PaperOrderStatus.TAKEN_PROFIT``,
                ``PaperOrderStatus.STOPPED``, or
                ``PaperOrderStatus.CLOSED_MANUAL``).
            exit_at: ISO-style timestamp for the exit. When ``None``,
                the current time is used.
        """
        self.exit_price = float(exit_price)
        self.status = status
        self.exit_at = exit_at or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.realized_pnl = self._realized_pnl_for(self.exit_price)

    def _realized_pnl_for(self, exit_price: float) -> float:
        """Compute realized PnL for a close at ``exit_price``.

        Mirrors ``unrealized_pnl_for`` but is used internally when
        recording an exit so the two calculations stay consistent.

        Args:
            exit_price: The closing price.

        Returns:
            Realized profit (positive) or loss (negative).
        """
        return self.unrealized_pnl_for(exit_price)

    # --- Convenience queries ------------------------------------------

    def is_open(self) -> bool:
        """Return True when the order is currently open."""
        return self.status == PaperOrderStatus.OPEN

    def is_closed(self) -> bool:
        """Return True when the order has been closed."""
        return self.status not in (PaperOrderStatus.PENDING, PaperOrderStatus.OPEN)

    def is_terminal(self) -> bool:
        """Return True when the order is in a terminal (non-open) state."""
        return not self.is_open()
