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

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from trader.paper.rules import is_valid_exit_price
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
    # Last validated market price recorded for this position. ``None`` means
    # no market mark has been recorded yet; the entry price is never used as
    # a substitute, so an unmarked position contributes 0.0 unrealized PnL.
    mark_price: Optional[float] = None
    # UTC timestamp of ``mark_price`` as an ISO 8601 string. ``None`` means
    # the mark has no verified age (for example a legacy saved account), so
    # sizing must not treat that mark as fresh.
    mark_at: Optional[str] = None
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
        if isinstance(current_price, bool) or not isinstance(
            current_price, (int, float)
        ):
            return 0.0
        if not math.isfinite(current_price) or current_price <= 0:
            return 0.0
        price_diff = current_price - self.entry_price
        if self.side == PaperOrderSide.SHORT:
            price_diff = -price_diff
        return price_diff * self.quantity

    def mark_to_market(
        self, current_price: float, now: datetime | None = None
    ) -> bool:
        """Record a validated market price and refresh unrealized PnL.

        Only a finite, strictly positive number is accepted. A rejected
        value leaves any previously recorded mark and timestamp untouched,
        so a bad market read can never refresh the age of a stale mark.

        Args:
            current_price: The market price to record for this position.
            now: Optional UTC timestamp for the mark. Defaults to the
                current UTC time. Naive datetimes are treated as UTC so
                stored timestamps always describe an absolute instant.

        Returns:
            True when the mark was recorded, False when the price was not
            usable and nothing changed.
        """
        if isinstance(current_price, bool) or not isinstance(
            current_price, (int, float)
        ):
            return False
        if not math.isfinite(current_price) or current_price <= 0:
            return False

        timestamp = now or datetime.now(timezone.utc)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        self.mark_price = float(current_price)
        self.mark_at = timestamp.astimezone(timezone.utc).isoformat()
        self.unrealized_pnl = self.unrealized_pnl_for(self.mark_price)
        return True

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
            exit_price: The price at which the position was closed. Must
                be a finite, strictly positive number.
            status: The terminal status for this order (e.g.
                ``PaperOrderStatus.TAKEN_PROFIT``,
                ``PaperOrderStatus.STOPPED``, or
                ``PaperOrderStatus.CLOSED_MANUAL``).
            exit_at: ISO-style timestamp for the exit. When ``None``,
                the current time is used.

        Raises:
            ValueError: If ``exit_price`` is not a finite, strictly
                positive number. Nothing is changed in that case.
        """
        if not is_valid_exit_price(exit_price):
            raise ValueError(
                "exit_price must be a finite, strictly positive number, "
                f"got {exit_price!r}."
            )

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


def paper_exit_status(order: object, default: str = "") -> str:
    """Return the exit-status label for a closed paper order.

    The canonical attribute on PaperOrder is ``status``: ``record_exit``
    sets ``status`` to the terminal PaperOrderStatus when a position
    closes. Some historical call sites and test mocks used an
    ``exit_status`` attribute that never existed on real orders, which
    made those code paths crash (or silently abort) on production data.

    This function is the single compatibility point for both shapes:
    it prefers ``exit_status`` when present (legacy/mock shape), falls
    back to ``status`` (real PaperOrder shape), and returns ``default``
    when neither carries a usable value. A non-string ``exit_status``
    value is treated as absent (this also makes the helper robust
    against auto-created mock attributes).

    Args:
        order: A PaperOrder or compatible object.
        default: Value returned when no usable status is found.

    Returns:
        The terminal status label, e.g. ``"STOPPED"`` or
        ``"TAKEN_PROFIT"``.
    """
    for attr in ("exit_status", "status"):
        status = getattr(order, attr, None)
        value = getattr(status, "value", status)
        if isinstance(value, str) and value:
            return value
    return default


def paper_exit_date(order: object, format: str = "%Y-%m-%d %H:%M", default: str = "N/A") -> str:
    """Return the formatted exit date for a closed paper order.

    PaperOrder.exit_at is always a naive timestamp *string*
    (``"YYYY-MM-DD HH:MM:SS"``, the format written by the paper order
    and journal layers), yet some display code called
    ``exit_at.strftime`` directly, which only works on datetime
    objects and crashes on production data. This helper accepts both
    shapes: datetime-like objects (via ``strftime``) and plain strings
    (truncated to 16 characters, i.e. ``YYYY-MM-DD HH:MM``).

    Args:
        order: A PaperOrder or compatible object.
        format: strftime format used when ``exit_at`` is datetime-like.
        default: Value returned when ``exit_at`` is missing or empty.

    Returns:
        The formatted exit timestamp, or ``default`` when unavailable.
    """
    exit_at = getattr(order, "exit_at", None)
    if not exit_at:
        return default
    strftime = getattr(exit_at, "strftime", None)
    if callable(strftime):
        return strftime(format)
    if isinstance(exit_at, str):
        return exit_at[:16]
    return default
