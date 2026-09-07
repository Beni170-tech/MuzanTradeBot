"""Daily loss circuit breaker for paper trading.

Sprint 8A: Halt new paper trades once the paper account's net realized
PnL for the current local calendar day reaches (or falls below) a
configured daily loss limit.

Rules:
    - "Today" is the local calendar day derived from ``datetime.now()``,
      matching the naive timestamps used across the paper order and
      journal records.
    - Net realized PnL for the day is the sum of ``realized_pnl`` over
      closed positions whose ``exit_at`` falls on that day. Realized
      profits reduce the day's loss, realized losses increase it, and
      unrealized PnL on open positions is ignored — only settled
      results count.
    - The brake is triggered when the day's net realized PnL is
      ``<= -limit``: reaching the daily loss budget stops new entries
      for the rest of the day. A limit ``<= 0`` disables the brake.

The daily risk manager is a read-only analysis tool. It never places,
closes, or modifies orders, and it needs no persistent state: the rule
is re-evaluated from the closed-position history on every call, so the
brake resets automatically at the next calendar day. It only prevents
*new entries* — managing and closing existing open positions is never
affected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trader.paper.order import PaperOrder


# ---------------------------------------------------------------------------
# Pure per-day calculation helpers
# ---------------------------------------------------------------------------

def _day_of(timestamp: str | None) -> str:
    """Extract the ``YYYY-MM-DD`` day label from a naive timestamp.

    Args:
        timestamp: A timestamp string such as ``2026-09-07 14:30:00``
            (the format written by the paper order and journal layers),
            or None for an order that was never stamped.

    Returns:
        The first 10 characters (the date) when the timestamp is a
        string of sufficient length, otherwise an empty string so the
        order is safely excluded from every day's totals.
    """
    if isinstance(timestamp, str) and len(timestamp) >= 10:
        return timestamp[:10]
    return ""


def realized_pnl_on_day(
    closed_positions: "Sequence[PaperOrder]", day: str
) -> float:
    """Sum the net realized PnL of positions closed on a given day.

    Args:
        closed_positions: The account's closed paper orders.
        day: The ``YYYY-MM-DD`` day label to filter by.

    Returns:
        Net realized PnL (profit positive, loss negative) across all
        orders whose exit date matches ``day``. Orders without an
        ``exit_at`` timestamp are excluded.
    """
    return sum(
        order.realized_pnl
        for order in closed_positions
        if _day_of(order.exit_at) == day
    )


def loss_limit_reached(net_pnl: float, limit: float) -> bool:
    """Check whether the day's net realized PnL has reached the limit.

    The brake engages the moment the daily loss budget is fully spent:
    a net PnL of exactly ``-limit`` stops trading, matching how
    professional daily loss limiters behave.

    Args:
        net_pnl: Net realized PnL for the day (negative when losing).
        limit: The configured daily loss limit. When <= 0 the brake is
            disabled and never triggers.

    Returns:
        True when the loss limit is configured and ``net_pnl`` is
        ``<= -limit``, False otherwise.
    """
    if limit <= 0:
        return False
    return net_pnl <= -limit


def remaining_loss_budget(net_pnl: float, limit: float) -> float:
    """Compute the loss budget still available before the brake trips.

    A profitable day increases the budget (profits earned today may be
    lost again before the daily loss limit is reached); once the limit
    is reached the remaining budget floors at zero.

    Args:
        net_pnl: Net realized PnL for the day.
        limit: The configured daily loss limit. When <= 0, returns 0.0.

    Returns:
        The remaining loss budget: ``limit + net_pnl`` clamped at zero,
        or 0.0 when the limit is disabled.
    """
    if limit <= 0:
        return 0.0
    return max(0.0, limit + net_pnl)


# ---------------------------------------------------------------------------
# Result and metrics dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DailyLossCheck:
    """Outcome of evaluating the daily loss brake.

    Attributes:
        allowed: True when a new trade may be placed (the brake is
            disabled, or today's losses have not reached the limit).
    """

    allowed: bool


@dataclass(frozen=True, slots=True)
class DailyRiskMetrics:
    """Daily loss snapshot for display and enforcement.

    Attributes:
        day: The ``YYYY-MM-DD`` day label these metrics describe.
        realized_pnl_today: Net realized PnL settled today.
        loss_limit: The configured daily loss limit. 0 when disabled.
        loss_usage_percent: Today's loss as a percentage of the limit.
            0.0 when no limit is configured.
        remaining_budget: Loss budget remaining before the brake trips.
            0.0 when no limit is configured.
        breached: True when the day's loss has reached the limit and
            new trades are halted.
    """

    day: str
    realized_pnl_today: float
    loss_limit: float
    loss_usage_percent: float
    remaining_budget: float
    breached: bool


# ---------------------------------------------------------------------------
# DailyRiskManager
# ---------------------------------------------------------------------------

class DailyRiskManager:
    """Daily loss brake for paper trading.

    Wraps a PaperBroker plus a configurable daily loss limit and
    answers whether new paper trades may still be opened today. The
    brake is entry-only: it never interferes with managing or closing
    existing open positions.

    When the limit is 0 or negative, the brake is disabled and all
    checks pass unconditionally. This preserves backward compatibility
    — existing installations without a configured limit are unaffected.

    No state is persisted: each call re-derives today's realized PnL
    from the broker's closed-position history, so the brake resets
    automatically at the next local calendar day.

    Usage:
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        if manager.evaluate().allowed:
            broker.place_order(trade)
        else:
            # reject the trade until tomorrow
    """

    def __init__(self, broker, daily_loss_limit: float = 0.0) -> None:
        """Initialize the daily risk manager.

        Args:
            broker: The paper broker to monitor. Its account's closed
                positions supply the realized-PnL history.
            daily_loss_limit: Maximum net realized loss allowed per
                calendar day. When <= 0, the brake is disabled.
        """
        self._broker = broker
        self._daily_loss_limit = daily_loss_limit

    @property
    def daily_loss_limit(self) -> float:
        """The configured maximum daily net realized loss."""
        return self._daily_loss_limit

    def _default_day(self) -> str:
        """The current local calendar day label (``YYYY-MM-DD``)."""
        return datetime.now().strftime("%Y-%m-%d")

    def _closed_positions(self) -> "Sequence[PaperOrder]":
        """The broker's closed-position history."""
        return self._broker.account.closed_positions

    def _net_pnl_today(self, day: str) -> float:
        """Net realized PnL settled on the given day."""
        return realized_pnl_on_day(self._closed_positions(), day)

    def evaluate(self, day: str | None = None) -> DailyLossCheck:
        """Evaluate whether new trades are still allowed today.

        When the limit is disabled (<= 0), this always allows trades.

        Args:
            day: The ``YYYY-MM-DD`` day to evaluate. Defaults to the
                current local calendar day.

        Returns:
            A DailyLossCheck describing whether a new trade may be
            placed. ``allowed`` is False once the day's net realized
            PnL reaches ``-limit``.
        """
        if self._daily_loss_limit <= 0:
            return DailyLossCheck(allowed=True)
        net_pnl = self._net_pnl_today(day or self._default_day())
        return DailyLossCheck(
            allowed=not loss_limit_reached(net_pnl, self._daily_loss_limit)
        )

    def is_halted(self, day: str | None = None) -> bool:
        """Return True when the daily loss brake is currently tripped.

        Args:
            day: The ``YYYY-MM-DD`` day to evaluate. Defaults to the
                current local calendar day.

        Returns:
            True when new trades are halted for the day, False when
            they are still allowed (or the brake is disabled).
        """
        return not self.evaluate(day).allowed

    def metrics(self, day: str | None = None) -> DailyRiskMetrics:
        """Build a snapshot of the daily loss state.

        Args:
            day: The ``YYYY-MM-DD`` day to describe. Defaults to the
                current local calendar day.

        Returns:
            A DailyRiskMetrics with the day's realized PnL, limit,
            usage percentage, remaining budget, and breach status.
        """
        resolved_day = day or self._default_day()
        net_pnl = self._net_pnl_today(resolved_day)
        limit = self._daily_loss_limit

        if limit > 0:
            loss_today = max(0.0, -net_pnl)
            usage_percent = (loss_today / limit) * 100.0
            remaining = max(0.0, limit + net_pnl)
        else:
            usage_percent = 0.0
            remaining = 0.0

        return DailyRiskMetrics(
            day=resolved_day,
            realized_pnl_today=net_pnl,
            loss_limit=limit,
            loss_usage_percent=usage_percent,
            remaining_budget=remaining,
            breached=loss_limit_reached(net_pnl, limit),
        )
