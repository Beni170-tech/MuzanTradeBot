"""Daily loss circuit breaker for paper trading.

Sprint 8A: Halt new paper trades once the paper account's loss for the
current local calendar day reaches (or falls below) a configured daily
loss limit.

Rules:
    - "Today" is the local calendar day derived from ``datetime.now()``,
      matching the naive timestamps used across the paper order and
      journal records.
    - The day's loss has two parts. Settled PnL is the sum of
      ``realized_pnl`` over closed positions whose ``exit_at`` falls on
      that day. Unsettled PnL is the sum of realized PnL for every open
      position at its recorded market mark. Both count, so a day that is
      only bleeding on open positions is visible before those positions
      are closed. Realized profits and floating gains offset losses,
      exactly as realized profits always have.
    - A missing or unusable mark on an open position is *not* treated as
      zero loss. The day cannot be measured, so the brake fails closed:
      new entries are halted until every open position has a real mark.
      ``unpriced_symbols`` names the positions responsible.
    - The brake is triggered when the combined day figure is
      ``<= -limit``: reaching the daily loss budget stops new entries
      for the rest of the day. A limit ``<= 0`` disables the brake.

The daily risk manager is a read-only analysis tool. It never places,
closes, or modifies orders, and it needs no persistent state: the rule
is re-evaluated from the account on every call, so the brake resets
automatically at the next calendar day. It only prevents *new entries*
— managing and closing existing open positions is never affected.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trader.paper.order import PaperOrder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure per-day calculation helpers
# ---------------------------------------------------------------------------

def trade_day(timestamp: str | None) -> str:
    """Extract the ``YYYY-MM-DD`` day label from a naive timestamp.

    Shared by the daily loss brake and the performance report so both
    read exit dates identically.

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


# Backwards-compatible private alias for existing internal callers.
_day_of = trade_day


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


def usable_mark(position: object) -> float | None:
    """Return the position's recorded market mark, or None when unusable.

    Only a finite, strictly positive number recorded on the position
    counts as a real mark. A missing mark, a boolean, a string, zero,
    NaN or infinity is unusable: the position's floating result cannot
    be measured and must not be assumed to be zero.

    Args:
        position: An open paper position.

    Returns:
        The mark price as a float, or None when there is no usable mark.
    """
    value = getattr(position, "mark_price", None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value) or value <= 0:
            return None
    except OverflowError:
        return None
    return float(value)


def open_unrealized_pnl(
    positions: "Sequence[PaperOrder]",
) -> tuple[float, tuple[str, ...]]:
    """Sum floating PnL across open positions at their recorded marks.

    A position without a usable mark is never counted as zero: its
    symbol is returned in the unpriced tuple so callers can fail closed
    instead of pretending the loss does not exist.

    Args:
        positions: The account's open paper positions.

    Returns:
        A ``(total, unpriced_symbols)`` pair. ``total`` sums
        ``unrealized_pnl_for(mark)`` over every position with a usable
        mark; ``unpriced_symbols`` names the positions that could not be
        valued.
    """
    total = 0.0
    unpriced: list[str] = []

    for position in positions:
        symbol = str(getattr(position, "symbol", "?"))
        mark = usable_mark(position)
        if mark is None:
            unpriced.append(symbol)
            continue

        pnl_for = getattr(position, "unrealized_pnl_for", None)
        if not callable(pnl_for):
            unpriced.append(symbol)
            continue
        try:
            value = float(pnl_for(mark))
        except (TypeError, ValueError, OverflowError):
            unpriced.append(symbol)
            continue
        if not math.isfinite(value):
            unpriced.append(symbol)
            continue
        total += value

    return total, tuple(unpriced)


def loss_limit_reached(net_pnl: float, limit: float) -> bool:
    """Check whether the day's net PnL has reached the limit.

    ``net_pnl`` is the day's combined figure: settled PnL plus floating
    PnL at recorded marks.


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
        reason: Why entries are blocked when ``allowed`` is False and the
            cause is not a breached limit (for example an open position
            with no usable mark). Empty otherwise.
    """

    allowed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class DailyRiskMetrics:
    """Daily loss snapshot for display and enforcement.

    Attributes:
        day: The ``YYYY-MM-DD`` day label these metrics describe.
        realized_pnl_today: Net realized PnL settled today.
        loss_limit: The configured daily loss limit. 0 when disabled.
        loss_usage_percent: Today's loss as a percentage of the limit.
            0.0 when no limit is configured. Computed from
            ``day_loss``, so floating losses count too.
        remaining_budget: Loss budget remaining before the brake trips.
            0.0 when no limit is configured.
        breached: True when new entries are halted: the day's combined
            loss has reached the limit, or an open position has no
            usable mark so the day cannot be measured.
        open_unrealized_pnl: Floating PnL across open positions at their
            recorded marks. 0.0 when there are no open positions.
        day_loss: ``realized_pnl_today + open_unrealized_pnl``; the figure
            the brake actually tests.
        unpriced_symbols: Open positions that have no usable mark, so
            their floating result is unknown. Non-empty means the brake
            failed closed.
        day_unmeasurable: True when the account could not report its open
            positions at all, so the floating result is unknown. Also
            fail-closed.
    """

    day: str
    realized_pnl_today: float
    loss_limit: float
    loss_usage_percent: float
    remaining_budget: float
    breached: bool
    open_unrealized_pnl: float = 0.0
    day_loss: float = 0.0
    unpriced_symbols: tuple[str, ...] = ()
    day_unmeasurable: bool = False


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

    def _open_positions(self) -> "Sequence[PaperOrder] | None":
        """The broker's open positions, or None when it cannot report them.

        ``None`` means the floating side of the day is unknown, which the
        brake treats as fail-closed rather than as zero loss.
        """
        getter = getattr(self._broker, "get_open_positions", None)
        if not callable(getter):
            return None
        try:
            return tuple(getter() or ())
        except Exception:
            logger.warning(
                "Could not read open paper positions for the daily loss "
                "brake; treating the day's floating result as unknown.",
                exc_info=True,
            )
            return None

    def _net_pnl_today(self, day: str) -> float:
        """Net realized PnL settled on the given day."""
        return realized_pnl_on_day(self._closed_positions(), day)

    def _day_state(
        self, day: str
    ) -> tuple[float, float, tuple[str, ...], bool]:
        """Return ``(realized, floating, unpriced, measurable)`` for a day.

        ``measurable`` is False when the account could not report its open
        positions, so the floating side of the day is unknown rather than
        zero.
        """
        realized = self._net_pnl_today(day)
        positions = self._open_positions()
        if positions is None:
            return realized, 0.0, (), False
        floating, unpriced = open_unrealized_pnl(positions)
        return realized, floating, unpriced, True

    def evaluate(self, day: str | None = None) -> DailyLossCheck:
        """Evaluate whether new trades are still allowed today.

        When the limit is disabled (<= 0), this always allows trades and
        performs no account reads.

        The brake tests settled plus floating PnL. When an open position
        has no usable mark, the day cannot be measured and entries are
        blocked with a reason instead of assuming the floating loss is
        zero.

        Args:
            day: The ``YYYY-MM-DD`` day to evaluate. Defaults to the
                current local calendar day.

        Returns:
            A DailyLossCheck describing whether a new trade may be
            placed. ``allowed`` is False once the day's combined loss
            reaches ``-limit``, or when an open position cannot be
            priced.
        """
        if self._daily_loss_limit <= 0:
            return DailyLossCheck(allowed=True)

        realized, floating, unpriced, measurable = self._day_state(
            day or self._default_day()
        )
        if not measurable:
            return DailyLossCheck(
                allowed=False,
                reason=(
                    "the paper account could not report open positions, so "
                    "the day's floating loss is unknown"
                ),
            )
        if unpriced:
            return DailyLossCheck(
                allowed=False,
                reason=(
                    "open position has no usable market mark: "
                    + ", ".join(unpriced)
                ),
            )
        return DailyLossCheck(
            allowed=not loss_limit_reached(
                realized + floating, self._daily_loss_limit
            )
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

        floating = 0.0
        unpriced: tuple[str, ...] = ()
        measurable = True
        if limit > 0:
            _, floating, unpriced, measurable = self._day_state(resolved_day)

        combined = net_pnl + floating

        if limit > 0:
            loss_today = max(0.0, -combined)
            usage_percent = (loss_today / limit) * 100.0
            remaining = max(0.0, limit + combined)
        else:
            usage_percent = 0.0
            remaining = 0.0

        return DailyRiskMetrics(
            day=resolved_day,
            realized_pnl_today=net_pnl,
            loss_limit=limit,
            loss_usage_percent=usage_percent,
            remaining_budget=remaining,
            breached=(
                loss_limit_reached(combined, limit)
                or (limit > 0 and bool(unpriced))
                or (limit > 0 and not measurable)
            ),
            open_unrealized_pnl=floating,
            day_loss=combined,
            unpriced_symbols=unpriced,
            day_unmeasurable=limit > 0 and not measurable,
        )
