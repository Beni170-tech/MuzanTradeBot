"""
MuzanTradeBot - Background Paper Position Monitor (Sprint 9B)

Monitors all open paper-trading positions automatically in the
background and closes them when stop-loss or take-profit levels are
hit, and updates trailing stops when appropriate.

Every cycle obtains a fresh market price for each position through the
existing CoinGecko integration. If a reliable price cannot be obtained
or is invalid, that position is skipped for the cycle and retried
later; the entry price is never used as a substitute.

The monitor runs as an asyncio task in the Telegram event loop on a
configurable interval, so it never blocks Telegram polling. It reuses
the existing paper-trading SL/TP and trailing-stop logic from the
handler layer rather than duplicating it.

Guardian Mode: the monitor never bypasses any risk check. It only
performs exit actions that the existing /analyze flow already performs
for interactive requests.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Callable

from market.analyzer import get_current_price

if TYPE_CHECKING:
    from alerts.manager import AlertManager
    from trader.paper.service import PaperTradingService

logger = logging.getLogger(__name__)

# Type aliases for the handler functions we depend on.
_SltpCheckFn = Callable[..., object]
_TrailingFn = Callable[..., object | None]


def _default_sltp_check():
    """Lazily import the real _before_analysis_paper_check from handlers."""
    from telegram_bot.handlers import _before_analysis_paper_check as _fn
    return _fn


def _default_trailing():
    """Lazily import the real _apply_trailing_stop from handlers."""
    from telegram_bot.handlers import _apply_trailing_stop as _fn
    return _fn


class PaperPositionMonitor:
    """Background monitor for open paper-trading positions.

    The monitor periodically checks every open paper position for
    stop-loss / take-profit breaches and trailing-stop updates,
    using the existing handler logic from :mod:`telegram_bot.handlers`.

    The monitor runs as an asyncio task in the Telegram event loop, so
    it never blocks Telegram polling. AlertManager coroutine methods
    are properly awaited within the same event loop.

    The monitor is designed to be started and stopped cleanly, to be
    safe to call start() multiple times (duplicate-start protection),
    and to shut down predictably on stop().

    The monitoring loop is resilient:
        - A failure on one position does not stop other positions from
          being monitored in the same cycle.
        - An unexpected error during a full cycle is logged and the
          next cycle runs normally.

    For testing, the SL/TP check and trailing-stop functions can be
    overridden via the constructor so tests can inject mocks without
    patching module internals.
    """

    def __init__(
        self,
        service: "PaperTradingService",
        alert_manager: "AlertManager",
        interval_seconds: float = 45.0,
        *,
        sltp_check: _SltpCheckFn | None = None,
        trailing_fn: _TrailingFn | None = None,
        price_provider: Callable[[str], float | None] | None = None,
    ) -> None:
        """Initialize the monitor.

        Args:
            service: The PaperTradingService to monitor and close
                positions through.
            alert_manager: The AlertManager used to push SL/TP and
                trailing-stop alerts when the monitor triggers them.
            interval_seconds: How long to wait between monitoring
                cycles. Must be positive.
            sltp_check: Optional override for the SL/TP check function.
                Defaults to the real ``_before_analysis_paper_check``
                from the handler layer.
            trailing_fn: Optional override for the trailing-stop function.
                Defaults to the real ``_apply_trailing_stop`` from the
                handler layer.
            price_provider: Optional synchronous callable returning a
                fresh market price for a symbol, or None when no reliable
                price is available. It is invoked in a worker thread so a
                slow lookup never blocks the event loop. Defaults to
                :func:`market.analyzer.get_current_price`.
        """
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")

        self._service = service
        self._alerts = alert_manager
        self._interval = float(interval_seconds)

        # Allow tests to inject mock SL/TP and trailing-stop functions.
        self._sltp_check = sltp_check or _default_sltp_check()
        self._trailing_fn = trailing_fn or _default_trailing()

        # Market price source. The default reuses the existing CoinGecko
        # integration; tests inject a deterministic provider.
        self._price_provider = price_provider or get_current_price

        self._task: asyncio.Task[None] | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background monitoring loop as an asyncio task.

        This method is safe to call multiple times — duplicate starts
        are protected. The task is created in the current event loop.

        Note: When the monitor is managed by the Telegram bot, the task
        is created directly in the bot's async context rather than via
        this method, so that the task is properly attached to the running
        event loop.
        """
        if self._running:
            logger.info(
                "Paper position monitor already running; "
                "skipping duplicate start."
            )
            return

        self._running = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running; create a new one for testing purposes.
            # In production, the bot's event loop will be used.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._task = loop.create_task(self._run_loop(), name="PaperPositionMonitor")
        logger.info(
            "Paper position monitor started (interval=%.1fs).",
            self._interval,
        )

    def stop(self) -> None:
        """Signal the monitor to stop. The task is cancelled and awaited.

        Calls :meth:`stop` are idempotent. When the monitor is not
        running, this is a no-op.

        Note: In production, the bot's shutdown path handles task
        cancellation directly. This method is primarily for tests and
        direct usage.
        """
        if not self._running:
            return

        self._running = False
        task = self._task
        self._task = None

        if task is not None and not task.done():
            task.cancel()
            try:
                # Try to run the loop to completion; this may fail if
                # no loop is running, which is fine.
                asyncio.get_event_loop().run_until_complete(task)
            except (RuntimeError, asyncio.CancelledError):
                pass
            except Exception:
                logger.exception("Monitor task error during stop.")

        logger.info("Paper position monitor stopped.")

    @property
    def is_running(self) -> bool:
        """Return True when the monitor is currently running."""
        return self._running

    # ------------------------------------------------------------------
    # Internal async loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Main monitoring loop, run as an asyncio task."""
        logger.info("Paper position monitor loop entered.")
        while self._running:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                logger.info("Paper position monitor loop cancelled.")
                break
            except Exception:
                logger.exception(
                    "Paper position monitor cycle failed; continuing."
                )
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                logger.info("Paper position monitor loop cancelled during sleep.")
                break
        logger.info("Paper position monitor loop exited.")

    async def _run_cycle(self) -> None:
        """Run one monitoring cycle over all open positions.

        A failure on one position does not prevent other positions from
        being monitored in the same cycle.
        """
        open_positions = self._service.open_positions()
        if not open_positions:
            return

        for position in open_positions:
            try:
                await self._monitor_position(position)
            except Exception:
                logger.exception(
                    "Failed to monitor position %s (%s); "
                    "continuing with other positions.",
                    position.order_id,
                    position.symbol,
                )

    async def _market_price(self, position: "PaperOrder") -> float | None:
        """Return a fresh, validated market price for a position.

        The price provider is synchronous (it performs an HTTP request),
        so it runs in a worker thread via ``asyncio.to_thread``. This
        keeps a slow market-data request from blocking the Telegram
        event loop and delaying polling.

        Failures and invalid values return None. The caller must skip the
        position for this cycle; the entry price is never used as a
        fallback, because a stale price would make stop-loss and
        take-profit checks meaningless.

        Args:
            position: The open PaperOrder being monitored.

        Returns:
            A positive finite float price, or None when no reliable price
            could be obtained.
        """
        symbol = position.symbol
        try:
            price = await asyncio.to_thread(self._price_provider, symbol)
        except Exception:
            logger.warning(
                "Market price lookup failed for %s; skipping this "
                "position for this cycle.",
                symbol,
                exc_info=True,
            )
            return None

        if isinstance(price, bool) or not isinstance(price, (int, float)):
            logger.warning(
                "Market price for %s is not a usable number (%r); "
                "skipping this position for this cycle.",
                symbol,
                price,
            )
            return None
        if not math.isfinite(price) or price <= 0:
            logger.warning(
                "Market price for %s is invalid (%r); skipping this "
                "position for this cycle.",
                symbol,
                price,
            )
            return None

        return float(price)

    async def _monitor_position(self, position: "PaperOrder") -> None:
        """Monitor a single open position for SL/TP and trailing-stop.

        A fresh market price is required for every decision. When no
        reliable price is available, the position is skipped for this
        cycle and retried on the next one.

        Args:
            position: The open PaperOrder to monitor.
        """
        symbol = position.symbol

        price = await self._market_price(position)
        if price is None:
            logger.warning(
                "Skipping paper position %s (%s): no reliable market "
                "price available; it will be retried next cycle.",
                position.order_id,
                symbol,
            )
            return

        # 1. Check stop-loss / take-profit first (Guardian Mode priority).
        closed = self._sltp_check(self._service, symbol, price)
        if closed is not None:
            # The helper already saved the account and closed the position.
            # Push the matching alert so the user is notified even when
            # the close happens in the background.
            from trader.paper.order import PaperOrderStatus

            exit_value = closed.status.value
            if exit_value == PaperOrderStatus.STOPPED.value:
                await self._alerts.stop_loss_triggered(closed)
            elif exit_value == PaperOrderStatus.TAKEN_PROFIT.value:
                await self._alerts.take_profit_triggered(closed)
            return

        # 2. Apply trailing stop (if applicable). When it changes the
        #    stop, push an alert.
        trail_result = self._trailing_fn(self._service, symbol, price)
        if trail_result is not None:
            changed, old_stop, new_stop = trail_result
            if changed:
                await self._alerts.trailing_stop_updated(
                    symbol, old_stop, new_stop
                )

    # ------------------------------------------------------------------
    # Monitoring without alerts (for tests that do not wire alerts)
    # ------------------------------------------------------------------

    async def monitor_open_positions(self) -> int:
        """Run a single monitoring cycle over all open positions.

        This is a convenience method for tests and for callers that want
        to run a single cycle without waiting for the background loop.
        It returns the number of positions that were checked.

        Returns:
            The number of open positions that were monitored during this
            cycle.
        """
        open_positions = self._service.open_positions()
        if not open_positions:
            return 0
        for position in open_positions:
            try:
                await self._monitor_position(position)
            except Exception:
                logger.exception(
                    "Failed to monitor position %s (%s); "
                    "continuing with other positions.",
                    position.order_id,
                    position.symbol,
                )
        return len(open_positions)
