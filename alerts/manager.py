"""
MuzanTradeBot - Alert manager (Sprint 9A)

Composes the pure message builders in :mod:`alerts.messages` with a
notifier from :mod:`alerts.notifier` and exposes one semantic method
per alert event:

    - order_placed
    - stop_loss_triggered
    - take_profit_triggered
    - trailing_stop_updated
    - trade_rejected
    - daily_loss_breaker
    - max_open_positions_reached

The manager is fail-safe by design: neither message building nor
delivery may ever raise into the trading flow. A disabled manager
skips everything, matching the ALERTS_ENABLED=false behavior.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from alerts.messages import (
    daily_loss_breaker_message,
    max_open_positions_message,
    order_placed_message,
    stop_loss_triggered_message,
    take_profit_triggered_message,
    trailing_stop_updated_message,
    trade_rejected_message,
)
from alerts.notifier import NullNotifier, build_notifier_from_settings

if TYPE_CHECKING:
    from trader.paper.daily_risk import DailyRiskMetrics
    from trader.paper.max_open_positions import MaxOpenPositionsResult
    from trader.paper.order import PaperOrder
    from trader.trade import Trade

logger = logging.getLogger(__name__)


class AlertManager:
    """Fan-out point for trading alerts.

    Wraps a notifier and builds the right message for each alert
    event. Every method is safe to call unconditionally: failures are
    logged and swallowed, and a disabled manager does nothing.
    """

    def __init__(self, notifier: Any, enabled: bool = True) -> None:
        """Initialize the manager.

        Args:
            notifier: Any object with an async ``send(text)`` method
                (TelegramNotifier or NullNotifier).
            enabled: Master switch. When False, no alerts are built or
                sent.
        """
        self._notifier = notifier
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether alerts are currently enabled."""
        return self._enabled

    async def _emit(self, text: str) -> None:
        """Send one alert text, swallowing every failure.

        Args:
            text: The alert text to deliver.
        """
        if not self._enabled:
            return
        try:
            await self._notifier.send(text)
        except Exception:
            logger.exception("Alert delivery failed; alert dropped.")

    def _build(self, builder, *args) -> str | None:
        """Build an alert text, swallowing every failure.

        Args:
            builder: The message builder function to call.
            *args: Arguments forwarded to the builder.

        Returns:
            The alert text, or None when building failed (e.g. a
            malformed domain object).
        """
        try:
            return builder(*args)
        except Exception:
            logger.exception(
                "Alert message build failed for %s; alert dropped.",
                getattr(builder, "__name__", builder),
            )
            return None

    # ------------------------------------------------------------------
    # Alert events
    # ------------------------------------------------------------------

    async def order_placed(self, order: "PaperOrder") -> None:
        """Alert: a paper order was placed."""
        text = self._build(order_placed_message, order)
        if text is not None:
            await self._emit(text)

    async def stop_loss_triggered(self, order: "PaperOrder") -> None:
        """Alert: a position was closed by its stop loss."""
        text = self._build(stop_loss_triggered_message, order)
        if text is not None:
            await self._emit(text)

    async def take_profit_triggered(self, order: "PaperOrder") -> None:
        """Alert: a position was closed by its take profit."""
        text = self._build(take_profit_triggered_message, order)
        if text is not None:
            await self._emit(text)

    async def trailing_stop_updated(
        self, symbol: str, old_stop: float, new_stop: float
    ) -> None:
        """Alert: a position's trailing stop moved."""
        text = self._build(
            trailing_stop_updated_message, symbol, old_stop, new_stop
        )
        if text is not None:
            await self._emit(text)

    async def trade_rejected(self, trade: "Trade", reason: str) -> None:
        """Alert: Guardian Mode / a risk limit rejected a trade."""
        text = self._build(trade_rejected_message, trade, reason)
        if text is not None:
            await self._emit(text)

    async def daily_loss_breaker(self, metrics: "DailyRiskMetrics") -> None:
        """Alert: the daily loss breaker activated."""
        text = self._build(daily_loss_breaker_message, metrics)
        if text is not None:
            await self._emit(text)

    async def max_open_positions_reached(
        self, result: "MaxOpenPositionsResult"
    ) -> None:
        """Alert: the maximum open positions limit was reached."""
        text = self._build(max_open_positions_message, result)
        if text is not None:
            await self._emit(text)


def build_alert_manager(
    *, alerts_enabled: bool, chat_id: str | int | None, token: str | None
) -> AlertManager:
    """Build the AlertManager described by configuration values.

    Returns a disabled manager (backed by a NullNotifier) when alerts
    are disabled or unconfigured — never raises.

    Args:
        alerts_enabled: The ALERTS_ENABLED configuration value.
        chat_id: The TELEGRAM_ALERT_CHAT_ID configuration value.
        token: The TELEGRAM_BOT_TOKEN configuration value.

    Returns:
        An AlertManager ready for use.
    """
    notifier = build_notifier_from_settings(
        alerts_enabled=alerts_enabled, chat_id=chat_id, token=token
    )
    return AlertManager(notifier, enabled=alerts_enabled)
