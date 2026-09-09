"""
MuzanTradeBot - Alert notifiers (Sprint 9A)

Delivers alert texts to Telegram. Two implementations:

    - TelegramNotifier: wraps python-telegram-bot's Bot.send_message.
      A failed send is logged and swallowed — a broken notifier must
      never break trading.
    - NullNotifier: the no-op fallback used when alerts are disabled
      or unconfigured.

The module-level :func:`build_notifier_from_settings` factory returns
the right notifier from configuration values, and never raises.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _build_bot(token: str):
    """Build a python-telegram-bot Bot instance.

    Kept as an indirect import target so tests can fake bot
    construction without network access.
    """
    from telegram import Bot

    return Bot(token=token)


class TelegramNotifier:
    """Sends alert texts to a Telegram chat via Bot.send_message.

    The notifier is fail-safe by design: every failure is logged and
    swallowed so an alerts outage can never interrupt the trading
    flow.
    """

    def __init__(self, bot: Any, chat_id: str | int) -> None:
        """Initialize the notifier.

        Args:
            bot: A python-telegram-bot Bot (or compatible object with
                an async ``send_message`` coroutine).
            chat_id: The chat id alerts are sent to. An empty value
                disables sending.
        """
        self._bot = bot
        self._chat_id = chat_id

    def is_configured(self) -> bool:
        """Return True when both a bot and a chat id are available."""
        bot_ok = self._bot is not None
        chat_ok = self._chat_id is not None and self._chat_id != ""
        return bot_ok and chat_ok

    async def send(self, text: str) -> None:
        """Send one alert text to the configured chat.

        Never raises: transport failures are logged and swallowed.

        Args:
            text: The alert text to send.
        """
        if not self.is_configured():
            return
        try:
            await self._bot.send_message(chat_id=self._chat_id, text=text)
        except Exception:
            logger.exception(
                "Alert delivery to Telegram chat %s failed; alert dropped.",
                self._chat_id,
            )


class NullNotifier:
    """No-op notifier used when alerts are disabled or unconfigured."""

    def is_configured(self) -> bool:
        """Always False: the null notifier never sends anything."""
        return False

    async def send(self, text: str) -> None:
        """Do nothing. Alerts are intentionally dropped."""


def build_notifier_from_settings(
    *, alerts_enabled: bool, chat_id: str | int | None, token: str | None
):
    """Build the notifier described by configuration values.

    Returns a NullNotifier when alerts are disabled, the chat id is
    missing, or the bot token is missing — never raises.

    Args:
        alerts_enabled: The ALERTS_ENABLED configuration value.
        chat_id: The TELEGRAM_ALERT_CHAT_ID configuration value.
        token: The TELEGRAM_BOT_TOKEN configuration value.

    Returns:
        A TelegramNotifier when alerts are fully configured, otherwise
        a NullNotifier.
    """
    if not alerts_enabled:
        return NullNotifier()
    if not chat_id or not token:
        logger.info(
            "Alerts enabled but TELEGRAM_ALERT_CHAT_ID or bot token is "
            "missing; alerts are disabled for this run."
        )
        return NullNotifier()
    return TelegramNotifier(_build_bot(token), chat_id)
