"""
Tests for Sprint 9A - Telegram notifier.

TelegramNotifier wraps python-telegram-bot's Bot.send_message so alert
messages reach the configured chat. NullNotifier is the no-op fallback
used when alerts are disabled or unconfigured.

Contract under test:
    - TelegramNotifier.send() delegates to Bot.send_message with the
      configured chat id and text.
    - TelegramNotifier.send() never raises: a failed send is logged and
      swallowed (Guardian Mode: a broken notifier must not break
      trading).
    - NullNotifier.send() does nothing and never raises.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from alerts.notifier import NullNotifier, TelegramNotifier


def _make_notifier(chat_id="12345"):
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock())
    notifier = TelegramNotifier(bot, chat_id=chat_id)
    return notifier, bot


# ---------------------------------------------------------------------------
# TelegramNotifier
# ---------------------------------------------------------------------------


class TestTelegramNotifier:
    def test_send_delegates_to_bot_send_message(self):
        notifier, bot = _make_notifier(chat_id="12345")
        import asyncio

        asyncio.run(notifier.send("hello alert"))
        bot.send_message.assert_awaited_once()

    def test_send_passes_chat_id_and_text(self):
        notifier, bot = _make_notifier(chat_id="12345")
        import asyncio

        asyncio.run(notifier.send("hello alert"))
        kwargs = bot.send_message.await_args.kwargs
        assert kwargs["chat_id"] == "12345"
        assert kwargs["text"] == "hello alert"

    def test_send_accepts_int_chat_id(self):
        bot = MagicMock()
        bot.send_message = AsyncMock()
        notifier = TelegramNotifier(bot, chat_id=987654)
        import asyncio

        asyncio.run(notifier.send("hi"))
        assert bot.send_message.await_args.kwargs["chat_id"] == 987654

    def test_send_swallows_exceptions(self):
        """A failed send must never raise to the caller."""
        bot = MagicMock()
        bot.send_message = AsyncMock(side_effect=RuntimeError("network down"))
        notifier = TelegramNotifier(bot, chat_id="12345")
        import asyncio

        # Must not raise.
        asyncio.run(notifier.send("hello alert"))

    def test_send_returns_none(self):
        notifier, _bot = _make_notifier()
        import asyncio

        assert asyncio.run(notifier.send("hi")) is None

    def test_missing_chat_id_disables_notifier(self):
        """A notifier without a chat id cannot send anything."""
        bot = MagicMock()
        bot.send_message = AsyncMock()
        notifier = TelegramNotifier(bot, chat_id="")
        import asyncio

        asyncio.run(notifier.send("hi"))
        bot.send_message.assert_not_awaited()

    def test_missing_bot_disables_notifier(self):
        notifier = TelegramNotifier(None, chat_id="12345")
        import asyncio

        # Must not raise even without a bot.
        asyncio.run(notifier.send("hi"))

    def test_is_configured_true_with_bot_and_chat(self):
        notifier, _bot = _make_notifier()
        assert notifier.is_configured() is True

    def test_is_configured_false_without_chat_id(self):
        bot = MagicMock()
        notifier = TelegramNotifier(bot, chat_id="")
        assert notifier.is_configured() is False

    def test_is_configured_false_without_bot(self):
        notifier = TelegramNotifier(None, chat_id="12345")
        assert notifier.is_configured() is False


# ---------------------------------------------------------------------------
# NullNotifier
# ---------------------------------------------------------------------------


class TestNullNotifier:
    def test_send_does_nothing(self):
        notifier = NullNotifier()
        import asyncio

        # Must not raise.
        asyncio.run(notifier.send("hi"))

    def test_is_configured_false(self):
        assert NullNotifier().is_configured() is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestBuildNotifierFromSettings:
    def test_disabled_returns_null_notifier(self):
        from alerts.notifier import build_notifier_from_settings

        notifier = build_notifier_from_settings(
            alerts_enabled=False, chat_id="12345", token="tok"
        )
        assert isinstance(notifier, NullNotifier)

    def test_missing_chat_id_returns_null_notifier(self):
        from alerts.notifier import build_notifier_from_settings

        notifier = build_notifier_from_settings(
            alerts_enabled=True, chat_id="", token="tok"
        )
        assert isinstance(notifier, NullNotifier)

    def test_missing_token_returns_null_notifier(self):
        from alerts.notifier import build_notifier_from_settings

        notifier = build_notifier_from_settings(
            alerts_enabled=True, chat_id="12345", token=""
        )
        assert isinstance(notifier, NullNotifier)

    def test_enabled_with_config_returns_telegram_notifier(self, monkeypatch):
        from alerts.notifier import build_notifier_from_settings

        captured = {}

        class _FakeBot:
            def __init__(self, token):
                captured["token"] = token
                self.send_message = AsyncMock()

        monkeypatch.setattr("alerts.notifier._build_bot", _FakeBot)
        notifier = build_notifier_from_settings(
            alerts_enabled=True, chat_id="12345", token="tok"
        )
        assert isinstance(notifier, TelegramNotifier)
        assert captured["token"] == "tok"
