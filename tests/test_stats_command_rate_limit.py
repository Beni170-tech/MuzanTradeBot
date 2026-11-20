"""Rate-limit tests for the /stats dashboard command.

/stats must use the same shared limiter as the other commands: an
allowed request renders the dashboard, and a throttled one replies with
the standard message without touching the paper service or its account.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import telegram_bot.handlers as handlers
from trader.paper.service import PaperTradingService


def _update() -> SimpleNamespace:
    reply_text = AsyncMock()
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=1),
        message=SimpleNamespace(reply_text=reply_text),
        effective_message=SimpleNamespace(reply_text=reply_text),
    )


def test_allowed_stats_renders_dashboard(monkeypatch) -> None:
    service = PaperTradingService.create_new_account(1000.0)
    update = _update()
    seen = []

    def _allow(chat_id, config) -> bool:
        seen.append((chat_id, config))
        return True

    monkeypatch.setattr(handlers, "check_rate_limit", _allow)
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)

    asyncio.run(handlers.stats_command(update, SimpleNamespace()))

    text = update.message.reply_text.await_args[0][0]
    assert "📊 Paper Trading Performance" in text
    assert "Balance: $1,000.00" in text
    # The shared helper was used with this chat and the configured rule.
    assert seen == [(1, handlers._rate_limit_config)]


def test_rate_limited_stats_does_not_load_the_paper_service(monkeypatch) -> None:
    update = _update()

    def _service_called():
        raise AssertionError("paper service must not be touched")

    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: False)
    monkeypatch.setattr(handlers, "_get_paper_service", _service_called)

    asyncio.run(handlers.stats_command(update, SimpleNamespace()))

    assert (
        update.message.reply_text.await_args[0][0]
        == "⏳ Too many requests. Please wait a moment before trying again."
    )
