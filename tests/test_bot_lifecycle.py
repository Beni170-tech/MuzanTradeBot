"""Lifecycle tests for Telegram bot and paper position monitor shutdown.

These tests exercise the real run_bot() path with a fake Application that
mimics python-telegram-bot's ordering: post_init is awaited on the polling
loop, then post_shutdown is awaited on that same still-open loop before the
loop is closed afterwards.
"""

from __future__ import annotations

import asyncio
import logging

import telegram_bot.bot as bot
from trader.paper.monitor import PaperPositionMonitor


class _NoPositionsService:
    def open_positions(self):
        return []

    def get_position(self, symbol):
        return None


class _Alerts:
    """Monitor is stopped before any alert would be sent."""


def _monitor() -> PaperPositionMonitor:
    return PaperPositionMonitor(
        _NoPositionsService(), _Alerts(), interval_seconds=30.0
    )


class _FakeApplication:
    """Stand-in for PTB Application with the real shutdown ordering."""

    def __init__(self) -> None:
        self.post_init = None
        self.post_shutdown = None
        self.loop = None
        self.task = None
        self.task_done_at_shutdown = False

    def run_polling(self, **kwargs) -> None:
        assert self.post_init is not None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            # PTB runs post_init on the polling loop, before updates.
            self.loop.run_until_complete(self.post_init(self))
            assert bot._paper_monitor is not None
            self.task = bot._paper_monitor._task
            assert self.task is not None
            assert self.task.get_loop() is self.loop
            self.loop.run_until_complete(asyncio.sleep(0.01))

            assert self.post_shutdown is not None
            self.loop.run_until_complete(self.post_shutdown(self))

            self.task_done_at_shutdown = self.task.done()
        finally:
            self.loop.close()
            asyncio.set_event_loop(None)


def test_run_bot_awaits_monitor_before_closing_loop(monkeypatch):
    monitor = _monitor()
    app = _FakeApplication()

    monkeypatch.setattr(bot, "build_application", lambda: app)
    monkeypatch.setattr(bot, "_build_paper_monitor", lambda: monitor)
    monkeypatch.setattr(bot, "_paper_monitor", None)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    bot.run_bot()

    assert app.loop is not None
    assert app.loop.is_closed()
    assert app.task is not None
    assert app.task_done_at_shutdown is True
    assert app.task.done()
    assert monitor._task is None
    assert monitor.is_running is False


def test_post_shutdown_callback_awaits_aclose(monkeypatch):
    monitor = _monitor()
    monkeypatch.setattr(bot, "_paper_monitor", monitor)

    async def scenario():
        monitor.start()
        task = monitor._task
        await bot._shutdown_paper_monitor(None)
        return task

    task = asyncio.run(scenario())

    assert task.done()
    assert monitor._task is None
    assert monitor.is_running is False


def test_shutdown_callback_without_monitor_is_safe(monkeypatch):
    monkeypatch.setattr(bot, "_paper_monitor", None)

    asyncio.run(bot._shutdown_paper_monitor(None))


def test_shutdown_callback_logs_failures_without_raising(
    monkeypatch, caplog
):
    class _FailingMonitor:
        async def aclose(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(bot, "_paper_monitor", _FailingMonitor())

    with caplog.at_level(logging.ERROR, logger="telegram_bot.bot"):
        asyncio.run(bot._shutdown_paper_monitor(None))

    assert "Paper position monitor shutdown failed" in caplog.text
