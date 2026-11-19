"""
Tests for the ``/status`` command (Milestone M-C).

The status report must make degraded operations visible: a paused or
unreadable pause control, a stopped monitor, and a failing journal all
appear explicitly, while credentials never do.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from dashboard.status import format_status_message
from journal.journal import JournalStatus
from telegram_bot import handlers
from trader.paper.pause import UNREADABLE_REASON, PauseState


def _journal(**overrides):
    values = dict(
        path=Path("logs/trade_journal.jsonl"),
        exists=True,
        size_bytes=2048,
        rotated_files=2,
        last_write_at="2026-11-19 10:00:00",
        last_error=None,
    )
    values.update(overrides)
    return JournalStatus(**values)


def _render(**overrides):
    values = dict(
        execution_mode="PAPER",
        paper_trading=True,
        guardian_mode=True,
        authorization="1 chat id configured",
        pause=PauseState(paused=False),
        monitor_status="running",
        monitor_interval=45.0,
        balance=10500.0,
        open_positions=2,
        closed_trades=7,
        journal=_journal(),
        rate_limit_commands=12,
        rate_limit_window=10.0,
    )
    values.update(overrides)
    return format_status_message(**values)


def test_status_reports_allowed_entries_and_running_monitor():
    message = _render()

    assert "Operational Status" in message
    assert "Execution Mode: PAPER" in message
    assert "Paper Trading: Enabled" in message
    assert "Guardian Mode: Active" in message
    assert "Allowed" in message
    assert "Monitor: Running (every 45s)" in message
    assert "Balance: $10,500.00" in message
    assert "Open Positions: 2" in message
    assert "Closed Trades: 7" in message
    assert "logs/trade_journal.jsonl" in message
    assert "2.0 KB" in message
    assert "12 commands / 10s per chat" in message


def test_status_shows_pause_reason():
    message = _render(pause=PauseState(paused=True, reason="maintenance"))

    assert "Paused" in message
    assert "Reason: maintenance" in message


def test_status_marks_unreadable_pause_as_blocked():
    message = _render(
        pause=PauseState(
            paused=True, reason=UNREADABLE_REASON, degraded=True
        )
    )

    assert "BLOCKED (pause control state is unreadable)" in message
    assert "/resume" in message


def test_status_shows_stopped_monitor_and_journal_error():
    message = _render(
        monitor_status="stopped",
        journal=_journal(last_error="OSError: disk full"),
    )

    assert "Monitor: Stopped" in message
    assert "Last Error: OSError: disk full" in message


def test_status_handles_missing_journal_and_disabled_rate_limit():
    message = _render(
        journal=_journal(
            exists=False, size_bytes=0, rotated_files=0, last_write_at=None
        ),
        rate_limit_commands=0,
    )

    assert "Size: not written yet" in message
    assert "Rate Limit" in message
    assert "Disabled" in message


def _update():
    reply_text = AsyncMock()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1),
        message=SimpleNamespace(reply_text=reply_text),
    )
    return update, reply_text


def test_status_command_replies_with_operational_state(monkeypatch):
    update, reply_text = _update()
    service = SimpleNamespace(
        summary=lambda: SimpleNamespace(current_balance=1234.0),
        open_positions=lambda: ["a"],
        closed_positions=lambda: ["b", "c"],
    )

    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)
    monkeypatch.setattr(handlers, "_monitor_status_provider", lambda: "running")
    monkeypatch.setattr(handlers, "journal_status", _journal)
    monkeypatch.setattr(
        handlers, "_paper_pause_state", lambda: PauseState(paused=True, reason="x")
    )

    asyncio.run(handlers.status_command(update, SimpleNamespace()))

    message = reply_text.await_args.args[0]
    assert "Operational Status" in message
    assert "Paused" in message
    assert "Monitor: Running" in message
    assert "Balance: $1,234.00" in message
    assert "Open Positions: 1" in message
    assert "Closed Trades: 2" in message


def test_status_command_is_rate_limited_first(monkeypatch):
    update, reply_text = _update()

    def _service_called():
        raise AssertionError("paper service must not be touched")

    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: False)
    monkeypatch.setattr(handlers, "_get_paper_service", _service_called)

    asyncio.run(handlers.status_command(update, SimpleNamespace()))

    assert "Too many requests" in reply_text.await_args.args[0]
