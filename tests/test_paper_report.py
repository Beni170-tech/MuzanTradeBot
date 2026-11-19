"""
Tests for the ``/report`` command (Milestone M-C).

The report reuses the existing paper analytics: it only selects periods
(all time, today, last seven days, per symbol) and renders them. These
tests cover the window selection, the rendered sections, and the handler
wiring (rate limit first, then one reply).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from dashboard.report import format_report_message
from telegram_bot import handlers
from trader.paper.read_model import report_view


def _order(symbol="BTC", realized_pnl=0.0, exit_at="2026-11-19 10:00:00"):
    return SimpleNamespace(
        symbol=symbol, realized_pnl=realized_pnl, exit_at=exit_at
    )


def _service(closed=(), open_positions=(), balance=11000.0, initial=10000.0):
    service = SimpleNamespace(
        summary=lambda: SimpleNamespace(
            current_balance=balance, initial_balance=initial
        ),
        open_positions=lambda: list(open_positions),
        closed_positions=lambda: list(closed),
    )
    return service


def _view(closed, now=datetime(2026, 11, 19, 12, 0), open_positions=()):
    return report_view(
        _service(closed, open_positions=open_positions), now=now
    )


def test_report_windows_select_periods_by_exit_date():
    view = _view(
        [
            _order("BTC", 100.0, "2026-11-19 09:00:00"),  # today
            _order("ETH", -40.0, "2026-11-15 09:00:00"),  # this week
            _order("SOL", 25.0, "2026-10-01 09:00:00"),  # older
            _order("XRP", 10.0, None),  # never stamped: excluded everywhere
        ]
    )

    assert view.today_label == "2026-11-19"
    assert view.week_start_label == "2026-11-13"
    assert view.all_time.overall.total_trades == 4
    assert view.today.overall.total_trades == 1
    assert view.today.overall.total_realized_pnl == 100.0
    assert view.week.overall.total_trades == 2
    assert view.week.overall.total_realized_pnl == 60.0
    assert view.symbol_count == 4
    assert view.total_return == 10.0


def test_report_without_closed_trades_has_no_windows():
    view = _view([], open_positions=[_order("BTC", exit_at=None)])

    assert view.all_time is None
    assert view.today.overall.total_trades == 0
    assert view.symbol_groups == []

    message = format_report_message(view)
    assert "Paper Trading Report" in message
    assert "Open Positions: 1" in message
    assert "No closed trades yet." in message
    assert "By Symbol" not in message


def test_report_renders_all_periods_and_symbols():
    view = _view(
        [
            _order("BTC", 100.0, "2026-11-19 09:00:00"),
            _order("BTC", -20.0, "2026-11-16 09:00:00"),
        ]
    )

    message = format_report_message(view)

    assert "All Time" in message
    assert "Closed Trades: 2" in message
    assert "Today (2026-11-19)" in message
    assert "Last 7 Days (2026-11-13 to 2026-11-19)" in message
    assert "BTC: 2 trades" in message
    assert "+$80.00" in message
    assert "no live orders are placed" in message
    # Gate metrics stay on /stats.
    assert "Exposure" not in message


def test_report_caps_listed_symbols_and_summarises_the_rest():
    closed = [
        _order(f"SYM{index}", float(index), "2026-11-18 09:00:00")
        for index in range(7)
    ]

    view = _view(closed)
    message = format_report_message(view)

    assert view.symbol_count == 7
    assert len(view.symbol_groups) == 5
    assert "(+2 more symbols)" in message


def test_report_command_is_rate_limited_first(monkeypatch):
    reply_text = AsyncMock()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1),
        message=SimpleNamespace(reply_text=reply_text),
    )

    def _service_called():
        raise AssertionError("paper service must not be touched")

    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: False)
    monkeypatch.setattr(handlers, "_get_paper_service", _service_called)

    asyncio.run(handlers.report_command(update, SimpleNamespace()))

    reply_text.assert_awaited_once_with(
        "⏳ Too many requests. Please wait a moment before trying again."
    )


def test_report_command_replies_with_the_rendered_report(monkeypatch):
    reply_text = AsyncMock()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1),
        message=SimpleNamespace(reply_text=reply_text),
    )
    service = _service([_order("BTC", 5.0, "2026-11-19 09:00:00")])

    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)

    asyncio.run(handlers.report_command(update, SimpleNamespace()))

    reply_text.assert_awaited_once()
    message = reply_text.await_args.args[0]
    assert "Paper Trading Report" in message
    assert "BTC: 1 trades" in message
