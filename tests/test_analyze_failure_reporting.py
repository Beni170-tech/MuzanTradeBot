"""Focused tests for honest /analyze paper-execution failure reporting.

Before Milestone M-A a failure inside the paper-trading branch was only
logged; the Telegram reply still showed the approved-risk block and no
placement result, which reads like success. These tests pin the new
behaviour: the reply says plainly that the step failed and no order was
confirmed, no exception internals leak, and the normal success text is
untouched.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import telegram_bot.handlers as handlers
from config import settings as cfg_settings
from trader.paper.service import PaperTradingService


_SETTING_NAMES = (
    "PAPER_TRADING",
    "ALERTS_ENABLED",
    "PORTFOLIO_MAX_OPEN_POSITIONS",
    "PORTFOLIO_DAILY_LOSS_LIMIT",
    "PORTFOLIO_RISK_LIMIT",
    "PORTFOLIO_EXPOSURE_LIMIT",
    "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
    "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
)


def _analysis() -> dict:
    return {
        "symbol": "BTC",
        "price": 100.0,
        "trend": "Bullish",
        "signal": "BUY",
        "confidence": 80,
        "indicators": None,
    }


def _run_analyze(monkeypatch, tmp_path, service):
    """Run the real /analyze pipeline with a controlled paper service."""
    originals = {
        name: getattr(cfg_settings.settings, name) for name in _SETTING_NAMES
    }
    try:
        cfg_settings.settings.PAPER_TRADING = True
        cfg_settings.settings.ALERTS_ENABLED = False
        for name in _SETTING_NAMES[2:]:
            setattr(
                cfg_settings.settings,
                name,
                0 if not name.endswith("LIMIT") else 0.0,
            )

        monkeypatch.setattr(
            handlers, "check_rate_limit", lambda *a, **k: True
        )
        monkeypatch.setattr(
            handlers, "analyze_coin", MagicMock(return_value=_analysis())
        )
        monkeypatch.setattr(handlers, "get_current_price", lambda _: 105.0)
        monkeypatch.setattr(handlers, "log_trade", MagicMock())
        monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)
        monkeypatch.setattr(
            handlers, "PAPER_STATE_PATH", tmp_path / "paper_state.json"
        )

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        update.message.chat.id = 123
        context = MagicMock()
        context.args = ["BTC"]
        asyncio.run(handlers.analyze(update, context))
        return update.message.reply_text.await_args[0][0]
    finally:
        for name, value in originals.items():
            setattr(cfg_settings.settings, name, value)


def test_paper_failure_is_reported_and_never_looks_placed(
    monkeypatch, tmp_path
):
    service = PaperTradingService.create_new_account(1000.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(handlers, "_before_analysis_paper_check", _boom)

    message = _run_analyze(monkeypatch, tmp_path, service)

    assert "❌ Paper Execution Failed" in message
    assert "No confirmed order is reported as placed." in message
    assert "Check /positions before retrying" in message
    # The success banner must not appear when execution failed.
    assert "Paper Order: PLACED" not in message
    # Internals stay in the logs, never in the Telegram reply.
    assert "secret internal detail" not in message
    assert "Traceback" not in message
    # The analysis itself is still delivered.
    assert "Analysis for BTC" in message


def test_execution_failure_is_logged_with_traceback(
    monkeypatch, tmp_path, caplog
):
    service = PaperTradingService.create_new_account(1000.0)

    def _boom(*args, **kwargs):
        raise RuntimeError("paper service exploded")

    monkeypatch.setattr(handlers, "_before_analysis_paper_check", _boom)

    with caplog.at_level("ERROR", logger="telegram_bot.handlers"):
        _run_analyze(monkeypatch, tmp_path, service)

    assert "Paper trading integration failed during /analyze" in caplog.text
    assert "paper service exploded" in caplog.text


def test_successful_paper_flow_has_no_failure_notice(monkeypatch, tmp_path):
    service = PaperTradingService.create_new_account(1000.0)
    message = _run_analyze(monkeypatch, tmp_path, service)

    assert "❌ Paper Execution Failed" not in message
    assert "Paper Order: PLACED" in message


def test_rejected_placement_is_not_reported_as_placed(
    monkeypatch, tmp_path
):
    service = MagicMock()
    service.summary.return_value = MagicMock(equity=1000.0)
    service.open_positions.return_value = []
    service.get_position.return_value = None
    service.place_trade.return_value = None
    service.execution_backend.return_value = MagicMock()
    service.save_account = MagicMock()

    message = _run_analyze(monkeypatch, tmp_path, service)

    assert "Paper Order: PLACED" not in message
    assert "❌ Paper Execution Failed" not in message


def test_paper_disabled_path_is_unchanged(monkeypatch, tmp_path):
    originals = {
        name: getattr(cfg_settings.settings, name) for name in _SETTING_NAMES
    }
    try:
        cfg_settings.settings.PAPER_TRADING = False
        cfg_settings.settings.ALERTS_ENABLED = False
        monkeypatch.setattr(
            handlers, "check_rate_limit", lambda *a, **k: True
        )
        monkeypatch.setattr(
            handlers, "analyze_coin", MagicMock(return_value=_analysis())
        )
        monkeypatch.setattr(handlers, "get_current_price", lambda _: 105.0)
        monkeypatch.setattr(handlers, "log_trade", MagicMock())

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        update.message.chat.id = 123
        context = MagicMock()
        context.args = ["BTC"]
        asyncio.run(handlers.analyze(update, context))
        message = update.message.reply_text.await_args[0][0]
    finally:
        for name, value in originals.items():
            setattr(cfg_settings.settings, name, value)

    assert "Analysis for BTC" in message
    assert "Paper Order" not in message
    assert "❌ Paper Execution Failed" not in message
