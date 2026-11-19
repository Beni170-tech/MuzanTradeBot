"""Regression tests for the final paper-trading readiness audit fixes.

Covers the two defects found in the audit that were not already tested:

* an unreadable saved account must block trading with a stated reason
  instead of raising into silence, and
* a close that cannot be persisted must not report an unqualified
  success, and must not mutate memory when storage is unusable.

Also pins the user-facing text for a rejected (not placed) paper order
and the presence of a Telegram error handler.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import telegram_bot.bot as bot
import telegram_bot.handlers as handlers
from trader.paper.analysis_service import AnalysisRun, AnalysisStatus
from trader.paper.read_model import execute_close
from trader.paper.service import PaperTradingService


def _trade() -> SimpleNamespace:
    return SimpleNamespace(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        approved=True,
        risk_verdict="PASS",
        risk_reasons=[],
        position_size=1.0,
        stop_loss=95.0,
        take_profit=110.0,
        side="LONG",
        timestamp="now",
    )


def _update() -> SimpleNamespace:
    message = SimpleNamespace(reply_text=AsyncMock())
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=1),
        message=message,
        effective_message=message,
    )


def _service_with_position() -> PaperTradingService:
    service = PaperTradingService.create_new_account(1000.0)
    assert service.place_trade(_trade()) is not None
    return service


# ---------------------------------------------------------------------------
# Unreadable paper state
# ---------------------------------------------------------------------------


def test_unreadable_state_raises_and_never_creates_an_account(
    monkeypatch, tmp_path
) -> None:
    bad = tmp_path / "paper_state.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", bad)
    monkeypatch.setattr(handlers, "_paper_service", None)

    with pytest.raises(handlers.PaperStateUnreadable) as excinfo:
        handlers._get_paper_service()

    message = str(excinfo.value).lower()
    assert "unreadable" in message
    assert "no trading action" in message
    # The cache stays empty, so the next attempt still fails closed.
    assert handlers._paper_service is None


def test_error_handler_explains_unreadable_state() -> None:
    update = _update()
    context = SimpleNamespace(
        error=handlers.PaperStateUnreadable(
            "the paper account state file is unreadable."
        )
    )

    asyncio.run(bot._error_handler(update, context))

    text = update.effective_message.reply_text.await_args[0][0]
    assert "unreadable" in text.lower()


def test_error_handler_never_claims_success() -> None:
    update = _update()
    context = SimpleNamespace(error=RuntimeError("boom"))

    asyncio.run(bot._error_handler(update, context))

    text = update.effective_message.reply_text.await_args[0][0]
    assert "failed" in text.lower()
    assert "No completed trading action" in text
    assert "✅" not in text


def test_build_application_registers_an_error_handler(monkeypatch) -> None:
    monkeypatch.setattr(bot.settings, "TELEGRAM_BOT_TOKEN", "123:abc")

    app = bot.build_application()

    assert app.error_handlers


def test_run_bot_refuses_to_start_on_unreadable_state(monkeypatch) -> None:
    monkeypatch.setattr(bot, "build_application", MagicMock())
    monkeypatch.setattr(
        bot,
        "_build_paper_monitor",
        MagicMock(side_effect=handlers.PaperStateUnreadable("unreadable")),
    )

    with pytest.raises(SystemExit) as excinfo:
        bot.run_bot()

    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# Close persistence
# ---------------------------------------------------------------------------


def test_close_with_unwritable_state_closes_nothing(tmp_path) -> None:
    service = _service_with_position()
    target = service.get_position("BTC")
    balance_before = service.summary().current_balance
    unwritable = tmp_path / "missing-dir" / "nested"
    unwritable.mkdir(parents=True)
    # A directory cannot be replaced by a file write.
    state_path = unwritable / "state.json"
    state_path.mkdir()

    result = execute_close(service, target, 101.0, state_path)

    assert result is None
    assert service.get_position("BTC") is not None
    assert service.summary().current_balance == balance_before


def test_close_reports_when_post_close_save_fails(monkeypatch, tmp_path) -> None:
    service = _service_with_position()
    target = service.get_position("BTC")

    calls = {"n": 0}

    def _save(path) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("disk full")

    monkeypatch.setattr(service, "save_account", _save)

    result = execute_close(service, target, 101.0, tmp_path / "state.json")

    assert result is not None
    assert result.persisted is False
    assert service.get_position("BTC") is None


def test_close_command_reports_unpersisted_close(monkeypatch, tmp_path) -> None:
    service = _service_with_position()
    monkeypatch.setattr(handlers, "_paper_service", service)

    calls = {"n": 0}

    def _save(path) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("disk full")

    monkeypatch.setattr(service, "save_account", _save)

    update = _update()
    context = SimpleNamespace(args=["BTC", "101"])

    asyncio.run(handlers.close_command(update, context))

    text = update.effective_message.reply_text.await_args[0][0]
    assert "could not be saved to disk" in text


# ---------------------------------------------------------------------------
# Rejected placement text
# ---------------------------------------------------------------------------


def test_not_placed_message_states_that_nothing_was_placed() -> None:
    run = AnalysisRun(
        AnalysisStatus.PAPER_NOT_PLACED, "BTC", trade=_trade()
    )

    message = handlers._format_analysis_message(run)

    assert "NOT PLACED" in message
    assert "No new paper position was opened" in message
    assert "PLACED" in message and "Paper Order: PLACED" not in message


def test_not_placed_message_does_not_look_like_a_failure() -> None:
    run = AnalysisRun(
        AnalysisStatus.PAPER_NOT_PLACED, "BTC", trade=_trade()
    )

    message = handlers._format_analysis_message(run)

    assert "Paper Execution Failed" not in message
