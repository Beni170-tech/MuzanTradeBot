"""Regression tests for /close persistence safety.

Pins the three user-visible outcomes of a manual close:

* closed and saved to disk,
* applied in memory but not persisted (never reported as saved), and
* rejected without any mutation.

Also pins retry behaviour: a retry after an unpersisted close must not
close the position twice or apply the balance change twice.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import telegram_bot.handlers as handlers
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


def _service_with_position() -> PaperTradingService:
    service = PaperTradingService.create_new_account(1000.0)
    assert service.place_trade(_trade()) is not None
    return service


def _update() -> SimpleNamespace:
    message = SimpleNamespace(reply_text=AsyncMock())
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=1),
        message=message,
        effective_message=message,
    )


def _close(monkeypatch, service, state_path, price: str = "101"):
    monkeypatch.setattr(handlers, "_paper_service", service)
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", state_path)
    # Keep the two-call retry test independent of the shared limiter.
    monkeypatch.setattr(
        handlers, "check_rate_limit", lambda chat_id, config: True
    )

    update = _update()
    context = SimpleNamespace(args=["BTC", price])
    asyncio.run(handlers.close_command(update, context))
    return update.effective_message.reply_text.await_args[0][0]


def test_persisted_close_is_reported_as_saved(monkeypatch, tmp_path) -> None:
    service = _service_with_position()
    state_path = tmp_path / "state.json"

    text = _close(monkeypatch, service, state_path)

    assert "✅ Position Closed and Saved" in text
    assert "NOT Persisted" not in text

    # The close really is on disk: a reload sees no open position.
    reloaded = PaperTradingService.load_account(state_path)
    assert reloaded.get_position("BTC") is None
    assert reloaded.summary().current_balance == service.summary().current_balance


def test_unpersisted_close_never_claims_a_saved_close(
    monkeypatch, tmp_path
) -> None:
    service = _service_with_position()
    calls = {"n": 0}

    def _save(path) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("disk full")

    monkeypatch.setattr(service, "save_account", _save)

    text = _close(monkeypatch, service, tmp_path / "state.json")

    assert "Close Applied in Memory — NOT Persisted" in text
    assert "✅" not in text
    assert "could not be saved to disk" in text
    assert "Retrying now is rejected" in text
    assert service.get_position("BTC") is None


def test_rejected_close_states_that_nothing_changed(monkeypatch, tmp_path) -> None:
    service = _service_with_position()
    balance_before = service.summary().current_balance
    # A directory cannot be replaced by a file write, so the pre-flight
    # save fails before any in-memory mutation.
    state_path = tmp_path / "state.json"
    state_path.mkdir()

    text = _close(monkeypatch, service, state_path)

    assert "Close Rejected — nothing was closed or changed" in text
    assert "No balance change was applied" in text
    assert service.get_position("BTC") is not None
    assert service.summary().current_balance == balance_before
    assert service.summary().closed_count == 0


def test_retry_after_unpersisted_close_cannot_close_twice(
    monkeypatch, tmp_path
) -> None:
    service = _service_with_position()
    calls = {"n": 0}

    def _save(path) -> None:
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("disk full")

    monkeypatch.setattr(service, "save_account", _save)

    first = _close(monkeypatch, service, tmp_path / "state.json")
    assert "NOT Persisted" in first

    after_first = service.summary()
    second = _close(monkeypatch, service, tmp_path / "state.json")

    # The retry finds no open position: no second close, no second
    # balance change and no duplicated realized PnL.
    assert "No open paper position found for BTC" in second
    final = service.summary()
    assert final.current_balance == after_first.current_balance
    assert final.closed_count == 1
    assert final.total_realized_pnl == after_first.total_realized_pnl
