"""
Tests for the global paper-trading pause (Milestone M-C).

The pause must persist, block only *new* entries, and fail closed: state
that cannot be read blocks entries, and a failed pause write must never
be reported as success.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from telegram_bot import handlers
from trader.paper.analysis_service import AnalysisStatus, run_analysis_pipeline
from trader.paper.pause import (
    UNREADABLE_REASON,
    PauseState,
    clear_pause,
    read_pause_state,
    request_pause,
)


# ---------------------------------------------------------------------------
# Control state
# ---------------------------------------------------------------------------


def test_missing_control_file_means_not_paused(tmp_path):
    state = read_pause_state(tmp_path / "paper_pause.json")

    assert state.paused is False
    assert state.degraded is False


def test_pause_and_resume_round_trip(tmp_path):
    path = tmp_path / "paper_pause.json"

    paused = request_pause("maintenance window", path)
    assert paused is not None
    assert paused.paused is True
    assert paused.updated_at is not None

    stored = read_pause_state(path)
    assert stored.paused is True
    assert stored.reason == "maintenance window"

    resumed = clear_pause(path)
    assert resumed is not None
    stored = read_pause_state(path)
    assert stored.paused is False
    assert stored.reason is None


def test_reason_is_cleaned_and_truncated(tmp_path):
    path = tmp_path / "paper_pause.json"

    state = request_pause("  line one\nline two  " + "x" * 500, path)

    assert state.reason.startswith("line one line two")
    assert "\n" not in state.reason
    assert len(state.reason) <= 200


def test_corrupt_control_state_fails_closed(tmp_path):
    path = tmp_path / "paper_pause.json"
    path.write_text("{not json", encoding="utf-8")

    state = read_pause_state(path)

    assert state.paused is True
    assert state.degraded is True
    assert state.reason == UNREADABLE_REASON


def test_unexpected_control_shape_fails_closed(tmp_path):
    path = tmp_path / "paper_pause.json"

    for payload in ('["paused"]', '{"reason": "x"}', '{"paused": 0}'):
        path.write_text(payload, encoding="utf-8")
        state = read_pause_state(path)
        assert state.paused is True, payload
        assert state.degraded is True, payload


def test_failed_write_returns_none_and_leaves_state(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    assert request_pause("x", blocker / "pause.json") is None
    assert clear_pause(blocker / "pause.json") is None


# ---------------------------------------------------------------------------
# Pipeline gate
# ---------------------------------------------------------------------------


def _noop_async(*args, **kwargs):
    async def _inner():
        return None

    return _inner()


def _pipeline_kwargs(tmp_path, pause_state, hooks):
    trade = SimpleNamespace(
        symbol="BTC",
        price=100.0,
        decision="BUY",
        risk_verdict="APPROVED",
        approved=True,
    )
    service = MagicMock()
    service.place_trade.return_value = None

    return dict(
        strategy=object(),
        paper_trading_enabled=True,
        analyze_coin=lambda coin, parameters=None: {"symbol": "BTC"},
        current_quote=lambda symbol: 100.0,
        make_decision=lambda result, parameters=None: object(),
        paper_mark_block=lambda: None,
        resolve_trading_policy=lambda strategy: SimpleNamespace(
            risk_profile=object()
        ),
        validate_trade=lambda decision, result, profile: object(),
        build_trade=lambda result, verdict, profile: trade,
        log_trade=lambda symbol, decision, verdict: True,
        paper_service_factory=lambda: service,
        paper_pause_state=lambda: pause_state,
        before_analysis_paper_check=hooks["before_check"],
        apply_trailing_stop=hooks["trailing"],
        evaluate_entry_gates=hooks["gates"],
        state_path=tmp_path / "paper_state.json",
        dispatch_alert=_noop_async,
        dispatch_entry_gate_alerts=_noop_async,
        portfolio_risk_limit=0.0,
        portfolio_exposure_limit=0.0,
        daily_loss_limit=0.0,
        max_open_positions=0,
        max_positions_per_symbol=0,
        max_exposure_per_symbol=0.0,
    )


def _hooks():
    closed = SimpleNamespace(symbol="BTC")
    return {
        "before_check": MagicMock(return_value=closed),
        "trailing": MagicMock(return_value=(True, 95.0, 99.0)),
        "gates": MagicMock(return_value=SimpleNamespace(allowed=True)),
    }


def test_pause_blocks_new_entry_but_still_protects_positions(tmp_path):
    hooks = _hooks()

    run = asyncio.run(
        run_analysis_pipeline(
            "BTC",
            **_pipeline_kwargs(
                tmp_path, PauseState(paused=True, reason="manual"), hooks
            ),
        )
    )

    assert run.status is AnalysisStatus.PAUSED
    assert run.sltp_closed is not None
    assert run.trail_result == (True, 95.0, 99.0)
    hooks["before_check"].assert_called_once()
    hooks["trailing"].assert_called_once()
    hooks["gates"].assert_not_called()


def test_unpaused_pipeline_reaches_the_entry_gates(tmp_path):
    hooks = _hooks()

    run = asyncio.run(
        run_analysis_pipeline(
            "BTC",
            **_pipeline_kwargs(tmp_path, PauseState(paused=False), hooks),
        )
    )

    assert run.status is AnalysisStatus.PAPER_NOT_PLACED
    hooks["gates"].assert_called_once()


def test_pause_read_failure_still_blocks_placement(tmp_path):
    hooks = _hooks()
    kwargs = _pipeline_kwargs(tmp_path, None, hooks)

    def _broken():
        raise OSError("control file unavailable")

    kwargs["paper_pause_state"] = _broken

    run = asyncio.run(run_analysis_pipeline("BTC", **kwargs))

    assert run.status is AnalysisStatus.PAPER_FAILED
    hooks["gates"].assert_not_called()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _update():
    reply_text = AsyncMock()
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1),
        message=SimpleNamespace(reply_text=reply_text),
    )
    return update, reply_text


def test_pause_command_persists_state(tmp_path, monkeypatch):
    update, reply_text = _update()
    pause_path = tmp_path / "paper_pause.json"

    monkeypatch.setattr(handlers, "PAUSE_STATE_PATH", pause_path)
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)

    asyncio.run(
        handlers.pause_command(
            update, SimpleNamespace(args=["maintenance", "window"])
        )
    )

    stored = read_pause_state(pause_path)
    assert stored.paused is True
    assert stored.reason == "maintenance window"
    message = reply_text.await_args.args[0]
    assert "paused" in message
    assert "maintenance window" in message
    assert "Existing positions are still monitored" in message


def test_pause_command_reports_write_failure_without_claiming_pause(
    tmp_path, monkeypatch
):
    update, reply_text = _update()
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    monkeypatch.setattr(handlers, "PAUSE_STATE_PATH", blocker / "pause.json")
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)

    asyncio.run(handlers.pause_command(update, SimpleNamespace(args=[])))

    message = reply_text.await_args.args[0]
    assert "Could not save the pause state" in message
    assert "Paper trading paused" not in message


def test_resume_command_clears_state(tmp_path, monkeypatch):
    update, reply_text = _update()
    pause_path = tmp_path / "paper_pause.json"
    request_pause("test", pause_path)

    monkeypatch.setattr(handlers, "PAUSE_STATE_PATH", pause_path)
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)

    asyncio.run(handlers.resume_command(update, SimpleNamespace(args=[])))

    assert read_pause_state(pause_path).paused is False
    assert "resumed" in reply_text.await_args.args[0]


def test_pause_command_is_rate_limited_first(tmp_path, monkeypatch):
    update, reply_text = _update()
    pause_path = tmp_path / "paper_pause.json"

    monkeypatch.setattr(handlers, "PAUSE_STATE_PATH", pause_path)
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: False)

    asyncio.run(handlers.pause_command(update, SimpleNamespace(args=[])))

    assert not pause_path.exists()
    assert "Too many requests" in reply_text.await_args.args[0]


def test_handler_pause_state_uses_patched_path(tmp_path, monkeypatch):
    pause_path = tmp_path / "paper_pause.json"
    monkeypatch.setattr(handlers, "PAUSE_STATE_PATH", pause_path)

    assert handlers._paper_pause_state().paused is False

    pause_path.write_text(json.dumps({"paused": True}), encoding="utf-8")
    state = handlers._paper_pause_state()
    assert state.paused is True
    assert state.reason is None
