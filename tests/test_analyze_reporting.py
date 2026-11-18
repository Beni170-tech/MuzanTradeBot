"""Focused tests for ``telegram_bot.analyze_reporting``.

The module owns the ``/analyze`` entry-gate rejection text and the
rejection alert sequence. These tests pin that text character for
character, pin the alert types/counts/payload order, and prove the
handler keeps its old bindings and injects its own dispatcher.
"""

import ast
import asyncio
import inspect
from types import SimpleNamespace

import pytest

import telegram_bot.handlers as handlers
from telegram_bot import analyze_reporting
from trader.paper.entry_gates import (
    RULE_DAILY_LOSS,
    RULE_INVALID_TRADE,
    RULE_MAX_OPEN_POSITIONS,
    RULE_PORTFOLIO_EXPOSURE,
    RULE_PORTFOLIO_RISK,
    RULE_SYMBOL_EXPOSURE,
    RULE_SYMBOL_POSITIONS,
    EntryGateOutcome,
)
from trader.paper.portfolio_risk import trade_risk
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(symbol="BTC", price=100.0, position_size=10.0, stop_loss=98.0):
    """Build a real Trade so risk/exposure math is the domain math."""
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=104.0,
        side="LONG",
        approved=True,
    )


def _outcome(rule, metrics):
    return EntryGateOutcome(
        allowed=False, rule=rule, detail="", metrics=metrics
    )


class _Recorder:
    """Records ``(event, args)`` in dispatch order."""

    def __init__(self):
        self.calls = []

    async def __call__(self, event, *args):
        self.calls.append((event, args))


# ---------------------------------------------------------------------------
# Message text (character for character)
# ---------------------------------------------------------------------------


class TestRejectionMessageText:
    def test_portfolio_risk_text(self):
        trade = _make_trade()
        metrics = SimpleNamespace(current_risk=1000.0, risk_limit=500.0)
        text = analyze_reporting.entry_gate_rejection_message(
            trade, _outcome(RULE_PORTFOLIO_RISK, metrics)
        )
        assert text == (
            "\n⚠️ Portfolio Risk Limit Exceeded\n"
            "   Current Risk: $1,000.00\n"
            "   Risk Limit: $500.00\n"
            "   Trade Risk: $20.00\n"
            "   The trade was not placed.\n"
            "   Close existing positions to free up risk budget."
        )

    def test_portfolio_exposure_text(self):
        trade = _make_trade()
        metrics = SimpleNamespace(
            current_exposure=2000.0, exposure_limit=5000.0
        )
        text = analyze_reporting.entry_gate_rejection_message(
            trade, _outcome(RULE_PORTFOLIO_EXPOSURE, metrics)
        )
        assert text == (
            "\n⚠️ Portfolio Exposure Limit Exceeded\n"
            "   Current Exposure: $2,000.00\n"
            "   Requested Exposure: $1,000.00\n"
            "   Exposure Limit: $5,000.00\n"
            "   The trade was not placed.\n"
            "   Close existing positions to free up exposure budget."
        )

    def test_daily_loss_text(self):
        trade = _make_trade()
        metrics = SimpleNamespace(
            day="2026-01-15",
            realized_pnl_today=-150.0,
            loss_limit=100.0,
        )
        text = analyze_reporting.entry_gate_rejection_message(
            trade, _outcome(RULE_DAILY_LOSS, metrics)
        )
        assert text == (
            "\n⛔ Daily Loss Limit Reached\n"
            "   Day: 2026-01-15\n"
            "   Today's Realized PnL: -$150.00\n"
            "   Daily Loss Limit: $100.00\n"
            "   The trade was not placed.\n"
            "   New paper trades resume on the next day."
        )

    def test_max_open_positions_text(self):
        trade = _make_trade()
        metrics = SimpleNamespace(current_open_positions=3, limit=3)
        text = analyze_reporting.entry_gate_rejection_message(
            trade, _outcome(RULE_MAX_OPEN_POSITIONS, metrics)
        )
        assert text == (
            "\n⛔ Maximum Open Positions Reached\n"
            "   Current Open Positions: 3\n"
            "   Max Open Positions: 3\n"
            "   The trade was not placed.\n"
            "   Close existing positions to free up "
            "open-position capacity."
        )

    def test_symbol_positions_text(self):
        trade = _make_trade()
        metrics = SimpleNamespace(open_positions=2, max_positions=2)
        text = analyze_reporting.entry_gate_rejection_message(
            trade, _outcome(RULE_SYMBOL_POSITIONS, metrics)
        )
        assert text == (
            "\n⚠️ Symbol Concentration Limit Exceeded\n"
            "   Symbol: BTC\n"
            "   Open Positions: 2\n"
            "   Max Positions: 2\n"
            "   The trade was not placed.\n"
            "   Close existing positions to free up "
            "concentration budget."
        )

    def test_symbol_exposure_text(self):
        trade = _make_trade()
        metrics = SimpleNamespace(
            symbol_exposure=800.0, exposure_limit=500.0
        )
        text = analyze_reporting.entry_gate_rejection_message(
            trade, _outcome(RULE_SYMBOL_EXPOSURE, metrics)
        )
        assert text == (
            "\n⚠️ Symbol Concentration Limit Exceeded\n"
            "   Symbol: BTC\n"
            "   Current Exposure: $800.00\n"
            "   Requested Exposure: $1,000.00\n"
            "   Exposure Limit: $500.00\n"
            "   The trade was not placed.\n"
            "   Close existing positions to free up "
            "concentration budget."
        )

    def test_unsupported_rule_still_fails_loudly(self):
        trade = _make_trade()
        with pytest.raises(ValueError) as excinfo:
            analyze_reporting.entry_gate_rejection_message(
                trade, _outcome(RULE_INVALID_TRADE, SimpleNamespace())
            )
        assert "unsupported entry gate rejection rule" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


class TestSharedHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [(1.5, "+$1.50"), (-2.0, "-$2.00"), (0.0, "$0.00")],
    )
    def test_format_pnl_value(self, value, expected):
        assert analyze_reporting.format_pnl_value(value) == expected

    def test_trade_risk_value_delegates_to_domain_math(self):
        trade = _make_trade()
        assert analyze_reporting.trade_risk_value(trade) == trade_risk(trade)


# ---------------------------------------------------------------------------
# Alert dispatch: types, counts, payloads, ordering
# ---------------------------------------------------------------------------


class TestRejectionAlerts:
    def _dispatch(self, rule, metrics, trade=None):
        trade = trade or _make_trade()
        recorder = _Recorder()
        asyncio.run(
            analyze_reporting.dispatch_entry_gate_alerts(
                trade,
                _outcome(rule, metrics),
                dispatch_alert=recorder,
            )
        )
        return trade, metrics, recorder.calls

    def test_daily_loss_breaker_precedes_rejection(self):
        metrics = SimpleNamespace(
            day="2026-01-15",
            realized_pnl_today=-150.0,
            loss_limit=100.0,
        )
        trade, metrics, calls = self._dispatch(RULE_DAILY_LOSS, metrics)
        assert [event for event, _ in calls] == [
            "daily_loss_breaker",
            "trade_rejected",
        ]
        assert calls[0][1] == (metrics,)
        assert calls[1][1] == (
            trade,
            "daily loss breaker active for today",
        )

    def test_max_open_positions_precedes_rejection(self):
        metrics = SimpleNamespace(current_open_positions=3, limit=3)
        trade, metrics, calls = self._dispatch(
            RULE_MAX_OPEN_POSITIONS, metrics
        )
        assert [event for event, _ in calls] == [
            "max_open_positions_reached",
            "trade_rejected",
        ]
        assert calls[0][1] == (metrics,)
        assert calls[1][1] == (
            trade,
            "maximum open positions limit reached",
        )

    @pytest.mark.parametrize(
        "rule,reason",
        [
            (RULE_PORTFOLIO_RISK, "portfolio risk limit exceeded"),
            (RULE_PORTFOLIO_EXPOSURE, "portfolio exposure limit exceeded"),
            (RULE_SYMBOL_POSITIONS, "symbol concentration limit exceeded"),
            (RULE_SYMBOL_EXPOSURE, "symbol concentration limit exceeded"),
        ],
    )
    def test_single_rejection_alert(self, rule, reason):
        metrics = SimpleNamespace()
        trade, _, calls = self._dispatch(rule, metrics)
        assert [event for event, _ in calls] == ["trade_rejected"]
        assert calls[0][1] == (trade, reason)


# ---------------------------------------------------------------------------
# Handler compatibility bindings
# ---------------------------------------------------------------------------


class TestHandlerBindings:
    def test_handler_renderer_matches_the_shared_implementation(self):
        trade = _make_trade()
        metrics = SimpleNamespace(current_risk=1000.0, risk_limit=500.0)
        outcome = _outcome(RULE_PORTFOLIO_RISK, metrics)
        assert handlers._entry_gate_rejection_message(
            trade, outcome
        ) == analyze_reporting.entry_gate_rejection_message(trade, outcome)

    def test_handler_pnl_formatter_is_the_shared_implementation(self):
        assert (
            handlers._format_pnl_value
            is analyze_reporting.format_pnl_value
        )

    def test_handler_trade_risk_value_delegates_to_handler_trade_risk(self):
        trade = _make_trade()
        assert handlers.trade_risk_value(trade) == trade_risk(trade)

    def test_handler_wrapper_injects_the_live_dispatcher(self, monkeypatch):
        recorder = _Recorder()
        monkeypatch.setattr(handlers, "_dispatch_alert", recorder)
        trade = _make_trade()
        metrics = SimpleNamespace(current_risk=1000.0, risk_limit=500.0)
        asyncio.run(
            handlers._dispatch_entry_gate_alerts(
                trade, _outcome(RULE_PORTFOLIO_RISK, metrics)
            )
        )
        assert [event for event, _ in recorder.calls] == ["trade_rejected"]
        assert recorder.calls[0][1] == (
            trade,
            "portfolio risk limit exceeded",
        )

    def test_reporting_module_does_not_import_the_handler_layer(self):
        tree = ast.parse(inspect.getsource(analyze_reporting))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        # The alert dispatcher must be injected by the handler, never
        # imported back: that would create a cycle and would bypass the
        # handler's live ALERTS_ENABLED gate.
        assert not any(
            name == "telegram_bot.handlers"
            or name.startswith("telegram_bot.handlers.")
            for name in imported
        )
        assert not any(
            name == "config.settings"
            or name.startswith("config.settings.")
            for name in imported
        )


# ---------------------------------------------------------------------------
# Handler-level monkeypatch seams (pre-extraction runtime lookup)
# ---------------------------------------------------------------------------


class TestHandlerMonkeypatchSeams:
    """The handler renderer resolves its helpers from handler globals.

    Before the extraction the renderer read ``trade_risk_value``,
    ``trade_exposure`` and ``_format_pnl_value`` out of
    ``telegram_bot.handlers`` at call time. Patching each of those names
    must still change the rendered text.
    """

    def _render(self, rule, metrics, trade=None):
        return handlers._entry_gate_rejection_message(
            trade or _make_trade(), _outcome(rule, metrics)
        )

    def test_patched_pnl_formatter_changes_daily_loss_text(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            handlers, "_format_pnl_value", lambda value: "PATCHED"
        )
        metrics = SimpleNamespace(
            day="2026-01-15", realized_pnl_today=-150.0, loss_limit=100.0
        )
        text = self._render(RULE_DAILY_LOSS, metrics)
        assert "Today's Realized PnL: PATCHED\n" in text
        assert "-$150.00" not in text

    def test_patched_trade_risk_value_changes_portfolio_risk_text(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            handlers, "trade_risk_value", lambda trade: 1234.5
        )
        metrics = SimpleNamespace(current_risk=1.0, risk_limit=2.0)
        text = self._render(RULE_PORTFOLIO_RISK, metrics)
        assert "Trade Risk: $1,234.50\n" in text

    def test_patched_trade_risk_changes_portfolio_risk_text(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            handlers, "trade_risk", lambda trade: 555.0
        )
        metrics = SimpleNamespace(current_risk=1.0, risk_limit=2.0)
        text = self._render(RULE_PORTFOLIO_RISK, metrics)
        assert "Trade Risk: $555.00\n" in text
        assert "$20.00" not in text

    def test_patched_trade_risk_reaches_the_handler_wrapper(self, monkeypatch):
        monkeypatch.setattr(
            handlers, "trade_risk", lambda trade: 42.0
        )
        assert handlers.trade_risk_value(_make_trade()) == 42.0

    def test_patched_trade_exposure_changes_symbol_exposure_text(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            handlers, "trade_exposure", lambda trade: 7.0
        )
        metrics = SimpleNamespace(symbol_exposure=1.0, exposure_limit=2.0)
        text = self._render(RULE_SYMBOL_EXPOSURE, metrics)
        assert "Requested Exposure: $7.00\n" in text

    def test_all_three_seams_patch_together(self, monkeypatch):
        monkeypatch.setattr(
            handlers, "_format_pnl_value", lambda value: "P"
        )
        monkeypatch.setattr(
            handlers, "trade_risk_value", lambda trade: 1.0
        )
        monkeypatch.setattr(
            handlers, "trade_exposure", lambda trade: 2.0
        )
        metrics = SimpleNamespace(
            day="d", realized_pnl_today=0.0, loss_limit=100.0
        )
        assert "Today's Realized PnL: P\n" in self._render(
            RULE_DAILY_LOSS, metrics
        )

    def test_module_default_ignores_handler_patches(self, monkeypatch):
        monkeypatch.setattr(
            handlers, "_format_pnl_value", lambda value: "PATCHED"
        )
        metrics = SimpleNamespace(
            day="2026-01-15", realized_pnl_today=-150.0, loss_limit=100.0
        )
        text = analyze_reporting.entry_gate_rejection_message(
            _make_trade(), _outcome(RULE_DAILY_LOSS, metrics)
        )
        assert "Today's Realized PnL: -$150.00\n" in text

    def test_module_defaults_stay_patchable_in_the_module(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            analyze_reporting, "format_pnl_value", lambda value: "MODULE"
        )
        metrics = SimpleNamespace(
            day="d", realized_pnl_today=1.0, loss_limit=100.0
        )
        text = analyze_reporting.entry_gate_rejection_message(
            _make_trade(), _outcome(RULE_DAILY_LOSS, metrics)
        )
        assert "Today's Realized PnL: MODULE\n" in text

    def test_handler_trade_exposure_is_the_domain_function(self):
        from trader.paper.correlation_risk import trade_exposure

        assert handlers.trade_exposure is trade_exposure
