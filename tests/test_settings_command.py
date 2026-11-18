"""Tests for /settings and its pure formatter (Milestone M-A).

The report must show execution mode, every configured limit (with
disabled limits visible), authorization status and alert status, without
ever printing credentials.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import telegram_bot.handlers as handlers
from config.settings import settings
from telegram_bot import settings_report


def _formatted(**overrides):
    values = dict(
        execution_mode="PAPER",
        paper_trading=True,
        guardian_mode=True,
        portfolio_risk_limit=0.0,
        portfolio_exposure_limit=0.0,
        daily_loss_limit=0.0,
        max_open_positions=0,
        max_positions_per_symbol=0,
        max_exposure_per_symbol=0.0,
        authorization_status="locked (no authorized IDs configured)",
        alerts_enabled=False,
        alert_chat_configured=False,
    )
    values.update(overrides)
    return settings_report.format_settings_message(**values)


def _make_update():
    update = MagicMock()
    update.effective_chat.id = 123
    update.message.reply_text = AsyncMock()
    return update


class TestSettingsReport:
    def test_shows_execution_mode_and_status(self):
        message = _formatted()
        assert "⚙️ Bot Settings" in message
        assert "Mode: PAPER" in message
        assert "Paper Trading: Enabled" in message
        assert "Guardian Mode: Active" in message

    def test_disabled_limits_are_explicit(self):
        message = _formatted()
        for label in (
            "Portfolio Risk Limit",
            "Portfolio Exposure Limit",
            "Daily Loss Limit",
            "Max Open Positions",
            "Max Positions per Symbol",
            "Max Exposure per Symbol",
        ):
            assert f"{label}: disabled" in message

    def test_configured_limits_are_formatted(self):
        message = _formatted(
            portfolio_risk_limit=250.0,
            portfolio_exposure_limit=5000.0,
            daily_loss_limit=125.5,
            max_open_positions=3,
            max_positions_per_symbol=2,
            max_exposure_per_symbol=1500.0,
        )
        assert "Portfolio Risk Limit: $250.00" in message
        assert "Portfolio Exposure Limit: $5,000.00" in message
        assert "Daily Loss Limit: $125.50" in message
        assert "Max Open Positions: 3" in message
        assert "Max Positions per Symbol: 2" in message
        assert "Max Exposure per Symbol: $1,500.00" in message

    def test_authorization_status_is_shown(self):
        message = _formatted(authorization_status="2 authorized ID(s)")
        assert "Authorization: 2 authorized ID(s)" in message

    def test_alerts_disabled_state(self):
        message = _formatted()
        assert "Alerts: Disabled" in message
        assert "Alert Chat: n/a (alerts disabled)" in message

    def test_alerts_enabled_and_configured(self):
        message = _formatted(
            alerts_enabled=True, alert_chat_configured=True
        )
        assert "Alerts: Enabled" in message
        assert "Alert Chat: Configured" in message

    def test_alerts_enabled_without_chat_is_visible(self):
        message = _formatted(
            alerts_enabled=True, alert_chat_configured=False
        )
        assert "Alert Chat: Not configured" in message

    def test_report_contains_no_credentials(self):
        message = _formatted()
        lowered = message.lower()
        assert "token" not in lowered
        assert "secret" not in lowered
        assert "api" not in lowered


class TestSettingsCommand:
    @pytest.mark.asyncio
    async def test_reports_live_configuration_without_secrets(
        self, monkeypatch
    ):
        monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
        monkeypatch.setattr(settings, "EXECUTION_MODE", "PAPER")
        monkeypatch.setattr(settings, "PAPER_TRADING", True)
        monkeypatch.setattr(settings, "GUARDIAN_MODE", True)
        monkeypatch.setattr(settings, "PORTFOLIO_RISK_LIMIT", 100.0)
        monkeypatch.setattr(settings, "PORTFOLIO_MAX_OPEN_POSITIONS", 2)
        monkeypatch.setattr(settings, "ALERTS_ENABLED", True)
        monkeypatch.setattr(settings, "TELEGRAM_ALERT_CHAT_ID", "999")
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "123,456")
        monkeypatch.setattr(
            settings, "TELEGRAM_BOT_TOKEN", "secret-token-value"
        )

        update = _make_update()
        await handlers.settings_command(update, MagicMock())

        message = update.message.reply_text.await_args[0][0]
        assert "Mode: PAPER" in message
        assert "Portfolio Risk Limit: $100.00" in message
        assert "Max Open Positions: 2" in message
        assert "Alerts: Enabled" in message
        assert "Alert Chat: Configured" in message
        assert "2 authorized ID(s)" in message
        assert "secret-token-value" not in message

    @pytest.mark.asyncio
    async def test_locked_authorization_is_visible(self, monkeypatch):
        monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
        monkeypatch.setattr(settings, "TELEGRAM_ALLOWED_CHAT_IDS", "")

        update = _make_update()
        await handlers.settings_command(update, MagicMock())

        message = update.message.reply_text.await_args[0][0]
        assert "Authorization: locked" in message

    @pytest.mark.asyncio
    async def test_settings_command_is_rate_limited(self, monkeypatch):
        monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: False)

        update = _make_update()
        await handlers.settings_command(update, MagicMock())

        message = update.message.reply_text.await_args[0][0]
        assert "Too many requests" in message

    @pytest.mark.asyncio
    async def test_defaults_show_disabled_limits(self, monkeypatch):
        monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
        for name, value in (
            ("PORTFOLIO_RISK_LIMIT", 0.0),
            ("PORTFOLIO_EXPOSURE_LIMIT", 0.0),
            ("PORTFOLIO_DAILY_LOSS_LIMIT", 0.0),
            ("PORTFOLIO_MAX_OPEN_POSITIONS", 0),
            ("PORTFOLIO_MAX_POSITIONS_PER_SYMBOL", 0),
            ("PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL", 0.0),
        ):
            monkeypatch.setattr(settings, name, value)

        update = _make_update()
        await handlers.settings_command(update, MagicMock())

        message = update.message.reply_text.await_args[0][0]
        assert message.count("disabled") >= 6


def test_help_command_lists_settings(monkeypatch):
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    update = _make_update()
    import asyncio

    asyncio.run(handlers.help_command(update, MagicMock()))
    message = update.message.reply_text.await_args[0][0]
    assert "/settings" in message
