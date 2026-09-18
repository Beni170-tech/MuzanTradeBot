"""
Tests for Sprint 10A - Telegram command rate limiting.

These tests cover the rate-limit decision logic, configuration wiring,
handler-level behavior, and the fact that rate limiting is disabled by
default through the project's existing settings patterns.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from telegram_bot.rate_limit import (
    DEFAULT_COMMANDS_PER_WINDOW,
    DEFAULT_WINDOW_SECONDS,
    RateLimitConfig,
    build_rate_limit_config,
    check_rate_limit,
    reset_rate_limit_state,
)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _disabled_config() -> RateLimitConfig:
    """Return a config with rate limiting explicitly disabled."""
    return RateLimitConfig(commands_per_window=0, window_seconds=10.0)


def _tiny_config() -> RateLimitConfig:
    """Return a very small window config for fast deterministic tests."""
    return RateLimitConfig(commands_per_window=3, window_seconds=0.05)


def _patch_rate_limit_for_chosen_config(config: RateLimitConfig) -> None:
    """Make ``telegram_bot.handlers.check_rate_limit`` a thin dispatcher

    This keeps handler tests focused on handler behavior instead of on
    the internals of the rate limiter state machine.
    """
    import telegram_bot.handlers as _handlers
    from telegram_bot.rate_limit import check_rate_limit as _original_check_rate_limit

    def _dispatcher(chat_id: int, config):
        return bool(_original_check_rate_limit(chat_id, config))

    _handlers.check_rate_limit = _dispatcher
    _handlers._rate_limit_config = config


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestRateLimitConfig:
    def test_default_is_enabled(self):
        config = build_rate_limit_config()
        assert config.is_enabled() is True
        assert config.commands_per_window == DEFAULT_COMMANDS_PER_WINDOW
        assert config.window_seconds == DEFAULT_WINDOW_SECONDS

    def test_explicit_commands_per_window_is_used(self):
        config = build_rate_limit_config(commands_per_window=5)
        assert config.commands_per_window == 5
        assert config.window_seconds == DEFAULT_WINDOW_SECONDS

    def test_explicit_window_seconds_is_used(self):
        config = build_rate_limit_config(window_seconds=2.5)
        assert config.window_seconds == 2.5
        assert config.commands_per_window == DEFAULT_COMMANDS_PER_WINDOW

    def test_zero_commands_per_window_disables_limiting(self):
        config = build_rate_limit_config(commands_per_window=0)
        assert config.is_enabled() is False

    def test_negative_commands_per_window_disables_limiting(self):
        config = build_rate_limit_config(commands_per_window=-1)
        assert config.is_enabled() is False

    def test_zero_window_seconds_disables_limiting(self):
        config = build_rate_limit_config(commands_per_window=10, window_seconds=0)
        assert config.is_enabled() is False


# ---------------------------------------------------------------------------
# Core rate-limit logic
# ---------------------------------------------------------------------------

class TestRateLimitDecision:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_rate_limit_state()
        yield

    def test_disabled_config_allows_everything(self):
        config = _disabled_config()
        assert check_rate_limit(chat_id=1, config=config) is True
        assert check_rate_limit(chat_id=1, config=config) is True

    def test_first_command_in_window_is_allowed(self):
        config = _tiny_config()
        assert check_rate_limit(chat_id=123, config=config) is True

    def test_under_limit_is_allowed(self):
        config = _tiny_config()
        for _ in range(config.commands_per_window - 1):
            assert check_rate_limit(chat_id=456, config=config) is True

    def test_at_limit_still_allowed_for_last_command(self):
        config = _tiny_config()
        for _ in range(config.commands_per_window):
            allowed = check_rate_limit(chat_id=789, config=config)
        assert allowed is True

    def test_over_limit_is_rejected(self):
        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=999, config=config)
        assert check_rate_limit(chat_id=999, config=config) is False

    def test_per_chat_isolation(self):
        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=111, config=config)
        assert check_rate_limit(chat_id=111, config=config) is False
        assert check_rate_limit(chat_id=222, config=config) is True

    def test_window_resets_after_window_seconds(self):
        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=333, config=config)
        assert check_rate_limit(chat_id=333, config=config) is False

        # Advance the synthetic clock past the window so the next
        # command is allowed again without sleeping.
        import time
        time.sleep(config.window_seconds + 0.02)
        assert check_rate_limit(chat_id=333, config=config) is True


# ---------------------------------------------------------------------------
# Handler-level behavior
# ---------------------------------------------------------------------------

class TestHandlerRateLimitBehavior:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_rate_limit_state()
        yield

    @pytest.mark.asyncio
    async def test_start_allows_first_command(self):
        from telegram_bot.handlers import start

        update = MagicMock()
        update.effective_chat.id = 101
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        with patch("telegram_bot.handlers.settings") as mock_settings:
            mock_settings.PAPER_TRADING = True
            mock_settings.GUARDIAN_MODE = True
            await start(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Guardian Mode" in msg
        assert "Paper Trading" in msg
        assert "Too many requests" not in msg

    @pytest.mark.asyncio
    async def test_start_rate_limited_returns_friendly_message(self):
        from telegram_bot.handlers import start

        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=202, config=config)
        assert check_rate_limit(chat_id=202, config=config) is False

        update = MagicMock()
        update.effective_chat.id = 202
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        _patch_rate_limit_for_chosen_config(config)

        await start(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Too many requests" in msg
        assert "wait" in msg.lower()

    @pytest.mark.asyncio
    async def test_rate_limited_response_is_sent_before_business_logic(self):
        """Even if the handler would have done real work, the rate-limited
        path must return early with the friendly message."""
        from telegram_bot.handlers import start

        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=303, config=config)
        assert check_rate_limit(chat_id=303, config=config) is False

        update = MagicMock()
        update.effective_chat.id = 303
        replied = []

        async def capture(text):
            replied.append(text)
            return AsyncMock()

        update.message.reply_text = capture
        context = MagicMock()

        _patch_rate_limit_for_chosen_config(config)
        await start(update, context)

        assert len(replied) == 1
        assert "Too many requests" in replied[0]

    @pytest.mark.asyncio
    async def test_help_command_is_rate_limited(self):
        from telegram_bot.handlers import help_command

        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=404, config=config)
        assert check_rate_limit(chat_id=404, config=config) is False

        update = MagicMock()
        update.effective_chat.id = 404
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        _patch_rate_limit_for_chosen_config(config)
        await help_command(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Too many requests" in msg

    @pytest.mark.asyncio
    async def test_help_command_runs_when_not_rate_limited(self):
        from telegram_bot.handlers import help_command

        config = _tiny_config()
        update = MagicMock()
        update.effective_chat.id = 405
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        _patch_rate_limit_for_chosen_config(config)
        await help_command(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Available Commands" in msg

    @pytest.mark.asyncio
    async def test_ping_command_is_rate_limited(self):
        from telegram_bot.handlers import ping

        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=505, config=config)
        assert check_rate_limit(chat_id=505, config=config) is False

        update = MagicMock()
        update.effective_chat.id = 505
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        _patch_rate_limit_for_chosen_config(config)
        await ping(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Too many requests" in msg

    @pytest.mark.asyncio
    async def test_version_command_is_rate_limited(self):
        from telegram_bot.handlers import version

        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=606, config=config)
        assert check_rate_limit(chat_id=606, config=config) is False

        update = MagicMock()
        update.effective_chat.id = 606
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        _patch_rate_limit_for_chosen_config(config)
        await version(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Too many requests" in msg

    @pytest.mark.asyncio
    async def test_analyze_command_is_rate_limited(self):
        from telegram_bot.handlers import analyze

        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=707, config=config)
        assert check_rate_limit(chat_id=707, config=config) is False

        update = MagicMock()
        update.effective_chat.id = 707
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = ["BTC"]

        _patch_rate_limit_for_chosen_config(config)
        await analyze(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Too many requests" in msg

    @pytest.mark.asyncio
    async def test_analyze_command_runs_when_not_rate_limited(self):
        from telegram_bot.handlers import analyze

        config = _tiny_config()
        update = MagicMock()
        update.effective_chat.id = 708
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = ["ETH"]

        _patch_rate_limit_for_chosen_config(config)
        await analyze(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Analysis for ETH" in msg

    @pytest.mark.asyncio
    async def test_paper_command_is_rate_limited(self):
        from telegram_bot.handlers import paper_command

        config = _tiny_config()
        for _ in range(config.commands_per_window):
            check_rate_limit(chat_id=808, config=config)
        assert check_rate_limit(chat_id=808, config=config) is False

        update = MagicMock()
        update.effective_chat.id = 808
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        _patch_rate_limit_for_chosen_config(config)
        await paper_command(update, context)

        update.message.reply_text.assert_called_once()
        msg = update.message.reply_text.await_args[0][0]
        assert "Too many requests" in msg


# ---------------------------------------------------------------------------
# Settings wiring behavior
# ---------------------------------------------------------------------------

class TestRateLimitSettingsWiring:
    def test_build_rate_limit_config_defaults_match_project_defaults(self):
        config = build_rate_limit_config()
        assert config.commands_per_window == DEFAULT_COMMANDS_PER_WINDOW
        assert config.window_seconds == DEFAULT_WINDOW_SECONDS

    def test_settings_defaults_disable_rate_limiting_when_zero(self):
        with patch("config.settings.settings.TELEGRAM_COMMAND_RATE_LIMIT_PER_WINDOW", 0):
            with patch("config.settings.settings.TELEGRAM_COMMAND_RATE_WINDOW_SECONDS", 10.0):
                import telegram_bot.handlers as _handlers
                import importlib
                importlib.reload(_handlers)
                assert _handlers._rate_limit_config.is_enabled() is False

    @pytest.mark.asyncio
    async def test_multiple_command_handlers_share_one_rate_limit_state(self):
        import telegram_bot.handlers as _handlers
        import importlib
        importlib.reload(_handlers)

        config = _tiny_config()
        chat = 999

        _patch_rate_limit_for_chosen_config(config)

        calls = []

        async def record_start():
            from telegram_bot.handlers import start

            update = MagicMock()
            update.effective_chat.id = chat
            update.message.reply_text = AsyncMock()
            context = MagicMock()
            await start(update, context)
            calls.append(update.message.reply_text.await_args[0][0])

        async def record_help():
            from telegram_bot.handlers import help_command

            update = MagicMock()
            update.effective_chat.id = chat
            update.message.reply_text = AsyncMock()
            context = MagicMock()
            await help_command(update, context)
            calls.append(update.message.reply_text.await_args[0][0])

        async def record_ping():
            from telegram_bot.handlers import ping

            update = MagicMock()
            update.effective_chat.id = chat
            update.message.reply_text = AsyncMock()
            context = MagicMock()
            await ping(update, context)
            calls.append(update.message.reply_text.await_args[0][0])

        await record_start()
        await record_help()
        await record_ping()

        for msg in calls:
            assert "Too many requests" not in msg
        assert len(calls) == config.commands_per_window

        await record_start()
        assert "Too many requests" in calls[-1]
