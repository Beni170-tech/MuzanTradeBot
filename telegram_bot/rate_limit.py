"""
MuzanTradeBot - Telegram command rate limiting (Sprint 10A)

A small, configurable per-chat rate limiter that protects Telegram
command handlers from accidental or abusive spam without blocking
normal use.

Design notes:
- Per-chat windows only. No cross-chat global buckets, no user identity
  beyond the Telegram chat id.
- Pure functions for the rate-limit decision so the logic is easy to
  test without touching the rest of the bot.
- The handler integration is explicit: handlers call the helper and
  return early with a friendly message when rate limited.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#---------------------------------------------------------------------------
# Configurable defaults
#---------------------------------------------------------------------------

# Default per-chat command rate limit. When <= 0, rate limiting is
# disabled entirely. Expressed as commands allowed in the configured
# window.
DEFAULT_COMMANDS_PER_WINDOW = 12

# Default window length in seconds.
DEFAULT_WINDOW_SECONDS = 10.0


@dataclass
class RateLimitConfig:
    """Runtime rate-limit configuration.

    The defaults match the Settings-based configuration, but this object
    is accepted directly by the rate limiter so tests can inject values
    without patching environment settings.
    """

    commands_per_window: int = DEFAULT_COMMANDS_PER_WINDOW
    window_seconds: float = DEFAULT_WINDOW_SECONDS

    def is_enabled(self) -> bool:
        """Rate limiting is active when the limit is positive and the
        window is positive.

        Either value being <= 0 effectively disables limiting, which
        keeps the Settings-driven wiring simple: setting either one to
        a non-positive value is enough to turn the limiter off.
        """
        return self.commands_per_window > 0 and self.window_seconds > 0


#---------------------------------------------------------------------------
# State
#---------------------------------------------------------------------------

@dataclass
class _ChatWindow:
    """Recency state for one chat.

    Kept intentionally minimal: we only track the last window reset and
    the command count since that reset.
    """

    window_start: float
    count: int = 0


class _RateLimitState:
    """In-memory per-chat rate-limit state.

    This is process-local state, which is appropriate for a single-process
    polling bot. If the process restarts, the windows reset, which is the
    safe behavior for a spam guard.
    """

    def __init__(self) -> None:
        self._windows: dict[int, _ChatWindow] = {}

    def clear(self) -> None:
        self._windows.clear()

    def record_and_check(self, now: float) -> None:
        """Set the synthetic clock used by this state.

        This exists so tests can control time without touching
        ``time.monotonic`` directly.
        """
        self._now = now

    def record_and_check(
        self,
        chat_id: int,
        config: RateLimitConfig,
        now: float,
    ) -> bool:
        """Record a command attempt and return True if it is allowed.

        Args:
            chat_id: Telegram chat id.
            config: Active rate-limit configuration.
            now: Current time as returned by time.monotonic().

        Returns:
            True when the command is allowed, False when the chat is
            temporarily rate limited.
        """
        if not config.is_enabled():
            return True

        window = self._windows.get(chat_id)
        if window is None:
            self._windows[chat_id] = _ChatWindow(window_start=now, count=1)
            return True

        if now - window.window_start >= config.window_seconds:
            window.window_start = now
            window.count = 1
            return True

        if window.count >= config.commands_per_window:
            return False

        window.count += 1
        return True

    def clear(self) -> None:
        self._windows.clear()

    def advance_clock(self, now: float) -> None:
        """Set the synthetic clock used by this state.

        This exists so tests can control time without touching
        ``time.monotonic`` directly.
        """
        self._now = now

    def clear(self) -> None:
        self._windows.clear()

    def advance_clock(self, now: float) -> None:
        """Set the synthetic clock used by this state.

        This exists so tests can control time without touching
        ``time.monotonic`` directly.
        """
        self._now = now

    def advance_clock(self, now: float) -> None:
        """Set the synthetic clock used by this state.

        This exists so tests can control time without touching
        ``time.monotonic`` directly.
        """
        self._now = now


_rate_limit_state = _RateLimitState()


#---------------------------------------------------------------------------
# Public API
#---------------------------------------------------------------------------

def build_rate_limit_config(
    *,
    commands_per_window: int | None = None,
    window_seconds: float | None = None,
) -> RateLimitConfig:
    """Build a RateLimitConfig from explicit overrides.

    Any value left as None falls back to the safe defaults. This helper
    exists so callers (including Settings-based wiring) can express
    "use configuration when present, otherwise use the safe default"
    without branching in multiple places.
    """
    return RateLimitConfig(
        commands_per_window=(
            commands_per_window if commands_per_window is not None
            else DEFAULT_COMMANDS_PER_WINDOW
        ),
        window_seconds=(
            window_seconds if window_seconds is not None
            else DEFAULT_WINDOW_SECONDS
        ),
    )


def check_rate_limit(
    chat_id: int,
    config: RateLimitConfig,
) -> bool:
    """Check whether a command from *chat_id* is currently allowed.

    This records the attempt and returns the allow/deny decision.

    Args:
        chat_id: Telegram chat id.
        config: Active rate-limit configuration.

    Returns:
        True when the command is allowed, False when the chat is
        temporarily rate limited.
    """
    return _rate_limit_state.record_and_check(
        chat_id,
        config,
        time.monotonic(),
    )


def reset_rate_limit_state() -> None:
    """Clear all in-memory rate-limit state.

    Provided mainly for testing and for hard reset scenarios. Normal
    operation never needs to call this.
    """
    _rate_limit_state._windows.clear()
