"""Tests for execution-mode / PAPER_TRADING consistency safety (Sprint 10.2)."""

from __future__ import annotations

import sys
import os

# Ensure project root is importable for config monkeypatching below.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import Settings
import pytest


def _set(paper: bool, mode: str) -> tuple[bool, str]:
    """Snapshot and restore Settings class-level config for the test body."""
    original_paper = Settings.PAPER_TRADING
    original_mode = Settings.EXECUTION_MODE
    try:
        Settings.PAPER_TRADING = paper
        Settings.EXECUTION_MODE = mode
        return original_paper, original_mode
    finally:
        Settings.PAPER_TRADING = original_paper
        Settings.EXECUTION_MODE = original_mode


def test_default_config_is_consistent_and_safe() -> None:
    # Default environment gives PAPER_TRADING=True and EXECUTION_MODE=PAPER.
    # That combination is valid and must never be treated as live-enabled.
    assert Settings.PAPER_TRADING is True
    assert Settings.EXECUTION_MODE == "PAPER"
    assert Settings.is_paper_mode() is True
    assert Settings.is_live_mode() is False
    assert Settings.is_execution_mode_valid() is True
    assert Settings.execution_mode_is_safe_for_current_intent() is True


def test_paper_true_requires_paper_mode() -> None:
    # PAPER_TRADING=True must be paired with EXECUTION_MODE=PAPER.
    # Any other EXECUTION_MODE while PAPER_TRADING=True is an unsafe mismatch.
    for bad_mode in ("", "live", "LIVE", "pApEr", "FOO", "1", "PaperTrading"):
        Settings.PAPER_TRADING = True
        Settings.EXECUTION_MODE = bad_mode
        assert Settings.is_execution_mode_valid() is False
        assert Settings.execution_mode_is_safe_for_current_intent() is False
        assert Settings.is_live_mode() is False
    Settings.PAPER_TRADING = True
    Settings.EXECUTION_MODE = "PAPER"
    assert Settings.execution_mode_is_safe_for_current_intent() is True


def test_paper_false_is_not_live_and_not_implicitly_safe() -> None:
    # PAPER_TRADING=False is currently an "off" config, not a live config.
    # Because live execution is not implemented, PAPER_TRADING=False must still
    # not be silently treated as a valid live execution mode.
    Settings.PAPER_TRADING = False
    Settings.EXECUTION_MODE = "PAPER"
    # With EXECUTION_MODE=PAPER, is_paper_mode() is still True by definition,
    # even though paper trading is currently disabled at runtime. That is fine:
    # is_paper_mode() describes the execution mode label, not the runtime
    # PAPER_TRADING switch.
    assert Settings.is_paper_mode() is True
    assert Settings.is_live_mode() is False
    assert Settings.is_execution_mode_valid() is True
    # A valid-but-disabled paper config is currently considered safe to start,
    # because no live mode exists yet. The key safety guarantee is that it is
    # not treated as live-enabled.
    assert Settings.execution_mode_is_safe_for_current_intent() is True
    assert Settings.is_live_mode() is False


@pytest.mark.parametrize(
    "paper,mode,safe,live",
    [
        (True, "PAPER", True, False),
        (True, "LIVE", False, False),
        (True, "pApEr", False, False),
        (True, "", False, False),
        (False, "PAPER", True, False),
        (False, "LIVE", False, False),
        (False, "", False, False),
    ],
)
def test_execution_mode_safety_table(paper: bool, mode: str, safe: bool, live: bool) -> None:
    Settings.PAPER_TRADING = paper
    Settings.EXECUTION_MODE = mode
    assert Settings.is_live_mode() is live
    assert Settings.execution_mode_is_safe_for_current_intent() is safe


# ---------------------------------------------------------------------------
# Runtime enforcement (Sprint 10.2)
# ---------------------------------------------------------------------------

def test_runtime_safety_check_blocks_unsafe_config() -> None:
    """An unsafe execution-mode config must fail the safety check.

    This mirrors the gate used in main.py at startup: when the configured
    execution mode is invalid or inconsistent with PAPER_TRADING, the bot
    must not start. The check itself is pure config validation and never
    enables live trading.
    """
    # Safe default: PAPER_TRADING=True, EXECUTION_MODE=PAPER
    Settings.PAPER_TRADING = True
    Settings.EXECUTION_MODE = "PAPER"
    assert Settings.execution_mode_is_safe_for_current_intent() is True

    # Unsafe: PAPER_TRADING=True with a non-PAPER mode
    Settings.EXECUTION_MODE = "LIVE"
    assert Settings.execution_mode_is_safe_for_current_intent() is False

    # Unsafe: unrecognized mode, even with PAPER_TRADING=False
    Settings.PAPER_TRADING = False
    Settings.EXECUTION_MODE = "UNKNOWN"
    assert Settings.execution_mode_is_safe_for_current_intent() is False

    # Safe: PAPER_TRADING=False with a valid (PAPER) mode — off config,
    # not live. Safe to start under current rules.
    Settings.EXECUTION_MODE = "PAPER"
    assert Settings.execution_mode_is_safe_for_current_intent() is True
    assert Settings.is_live_mode() is False


def test_runtime_safety_check_never_enables_live() -> None:
    """The safety check gates startup; it never turns on live trading.

    Even when the config passes the safety check, the bot remains in
    paper mode. A passing safety check under the current code means the
    config is consistent and valid, not that live execution is enabled.
    """
    Settings.PAPER_TRADING = True
    Settings.EXECUTION_MODE = "PAPER"
    assert Settings.execution_mode_is_safe_for_current_intent() is True
    assert Settings.is_live_mode() is False
    assert Settings.is_paper_mode() is True

    # An invalid mode must fail the safety check and still not be live.
    Settings.EXECUTION_MODE = "LIVE"
    assert Settings.execution_mode_is_safe_for_current_intent() is False
    assert Settings.is_live_mode() is False
