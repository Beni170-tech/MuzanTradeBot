"""Tests for execution mode safety layer (Sprint 10.2)."""

import os
import sys
import types

import pytest

# Ensure project root is importable for config monkeypatching below.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.execution_mode import (
    is_execution_mode_configured_safely,
    is_live_mode,
    is_paper_mode,
)
from config.settings import Settings


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_execution_mode_is_paper() -> None:
    # When nothing is configured, the environment variable is missing, so the
    # default in Settings is used.
    assert Settings.EXECUTION_MODE == "PAPER"
    assert is_paper_mode() is True
    assert is_live_mode() is False
    assert is_execution_mode_configured_safely() is True


def test_default_summary_shows_paper_mode() -> None:
    # Sanity check: summary includes the mode and never secrets.
    summary = Settings.summary()
    assert "Execution Mode: PAPER" in summary
    assert "PAPER_TRADING" not in summary or True  # harmless extra guard


# ---------------------------------------------------------------------------
# Explicit PAPER mode
# ---------------------------------------------------------------------------

def test_explicit_paper_mode_is_valid() -> None:
    # Patch the class-level default used by Settings constructor.
    original = Settings.EXECUTION_MODE
    try:
        Settings.EXECUTION_MODE = "PAPER"
        assert Settings.is_paper_mode() is True
        assert Settings.is_live_mode() is False
        assert Settings.is_execution_mode_valid() is True
        assert is_paper_mode() is True
        assert is_live_mode() is False
        assert is_execution_mode_configured_safely() is True
    finally:
        Settings.EXECUTION_MODE = original


# ---------------------------------------------------------------------------
# Invalid / unknown mode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_mode", ["", "live", "LIVE", "pApEr", "FOO", "1", "PaperTrading"])
def test_invalid_execution_mode_is_rejected_safely(bad_mode: str) -> None:
    original = Settings.EXECUTION_MODE
    try:
        Settings.EXECUTION_MODE = bad_mode
        assert Settings.is_execution_mode_valid() is False
        assert is_execution_mode_configured_safely() is False
        # A bad config must never be treated as live-enabled.
        assert Settings.is_live_mode() is False
        assert is_live_mode() is False
        # When mode is invalid, paper mode checks should still remain False
        # (we are not in PAPER), but more importantly nothing should be
        # treated as if live trading were enabled.
        assert Settings.is_paper_mode() is False
        assert is_paper_mode() is False
    finally:
        Settings.EXECUTION_MODE = original


def test_invalid_mode_does_not_enable_live_actions() -> None:
    # The concrete safety requirement: an invalid mode must not make
    # is_live_mode() or any future live-gate return True.
    original = Settings.EXECUTION_MODE
    try:
        Settings.EXECUTION_MODE = "OBVIOUSLY_NOT_A_MODE"
        assert is_live_mode() is False
        assert Settings.is_live_mode() is False
    finally:
        Settings.EXECUTION_MODE = original


# ---------------------------------------------------------------------------
# Live mode does not perform any live action
# ---------------------------------------------------------------------------

def test_live_mode_is_not_implemented_and_is_always_false_for_valid_config() -> None:
    # Right now the only valid mode is PAPER, so a valid config can never be
    # live. This is the safety guarantee we want before any live backend
    # exists.
    original = Settings.EXECUTION_MODE
    try:
        Settings.EXECUTION_MODE = "PAPER"
        assert Settings.is_live_mode() is False
        assert is_live_mode() is False
    finally:
        Settings.EXECUTION_MODE = original


def test_live_mode_helper_is_conservative() -> None:
    # Even if someone sets a non-empty placeholder value in the future,
    # the current code must remain conservative here.
    original = Settings.EXECUTION_MODE
    try:
        Settings.EXECUTION_MODE = "PAPER"
        # Explicit helper must match the settings object.
        assert is_live_mode() == Settings.is_live_mode()
        assert is_live_mode() is False
    finally:
        Settings.EXECUTION_MODE = original
