"""Execution mode safety helpers (Sprint 10.2).

This module is a small, explicit layer for execution mode selection. It
keeps PAPER as the safe default and makes invalid or future live modes
impossible to enable by accident.

It does not place, close, persist, or notify. It only interprets the
configured execution mode and answers safe questions about it.
"""

from __future__ import annotations

from config.settings import settings


def is_paper_mode() -> bool:
    """Return True when the bot is running in PAPER execution mode.

    PAPER is the safe default and the only valid mode right now.
    """
    return bool(settings.is_paper_mode())


def is_live_mode() -> bool:
    """Return True when the bot is running in a live execution mode.

    There is no live mode implemented yet, so this always returns False
    for valid configurations. This is intentionally conservative: a
    misconfigured module must never act as if live trading were enabled.
    """
    return bool(settings.is_live_mode())


def is_execution_mode_configured_safely() -> bool:
    """Return True when the configured execution mode is valid.

    An unrecognized execution mode is unsafe and must not enable any
    live behavior.
    """
    return bool(settings.is_execution_mode_valid())
