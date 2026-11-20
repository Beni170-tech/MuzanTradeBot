"""Shared pytest configuration.

Besides enabling pytest-asyncio, this module guards the operator's real
paper-trading state. Tests must never write the files the running bot
uses: a test run in a checkout that has live paper state would otherwise
overwrite the account and the journal history (audit finding D1).

Tests that touch paper state redirect ``telegram_bot.handlers.PAPER_STATE_PATH``
(and ``log_trade`` where the analysis pipeline runs) to ``tmp_path``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

pytest_plugins = ("pytest_asyncio",)

#: State files the running bot writes, relative to the project root.
REAL_PAPER_STATE_FILES = (
    Path("logs/paper_state.json"),
    Path("logs/trade_journal.jsonl"),
)


def _fingerprint(path: Path) -> str | None:
    """Return a content hash for ``path``, or None when it does not exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


@pytest.fixture(scope="session", autouse=True)
def real_paper_state_is_never_modified():
    """Fail the session when any test writes the real paper state files."""
    before = {path: _fingerprint(path) for path in REAL_PAPER_STATE_FILES}
    yield
    changed = [
        path
        for path in REAL_PAPER_STATE_FILES
        if _fingerprint(path) != before[path]
    ]
    assert not changed, (
        "tests modified the real paper-trading state: "
        + ", ".join(str(path) for path in changed)
        + ". Redirect telegram_bot.handlers.PAPER_STATE_PATH (and log_trade "
        "where the analysis pipeline runs) to tmp_path in the offending test."
    )
