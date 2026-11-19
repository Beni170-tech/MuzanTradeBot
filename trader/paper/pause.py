"""Global paper-trading pause control (Milestone M-C).

Operators can stop *new* paper entries without restarting the bot. The
pause is persisted to a small JSON control file so it survives restarts,
and it never touches existing positions: the background monitor, the
SL/TP and trailing-stop checks and ``/close`` all keep running.

Safety rules:

* A pause only ever blocks new entries. Nothing here can close, alter or
  protect a position, and nothing here enables trading.
* State that cannot be read (corrupt JSON, an unexpected shape, an I/O
  error) is treated as **paused**. A typo in the control file must never
  be the reason a new entry is allowed; only an explicit, valid
  ``{"paused": false}`` record resumes trading.
* A missing file means no pause was ever requested, so a fresh install
  keeps trading. Failure to *write* a pause leaves the previous state
  untouched, and the caller is told the request did not take effect.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from trader.paper.storage import _atomic_write_text

logger = logging.getLogger(__name__)

#: Persisted pause control. Relative to the bot's working directory,
#: like the other ``logs/`` state files.
DEFAULT_PAUSE_STATE_PATH = Path("logs/paper_pause.json")

#: Longest reason kept, so one command cannot bloat the state file.
MAX_REASON_LENGTH = 200

#: Reason recorded when the control file exists but cannot be trusted.
UNREADABLE_REASON = "pause control state is unreadable"


@dataclass(frozen=True, slots=True)
class PauseState:
    """Whether new paper entries are paused, and why.

    Attributes:
        paused: True when new entries must be blocked.
        reason: The operator-supplied reason, when one was given.
        updated_at: When the state was last written, when known.
        degraded: True when the outcome comes from unreadable control
            state rather than an explicit record. ``paused`` is then
            True, and the value must not be treated as a resume.
    """

    paused: bool
    reason: str | None = None
    updated_at: str | None = None
    degraded: bool = False


def _now_label() -> str:
    """The naive local timestamp used by the other paper state files."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clean_reason(reason: str | None) -> str | None:
    """Normalise an operator-supplied reason, or return None.

    Newlines and surrounding whitespace are collapsed so a reason stays a
    single readable line, and long input is truncated.
    """
    if reason is None:
        return None
    text = " ".join(reason.split())
    if not text:
        return None
    return text[:MAX_REASON_LENGTH]


def _degraded_state() -> PauseState:
    """Build the fail-closed state used when control state is unusable."""
    return PauseState(paused=True, reason=UNREADABLE_REASON, degraded=True)


def read_pause_state(path: str | Path | None = None) -> PauseState:
    """Read the pause control state. Never raises.

    A missing file is an explicit "no pause requested". Any other problem
    (unreadable file, invalid JSON, wrong shape) fails closed: the result
    is paused, so unreadable state can never enable trading.

    Args:
        path: The control file. Defaults to ``logs/paper_pause.json``.

    Returns:
        The pause state, with ``degraded`` set when the file could not be
        trusted.
    """
    state_path = Path(path) if path is not None else DEFAULT_PAUSE_STATE_PATH

    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return PauseState(paused=False)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Paper pause state %s is unreadable; failing closed (paused): %s",
            state_path,
            exc,
        )
        return _degraded_state()

    paused = data.get("paused") if isinstance(data, dict) else None
    if type(paused) is not bool:
        logger.warning(
            "Paper pause state %s has no boolean 'paused' field; failing "
            "closed (paused).",
            state_path,
        )
        return _degraded_state()

    reason = data.get("reason")
    updated_at = data.get("updated_at")
    return PauseState(
        paused=paused,
        reason=reason if isinstance(reason, str) and reason else None,
        updated_at=(
            updated_at if isinstance(updated_at, str) and updated_at else None
        ),
    )


def _write_pause_state(
    paused: bool, reason: str | None, path: str | Path | None
) -> PauseState | None:
    """Persist one pause state atomically.

    Returns:
        The written state, or None when the write failed. A failed write
        leaves whatever state was already on disk untouched.
    """
    state_path = Path(path) if path is not None else DEFAULT_PAUSE_STATE_PATH
    state = PauseState(
        paused=paused, reason=reason, updated_at=_now_label()
    )
    payload = json.dumps(
        {
            "paused": state.paused,
            "reason": state.reason,
            "updated_at": state.updated_at,
        },
        indent=2,
        ensure_ascii=False,
    )
    try:
        _atomic_write_text(state_path, payload)
    except OSError as exc:
        logger.warning(
            "Could not write paper pause state %s: %s", state_path, exc
        )
        return None
    return state


def request_pause(
    reason: str | None = None, path: str | Path | None = None
) -> PauseState | None:
    """Persist a request to block new paper entries.

    Returns:
        The stored state, or None when it could not be written (paper
        trading is then unchanged and the caller must say so).
    """
    return _write_pause_state(True, clean_reason(reason), path)


def clear_pause(path: str | Path | None = None) -> PauseState | None:
    """Persist an explicit resume of new paper entries.

    Returns:
        The stored state, or None when it could not be written.
    """
    return _write_pause_state(False, None, path)
