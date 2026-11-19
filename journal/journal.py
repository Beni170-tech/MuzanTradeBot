"""
MuzanTradeBot - Trade Journal

Records every validated trade decision so the bot's activity can be
reviewed after the fact. Guardian Mode relies on this journal for
accountability and auditing.

Milestone M-C makes the journal structured and operational:

* one JSON object per line (JSONL), so records can be read back and
  rotated without parsing free text;
* size-based rotation to ``trade_journal.jsonl.1`` .. ``.N`` so the
  journal cannot grow without bound;
* write failures are logged and reported, never raised: a broken
  journal must not break an ``/analyze`` reply or look like a placed
  trade. The outcome is returned to the caller and exposed through
  :func:`journal_status` for ``/status``.

The legacy plain-text ``logs/trade_journal.txt`` is left untouched; this
module only ever writes the structured file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: Structured journal file. Relative to the bot's working directory,
#: like the other ``logs/`` state files.
DEFAULT_JOURNAL_PATH = Path("logs/trade_journal.jsonl")

#: Rotate once the active file would grow past this many bytes.
DEFAULT_MAX_BYTES = 1_000_000

#: How many rotated files to keep (``.1`` is the newest).
DEFAULT_BACKUP_COUNT = 5

#: Last write error seen by this process, for ``/status``. Cleared by a
#: successful write so a transient failure does not linger.
_last_error: str | None = None


@dataclass(frozen=True)
class JournalStatus:
    """Read-only snapshot of the journal's operational health.

    Attributes:
        path: The active journal file.
        exists: Whether the active file exists on disk.
        size_bytes: Size of the active file (0 when missing).
        rotated_files: How many rotated backups are present.
        last_write_at: Modification time of the active file, formatted
            ``YYYY-MM-DD HH:MM:SS``, or None when it does not exist.
        last_error: The last write error recorded by this process, or
            None when the most recent write (if any) succeeded.
    """

    path: Path
    exists: bool
    size_bytes: int
    rotated_files: int
    last_write_at: str | None
    last_error: str | None


def rotated_path(path: Path, index: int) -> Path:
    """Return the path of rotated backup ``index`` for ``path``."""
    return path.with_name(f"{path.name}.{index}")


def rotate(path: Path, *, backup_count: int = DEFAULT_BACKUP_COUNT) -> None:
    """Rotate ``path`` into numbered backups, oldest first.

    ``backup_count`` of 0 or less discards the active file instead of
    keeping backups, so rotation can never retain unbounded history.

    Raises:
        OSError: If a file operation fails. Callers decide how to react.
    """
    if backup_count <= 0:
        path.unlink(missing_ok=True)
        return

    rotated_path(path, backup_count).unlink(missing_ok=True)
    for index in range(backup_count - 1, 0, -1):
        source = rotated_path(path, index)
        if source.exists():
            source.replace(rotated_path(path, index + 1))
    path.replace(rotated_path(path, 1))


def append_entry(
    entry: dict,
    *,
    path: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> bool:
    """Append one structured record as a JSON line, rotating if needed.

    Never raises for a write or serialization failure: the problem is
    logged, remembered for :func:`journal_status`, and reported through
    the return value. Success is only reported after the record has been
    flushed and fsynced to disk, and the handle is closed on every path,
    including a failed flush or fsync.

    Returns:
        True when the record was written and forced to disk, False when
        it was not.
    """
    global _last_error

    journal_path = Path(path) if path is not None else DEFAULT_JOURNAL_PATH
    try:
        line = json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n"
        encoded = line.encode("utf-8")

        journal_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            max_bytes > 0
            and journal_path.exists()
            and journal_path.stat().st_size + len(encoded) > max_bytes
        ):
            # Rotate before appending so a single oversized record still
            # lands in an empty active file rather than looping forever.
            rotate(journal_path, backup_count=backup_count)

        with open(journal_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            # A record that only reached a process buffer can be lost in
            # a crash, so flush and fsync before reporting success. The
            # context manager still closes the handle if either fails.
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError) as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Trade journal write failed (%s): %s", journal_path, exc)
        return False

    _last_error = None
    return True


def log_trade(
    symbol: str,
    decision: str,
    risk_verdict: str,
    *,
    path: str | Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> bool:
    """Append one trade entry to the structured trade journal.

    Args:
        symbol: The cryptocurrency symbol (e.g. "BTC").
        decision: The decision from the decision engine.
        risk_verdict: The verdict from the risk manager.

    Returns:
        True when the entry was written, False when the journal was
        unavailable. A failed write is reported, never raised, so a
        broken journal cannot block an analysis reply or appear to be a
        placed trade.
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "decision": decision,
        "risk_verdict": risk_verdict,
    }
    return append_entry(
        entry,
        path=path,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )


def journal_status(path: str | Path | None = None) -> JournalStatus:
    """Return the journal's path, size, rotation and last-write state.

    Never raises: missing or unreadable files are reported as such so
    ``/status`` can show a degraded journal instead of failing.
    """
    journal_path = Path(path) if path is not None else DEFAULT_JOURNAL_PATH

    exists = False
    size_bytes = 0
    last_write_at: str | None = None
    try:
        stat = journal_path.stat()
    except OSError:
        pass
    else:
        exists = True
        size_bytes = stat.st_size
        last_write_at = datetime.fromtimestamp(stat.st_mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    rotated_files = 0
    for index in range(1, DEFAULT_BACKUP_COUNT + 1):
        try:
            if rotated_path(journal_path, index).exists():
                rotated_files += 1
        except OSError:
            break

    return JournalStatus(
        path=journal_path,
        exists=exists,
        size_bytes=size_bytes,
        rotated_files=rotated_files,
        last_write_at=last_write_at,
        last_error=_last_error,
    )
