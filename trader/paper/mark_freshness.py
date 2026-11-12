"""Freshness checks for recorded paper market marks.

Equity-based paper position sizing values open positions at their last
recorded market marks. A mark is only safe to size from while it is
recent: if the background position monitor stops updating, an old mark
would otherwise keep approving trades against prices the market has
already left behind.

Rules:
    - Every open position must carry a valid, strictly positive
      ``mark_price`` AND a parseable UTC ``mark_at`` timestamp.
    - A missing timestamp is never treated as fresh. This also covers
      accounts saved before mark timestamps existed.
    - A mark older than the configured maximum age is stale.
    - The maximum age is two monitor polling intervals
      (``PAPER_MONITOR_INTERVAL_SECONDS``): one missed cycle plus
      scheduling jitter. When the monitor is disabled (interval <= 0),
      a conservative default of two default intervals (90 seconds) is
      used for marks recorded by the analysis flow.
    - No open positions means there is nothing marked to verify, so
      sizing may proceed from the settled balance.

This module is pure: no I/O, no side effects. ``now`` is injectable so
the freshness decision is deterministic in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

# A mark may be up to two monitor cycles old (one missed refresh plus
# jitter) before the bot refuses to size from it.
MISSED_CYCLES_ALLOWED = 2

# Default monitor interval, used as the freshness basis when the
# background monitor is disabled. Matches Settings' default of 45s.
DEFAULT_MONITOR_INTERVAL_SECONDS = 45.0
DEFAULT_MAX_MARK_AGE_SECONDS = (
    DEFAULT_MONITOR_INTERVAL_SECONDS * MISSED_CYCLES_ALLOWED
)


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def max_mark_age_seconds(monitor_interval_seconds: Any) -> float:
    """Return the maximum accepted mark age in seconds.

    Args:
        monitor_interval_seconds: The configured background monitor
            interval (``PAPER_MONITOR_INTERVAL_SECONDS``). Values that
            are missing, non-numeric, or <= 0 fall back to the default
            interval, because a disabled monitor cannot refresh marks.

    Returns:
        Two monitor intervals, i.e. the age after which a mark is stale.
    """
    if (
        isinstance(monitor_interval_seconds, bool)
        or not isinstance(monitor_interval_seconds, (int, float))
        or not math.isfinite(monitor_interval_seconds)
        or monitor_interval_seconds <= 0
    ):
        return DEFAULT_MAX_MARK_AGE_SECONDS
    return float(monitor_interval_seconds) * MISSED_CYCLES_ALLOWED


def parse_mark_timestamp(value: Any) -> datetime | None:
    """Parse a stored mark timestamp, or return None when unusable.

    Accepts ISO 8601 strings (the format written by
    ``PaperOrder.mark_to_market``). A naive timestamp is interpreted as
    UTC so a stored value always maps to an absolute instant.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def valid_mark_price(value: Any) -> float | None:
    """Return the mark price as a float, or None when it is unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return float(value)


@dataclass(frozen=True)
class MarksFreshnessResult:
    """Outcome of checking every open paper position's mark.

    Attributes:
        fresh: True only when every open position has a valid mark with
            a recent timestamp (or there are no open positions).
        reason: Human-readable explanation when ``fresh`` is False.
        max_age_seconds: The maximum age applied to this check.
        stale_symbols: Symbols whose mark is too old (or from the future).
        missing_mark_symbols: Symbols with no usable mark price.
        missing_timestamp_symbols: Symbols with a mark but no timestamp.
    """

    fresh: bool
    reason: str
    max_age_seconds: float
    stale_symbols: tuple[str, ...] = ()
    missing_mark_symbols: tuple[str, ...] = ()
    missing_timestamp_symbols: tuple[str, ...] = ()


def check_marks_freshness(
    positions: Iterable[Any] | None,
    *,
    now: datetime | None = None,
    max_age_seconds: float | None = None,
) -> MarksFreshnessResult:
    """Check that every open position has a fresh, verified market mark.

    Args:
        positions: The account's open paper positions.
        now: UTC reference time. Defaults to the current UTC time.
        max_age_seconds: Maximum accepted mark age. Defaults to
            :data:`DEFAULT_MAX_MARK_AGE_SECONDS`.

    Returns:
        A frozen result. ``fresh`` is False when any position lacks a
        usable mark or a recent timestamp, so callers can refuse to size
        a trade instead of falling back to a stale value.
    """
    reference = now or utc_now()
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    else:
        reference = reference.astimezone(timezone.utc)

    max_age = (
        DEFAULT_MAX_MARK_AGE_SECONDS
        if max_age_seconds is None
        else float(max_age_seconds)
    )

    open_positions = list(positions or [])
    if not open_positions:
        return MarksFreshnessResult(True, "", max_age)

    stale: list[str] = []
    missing_mark: list[str] = []
    missing_timestamp: list[str] = []

    for position in open_positions:
        symbol = str(getattr(position, "symbol", "?"))
        if valid_mark_price(getattr(position, "mark_price", None)) is None:
            missing_mark.append(symbol)
            continue

        marked_at = parse_mark_timestamp(getattr(position, "mark_at", None))
        if marked_at is None:
            missing_timestamp.append(symbol)
            continue

        age_seconds = (reference - marked_at).total_seconds()
        # A mark dated in the future cannot be trusted either (clock skew
        # or corrupted state); treat it as stale rather than fresh.
        if age_seconds < 0 or age_seconds > max_age:
            stale.append(symbol)

    if not (stale or missing_mark or missing_timestamp):
        return MarksFreshnessResult(True, "", max_age)

    reasons: list[str] = []
    if missing_mark:
        reasons.append(
            "no market mark: " + ", ".join(missing_mark)
        )
    if missing_timestamp:
        reasons.append(
            "no mark timestamp: " + ", ".join(missing_timestamp)
        )
    if stale:
        reasons.append(
            f"mark older than {max_age:.0f}s: " + ", ".join(stale)
        )

    return MarksFreshnessResult(
        fresh=False,
        reason="; ".join(reasons),
        max_age_seconds=max_age,
        stale_symbols=tuple(stale),
        missing_mark_symbols=tuple(missing_mark),
        missing_timestamp_symbols=tuple(missing_timestamp),
    )
