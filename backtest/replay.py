"""Chronological replay of historical bars through the decision pipeline.

This module is read-only with respect to market history: each eligible
bar is analyzed using only that bar and the bars before it. It does not
simulate entries, exits, fees, or profit and loss.
"""

from dataclasses import dataclass
from datetime import datetime
import math

from brain.decision_engine import make_decision
from market.indicators import MIN_CANDLES, compute_analysis


@dataclass(frozen=True, slots=True)
class HistoricalBar:
    """One immutable historical observation."""

    timestamp: datetime
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """The strategy output recorded at one eligible historical bar."""

    timestamp: datetime
    signal: str
    trend: str
    confidence: float
    decision: str


def _is_finite_number(value) -> bool:
    """Return True for finite int/float values, rejecting bools."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_bars(bars) -> None:
    """Validate bar values and chronological ordering without sorting."""
    previous_timestamp = None
    for index, bar in enumerate(bars):
        if not isinstance(bar, HistoricalBar):
            raise TypeError(
                f"bars[{index}] must be a HistoricalBar, got "
                f"{type(bar).__name__}"
            )
        if not isinstance(bar.timestamp, datetime):
            raise TypeError(
                f"bars[{index}].timestamp must be a datetime, got "
                f"{type(bar.timestamp).__name__}"
            )
        if not _is_finite_number(bar.close) or bar.close <= 0:
            raise ValueError(
                f"bars[{index}].close must be a positive finite number"
            )
        if not _is_finite_number(bar.volume) or bar.volume < 0:
            raise ValueError(
                f"bars[{index}].volume must be a finite non-negative number"
            )

        if previous_timestamp is not None:
            try:
                in_order = bar.timestamp > previous_timestamp
            except TypeError as exc:
                raise ValueError(
                    "bar timestamps must be mutually comparable"
                ) from exc
            if not in_order:
                raise ValueError(
                    "bar timestamps must be strictly increasing in input order"
                )
        previous_timestamp = bar.timestamp


def replay_signals(bars) -> list[SignalRecord]:
    """Replay historical bars chronologically and record strategy decisions.

    Args:
        bars: An ordered list or tuple of HistoricalBar values, oldest first.
            The input is never sorted or modified.

    Returns:
        One SignalRecord per bar with at least MIN_CANDLES of history,
        starting with the MIN_CANDLES-th bar. Returns an empty list when
        fewer than MIN_CANDLES bars are supplied.

    Raises:
        TypeError: If ``bars`` is not a list/tuple or contains a non-bar, or
            if a bar timestamp is not a datetime.
        ValueError: If a bar has an invalid close/volume, or timestamps are
            not strictly increasing in the supplied order.
    """
    if not isinstance(bars, (list, tuple)):
        raise TypeError("bars must be a list or tuple of HistoricalBar values")

    _validate_bars(bars)

    if len(bars) < MIN_CANDLES:
        return []

    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    records = []

    for index in range(MIN_CANDLES - 1, len(bars)):
        analysis = compute_analysis(
            closes[: index + 1],
            volumes[: index + 1],
        )
        decision = make_decision(analysis)
        records.append(
            SignalRecord(
                timestamp=bars[index].timestamp,
                signal=analysis["signal"],
                trend=analysis["trend"],
                confidence=analysis["confidence"],
                decision=decision,
            )
        )

    return records
