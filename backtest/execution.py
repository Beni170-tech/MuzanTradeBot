"""Deterministic historical exit simulation.

The rules mirror the paper position monitor: existing stop-loss and
take-profit levels are checked before trailing-stop updates, and a stop
updated on one bar is not eligible until the next bar. Only closing
prices are used. This module performs no I/O and mutates no live state.
"""

from dataclasses import dataclass
from datetime import datetime
import math

from trader.paper.order import PaperOrderStatus

from backtest.replay import HistoricalBar


@dataclass(frozen=True, slots=True)
class BacktestPosition:
    """The exit-relevant state of one historical position.

    Directional level validation applies to the initial position, as it
    does when a paper order is placed. ``initial_risk_distance`` may be
    supplied for a legacy position; otherwise it is derived from entry
    and the original stop.
    """

    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    initial_risk_distance: float | None = None


@dataclass(frozen=True, slots=True)
class ExitEvent:
    """A stop-loss or take-profit exit at one historical bar."""

    timestamp: datetime
    price: float
    status: PaperOrderStatus


@dataclass(frozen=True, slots=True)
class ExitSimulation:
    """The deterministic final state of one exit simulation."""

    exit_event: ExitEvent | None
    final_stop_loss: float
    final_take_profit: float
    initial_risk_distance: float
    trailing_active: bool
    bars_processed: int


def _is_finite_positive(value) -> bool:
    """Return True only for a finite, strictly positive int/float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _normalize_side(side) -> str:
    if not isinstance(side, str):
        raise ValueError("side must be 'LONG' or 'SHORT'")
    normalized = side.strip().upper()
    if normalized not in ("LONG", "SHORT"):
        raise ValueError("side must be 'LONG' or 'SHORT'")
    return normalized


def _validate_position(position: BacktestPosition) -> None:
    if not isinstance(position, BacktestPosition):
        raise TypeError("position must be a BacktestPosition")

    side = _normalize_side(position.side)
    if not _is_finite_positive(position.entry_price):
        raise ValueError("entry_price must be a finite positive number")
    if not _is_finite_positive(position.stop_loss):
        raise ValueError("stop_loss must be a finite positive number")
    if not _is_finite_positive(position.take_profit):
        raise ValueError("take_profit must be a finite positive number")

    if side == "LONG":
        if not position.stop_loss < position.entry_price:
            raise ValueError("LONG stop_loss must be below entry_price")
        if not position.take_profit > position.entry_price:
            raise ValueError("LONG take_profit must be above entry_price")
    else:
        if not position.stop_loss > position.entry_price:
            raise ValueError("SHORT stop_loss must be above entry_price")
        if not position.take_profit < position.entry_price:
            raise ValueError("SHORT take_profit must be below entry_price")

    if position.initial_risk_distance is not None and not _is_finite_positive(
        position.initial_risk_distance
    ):
        raise ValueError(
            "initial_risk_distance must be a finite positive number"
        )


def _validate_bars(bars) -> None:
    if not isinstance(bars, (list, tuple)):
        raise TypeError("bars must be a list or tuple of HistoricalBar values")

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
        if not _is_finite_positive(bar.close):
            raise ValueError(
                f"bars[{index}].close must be a positive finite number"
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


def check_exit(
    side: str,
    current_price: float,
    stop_loss: float,
    take_profit: float,
) -> PaperOrderStatus | None:
    """Return the exit status triggered by one closing price.

    Stop loss is evaluated before take profit, matching the paper
    monitor. The returned status uses the existing paper order statuses.
    """
    side = _normalize_side(side)
    for name, value in (
        ("current_price", current_price),
        ("stop_loss", stop_loss),
        ("take_profit", take_profit),
    ):
        if not _is_finite_positive(value):
            raise ValueError(f"{name} must be a finite positive number")

    if side == "LONG":
        if current_price <= stop_loss:
            return PaperOrderStatus.STOPPED
        if current_price >= take_profit:
            return PaperOrderStatus.TAKEN_PROFIT
    else:
        if current_price >= stop_loss:
            return PaperOrderStatus.STOPPED
        if current_price <= take_profit:
            return PaperOrderStatus.TAKEN_PROFIT
    return None


def simulate_exit(position: BacktestPosition, bars) -> ExitSimulation:
    """Simulate one position's exit over chronological historical bars.

    Existing SL/TP levels are checked using each bar's close before any
    trailing update. A trailing stop changed on a bar can only trigger
    on a later bar. Exits fill at the stored stop/take-profit level.

    Args:
        position: The initial position and its initial exit levels.
        bars: Chronological HistoricalBar values to evaluate. They are
            never sorted or modified.

    Returns:
        An immutable ExitSimulation. ``exit_event`` is None when no
        exit level was touched.

    Raises:
        TypeError: If the position or an item in ``bars`` is the wrong
            type, or a timestamp is not a datetime.
        ValueError: If position levels or bar prices are invalid, or bar
            timestamps are not strictly increasing in input order.
    """
    _validate_position(position)
    _validate_bars(bars)

    side = _normalize_side(position.side)
    entry = float(position.entry_price)
    stop = float(position.stop_loss)
    take_profit = float(position.take_profit)
    risk_distance = (
        float(position.initial_risk_distance)
        if position.initial_risk_distance is not None
        else abs(entry - stop)
    )

    trailing_active = False
    bars_processed = 0

    for bar in bars:
        bars_processed += 1

        status = check_exit(side, bar.close, stop, take_profit)
        if status is PaperOrderStatus.STOPPED:
            return ExitSimulation(
                exit_event=ExitEvent(
                    timestamp=bar.timestamp,
                    price=stop,
                    status=status,
                ),
                final_stop_loss=stop,
                final_take_profit=take_profit,
                initial_risk_distance=risk_distance,
                trailing_active=trailing_active,
                bars_processed=bars_processed,
            )
        if status is PaperOrderStatus.TAKEN_PROFIT:
            return ExitSimulation(
                exit_event=ExitEvent(
                    timestamp=bar.timestamp,
                    price=take_profit,
                    status=status,
                ),
                final_stop_loss=stop,
                final_take_profit=take_profit,
                initial_risk_distance=risk_distance,
                trailing_active=trailing_active,
                bars_processed=bars_processed,
            )

        if not trailing_active:
            if side == "LONG":
                should_activate = bar.close > entry + risk_distance
                if should_activate:
                    stop = max(stop, entry)
            else:
                should_activate = bar.close < entry - risk_distance
                if should_activate:
                    stop = min(stop, entry)
            if should_activate:
                trailing_active = True
            continue

        if side == "LONG":
            candidate = bar.close - risk_distance
            if candidate > stop:
                stop = candidate
        else:
            candidate = bar.close + risk_distance
            if candidate < stop:
                stop = candidate

    return ExitSimulation(
        exit_event=None,
        final_stop_loss=stop,
        final_take_profit=take_profit,
        initial_risk_distance=risk_distance,
        trailing_active=trailing_active,
        bars_processed=bars_processed,
    )
