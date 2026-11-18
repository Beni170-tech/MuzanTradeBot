"""
MuzanTradeBot - Paper Position Lifecycle

Owns the per-position lifecycle decisions for paper trading: market
marks, stop-loss / take-profit checks, and trailing-stop updates.

The Telegram handler layer and the background position monitor both
call these functions, so the rules live in exactly one place. The
functions take an already-fetched market price; they never fetch prices
themselves, and they only ever touch the exact position they are given.

Guardian Mode: stop loss always takes priority over take profit, and a
trailing stop never moves backward. Nothing here bypasses a risk check.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable

from trader.paper.order import PaperOrder, PaperOrderStatus
from trader.paper.service import PaperTradingService

logger = logging.getLogger(__name__)

# Canonical path for persisting paper account state. The Telegram
# handler layer keeps its own patchable ``PAPER_STATE_PATH`` and
# registers it below through :func:`bind_state_path`, so both layers
# always resolve the same file at call time.
PAPER_STATE_PATH = Path("logs/paper_state.json")

# Optional provider registered by the handler layer. It is consulted at
# call time (never at import time), so monkeypatching the handler
# module's ``PAPER_STATE_PATH`` keeps redirecting every write here too.
_state_path_provider: Callable[[], str | Path] | None = None


def bind_state_path(provider: Callable[[], str | Path] | None) -> None:
    """Register the callable that resolves the shared state path.

    Args:
        provider: A zero-argument callable returning the state path, or
            ``None`` to fall back to this module's ``PAPER_STATE_PATH``.
    """
    global _state_path_provider
    _state_path_provider = provider


def default_state_path() -> str | Path:
    """Return the state path, honouring a registered provider.

    A provider failure falls back to this module's constant rather than
    breaking a monitoring cycle.
    """
    provider = _state_path_provider
    if provider is None:
        return PAPER_STATE_PATH
    try:
        return provider()
    except Exception:
        logger.warning(
            "Paper state path provider failed; using the default path.",
            exc_info=True,
        )
        return PAPER_STATE_PATH


def resolve_state_path(state_path: str | Path | None = None) -> str | Path:
    """Return an explicit state path, or the shared default."""
    return state_path if state_path is not None else default_state_path()


def valid_price(value: object) -> float | None:
    """Return a usable, finite, strictly positive price, or None.

    Anything else (missing, bool, string, zero, negative, NaN,
    infinity) is unusable and must not drive exits or stops.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        if not math.isfinite(value) or value <= 0:
            return None
        return float(value)
    except (OverflowError, ValueError):
        # An int too large for a float is not a usable price; never let
        # the monitor crash on malformed data.
        return None


def initial_risk_distance(position: object) -> float | None:
    """Return the fixed entry-to-stop risk distance for a position.

    New positions record ``initial_risk_distance`` when they are placed.
    Legacy saved positions do not have that field, so the original
    stop-loss on the stored trade reference is used when available. The
    original distance must never be recalculated from a stop that has
    already trailed.
    """
    stored = valid_price(getattr(position, "initial_risk_distance", None))
    if stored is not None:
        return stored

    trade = getattr(position, "trade", None)
    if trade is None:
        return None

    original_stop = valid_price(getattr(trade, "stop_loss", None))
    entry = valid_price(getattr(position, "entry_price", None))
    if original_stop is None or entry is None:
        return None

    distance = abs(entry - original_stop)
    return distance if distance > 0 else None


def is_trail_active(position: object) -> bool:
    """Return True when the position's trailing stop is already active.

    Uses a notes-entry marker so trailing state survives serialization
    through the JSON storage layer without requiring changes to the
    PaperOrder dataclass.
    """
    notes = getattr(position, "notes", None) or []
    return any(isinstance(n, str) and n == "trailing_active" for n in notes)


def mark_trail_active(position: object) -> None:
    """Mark the position as having an active trailing stop.

    Stores a marker string in the order's notes list so the state
    survives JSON round-trips.
    """
    notes = getattr(position, "notes", None)
    if notes is None:
        notes = []
        position.notes = notes
    if "trailing_active" not in notes:
        notes.append("trailing_active")


def record_market_mark(position: object, current_price: object) -> bool:
    """Record a validated current market price on a paper position.

    The price handed in already comes from the analysis flow, so this
    adds no network request. The position's own mark mechanism rejects
    unusable values, and objects without it (test doubles, legacy
    shapes) are left untouched.

    Args:
        position: The open paper position to mark.
        current_price: The current market price from the analysis flow.

    Returns:
        True when the mark was recorded, False otherwise.
    """
    marker = getattr(position, "mark_to_market", None)
    if not callable(marker):
        return False
    try:
        return bool(marker(current_price))
    except (TypeError, ValueError):
        return False


def apply_trailing_stop(
    service: PaperTradingService,
    symbol: str,
    current_price: float,
    *,
    state_path: str | Path | None = None,
) -> tuple[bool, float | None, float | None] | None:
    """Apply trailing stop logic to the first open position for a symbol.

    This is the analysis-flow wrapper around
    :func:`apply_trailing_stop_to_position`. The background monitor uses
    the position-scoped function directly so several positions on one
    symbol are never confused with each other.
    """
    position = service.get_position(symbol)
    if position is None:
        return None
    return apply_trailing_stop_to_position(
        service, position, current_price, state_path=state_path
    )


def apply_trailing_stop_to_position(
    service: PaperTradingService,
    position: PaperOrder,
    current_price: float,
    *,
    state_path: str | Path | None = None,
) -> tuple[bool, float | None, float | None] | None:
    """Apply trailing stop logic to one exact open paper position.

    Runs during /analyze and from the background monitor using the
    already-fetched market price. No additional market API calls are
    made, and only the supplied position can be modified.

    Behavior:
        - For LONG: if price moves above entry by at least 1R, where 1R
          is the original entry-to-stop distance captured when the
          position was opened, activate trailing and move the stop to
          break-even (or keep an already tighter stop). After
          activation, trail the stop upward by that same fixed distance
          as price rises, never moving it backward.
        - For SHORT: mirror behavior — activate at 1R profit, then trail
          downward, never raising an already tighter stop.
        - Never moves the stop backward.
        - Never modifies take profit.
        - Activation always saves the paper account so the trailing
          marker and initial risk distance survive a restart. After
          activation, saves only when the stop actually changes.

    Args:
        service: The paper trading service.
        position: The exact open position to evaluate and trail.
        current_price: The current market price from the ongoing
            analysis. Must already be fetched; this function does not
            fetch prices itself.
        state_path: Optional state file to save to. Defaults to the
            shared paper state path.

    Returns:
        None when there is no position, no valid stop loss, or
        no trailing action is warranted.
        Otherwise a tuple of (changed, old_stop, new_stop) where
        changed indicates whether the stop was modified and the old/new
        stop values are provided for reporting.
    """
    if position is None:
        return None

    # Defensive: skip if there is no valid stop loss to trail
    if not position.stop_loss or position.stop_loss <= 0:
        return None

    side = (
        position.side.value
        if hasattr(position.side, "value")
        else str(position.side).upper()
    )
    entry = position.entry_price
    stop = position.stop_loss

    path = resolve_state_path(state_path)

    # Use the risk distance captured when the position was opened. For
    # legacy positions without it, fall back to the current stop only
    # while trailing has not moved that stop yet.
    stop_distance = initial_risk_distance(position)
    if stop_distance is None:
        try:
            fallback_distance = abs(float(entry) - float(stop))
        except (TypeError, ValueError, OverflowError):
            return None
        stop_distance = valid_price(fallback_distance)
    if stop_distance is None:
        return None

    # Backfill legacy positions as soon as the original distance is known,
    # so it survives the next save instead of being re-derived forever.
    # Treat an invalid stored value the same as a missing one so a bad
    # value can never be persisted again.
    if valid_price(getattr(position, "initial_risk_distance", None)) is None:
        position.initial_risk_distance = stop_distance

    # 1R threshold: entry plus/minus one original stop-loss distance
    if side == "LONG":
        one_r_price = entry + stop_distance
        # Not yet in 1R profit (strictly below threshold)
        if current_price <= one_r_price:
            return None

        # First time crossing 1R → lock at break-even, but never lower
        # a stop that is already tighter than entry (manual/legacy).
        if not is_trail_active(position):
            new_stop = max(stop, entry)
            old_stop = stop
            position.stop_loss = new_stop
            position.initial_risk_distance = stop_distance
            mark_trail_active(position)
            # Always persist activation so the marker and the original
            # risk distance survive a restart, even if the stop was
            # already at breakeven.
            service.save_account(path)
            return (True, old_stop, new_stop)

        # Already trail-active — move stop up only if price allows,
        # and only if it improves the stop (never backward).
        new_stop = max(stop, current_price - stop_distance)
        if new_stop > stop:
            old_stop = stop
            position.stop_loss = new_stop
            service.save_account(path)
            return (True, old_stop, new_stop)
        return None

    else:
        # SHORT (and any non-LONG side): mirror the logic.
        one_r_price = entry - stop_distance
        if current_price >= one_r_price:
            return None

        if not is_trail_active(position):
            new_stop = min(stop, entry)
            old_stop = stop
            position.stop_loss = new_stop
            position.initial_risk_distance = stop_distance
            mark_trail_active(position)
            # Always persist activation so the marker and the original
            # risk distance survive a restart, even if the stop was
            # already at breakeven.
            service.save_account(path)
            return (True, old_stop, new_stop)

        new_stop = min(stop, current_price + stop_distance)
        if new_stop < stop:
            old_stop = stop
            position.stop_loss = new_stop
            service.save_account(path)
            return (True, old_stop, new_stop)
        return None


def before_analysis_paper_check(
    service: PaperTradingService,
    symbol: str,
    current_price: float,
    *,
    state_path: str | Path | None = None,
) -> PaperOrder | None:
    """Check the first open position for a symbol for SL/TP breach.

    This is the analysis-flow wrapper around
    :func:`paper_position_check`. The background monitor uses the
    position-scoped function directly so it can never close a different
    position than the one it is processing.

    Runs BEFORE any new order is placed for the current analysis.
    Only evaluates positions that existed before this /analyze request.
    If the current price hits the stop loss, close with STOPPED.
    If the current price hits the take profit, close with TAKEN_PROFIT.
    Stop loss takes priority over take profit (Guardian Mode: protect capital).
    Returns the closed order if a closure happened, else None.
    """
    existing = service.get_position(symbol)
    if existing is None:
        return None
    return paper_position_check(
        service,
        existing,
        current_price,
        by_id=False,
        symbol=symbol,
        state_path=state_path,
    )


def paper_position_check(
    service: PaperTradingService,
    position: PaperOrder,
    current_price: float,
    *,
    by_id: bool = True,
    symbol: str | None = None,
    state_path: str | Path | None = None,
) -> PaperOrder | None:
    """Check one exact open paper position for a stop-loss / take-profit.

    Only ``position`` is marked, evaluated, and (if breached) closed, so
    other positions on the same symbol are untouched. Stop loss takes
    priority over take profit (Guardian Mode: protect capital).

    Args:
        service: The paper trading service.
        position: The exact position to evaluate.
        current_price: The already-fetched market price for the symbol.
        by_id: When True (the monitor path), close the exact order id.
            When False, keep the analysis flow's historical behaviour of
            closing by symbol.
        symbol: Symbol to pass to the symbol-based close path. Defaults
            to the position's own symbol.
        state_path: Optional state file to save to after a close.
            Defaults to the shared paper state path.

    Returns:
        The closed order if a closure happened, else None.
    """
    if position is None:
        return None

    # Record the analysis price as this position's market mark before any
    # exit decision. The price was already fetched for the analysis, so
    # this adds no blocking network request.
    record_market_mark(position, current_price)

    side = (
        position.side.value.upper()
        if hasattr(position.side, "value")
        else str(position.side).upper()
    )

    # Guard: no valid SL/TP levels configured
    if position.stop_loss <= 0 and position.take_profit <= 0:
        return None

    closed_order: PaperOrder | None = None

    # For LONG: SL is below entry, TP is above → price must drop to hit SL,
    # rise to hit TP.
    # For SHORT: SL is above entry, TP is below → price must rise to hit SL,
    # drop to hit TP.
    if side == "LONG":
        sl_hit = position.stop_loss > 0 and current_price <= position.stop_loss
        tp_hit = (
            position.take_profit > 0 and current_price >= position.take_profit
        )
    else:
        # SHORT (and any non-LONG side): invert the comparisons
        sl_hit = position.stop_loss > 0 and current_price >= position.stop_loss
        tp_hit = (
            position.take_profit > 0 and current_price <= position.take_profit
        )

    if sl_hit:
        # Stop loss hit — close at stop loss price (Guardian Mode priority)
        closed_order = close_paper_position_for(
            service,
            position,
            position.stop_loss,
            PaperOrderStatus.STOPPED,
            by_id=by_id,
            symbol=symbol,
        )
    elif tp_hit:
        # Take profit hit — close at take profit price
        closed_order = close_paper_position_for(
            service,
            position,
            position.take_profit,
            PaperOrderStatus.TAKEN_PROFIT,
            by_id=by_id,
            symbol=symbol,
        )

    if closed_order is not None:
        service.save_account(resolve_state_path(state_path))

    return closed_order


def close_paper_position_for(
    service: PaperTradingService,
    position: PaperOrder,
    exit_price: float,
    status: PaperOrderStatus,
    *,
    by_id: bool,
    symbol: str | None = None,
) -> PaperOrder | None:
    """Close either the exact order id or the first position by symbol."""
    if by_id:
        return service.close_position_by_id(
            position.order_id, exit_price, status
        )
    return service.close_position(
        symbol or position.symbol, exit_price, status
    )
