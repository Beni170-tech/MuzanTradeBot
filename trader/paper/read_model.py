"""Read models for the paper trading Telegram commands.

The command handlers in :mod:`telegram_bot.handlers` keep their Telegram
formatting and replies. This module owns the data work behind them:

* which positions and history windows are shown,
* how mark-based equity/PnL is resolved,
* how manual close arguments are parsed and how a target position is
  selected (including the ambiguous-symbol guard), and
* the performance/gate metrics shown by ``/stats``.

Nothing here sends messages, parses Telegram objects or writes state.
Closing a position still goes through the paper service, so storage and
lifecycle behaviour are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from trader.paper.analytics import (
    OverallPerformance,
    RiskMetrics,
    SymbolPerformance,
    compute_overall_performance,
    compute_risk_metrics,
    get_recent_trades,
    group_by_symbol,
)
from trader.paper.entry_gates import collect_entry_gate_metrics
from trader.paper.order import PaperOrder, PaperOrderStatus
from trader.paper.rules import is_valid_exit_price
from trader.paper.summary import PaperAccountSummary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trader.paper.service import PaperTradingService


# Recent closed positions shown by /paper.
RECENT_CLOSED_LIMIT = 5
# Recent closed positions shown by /history.
RECENT_HISTORY_LIMIT = 10


# ---------------------------------------------------------------------------
# Account / position / history views
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PaperAccountView:
    """Snapshot backing the ``/paper`` account summary."""

    summary: PaperAccountSummary
    open_positions: list[PaperOrder]
    recent_closed: list[PaperOrder]


@dataclass(frozen=True, slots=True)
class PaperHistoryView:
    """Snapshot backing the ``/history`` closed-trade list."""

    recent_closed: list[PaperOrder]


def account_view(service: PaperTradingService) -> PaperAccountView:
    """Build the read model for ``/paper``.

    Keeps the original call order (summary, open, closed) and the
    most-recent-first five closed positions.
    """
    summary = service.summary()
    open_positions = service.open_positions()
    closed_positions = service.closed_positions()
    return PaperAccountView(
        summary=summary,
        open_positions=list(open_positions),
        recent_closed=list(closed_positions[:RECENT_CLOSED_LIMIT]),
    )


def open_positions_view(service: PaperTradingService) -> list[PaperOrder]:
    """Return the open positions shown by ``/positions``."""
    return list(service.open_positions())


def history_view(service: PaperTradingService) -> PaperHistoryView:
    """Build the read model for ``/history`` (most recent ten trades)."""
    closed_positions = service.closed_positions()
    return PaperHistoryView(
        recent_closed=list(closed_positions[:RECENT_HISTORY_LIMIT])
    )


# ---------------------------------------------------------------------------
# Mark-based displays
# ---------------------------------------------------------------------------


def recorded_mark(pos: object) -> float | None:
    """Return the position's usable recorded market mark, or None.

    Only a finite, strictly positive number counts as a mark. Anything
    else (missing, None, or an unexpected value) is treated as "no mark
    recorded yet" so displays can never invent a valuation.
    """
    mark = getattr(pos, "mark_price", None)
    if isinstance(mark, bool) or not isinstance(mark, (int, float)):
        return None
    if not math.isfinite(mark) or mark <= 0:
        return None
    return float(mark)


def positions_with_marks(open_positions: Sequence[object]) -> list:
    """Return the open positions that carry a recorded market mark."""
    return [pos for pos in open_positions if recorded_mark(pos) is not None]


# ---------------------------------------------------------------------------
# Manual close
# ---------------------------------------------------------------------------


class CloseRequestError(Enum):
    """Why a ``/close`` argument list could not be turned into a request."""

    USAGE = "usage"
    NOT_A_NUMBER = "not_a_number"
    INVALID_PRICE = "invalid_price"


@dataclass(frozen=True, slots=True)
class CloseRequest:
    """Parsed and validated ``/close`` arguments."""

    symbol: str
    exit_price: float
    order_id: str | None


@dataclass(frozen=True, slots=True)
class CloseParseResult:
    """Either a usable close request or the reason it was rejected."""

    request: CloseRequest | None
    error: CloseRequestError | None
    raw_price: str | None = None


class CloseTargetStatus(Enum):
    """Outcome of selecting one open position to close."""

    TARGET = "target"
    NO_POSITION = "no_position"
    AMBIGUOUS = "ambiguous"
    ID_NOT_FOUND = "id_not_found"


@dataclass(frozen=True, slots=True)
class CloseTargetResult:
    """Selected target plus the positions that were considered."""

    status: CloseTargetStatus
    target: PaperOrder | None = None
    symbol_positions: tuple[PaperOrder, ...] = ()


@dataclass(frozen=True, slots=True)
class CloseResult:
    """A closed position and the account balance after the close."""

    order: PaperOrder
    new_balance: float


def parse_close_request(
    args: Sequence[str],
    *,
    is_valid_price=is_valid_exit_price,
) -> CloseParseResult:
    """Parse ``/close <SYMBOL> <EXIT_PRICE> [ORDER_ID]`` arguments.

    The price is rejected before callers touch positions, balances,
    storage or alerts: argument count first, then float parsing, then
    the shared finite-positive price rule (the lower-level close path
    enforces the same rule independently).
    """
    if len(args) < 2:
        return CloseParseResult(
            request=None, error=CloseRequestError.USAGE
        )

    raw_price = args[1]
    try:
        exit_price = float(raw_price)
    except ValueError:
        return CloseParseResult(
            request=None,
            error=CloseRequestError.NOT_A_NUMBER,
            raw_price=raw_price,
        )

    if not is_valid_price(exit_price):
        return CloseParseResult(
            request=None,
            error=CloseRequestError.INVALID_PRICE,
            raw_price=raw_price,
        )

    order_id = args[2].strip() if len(args) > 2 else None
    return CloseParseResult(
        request=CloseRequest(
            symbol=args[0].strip().upper(),
            exit_price=exit_price,
            order_id=order_id,
        ),
        error=None,
    )


def select_close_target(
    service: PaperTradingService, request: CloseRequest
) -> CloseTargetResult:
    """Pick exactly one open position to close.

    A single open position for the symbol can be closed without an id,
    preserving the original behaviour. When several positions share the
    symbol, nothing is selected until the exact order id is supplied, so
    a manual close can never hit the wrong position or close them all.
    """
    symbol = request.symbol
    symbol_positions = tuple(
        position
        for position in service.open_positions()
        if str(position.symbol).strip().upper() == symbol
    )

    if not symbol_positions:
        return CloseTargetResult(CloseTargetStatus.NO_POSITION)

    if request.order_id is None:
        if len(symbol_positions) > 1:
            return CloseTargetResult(
                CloseTargetStatus.AMBIGUOUS,
                symbol_positions=symbol_positions,
            )
        return CloseTargetResult(
            CloseTargetStatus.TARGET,
            target=symbol_positions[0],
            symbol_positions=symbol_positions,
        )

    wanted = request.order_id.upper()
    matches = tuple(
        position
        for position in symbol_positions
        if str(position.order_id).strip().upper() == wanted
    )
    if not matches:
        return CloseTargetResult(
            CloseTargetStatus.ID_NOT_FOUND,
            symbol_positions=symbol_positions,
        )
    return CloseTargetResult(
        CloseTargetStatus.TARGET,
        target=matches[0],
        symbol_positions=symbol_positions,
    )


def execute_close(
    service: PaperTradingService,
    target: PaperOrder,
    exit_price: float,
    state_path,
) -> CloseResult | None:
    """Close the selected position, persist state and report the balance.

    Returns None when the service could not close the position, leaving
    storage untouched exactly as before.
    """
    order = service.close_position_by_id(
        target.order_id, exit_price, PaperOrderStatus.CLOSED_MANUAL
    )
    if order is None:
        return None

    service.save_account(state_path)
    return CloseResult(
        order=order, new_balance=service.summary().current_balance
    )


# ---------------------------------------------------------------------------
# Performance dashboard
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PaperStatsView:
    """Metrics backing the ``/stats`` performance dashboard.

    ``overall``/``risk``/``gate_metrics``/``symbol_groups``/``recent``
    are None or empty when there are no closed trades; the handler then
    shows only the account section, matching the original early return.
    """

    summary: PaperAccountSummary
    open_positions: list[PaperOrder]
    closed_positions: list[PaperOrder]
    total_return: float
    overall: OverallPerformance | None
    risk: RiskMetrics | None
    gate_metrics: object | None
    symbol_groups: list[SymbolPerformance]
    recent_trades: list[PaperOrder]


def stats_view(
    service: PaperTradingService,
    *,
    portfolio_risk_limit: float,
    portfolio_exposure_limit: float,
    daily_loss_limit: float,
    max_open_positions: int,
    max_positions_per_symbol: int,
    max_exposure_per_symbol: float,
    collect_gate_metrics=collect_entry_gate_metrics,
) -> PaperStatsView:
    """Build the metrics for ``/stats`` without formatting anything."""
    summary = service.summary()
    closed_positions = service.closed_positions()
    open_positions = service.open_positions()

    if summary.initial_balance != 0.0:
        total_return = (
            (summary.current_balance - summary.initial_balance)
            / summary.initial_balance
            * 100.0
        )
    else:
        total_return = 0.0

    if not closed_positions:
        return PaperStatsView(
            summary=summary,
            open_positions=list(open_positions),
            closed_positions=[],
            total_return=total_return,
            overall=None,
            risk=None,
            gate_metrics=None,
            symbol_groups=[],
            recent_trades=[],
        )

    overall = compute_overall_performance(closed_positions)
    risk = compute_risk_metrics(closed_positions)

    # One collector call reports every enabled gate without
    # short-circuiting; a gate whose limit is not configured (<= 0)
    # comes back as None and its section is omitted, exactly as when
    # each section read its own manager.
    gate_metrics = collect_gate_metrics(
        service.execution_backend(),
        portfolio_risk_limit=portfolio_risk_limit,
        portfolio_exposure_limit=portfolio_exposure_limit,
        daily_loss_limit=daily_loss_limit,
        max_open_positions=max_open_positions,
        max_positions_per_symbol=max_positions_per_symbol,
        max_exposure_per_symbol=max_exposure_per_symbol,
    )

    return PaperStatsView(
        summary=summary,
        open_positions=list(open_positions),
        closed_positions=list(closed_positions),
        total_return=total_return,
        overall=overall,
        risk=risk,
        gate_metrics=gate_metrics,
        symbol_groups=group_by_symbol(closed_positions),
        recent_trades=get_recent_trades(closed_positions, limit=10),
    )
