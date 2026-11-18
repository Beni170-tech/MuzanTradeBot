"""
MuzanTradeBot - Paper Analysis Orchestration

Owns the business sequence of one /analyze request: analysis, fresh
quote, decision, mark-freshness gate, equity-based sizing policy, risk
validation, then the paper-trading actions (SL/TP closure check,
trailing stop, portfolio entry gates, placement).

The Telegram handler layer still owns everything user-facing: command
parsing, rate limiting, message text and alert transport. Its callables
are injected into :func:`run_analysis_pipeline` at call time, which
keeps the handler module's monkeypatch seams — and therefore the same
decisions, ordering and alerts — exactly as before.

Guardian Mode: the sequence below never skips or reorders a gate, and a
paper-trading failure is reported, never treated as success.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from trader.paper.order import PaperOrderStatus, paper_exit_status

logger = logging.getLogger(__name__)


class AnalysisStatus(str, Enum):
    """Outcome of one /analyze orchestration run."""

    QUOTE_UNAVAILABLE = "quote_unavailable"
    STALE_MARKS = "stale_marks"
    SIZING_UNAVAILABLE = "sizing_unavailable"
    RISK_REJECTED = "risk_rejected"
    APPROVED = "approved"
    GATE_REJECTED = "gate_rejected"
    PAPER_PLACED = "paper_placed"
    PAPER_NOT_PLACED = "paper_not_placed"
    PAPER_FAILED = "paper_failed"


#: Statuses reached inside the paper-trading block.
PAPER_STATUSES = frozenset(
    {
        AnalysisStatus.GATE_REJECTED,
        AnalysisStatus.PAPER_PLACED,
        AnalysisStatus.PAPER_NOT_PLACED,
        AnalysisStatus.PAPER_FAILED,
    }
)


@dataclass(frozen=True)
class AnalysisRun:
    """Structured result of one /analyze request.

    The handler renders the message from these fields; nothing in this
    module formats Telegram text.

    Attributes:
        status: Which stage the run ended at.
        coin: The coin exactly as requested (upper-cased).
        analysis: The analysis dict after the fresh quote was applied.
        trade: The built trade, when one was built.
        stale_block: The mark-freshness result that blocked sizing.
        sltp_closed: A position closed by the pre-analysis SL/TP check.
        trail_result: The ``(changed, old_stop, new_stop)`` trailing
            result, when the helper returned one.
        gate_outcome: The entry-gate evaluation result.
        paper_order: The order placed by the paper service, if any.
        paper_error: The exception raised by the paper block, if any.
    """

    status: AnalysisStatus
    coin: str
    analysis: dict[str, Any] = field(default_factory=dict, repr=False)
    trade: Any | None = None
    stale_block: Any | None = None
    sltp_closed: Any | None = None
    trail_result: tuple[bool, float | None, float | None] | None = None
    gate_outcome: Any | None = None
    paper_order: Any | None = None
    paper_error: BaseException | None = None


async def run_analysis_pipeline(
    coin: str,
    *,
    strategy: Any,
    paper_trading_enabled: bool,
    analyze_coin: Callable[..., dict],
    current_quote: Callable[[str], float | None],
    make_decision: Callable[..., Any],
    paper_mark_block: Callable[[], Any | None],
    resolve_trading_policy: Callable[[Any], Any | None],
    validate_trade: Callable[..., Any],
    build_trade: Callable[..., Any],
    log_trade: Callable[..., Any],
    paper_service_factory: Callable[[], Any],
    before_analysis_paper_check: Callable[[Any, str, float], Any | None],
    apply_trailing_stop: Callable[
        [Any, str, float], tuple[bool, float | None, float | None] | None
    ],
    evaluate_entry_gates: Callable[..., Any],
    state_path: Any,
    dispatch_alert: Callable[..., Awaitable[None]],
    dispatch_entry_gate_alerts: Callable[..., Awaitable[None]],
    portfolio_risk_limit: float,
    portfolio_exposure_limit: float,
    daily_loss_limit: float,
    max_open_positions: int,
    max_positions_per_symbol: int,
    max_exposure_per_symbol: float,
) -> AnalysisRun:
    """Run one /analyze request through the full Guardian Mode pipeline.

    The callables are supplied by the handler layer so its dependency
    lookups (and test overrides) stay authoritative.

    Returns:
        The structured :class:`AnalysisRun` describing where the run
        stopped and what it produced. Never raises for a paper-trading
        failure: that is reported as :attr:`AnalysisStatus.PAPER_FAILED`
        with the exception attached.
    """
    # Historical candles drive trend/signal/confidence, but the trade must
    # be priced from a fresh market quote. The chart's last close is a
    # historical observation, not a current execution price.
    result = analyze_coin(coin, parameters=strategy)
    quote = current_quote(result.get("symbol", coin))
    if quote is None:
        logger.warning(
            "No usable current market quote for %s; trade blocked.",
            result.get("symbol", coin),
        )
        return AnalysisRun(
            AnalysisStatus.QUOTE_UNAVAILABLE, coin, analysis=dict(result)
        )

    result = {**result, "price": quote}
    decision = make_decision(result, parameters=strategy)

    # Refuse to size a new trade from stale or missing market marks. This
    # runs before sizing so no fallback balance can be used.
    mark_block = paper_mark_block()
    if mark_block is not None:
        logger.warning(
            "Paper trade for %s blocked: stale or missing market marks (%s).",
            result.get("symbol"),
            mark_block.reason,
        )
        return AnalysisRun(
            AnalysisStatus.STALE_MARKS,
            coin,
            analysis=result,
            stale_block=mark_block,
        )

    # Paper equity is read here, immediately before sizing, from the same
    # account whose marks were just verified. The strategy instance that
    # produced the signal is carried unchanged, and no fallback balance is
    # substituted when equity is unavailable.
    policy = resolve_trading_policy(strategy)
    if policy is None:
        logger.warning(
            "Paper account equity is unavailable for %s; trade blocked.",
            result.get("symbol", coin),
        )
        return AnalysisRun(
            AnalysisStatus.SIZING_UNAVAILABLE, coin, analysis=result
        )

    verdict = validate_trade(decision, result, policy.risk_profile)
    trade = build_trade(result, verdict, policy.risk_profile)
    log_trade(
        trade.symbol,
        trade.decision,
        trade.risk_verdict,
    )

    if not trade.approved:
        return AnalysisRun(
            AnalysisStatus.RISK_REJECTED, coin, analysis=result, trade=trade
        )

    if not paper_trading_enabled:
        return AnalysisRun(
            AnalysisStatus.APPROVED, coin, analysis=result, trade=trade
        )

    # Paper trading integration. A failure here must never break the
    # analysis reply, but it must never look like a successful placement
    # either.
    sltp_closed: Any | None = None
    trail_result: tuple[bool, float | None, float | None] | None = None
    gate_outcome: Any | None = None
    try:
        service = paper_service_factory()

        # Check pre-existing positions for SL/TP breach BEFORE placing
        # any new order. This ensures we never evaluate the order that
        # is about to be opened in this same request.
        sltp_closed = before_analysis_paper_check(
            service, trade.symbol, trade.price
        )
        if sltp_closed is not None:
            # Sprint 9A: push an SL/TP alert (never raises).
            exit_value = paper_exit_status(sltp_closed)
            if exit_value == PaperOrderStatus.STOPPED.value:
                await dispatch_alert("stop_loss_triggered", sltp_closed)
            elif exit_value == PaperOrderStatus.TAKEN_PROFIT.value:
                await dispatch_alert("take_profit_triggered", sltp_closed)

        # Apply trailing stop to any pre-existing position
        # before placing a new order. This runs after the
        # SL/TP closure check so a just-closed position is not
        # re-evaluated, and before place_trade so the new order
        # being opened in this request is never subject to
        # trailing in the same call.
        trail_result = apply_trailing_stop(
            service, trade.symbol, trade.price
        )
        if trail_result is not None:
            changed, old_stop, new_stop = trail_result
            # Sprint 9A: push a trailing-stop alert (never raises).
            await dispatch_alert(
                "trailing_stop_updated", trade.symbol, old_stop, new_stop
            )

        # Portfolio safety gates (Sprints 7A-8B): run after the
        # SL/TP closure check and the trailing stop update, but
        # before placing any new order. One domain evaluator owns
        # the gate order and thresholds; it stops at the first
        # rejection, so the reported rule is always the
        # highest-precedence failing gate. A limit that is not
        # configured (<= 0) skips its gate entirely.
        gate_outcome = evaluate_entry_gates(
            service.execution_backend(),
            trade,
            portfolio_risk_limit=portfolio_risk_limit,
            portfolio_exposure_limit=portfolio_exposure_limit,
            daily_loss_limit=daily_loss_limit,
            max_open_positions=max_open_positions,
            max_positions_per_symbol=max_positions_per_symbol,
            max_exposure_per_symbol=max_exposure_per_symbol,
        )
        if not gate_outcome.allowed:
            # Sprint 9A: push the gate's rejection alerts, in the
            # same order and count as the inline checks used.
            await dispatch_entry_gate_alerts(trade, gate_outcome)
            return AnalysisRun(
                AnalysisStatus.GATE_REJECTED,
                coin,
                analysis=result,
                trade=trade,
                sltp_closed=sltp_closed,
                trail_result=trail_result,
                gate_outcome=gate_outcome,
            )

        # Now place the new trade (if still approved)
        paper_order = service.place_trade(trade)
        status = AnalysisStatus.PAPER_NOT_PLACED
        if paper_order is not None:
            service.save_account(state_path)
            # Sprint 9A: push an order-placed alert (never raises).
            await dispatch_alert("order_placed", paper_order)
            status = AnalysisStatus.PAPER_PLACED
        return AnalysisRun(
            status,
            coin,
            analysis=result,
            trade=trade,
            sltp_closed=sltp_closed,
            trail_result=trail_result,
            gate_outcome=gate_outcome,
            paper_order=paper_order,
        )
    except Exception as exc:
        # Log the full traceback for diagnosis; the handler tells the
        # user plainly that no order was confirmed. Exception details
        # are never echoed to Telegram.
        logger.exception(
            "Paper trading integration failed during /analyze for %s: %s",
            trade.symbol,
            exc,
        )
        return AnalysisRun(
            AnalysisStatus.PAPER_FAILED,
            coin,
            analysis=result,
            trade=trade,
            sltp_closed=sltp_closed,
            trail_result=trail_result,
            gate_outcome=gate_outcome,
            paper_error=exc,
        )
