"""Chat rendering for the /analyze entry-gate rejection path.

This module owns the two presentation pieces ``/analyze`` uses when the
paper-trading entry gates reject a candidate trade:

    - :func:`entry_gate_rejection_message` renders the rejection text.
    - :func:`dispatch_entry_gate_alerts` dispatches the rejection alerts.

Both are the behaviour ``/analyze`` had before this extraction, moved
here unchanged. Everything is derived from the frozen outcome the
evaluator already returned, so nothing here re-reads account state,
re-evaluates a gate, or changes the trading decision.

Design constraints (Guardian Mode):
    - ``telegram_bot.handlers`` is never imported. The alert dispatcher
      and the three display helpers (``format_pnl_value``,
      ``trade_risk_value``, ``trade_exposure``) are injected by the
      caller, so the handler's live ``ALERTS_ENABLED`` gate, its
      patchable alert manager and its monkeypatchable display helpers
      keep applying exactly where they always did.
    - No settings, storage, broker or paper-execution imports: no I/O
      and no mutation happens here.
    - Message precedence follows the evaluator's gate order, because the
      rule is taken from the outcome rather than recomputed.
"""

from trader.paper.correlation_risk import trade_exposure
from trader.paper.entry_gates import (
    RULE_DAILY_LOSS,
    RULE_MAX_OPEN_POSITIONS,
    RULE_PORTFOLIO_EXPOSURE,
    RULE_PORTFOLIO_RISK,
    RULE_SYMBOL_EXPOSURE,
    RULE_SYMBOL_POSITIONS,
)
from trader.paper.portfolio_risk import trade_risk


def format_pnl_value(value: float) -> str:
    """Format a PnL value with sign and currency symbol."""
    if value > 0:
        return f"+${value:.2f}"
    if value < 0:
        return f"-${abs(value):.2f}"
    return "$0.00"


def trade_risk_value(trade) -> float:
    """Compute the dollar risk of a trade (for display in gate messages).

    Thin alias for :func:`trader.paper.portfolio_risk.trade_risk`, the
    single source of truth for a candidate trade's risk contribution.
    Keeping the name preserves the handler-level API used by callers
    and tests while removing the duplicated formula.
    """
    return trade_risk(trade)


def entry_gate_rejection_message(
    trade,
    outcome,
    *,
    pnl_formatter=None,
    risk_value=None,
    exposure=None,
) -> str:
    """Render the /analyze rejection text for one entry gate outcome.

    The text is byte-for-byte the text the inline gate checks appended
    before the extraction, produced from the frozen snapshot the gate
    returned instead of re-reading account state. ``outcome.rule``
    selects the block, so message precedence follows the evaluator's
    gate order.

    The three display helpers are injectable so the handler layer keeps
    owning their lookup: it passes its own module-level names, which
    preserves handler-level monkeypatching exactly as before the
    extraction. When omitted, the module's own implementations are used
    (resolved at call time, so they stay patchable here too).

    Args:
        trade: The candidate trade that was rejected.
        outcome: The frozen entry gate outcome carrying the rule and
            the metrics snapshot.
        pnl_formatter: Callable formatting a realized PnL value.
        risk_value: Callable returning a trade's dollar risk.
        exposure: Callable returning a trade's notional exposure.
    """
    if pnl_formatter is None:
        pnl_formatter = format_pnl_value
    if risk_value is None:
        risk_value = trade_risk_value
    if exposure is None:
        exposure = trade_exposure

    metrics = outcome.metrics

    if outcome.rule == RULE_PORTFOLIO_RISK:
        return (
            f"\n⚠️ Portfolio Risk Limit Exceeded\n"
            f"   Current Risk: ${metrics.current_risk:,.2f}\n"
            f"   Risk Limit: ${metrics.risk_limit:,.2f}\n"
            f"   Trade Risk: "
            f"${risk_value(trade):,.2f}\n"
            f"   The trade was not placed.\n"
            f"   Close existing positions to free up risk budget."
        )

    if outcome.rule == RULE_PORTFOLIO_EXPOSURE:
        return (
            f"\n⚠️ Portfolio Exposure Limit Exceeded\n"
            f"   Current Exposure: ${metrics.current_exposure:,.2f}\n"
            f"   Requested Exposure: "
            f"${trade.price * trade.position_size:,.2f}\n"
            f"   Exposure Limit: ${metrics.exposure_limit:,.2f}\n"
            f"   The trade was not placed.\n"
            f"   Close existing positions to free up exposure budget."
        )

    if outcome.rule == RULE_DAILY_LOSS:
        # Older snapshots and simple doubles may not carry the floating
        # fields; they then describe a settled-only day.
        unpriced = getattr(metrics, "unpriced_symbols", ())
        unmeasurable = getattr(metrics, "day_unmeasurable", False)
        floating_pnl = getattr(metrics, "open_unrealized_pnl", 0.0)
        if unmeasurable:
            return (
                f"\n⛔ Daily Loss Limit Reached\n"
                f"   Day: {metrics.day}\n"
                f"   Open positions could not be read.\n"
                f"   The day's floating loss cannot be measured.\n"
                f"   The trade was not placed.\n"
                f"   New paper trades resume once open positions can be "
                f"priced again."
            )
        if unpriced:
            return (
                f"\n⛔ Daily Loss Limit Reached\n"
                f"   Day: {metrics.day}\n"
                f"   Unpriced Positions: "
                f"{', '.join(unpriced)}\n"
                f"   The day's floating loss cannot be measured without a "
                f"market mark.\n"
                f"   The trade was not placed.\n"
                f"   New paper trades resume once every open position is "
                f"priced again."
            )
        floating_line = ""
        if floating_pnl:
            floating_line = (
                f"   Open Unrealized PnL: "
                f"{pnl_formatter(floating_pnl)}\n"
            )
        return (
            f"\n⛔ Daily Loss Limit Reached\n"
            f"   Day: {metrics.day}\n"
            f"   Today's Realized PnL: "
            f"{pnl_formatter(metrics.realized_pnl_today)}\n"
            f"{floating_line}"
            f"   Daily Loss Limit: ${metrics.loss_limit:,.2f}\n"
            f"   The trade was not placed.\n"
            f"   New paper trades resume on the next day."
        )

    if outcome.rule == RULE_MAX_OPEN_POSITIONS:
        return (
            f"\n⛔ Maximum Open Positions Reached\n"
            f"   Current Open Positions: "
            f"{metrics.current_open_positions}\n"
            f"   Max Open Positions: "
            f"{metrics.limit}\n"
            f"   The trade was not placed.\n"
            f"   Close existing positions to free up "
            f"open-position capacity."
        )

    if outcome.rule in (RULE_SYMBOL_POSITIONS, RULE_SYMBOL_EXPOSURE):
        block = (
            f"\n⚠️ Symbol Concentration Limit Exceeded\n"
            f"   Symbol: {trade.symbol}\n"
        )
        if outcome.rule == RULE_SYMBOL_POSITIONS:
            block += (
                f"   Open Positions: "
                f"{metrics.open_positions}\n"
                f"   Max Positions: "
                f"{metrics.max_positions}\n"
            )
        else:
            block += (
                f"   Current Exposure: "
                f"${metrics.symbol_exposure:,.2f}\n"
                f"   Requested Exposure: "
                f"${exposure(trade):,.2f}\n"
                f"   Exposure Limit: "
                f"${metrics.exposure_limit:,.2f}\n"
            )
        block += (
            f"   The trade was not placed.\n"
            f"   Close existing positions to free up "
            f"concentration budget."
        )
        return block

    raise ValueError(
        f"unsupported entry gate rejection rule: {outcome.rule!r}"
    )


async def dispatch_entry_gate_alerts(trade, outcome, *, dispatch_alert) -> None:
    """Dispatch the rejection alerts for one entry gate outcome.

    Alert count, ordering and payloads match the pre-extraction
    sequence: the daily-loss breaker alert precedes its rejection
    alert, and the max-open-positions alert precedes its rejection
    alert.

    Args:
        trade: The candidate trade that was rejected.
        outcome: The frozen entry gate outcome carrying the rule and
            the metrics snapshot used as the alert payload.
        dispatch_alert: The caller's alert dispatcher (the handler's
            ``_dispatch_alert``), invoked with ``(event, *args)``. It
            owns the enablement gate and the fail-safe behaviour, so
            this module never imports the handler layer.
    """
    if outcome.rule == RULE_DAILY_LOSS:
        await dispatch_alert("daily_loss_breaker", outcome.metrics)
        await dispatch_alert(
            "trade_rejected",
            trade,
            "daily loss breaker active for today",
        )
        return

    if outcome.rule == RULE_MAX_OPEN_POSITIONS:
        await dispatch_alert(
            "max_open_positions_reached", outcome.metrics
        )
        await dispatch_alert(
            "trade_rejected",
            trade,
            "maximum open positions limit reached",
        )
        return

    if outcome.rule == RULE_PORTFOLIO_RISK:
        await dispatch_alert(
            "trade_rejected", trade, "portfolio risk limit exceeded"
        )
        return

    if outcome.rule == RULE_PORTFOLIO_EXPOSURE:
        await dispatch_alert(
            "trade_rejected", trade, "portfolio exposure limit exceeded"
        )
        return

    if outcome.rule in (RULE_SYMBOL_POSITIONS, RULE_SYMBOL_EXPOSURE):
        await dispatch_alert(
            "trade_rejected", trade, "symbol concentration limit exceeded"
        )
