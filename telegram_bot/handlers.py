"""
MuzanTradeBot - Telegram Command Handlers
All user-facing command logic lives here.
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from telegram import Update
from telegram.ext import ContextTypes

from config.settings import settings
from market.analyzer import analyze_coin, get_current_price
from brain.decision_engine import make_decision
from risk.risk_manager import validate_trade
from strategy.parameters import (
    DEFAULT_STRATEGY_PARAMETERS,
    StrategyParameters,
)
from strategy.trading_policy import TradingPolicy
from trader.paper.service import PaperTradingService
from trader.paper.order import PaperOrder, PaperOrderStatus
from trader.paper.order import paper_exit_status, paper_exit_date
from trader.paper.rules import is_valid_exit_price
from trader.paper.sizing import (
    paper_mark_block,
    read_paper_equity,
    resolve_trading_policy,
)
from trader.paper.read_model import (
    CloseRequestError,
    CloseTargetStatus,
    account_view,
    execute_close,
    history_view,
    open_positions_view,
    parse_close_request,
    positions_with_marks,
    recorded_mark,
    report_view,
    select_close_target,
    stats_view,
)
from trader.paper.analytics import (
    OverallPerformance,
    RiskMetrics,
    SymbolPerformance,
    ExposureMetrics,
    compute_overall_performance,
    compute_risk_metrics,
    compute_symbol_performance,
    exposure_metrics,
    get_recent_trades,
    group_by_symbol,
)
# The risk/concentration managers and analytics helpers below stay
# imported at handler scope on purpose: the gate decisions now live in
# trader.paper.entry_gates, and these names remain part of this
# module's patchable surface for integrations and tests.
from trader.paper.correlation_risk import (
    CorrelationRiskManager,
    trade_exposure,
)
from trader.paper.daily_risk import DailyRiskManager
from trader.paper.entry_gates import (
    RULE_DAILY_LOSS,
    RULE_MAX_OPEN_POSITIONS,
    RULE_PORTFOLIO_EXPOSURE,
    RULE_PORTFOLIO_RISK,
    RULE_SYMBOL_EXPOSURE,
    RULE_SYMBOL_POSITIONS,
    collect_entry_gate_metrics,
    evaluate_entry_gates,
)
from trader.paper.lifecycle import (
    PAPER_STATE_PATH as DEFAULT_PAPER_STATE_PATH,
    apply_trailing_stop,
    apply_trailing_stop_to_position,
    before_analysis_paper_check,
    bind_state_path,
    close_paper_position_for,
    initial_risk_distance,
    is_trail_active,
    mark_trail_active,
    paper_position_check,
    record_market_mark,
    valid_price,
)
from trader.paper.pause import (
    DEFAULT_PAUSE_STATE_PATH,
    PauseState,
    clear_pause,
    read_pause_state,
    request_pause,
)
from trader.paper.mark_freshness import MarksFreshnessResult
from trader.paper.max_open_positions import MaxOpenPositionsManager
from trader.paper.portfolio_risk import (
    PortfolioRiskManager,
    total_portfolio_risk,
    trade_risk,
)
from trader.trade import build_trade
from journal.journal import log_trade
from alerts.manager import AlertManager, build_alert_manager
from telegram_bot.analyze_reporting import (
    dispatch_entry_gate_alerts,
    entry_gate_rejection_message,
    format_pnl_value as _format_pnl_value,
)
from trader.paper.analysis_service import (
    PAPER_STATUSES,
    AnalysisRun,
    AnalysisStatus,
    run_analysis_pipeline,
)
from telegram_bot.auth import authorization_status
from dashboard.report import format_report_message
from telegram_bot.settings_report import format_settings_message
from telegram_bot.rate_limit import (
    build_rate_limit_config,
    check_rate_limit,
    reset_rate_limit_state,
)

# Path for persisting paper account state. The value is shared with the
# paper lifecycle module through a call-time provider, so patching this
# attribute keeps redirecting handler and monitor writes alike.
PAPER_STATE_PATH = DEFAULT_PAPER_STATE_PATH
bind_state_path(lambda: PAPER_STATE_PATH)

# Pause control path. Read at call time, so patching this attribute
# redirects the pause gate and the /pause and /resume commands.
PAUSE_STATE_PATH = DEFAULT_PAUSE_STATE_PATH

# Module-level rate-limit configuration. This is built once at import time
# from settings, but the helper uses the live settings values so tests can
# monkeypatch settings before the handlers module is loaded or by patching
# telegram_bot.handlers.settings.
_rate_limit_config = build_rate_limit_config(
    commands_per_window=settings.TELEGRAM_COMMAND_RATE_LIMIT_PER_WINDOW,
    window_seconds=settings.TELEGRAM_COMMAND_RATE_WINDOW_SECONDS,
)

# Module-level paper service (lazy-loaded)
_paper_service: PaperTradingService | None = None


async def _rate_limited(update: Update) -> bool:
    """Send the friendly rate-limit reply when this chat must wait.

    The check and its reply are shared by every rate-limited command so
    the message text stays in one place. ``check_rate_limit`` and the
    configuration are read from this module at call time, preserving the
    existing patch seams.

    Args:
        update: The incoming Telegram update.

    Returns:
        True when the request was rejected and the reply was sent, so the
        handler returns without doing any other work.
    """
    if check_rate_limit(update.effective_chat.id, _rate_limit_config):
        return False
    await update.message.reply_text(
        "⏳ Too many requests. Please wait a moment before trying again."
    )
    return True


# Module-level alert manager (lazy-loaded, Sprint 9A)
_alert_manager: AlertManager | None = None


def _get_alert_manager() -> AlertManager:
    """Get or create the alert manager (lazy initialization).

    The manager is built from settings and never raises: when alerts
    are disabled or unconfigured it is backed by a NullNotifier and
    every call is a no-op.
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = build_alert_manager(
            alerts_enabled=settings.ALERTS_ENABLED,
            chat_id=settings.TELEGRAM_ALERT_CHAT_ID,
            token=settings.TELEGRAM_BOT_TOKEN,
        )
    return _alert_manager


def _alerts_active() -> bool:
    """Return True when alerts are enabled in settings right now.

    This is the live dispatch gate: the caller (this handler layer)
    is responsible for checking ALERTS_ENABLED at call time, exactly
    like the other settings gates in the analyze flow. The
    AlertManager's constructor-time ``enabled`` snapshot is only a
    second layer of defense and cannot be consulted when the manager
    is replaced (e.g. by tests) — the live settings check here is
    what guarantees no alert method is awaited when alerts are
    disabled.
    """
    return bool(settings.ALERTS_ENABLED)


async def _dispatch_alert(event: str, *args) -> None:
    """Dispatch one alert event when alerts are enabled (never raises).

    Args:
        event: The AlertManager method name to invoke (e.g.
            "order_placed", "trade_rejected").
        *args: Arguments forwarded to the alert method.
    """
    if not _alerts_active():
        return
    manager = _get_alert_manager()
    method = getattr(manager, event, None)
    if method is None:
        logger.warning("Unknown alert event %r; alert dropped.", event)
        return
    try:
        await method(*args)
    except Exception:
        logger.warning(
            "Alert dispatch failed for %s; alert dropped.",
            event,
            exc_info=True,
        )


def _paper_pause_state() -> PauseState:
    """Read the global paper pause control from this module's path.

    Never raises: unreadable control state comes back as paused, so a
    broken control file blocks new entries instead of allowing them.
    """
    return read_pause_state(PAUSE_STATE_PATH)


def _get_paper_service() -> PaperTradingService:
    """Get or create the paper trading service (lazy initialization)."""
    global _paper_service
    if _paper_service is None:
        if PAPER_STATE_PATH.exists():
            _paper_service = PaperTradingService.load_account(PAPER_STATE_PATH)
        else:
            _paper_service = PaperTradingService.create_new_account(
                settings.ACCOUNT_BALANCE
            )
    return _paper_service


def _paper_equity_for_sizing() -> float:
    """Return the paper account equity used to size a new trade.

    Delegates to :func:`trader.paper.sizing.read_paper_equity`; the
    service is resolved here, at call time, so patching
    ``_get_paper_service`` on this module keeps working.
    """
    return read_paper_equity(_get_paper_service)


def _resolve_trading_policy(
    strategy: StrategyParameters,
) -> TradingPolicy | None:
    """Resolve the validated policy that drives this trade's sizing.

    Delegates to :func:`trader.paper.sizing.resolve_trading_policy`. The
    live settings and this module's patchable equity reader are injected
    at call time, so ``PAPER_TRADING`` and monkeypatched seams still
    decide whether and how the account is read.
    """
    return resolve_trading_policy(
        strategy, settings, _paper_equity_for_sizing
    )


def _valid_current_price(value: object) -> float | None:
    """Return a usable execution price, or None."""
    return valid_price(value)


def _current_quote(symbol: str) -> float | None:
    """Fetch the fresh execution quote for a symbol, or None.

    Uses the existing simple-price helper (one request) and validates the
    result. Never falls back to a historical close or a stored price.
    """
    try:
        raw = get_current_price(symbol)
    except Exception:
        logger.warning(
            "Current market quote lookup failed for %s; trade blocked.",
            symbol,
            exc_info=True,
        )
        return None
    return _valid_current_price(raw)


def _quote_unavailable_message(analysis: dict) -> str:
    """Build the reply when the current execution quote is unavailable."""
    return (
        f"📊 Analysis for {analysis.get('symbol', '?')}\n\n"
        f"Trend: {analysis.get('trend', 'Unknown')}\n"
        f"Signal: {analysis.get('signal', 'Unknown')}\n"
        f"Confidence: {analysis.get('confidence', 'Unknown')}%\n\n"
        f"⛔ Trade Blocked: current market quote unavailable\n"
        "   A valid current price could not be obtained for this symbol.\n"
        "   The last historical close is never used as an execution price.\n"
        "   No trade was placed.\n"
        "   Try again once market data is available."
    )


def _paper_mark_block() -> MarksFreshnessResult | None:
    """Return the freshness result that forbids sizing, or None.

    Delegates to :func:`trader.paper.sizing.paper_mark_block` with the
    live settings and the patchable service factory, so the gate order,
    fail-closed behaviour and seam surface are unchanged.
    """
    return paper_mark_block(
        paper_trading=settings.PAPER_TRADING,
        monitor_interval_seconds=settings.PAPER_MONITOR_INTERVAL_SECONDS,
        service_factory=_get_paper_service,
    )


def _stale_mark_message(analysis: dict, block: MarksFreshnessResult) -> str:
    """Build the Telegram reply that explains a stale-mark block."""
    return (
        f"📊 Analysis for {analysis.get('symbol', '?')}\n\n"
        f"Price: {analysis.get('price')}\n"
        f"Trend: {analysis.get('trend', 'Unknown')}\n"
        f"Signal: {analysis.get('signal', 'Unknown')}\n"
        f"Confidence: {analysis.get('confidence', 'Unknown')}%\n\n"
        f"⛔ Trade Blocked: market-price data is stale or unavailable\n"
        f"   {block.reason}\n"
        f"   Sizing needs a verified market mark for every open position "
        f"no older than {block.max_age_seconds:.0f}s.\n"
        f"   The trade was not placed and no fallback balance was used.\n"
        f"   Wait for the position monitor to refresh prices, then try again."
    )


def _sizing_unavailable_message(symbol: str) -> str:
    """Build the reply when paper equity cannot drive position sizing."""
    return (
        f"📊 Analysis for {symbol}\n\n"
        f"⛔ Trade Blocked: paper account equity unavailable\n"
        "   Position sizing requires a positive account equity.\n"
        "   The paper account state could not be read or reported no "
        "usable capital.\n"
        "   No trade was placed and no fallback balance was used."
    )


def _format_pnl_label(order: PaperOrder) -> str:
    """Return WIN / LOSS / BREAKEVEN label for a closed order."""
    if order.realized_pnl > 0:
        return "WIN"
    if order.realized_pnl < 0:
        return "LOSS"
    return "BREAKEVEN"


# ---------------------------------------------------------------------------
# Paper position lifecycle
# ---------------------------------------------------------------------------
# The lifecycle rules (market marks, SL/TP checks, trailing stops) live in
# trader.paper.lifecycle so the background monitor and the command handlers
# share one implementation. The helpers below are the handler layer's
# patchable entry points: they forward the module's PAPER_STATE_PATH, so
# monkeypatching it still redirects every state write.


def _apply_trailing_stop(
    service: PaperTradingService, symbol: str, current_price: float
) -> tuple[bool, float | None, float | None] | None:
    """Apply trailing stop logic to the first open position for a symbol."""
    return apply_trailing_stop(
        service, symbol, current_price, state_path=PAPER_STATE_PATH
    )


def _initial_risk_distance(position: object) -> float | None:
    """Return the fixed entry-to-stop risk distance for a position."""
    return initial_risk_distance(position)


def _apply_trailing_stop_to_position(
    service: PaperTradingService,
    position: PaperOrder,
    current_price: float,
) -> tuple[bool, float | None, float | None] | None:
    """Apply trailing stop logic to one exact open paper position."""
    return apply_trailing_stop_to_position(
        service, position, current_price, state_path=PAPER_STATE_PATH
    )


def _is_trail_active(position: object) -> bool:
    """Return True when the position's trailing stop is already active."""
    return is_trail_active(position)


def _mark_trail_active(position: object) -> None:
    """Mark the position as having an active trailing stop."""
    mark_trail_active(position)


def _record_market_mark(position: object, current_price: object) -> bool:
    """Record a validated current market price on a paper position."""
    return record_market_mark(position, current_price)


def _before_analysis_paper_check(
    service: PaperTradingService, symbol: str, current_price: float
) -> PaperOrder | None:
    """Check the first open position for a symbol for SL/TP breach."""
    return before_analysis_paper_check(
        service, symbol, current_price, state_path=PAPER_STATE_PATH
    )


def _paper_position_check(
    service: PaperTradingService,
    position: PaperOrder,
    current_price: float,
    *,
    by_id: bool = True,
    symbol: str | None = None,
) -> PaperOrder | None:
    """Check one exact open paper position for a stop-loss / take-profit."""
    return paper_position_check(
        service,
        position,
        current_price,
        by_id=by_id,
        symbol=symbol,
        state_path=PAPER_STATE_PATH,
    )


def _close_paper_position_for(
    service: PaperTradingService,
    position: PaperOrder,
    exit_price: float,
    status: PaperOrderStatus,
    *,
    by_id: bool,
    symbol: str | None = None,
) -> PaperOrder | None:
    """Close either the exact order id or the first position by symbol."""
    return close_paper_position_for(
        service,
        position,
        exit_price,
        status,
        by_id=by_id,
        symbol=symbol,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if await _rate_limited(update):
        return

    paper_status = "Enabled" if settings.PAPER_TRADING else "Disabled"
    guardian_status = "Active" if settings.GUARDIAN_MODE else "Inactive"

    message = (
        f"🛡️ Guardian Mode {guardian_status}\n\n"
        "Welcome to MuzanTradeBot.\n\n"
        "Status:\n"
        "✅ Healthy\n"
        f"📄 Paper Trading: {paper_status}"
    )
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    if await _rate_limited(update):
        return

    message = (
        "📋 Available Commands:\n\n"
        "/start - Initialize bot and check status\n"
        "/help - Show this help message\n"
        "/ping - Check if bot is alive\n"
        "/version - Show bot version\n"
        "/analyze <COIN> - Analyze a coin (e.g. /analyze BTC)\n"
        "/paper - Show paper account summary\n"
        "/positions - Show open paper positions\n"
        "/history - Show closed paper trades\n"
        "/close <SYMBOL> <EXIT_PRICE> [ORDER_ID] - Close a paper position\n"
        "  (ORDER_ID is required when several positions share a symbol)\n"
        "/stats - Show paper trading performance dashboard\n"
        "/report - Show the paper performance report (all time, today, "
        "7 days, by symbol)\n"
        "/pause [reason] - Block new paper entries (positions stay open)\n"
        "/resume - Allow new paper entries again\n"
        "/settings - Show execution mode, limits, access and alerts"
    )
    await update.message.reply_text(message)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ping command."""
    if await _rate_limited(update):
        return

    await update.message.reply_text("🏓 Pong!")


async def version(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /version command."""
    if await _rate_limited(update):
        return

    from main import __version__
    await update.message.reply_text(f"MuzanTradeBot v{__version__}")


def _entry_gate_rejection_message(trade, outcome) -> str:
    """Render the /analyze rejection text for one entry gate outcome.

    The handler-level dependency lookup is preserved: the display
    helpers are resolved here, at call time, from this module's
    globals, so monkeypatching ``trade_risk_value``,
    ``trade_exposure`` or ``_format_pnl_value`` on
    ``telegram_bot.handlers`` still changes the rendered text exactly
    as it did before the extraction. The wording lives in
    :mod:`telegram_bot.analyze_reporting`.
    """
    return entry_gate_rejection_message(
        trade,
        outcome,
        pnl_formatter=_format_pnl_value,
        risk_value=trade_risk_value,
        exposure=trade_exposure,
    )


async def _dispatch_entry_gate_alerts(trade, outcome) -> None:
    """Dispatch the rejection alerts through this module's dispatcher.

    The two-argument handler entry point is preserved, and the handler's
    live ``_dispatch_alert`` is injected into the shared implementation.
    That keeps the ``ALERTS_ENABLED`` check, the patchable alert manager
    and the fail-safe dispatch semantics in the handler layer, while the
    alert types, payloads and ordering live in
    :mod:`telegram_bot.analyze_reporting`.
    """
    await dispatch_entry_gate_alerts(
        trade, outcome, dispatch_alert=_dispatch_alert
    )


def _format_analysis_message(run: AnalysisRun) -> str:
    """Render the Telegram reply for one /analyze orchestration result.

    All message text lives here: the orchestration service returns a
    structured result and never formats user-facing output.
    """
    if run.status is AnalysisStatus.QUOTE_UNAVAILABLE:
        return _quote_unavailable_message(run.analysis)
    if run.status is AnalysisStatus.STALE_MARKS:
        return _stale_mark_message(run.analysis, run.stale_block)
    if run.status is AnalysisStatus.SIZING_UNAVAILABLE:
        return _sizing_unavailable_message(run.coin)

    trade = run.trade
    message = (
        f"📊 Analysis for {trade.symbol}\n\n"
        f"Price: {trade.price}\n"
        f"Trend: {trade.trend}\n"
        f"Signal: {trade.signal}\n"
        f"Confidence: {trade.confidence}%\n\n"
        f"🧠 Decision: {trade.decision}\n"
    )

    if not trade.approved:
        message += f"🛑 Risk: {trade.risk_verdict}\n"
        if trade.risk_reasons:
            message += "\n".join(
                f"• {reason}" for reason in trade.risk_reasons
            )
        return message

    message += (
        f"✅ Risk: {trade.risk_verdict}\n"
        f"💰 Position Size: {trade.position_size}\n"
        f"🛑 Stop Loss: {trade.stop_loss}\n"
        f"🎯 Take Profit: {trade.take_profit}\n"
    )

    if run.status not in PAPER_STATUSES:
        return message

    if run.sltp_closed is not None:
        sltp_closed = run.sltp_closed
        message += (
            f"\n📄 Paper Order: CLOSED (SL/TP)\n"
            f"   Symbol: {sltp_closed.symbol}\n"
            f"   Side: {sltp_closed.side.value}\n"
            f"   Exit Price: ${sltp_closed.exit_price:.2f}\n"
            f"   Exit Status: {paper_exit_status(sltp_closed)}\n"
            f"   Realized PnL: ${sltp_closed.realized_pnl:.2f} "
            f"({_format_pnl_label(sltp_closed)})"
        )

    if run.trail_result is not None:
        _, old_stop, new_stop = run.trail_result
        message += (
            f"\n📄 Paper Order: TRAILING STOP UPDATED\n"
            f"   Symbol: {trade.symbol}\n"
            f"   Stop Loss: ${old_stop:.2f} → ${new_stop:.2f}"
        )

    if run.status is AnalysisStatus.GATE_REJECTED:
        message += _entry_gate_rejection_message(trade, run.gate_outcome)
    elif run.status is AnalysisStatus.PAUSED:
        message += (
            "\n\n⏸ Paper Trading Paused\n"
            "   No new paper entry was placed.\n"
            "   Existing positions remain monitored.\n"
            "   Use /resume to re-enable new entries."
        )
    elif run.status is AnalysisStatus.PAPER_PLACED:
        paper_order = run.paper_order
        message += (
            f"\n📄 Paper Order: PLACED\n"
            f"   Order ID: {paper_order.order_id}\n"
            f"   Status: {paper_order.status.value}\n"
            f"   Quantity: {paper_order.quantity}\n"
            f"   Entry: ${paper_order.entry_price:.2f}\n"
            f"   Stop Loss: ${paper_order.stop_loss:.2f}\n"
            f"   Take Profit: ${paper_order.take_profit:.2f}\n"
            f"   Side: {paper_order.side.value}"
        )
    elif run.status is AnalysisStatus.PAPER_FAILED:
        message += (
            "\n\n❌ Paper Execution Failed\n"
            "   The paper trading step did not complete.\n"
            "   No confirmed order is reported as placed.\n"
            "   Check /positions before retrying; "
            "see the bot logs for details."
        )

    return message


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze a cryptocurrency through the full Guardian Mode pipeline."""
    if await _rate_limited(update):
        return

    if context.args:
        coin = context.args[0].upper()
    else:
        await update.message.reply_text(
            "❌ Please provide a coin.\n\nExample:\n/analyze BTC"
        )
        return

    # One strategy instance drives analysis and the decision; the risk
    # profile is resolved only after the quote and freshness gates, so
    # paper equity is fresh at the moment sizing happens. Every
    # dependency is resolved here, at call time, so this module's
    # patchable surface stays authoritative.
    run = await run_analysis_pipeline(
        coin,
        strategy=DEFAULT_STRATEGY_PARAMETERS,
        paper_trading_enabled=settings.PAPER_TRADING,
        analyze_coin=analyze_coin,
        current_quote=_current_quote,
        make_decision=make_decision,
        paper_mark_block=_paper_mark_block,
        resolve_trading_policy=_resolve_trading_policy,
        validate_trade=validate_trade,
        build_trade=build_trade,
        log_trade=log_trade,
        paper_pause_state=_paper_pause_state,
        paper_service_factory=_get_paper_service,
        before_analysis_paper_check=_before_analysis_paper_check,
        apply_trailing_stop=_apply_trailing_stop,
        evaluate_entry_gates=evaluate_entry_gates,
        state_path=PAPER_STATE_PATH,
        dispatch_alert=_dispatch_alert,
        dispatch_entry_gate_alerts=_dispatch_entry_gate_alerts,
        portfolio_risk_limit=settings.PORTFOLIO_RISK_LIMIT,
        portfolio_exposure_limit=settings.PORTFOLIO_EXPOSURE_LIMIT,
        daily_loss_limit=settings.PORTFOLIO_DAILY_LOSS_LIMIT,
        max_open_positions=settings.PORTFOLIO_MAX_OPEN_POSITIONS,
        max_positions_per_symbol=(
            settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL
        ),
        max_exposure_per_symbol=settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL,
    )

    await update.message.reply_text(_format_analysis_message(run))


# ---------------------------------------------------------------------------
# Paper Trading Management Commands (Sprint 6B)
# ---------------------------------------------------------------------------

async def settings_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show operational settings without exposing credentials."""
    if await _rate_limited(update):
        return

    message = format_settings_message(
        execution_mode=settings.EXECUTION_MODE,
        paper_trading=settings.PAPER_TRADING,
        guardian_mode=settings.GUARDIAN_MODE,
        portfolio_risk_limit=settings.PORTFOLIO_RISK_LIMIT,
        portfolio_exposure_limit=settings.PORTFOLIO_EXPOSURE_LIMIT,
        daily_loss_limit=settings.PORTFOLIO_DAILY_LOSS_LIMIT,
        max_open_positions=settings.PORTFOLIO_MAX_OPEN_POSITIONS,
        max_positions_per_symbol=(
            settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL
        ),
        max_exposure_per_symbol=(
            settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL
        ),
        authorization_status=authorization_status(),
        alerts_enabled=settings.ALERTS_ENABLED,
        alert_chat_configured=bool(settings.TELEGRAM_ALERT_CHAT_ID),
    )
    await update.message.reply_text(message)


async def paper_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show paper account summary."""
    if await _rate_limited(update):
        return

    view = account_view(_get_paper_service())
    summary = view.summary
    open_positions = view.open_positions
    closed_positions = view.recent_closed

    message = (
        f"📄 Paper Account Summary\n\n"
        f"Balance: ${summary.current_balance:.2f}\n"
        f"{_equity_display(summary, open_positions)}\n"
        f"Open Positions: {len(open_positions)}\n"
        f"Closed Positions: {summary.closed_count}\n"
        f"Realized PnL: ${summary.total_realized_pnl:.2f}\n"
        f"{_unrealized_pnl_display(summary, open_positions)}"
    )

    if open_positions:
        message += "\n\n📌 Open Positions:\n"
        for pos in open_positions:
            message += (
                f"\n  {pos.symbol} {pos.side.value}\n"
                f"  Entry: ${pos.entry_price:.2f}\n"
                f"  Qty: {pos.quantity}\n"
                f"  {_position_mark_display(pos)}"
            )

    if closed_positions:
        message += "\n\n📜 Recently Closed:\n"
        for pos in closed_positions[:5]:
            message += (
                f"\n  {pos.symbol} - {pos.side.value}\n"
                f"  Exit: ${pos.exit_price:.2f} | "
                f"PnL: ${pos.realized_pnl:.2f} ({_format_pnl_label(pos)})"
            )

    await update.message.reply_text(message)


async def positions_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show all open paper positions."""
    if await _rate_limited(update):
        return

    open_positions = open_positions_view(_get_paper_service())

    if not open_positions:
        await update.message.reply_text("📭 No open paper positions.")
        return

    message = "📌 Open Paper Positions:\n\n"
    for pos in open_positions:
        message += (
            f"{pos.symbol} {pos.side.value}\n"
            f"  Entry: ${pos.entry_price:.2f}\n"
            f"  Qty: {pos.quantity}\n"
            f"  {_position_mark_display(pos)}\n\n"
        )

    await update.message.reply_text(message)


async def history_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show recently closed paper trades."""
    if await _rate_limited(update):
        return

    view = history_view(_get_paper_service())
    closed_positions = view.recent_closed

    if not closed_positions:
        await update.message.reply_text("📭 No closed paper trades yet.")
        return

    message = "📜 Closed Paper Trades (most recent first):\n\n"
    for pos in closed_positions:
        label = _format_pnl_label(pos)
        message += (
            f"{pos.symbol} {pos.side.value}\n"
            f"  Entry: ${pos.entry_price:.2f}\n"
            f"  Exit: ${pos.exit_price:.2f}\n"
            f"  Qty: {pos.quantity}\n"
            f"  Realized PnL: ${pos.realized_pnl:.2f} ({label})\n"
            f"  Closed: {paper_exit_status(pos)}\n"
            f"  Date: {paper_exit_date(pos)}\n\n"
        )

    await update.message.reply_text(message)


def _position_choice_line(position: PaperOrder) -> str:
    """Format one open position for a disambiguation list."""
    side = (
        position.side.value
        if hasattr(position.side, "value")
        else str(position.side)
    )
    return (
        f"  • {position.order_id} — {side} qty {position.quantity} "
        f"entry ${position.entry_price:.2f} "
        f"(SL ${position.stop_loss:.2f} / TP ${position.take_profit:.2f})"
    )


async def close_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Close one open paper position.

    Usage: /close <SYMBOL> <EXIT_PRICE> [ORDER_ID]

    A single open position for the symbol can be closed without an id,
    preserving the original behaviour. When several positions share the
    symbol, nothing is closed until the exact order id is supplied, so a
    manual close can never hit the wrong position or close them all.
    """
    if await _rate_limited(update):
        return

    # Everything up to the service lookup stays pure argument work, so a
    # rejected price never loads the account, touches positions, balances,
    # storage or alerts. The lower-level close path enforces the same rule,
    # so a direct caller cannot bypass it either.
    parsed = parse_close_request(
        context.args, is_valid_price=is_valid_exit_price
    )
    if parsed.error is CloseRequestError.USAGE:
        await update.message.reply_text(
            "❌ Usage: /close <SYMBOL> <EXIT_PRICE> [ORDER_ID]\n\n"
            "Examples:\n"
            "/close BTC 110.00\n"
            "/close BTC 110.00 <ORDER_ID>  (required when several "
            "positions share a symbol)"
        )
        return
    if parsed.error is CloseRequestError.NOT_A_NUMBER:
        await update.message.reply_text(
            f"❌ Invalid exit price: '{parsed.raw_price}'. "
            "Please provide a valid number."
        )
        return
    if parsed.error is CloseRequestError.INVALID_PRICE:
        await update.message.reply_text(
            f"❌ Invalid exit price: '{parsed.raw_price}'. "
            "Exit price must be a finite number greater than 0."
        )
        return

    request = parsed.request
    symbol = request.symbol
    exit_price = request.exit_price

    service = _get_paper_service()
    selection = select_close_target(service, request)

    if selection.status is CloseTargetStatus.NO_POSITION:
        await update.message.reply_text(
            f"❌ No open paper position found for {symbol}."
        )
        return

    if selection.status is CloseTargetStatus.AMBIGUOUS:
        choices = "\n".join(
            _position_choice_line(position)
            for position in selection.symbol_positions
        )
        await update.message.reply_text(
            f"⚠️ Multiple open positions for {symbol} — "
            "nothing was closed.\n\n"
            f"{choices}\n\n"
            "Re-run with the exact position ID:\n"
            f"/close {symbol} {exit_price:g} <ORDER_ID>"
        )
        return

    if selection.status is CloseTargetStatus.ID_NOT_FOUND:
        choices = "\n".join(
            _position_choice_line(position)
            for position in selection.symbol_positions
        )
        await update.message.reply_text(
            f"❌ No open {symbol} position with ID "
            f"'{request.order_id}' — nothing was closed.\n\n"
            f"Available {symbol} position IDs:\n{choices}\n\n"
            "Re-run with the exact position ID:\n"
            f"/close {symbol} {exit_price:g} <ORDER_ID>"
        )
        return

    target = selection.target
    result = execute_close(
        service,
        target,
        exit_price,
        PAPER_STATE_PATH,
    )

    if result is None:
        await update.message.reply_text(
            f"❌ Could not close {symbol} position "
            f"'{target.order_id}' — nothing was closed."
        )
        return

    order = result.order
    new_balance = result.new_balance

    await update.message.reply_text(
        f"✅ Position Closed\n\n"
        f"Symbol: {order.symbol}\n"
        f"Order ID: {order.order_id}\n"
        f"Side: {order.side.value}\n"
        f"Exit Price: ${order.exit_price:.2f}\n"
        f"Realized PnL: ${order.realized_pnl:.2f} ({_format_pnl_label(order)})\n"
        f"Updated Balance: ${new_balance:.2f}"
    )


# ---------------------------------------------------------------------------
# Paper Trading Performance Analytics (Sprint 6D)
# ---------------------------------------------------------------------------


def _format_streak(streak: int) -> str:
    """Format a win/loss streak for display."""
    if streak > 0:
        return f"+{streak} wins"
    if streak < 0:
        return f"{streak} losses"
    return "None"


def _format_infnan(value: float | None) -> str:
    """Format a float that may be inf or None for display."""
    if value is None:
        return "N/A"
    if value == float("inf"):
        return "∞"
    if value == float("-inf"):
        return "-∞"
    return f"{value:.2f}"


def _recorded_mark(pos) -> float | None:
    """Return the position's usable recorded market mark, or None.

    Only a finite, strictly positive number counts as a mark. Anything
    else (missing, None, or an unexpected value) is treated as "no mark
    recorded yet" so displays can never invent a valuation.
    """
    return recorded_mark(pos)


def _positions_with_marks(open_positions) -> list:
    """Return the open positions that carry a recorded market mark."""
    return positions_with_marks(open_positions)


def _equity_display(summary, open_positions) -> str:
    """Return the equity line, labelled by how fresh its valuation is.

    Equity only includes unrealized PnL for positions with a recorded
    market mark. Command handlers do not fetch prices, so marked equity
    is presented as based on last recorded marks rather than live
    prices, and unmarked positions are called out explicitly.
    """
    if not open_positions:
        return f"Equity: ${summary.equity:.2f}"

    marked = _positions_with_marks(open_positions)
    if not marked:
        return (
            f"Equity: ${summary.equity:.2f} "
            "(settled balance; no market marks recorded yet)"
        )

    line = (
        f"Equity: ${summary.equity:.2f} "
        "(based on last recorded marks, not live prices)"
    )
    missing = len(open_positions) - len(marked)
    if missing:
        line += f" — {missing} position(s) without a mark excluded"
    return line


def _unrealized_pnl_display(
    summary, open_positions, label: str = "Unrealized PnL"
) -> str:
    """Return the unrealized PnL line without faking missing marks."""
    if not open_positions:
        return f"{label}: ${summary.open_pnl:.2f}"

    marked = _positions_with_marks(open_positions)
    if not marked:
        return f"{label}: unavailable (no market mark recorded yet)"

    line = f"{label}: ${summary.open_pnl:.2f} (last recorded marks)"
    missing = len(open_positions) - len(marked)
    if missing:
        line += f" — {missing} position(s) without a mark excluded"
    return line


def _position_mark_display(pos) -> str:
    """Return the per-position mark/PnL lines for open position displays."""
    mark = _recorded_mark(pos)
    if mark is None:
        return "Unrealized PnL: unavailable (no market mark recorded yet)"
    return (
        f"Mark: ${mark:,.2f} (last recorded)\n"
        f"  Unrealized PnL: ${pos.unrealized_pnl_for(mark):,.2f}"
    )


def trade_risk_value(trade) -> float:
    """Compute the dollar risk of a trade (for display in gate messages).

    Thin alias for this module's :func:`trade_risk`, resolved at call
    time so patching ``telegram_bot.handlers.trade_risk`` keeps changing
    both this value and the entry-gate rejection text built from it.
    """
    return trade_risk(trade)


async def report_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show the paper trading performance report."""
    if await _rate_limited(update):
        return

    view = report_view(_get_paper_service())
    await update.message.reply_text(format_report_message(view))


async def pause_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Block new paper entries; existing positions stay managed."""
    if await _rate_limited(update):
        return

    reason = " ".join(context.args) if context.args else None
    state = request_pause(reason, PAUSE_STATE_PATH)
    if state is None:
        await update.message.reply_text(
            "❌ Could not save the pause state.\n"
            "   Paper trading is unchanged; check the bot logs and retry."
        )
        return

    message = "⏸ Paper trading paused: new entries are blocked."
    if state.reason:
        message += f"\nReason: {state.reason}"
    message += (
        "\nExisting positions are still monitored; SL/TP, trailing "
        "stops and /close keep working."
    )
    await update.message.reply_text(message)


async def resume_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Allow new paper entries again."""
    if await _rate_limited(update):
        return

    state = clear_pause(PAUSE_STATE_PATH)
    if state is None:
        await update.message.reply_text(
            "❌ Could not save the resume state.\n"
            "   Paper trading is unchanged; check the bot logs and retry."
        )
        return

    await update.message.reply_text(
        "▶️ Paper trading resumed: new entries are allowed."
    )


async def stats_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show paper trading performance dashboard."""
    view = stats_view(
        _get_paper_service(),
        portfolio_risk_limit=settings.PORTFOLIO_RISK_LIMIT,
        portfolio_exposure_limit=settings.PORTFOLIO_EXPOSURE_LIMIT,
        daily_loss_limit=settings.PORTFOLIO_DAILY_LOSS_LIMIT,
        max_open_positions=settings.PORTFOLIO_MAX_OPEN_POSITIONS,
        max_positions_per_symbol=(
            settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL
        ),
        max_exposure_per_symbol=settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL,
        collect_gate_metrics=collect_entry_gate_metrics,
    )
    summary = view.summary
    closed_positions = view.closed_positions
    open_positions = view.open_positions

    # --- Section 1: Account ---
    total_return = view.total_return
    return_str = _format_pnl_value(total_return)
    if total_return > 0:
        return_str = f"+{total_return:.2f}%"
    elif total_return < 0:
        return_str = f"{total_return:.2f}%"

    message = (
        f"📊 Paper Trading Performance\n\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"📋 Account\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Balance: ${summary.current_balance:,.2f}\n"
        f"Initial: ${summary.initial_balance:,.2f}\n"
        f"Return: {return_str}\n"
        f"Open Positions: {len(open_positions)}\n"
        f"{_unrealized_pnl_display(summary, open_positions, label='Open PnL')}\n"
    )

    if not closed_positions:
        message += "\n📭 No closed trades yet."
        await update.message.reply_text(message)
        return

    # --- Section 2: Overall Performance ---
    overall = view.overall
    message += (
        f"\n\n━━━━━━━━━━━━━━━━━━━\n"
        f"📈 Overall Performance\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Total Trades: {overall.total_trades}\n"
        f"Win Rate: {overall.win_rate:.1f}%\n"
        f"Wins: {overall.winning_trades}  "
        f"Losses: {overall.losing_trades}  "
        f"Breakevens: {overall.breakeven_trades}\n"
        f"Total PnL: {_format_pnl_value(overall.total_realized_pnl)}\n"
        f"Avg Win: ${overall.average_win:,.2f}\n"
        f"Avg Loss: ${overall.average_loss:,.2f}\n"
        f"Expectancy: {_format_pnl_value(overall.expectancy)}/trade\n"
        f"Largest Win: ${overall.largest_win:,.2f}\n"
        f"Largest Loss: ${overall.largest_loss:,.2f}\n"
    )

    # --- Section 3: Risk Metrics ---
    risk = view.risk
    message += (
        f"\n\n━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ Risk Metrics\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Profit Factor: {_format_infnan(risk.profit_factor)}\n"
        f"Win/Loss Ratio: {_format_infnan(risk.win_loss_ratio)}\n"
        f"Payoff Ratio: {_format_infnan(risk.payoff_ratio)}\n"
        f"Avg Trade PnL: {_format_pnl_value(risk.average_trade_pnl)}\n"
        f"Current Streak: {_format_streak(risk.current_streak)}\n"
        f"Max Consec. Losses: {risk.max_consecutive_losses}\n"
    )

    # --- Sections 3b-3f: Entry gate metrics (Sprints 7A-8B) ---
    gate_metrics = view.gate_metrics

    if gate_metrics.portfolio_risk is not None:
        prm = gate_metrics.portfolio_risk
        message += (
            f"\n\n━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Portfolio Risk\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Current Risk: ${prm.current_risk:,.2f}\n"
            f"Risk Limit: ${prm.risk_limit:,.2f}\n"
            f"Risk Usage: {prm.usage_percent:.1f}%\n"
            f"Remaining: ${prm.remaining:,.2f}\n"
        )

    if gate_metrics.portfolio_exposure is not None:
        em = gate_metrics.portfolio_exposure
        message += (
            f"\n\n━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Portfolio Exposure\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Current Exposure: ${em.current_exposure:,.2f}\n"
            f"Exposure Limit: ${em.exposure_limit:,.2f}\n"
            f"Exposure Usage: {em.usage_percent:.1f}%\n"
            f"Remaining: ${em.remaining:,.2f}\n"
        )

    if gate_metrics.symbol_concentration is not None:
        concentration_report = gate_metrics.symbol_concentration
        message += (
            f"\n\n━━━━━━━━━━━━━━━━━━━\n"
            f"🧲 Symbol Concentration\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
        )
        if not concentration_report:
            message += "No open positions.\n"
        for cm in concentration_report:
            message += f"{cm.symbol}:\n"
            if cm.max_positions > 0:
                message += (
                    f"   Positions: {cm.open_positions} / "
                    f"{cm.max_positions}\n"
                )
            if cm.exposure_limit > 0:
                message += (
                    f"   Exposure: ${cm.symbol_exposure:,.2f} / "
                    f"${cm.exposure_limit:,.2f} "
                    f"({cm.exposure_usage_percent:.1f}%)\n"
                )

    if gate_metrics.daily_loss is not None:
        daily_metrics = gate_metrics.daily_loss
        if daily_metrics.breached:
            daily_status = "⛔ Brake active — new trades halted until tomorrow."
        else:
            daily_status = "✅ New trades allowed."
        message += (
            f"\n\n━━━━━━━━━━━━━━━━━━━\n"
            f"🚦 Daily Risk\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Day: {daily_metrics.day}\n"
            f"Today's Realized PnL: "
            f"{_format_pnl_value(daily_metrics.realized_pnl_today)}\n"
            f"Daily Loss Limit: ${daily_metrics.loss_limit:,.2f}\n"
            f"Loss Usage: {daily_metrics.loss_usage_percent:.1f}%\n"
            f"Remaining Budget: ${daily_metrics.remaining_budget:,.2f}\n"
            f"Status: {daily_status}\n"
        )

    if gate_metrics.max_open_positions is not None:
        mop_metrics = gate_metrics.max_open_positions
        message += (
            f"\n\n━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Max Open Positions\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Current Open Positions: {mop_metrics.current_open_positions}\n"
            f"Limit: {mop_metrics.limit}\n"
            f"Capacity Used: {mop_metrics.usage_percent:.1f}%\n"
            f"Remaining: {mop_metrics.remaining}\n"
            f"Status: {mop_metrics.status}\n"
        )

    # --- Section 4: Per Symbol ---
    symbol_groups = view.symbol_groups
    message += "\n\n━━━━━━━━━━━━━━━━━━━\n📎 Per Symbol\n━━━━━━━━━━━━━━━━━━━\n"
    for sp in symbol_groups:
        pf_str = _format_infnan(sp.profit_factor)
        message += (
            f"{sp.symbol}: {sp.total_trades} trades, "
            f"{sp.winning_trades}W {sp.losing_trades}L "
            f"{sp.breakeven_trades}B, "
            f"WR {sp.win_rate:.1f}%, "
            f"PnL {_format_pnl_value(sp.total_pnl)}, "
            f"PF {pf_str}\n"
        )

    # --- Section 5: Recent Trades ---
    recent = view.recent_trades
    message += "\n\n━━━━━━━━━━━━━━━━━━━\n📜 Recent Trades (last 10)\n━━━━━━━━━━━━━━━━━━━\n"
    for pos in recent:
        label = _format_pnl_label(pos)
        entry_str = f"${pos.entry_price:,.2f}" if pos.entry_price else "N/A"
        exit_str = f"${pos.exit_price:,.2f}" if pos.exit_price else "N/A"
        date_str = paper_exit_date(pos, format="%Y-%m-%d")
        message += (
            f"{pos.symbol:<6} {pos.side.value:<6} "
            f"Entry {entry_str:<10} Exit {exit_str:<10} "
            f"Qty {pos.quantity:<6} "
            f"PnL {_format_pnl_value(pos.realized_pnl):<10} "
            f"{label:<9} {date_str}\n"
        )

    await update.message.reply_text(message)
