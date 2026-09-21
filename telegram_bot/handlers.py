"""
MuzanTradeBot - Telegram Command Handlers
All user-facing command logic lives here.
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from telegram import Update
from telegram.ext import ContextTypes

from config.settings import settings
from market.analyzer import analyze_coin
from brain.decision_engine import make_decision
from risk.risk_manager import validate_trade
from risk.risk_profile import RiskProfile
from trader.paper.service import PaperTradingService
from trader.paper.storage import load_account as storage_load_account
from trader.paper.storage import save_account as storage_save_account
from trader.paper.order import PaperOrder, PaperOrderStatus
from trader.paper.order import paper_exit_status, paper_exit_date
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
from trader.paper.correlation_risk import (
    CorrelationRiskManager,
    trade_exposure,
)
from trader.paper.daily_risk import DailyRiskManager
from trader.paper.max_open_positions import MaxOpenPositionsManager
from trader.paper.portfolio_risk import (
    PortfolioRiskManager,
    total_portfolio_risk,
)
from trader.trade import build_trade
from journal.journal import log_trade
from alerts.manager import AlertManager, build_alert_manager
from telegram_bot.rate_limit import (
    build_rate_limit_config,
    check_rate_limit,
    reset_rate_limit_state,
)

# Path for persisting paper account state
PAPER_STATE_PATH = Path("logs/paper_state.json")

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


def _format_pnl_label(order: PaperOrder) -> str:
    """Return WIN / LOSS / BREAKEVEN label for a closed order."""
    if order.realized_pnl > 0:
        return "WIN"
    if order.realized_pnl < 0:
        return "LOSS"
    return "BREAKEVEN"


def _apply_trailing_stop(
    service: PaperTradingService, symbol: str, current_price: float
) -> tuple[bool, float | None, float | None] | None:
    """Apply trailing stop logic to an open paper position.

    Runs during /analyze using the already-fetched market price.
    No additional market API calls are made.

    Behavior:
        - For LONG: if price moves above entry by at least 1R
          (entry - stop_loss distance), activate trailing and move
          stop to break-even. After activation, trail the stop upward
          as price rises, never moving it backward.
        - For SHORT: mirror behavior — activate at 1R profit,
          then trail downward.
        - Never moves the stop backward.
        - Never modifies take profit.
        - Saves the paper account only when the stop actually changes.

    Args:
        service: The paper trading service.
        symbol: The symbol to check.
        current_price: The current market price from the ongoing
            analysis. Must already be fetched; this function does not
            fetch prices itself.

    Returns:
        None when there is no open position, no valid stop loss, or
        no trailing action is warranted.
        Otherwise a tuple of (changed, old_stop, new_stop) where
        changed indicates whether the stop was modified and the old/new
        stop values are provided for reporting.
    """
    position = service.get_position(symbol)
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
    stop_distance = abs(entry - stop)

    # 1R threshold: entry plus/minus one stop-loss distance
    if side == "LONG":
        one_r_price = entry + stop_distance
        # Not yet in 1R profit (strictly below threshold)
        if current_price <= one_r_price:
            return None

        # First time crossing 1R → lock at break-even
        if not _is_trail_active(position):
            new_stop = entry
            old_stop = stop
            position.stop_loss = new_stop
            _mark_trail_active(position)
            # Always report activation — even if stop was already at
            # breakeven, the trail_active flag is new.
            if new_stop != old_stop:
                service.save_account(PAPER_STATE_PATH)
            return (True, old_stop, new_stop)

        # Already trail-active — move stop up only if price allows,
        # and only if it improves the stop (never backward).
        new_stop = max(stop, current_price - stop_distance)
        if new_stop > stop:
            old_stop = stop
            position.stop_loss = new_stop
            service.save_account(PAPER_STATE_PATH)
            return (True, old_stop, new_stop)
        return None

    else:
        # SHORT (and any non-LONG side): mirror the logic.
        one_r_price = entry - stop_distance
        if current_price >= one_r_price:
            return None

        if not _is_trail_active(position):
            new_stop = entry
            old_stop = stop
            position.stop_loss = new_stop
            _mark_trail_active(position)
            # Always report activation — even if stop was already at
            # breakeven, the trail_active flag is new.
            if new_stop != old_stop:
                service.save_account(PAPER_STATE_PATH)
            return (True, old_stop, new_stop)

        new_stop = min(stop, current_price + stop_distance)
        if new_stop < stop:
            old_stop = stop
            position.stop_loss = new_stop
            service.save_account(PAPER_STATE_PATH)
            return (True, old_stop, new_stop)
        return None


def _is_trail_active(position: object) -> bool:
    """Return True when the position's trailing stop is already active.

    Uses a notes-entry marker so trailing state survives serialization
    through the JSON storage layer without requiring changes to the
    PaperOrder dataclass.
    """
    notes = getattr(position, "notes", None) or []
    return any(
        isinstance(n, str) and n == "trailing_active" for n in notes
    )


def _mark_trail_active(position: object) -> None:
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


def _before_analysis_paper_check(
    service: PaperTradingService, symbol: str, current_price: float
) -> PaperOrder | None:
    """Check pre-existing paper positions for SL/TP breach.

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

    side = existing.side.value.upper() if hasattr(existing.side, "value") else str(existing.side).upper()

    # Guard: no valid SL/TP levels configured
    if existing.stop_loss <= 0 and existing.take_profit <= 0:
        return None

    closed_order: PaperOrder | None = None

    # For LONG: SL is below entry, TP is above → price must drop to hit SL,
    # rise to hit TP.
    # For SHORT: SL is above entry, TP is below → price must rise to hit SL,
    # drop to hit TP.
    if side == "LONG":
        sl_hit = existing.stop_loss > 0 and current_price <= existing.stop_loss
        tp_hit = existing.take_profit > 0 and current_price >= existing.take_profit
    else:
        # SHORT (and any non-LONG side): invert the comparisons
        sl_hit = existing.stop_loss > 0 and current_price >= existing.stop_loss
        tp_hit = existing.take_profit > 0 and current_price <= existing.take_profit

    if sl_hit:
        # Stop loss hit — close at stop loss price (Guardian Mode priority)
        closed_order = service.close_position(
            symbol, existing.stop_loss, PaperOrderStatus.STOPPED
        )
    elif tp_hit:
        # Take profit hit — close at take profit price
        closed_order = service.close_position(
            symbol, existing.take_profit, PaperOrderStatus.TAKEN_PROFIT
        )

    if closed_order is not None:
        service.save_account(PAPER_STATE_PATH)

    return closed_order


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
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
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
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
        "/close <SYMBOL> <EXIT_PRICE> - Close a paper position\n"
        "/stats - Show paper trading performance dashboard"
    )
    await update.message.reply_text(message)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ping command."""
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
        return

    await update.message.reply_text("🏓 Pong!")


async def version(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /version command."""
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
        return

    from main import __version__
    await update.message.reply_text(f"MuzanTradeBot v{__version__}")


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze a cryptocurrency through the full Guardian Mode pipeline."""
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
        return

    if context.args:
        coin = context.args[0].upper()
    else:
        await update.message.reply_text(
            "❌ Please provide a coin.\n\nExample:\n/analyze BTC"
        )
        return

    result = analyze_coin(coin)
    decision = make_decision(result)
    profile = RiskProfile.from_settings(settings)
    verdict = validate_trade(decision, result, profile)
    trade = build_trade(result, verdict, profile)
    log_trade(
        trade.symbol,
        trade.decision,
        trade.risk_verdict,
    )

    message = (
        f"📊 Analysis for {trade.symbol}\n\n"
        f"Price: {trade.price}\n"
        f"Trend: {trade.trend}\n"
        f"Signal: {trade.signal}\n"
        f"Confidence: {trade.confidence}%\n\n"
        f"🧠 Decision: {trade.decision}\n"
    )

    if trade.approved:
        message += (
            f"✅ Risk: {trade.risk_verdict}\n"
            f"💰 Position Size: {trade.position_size}\n"
            f"🛑 Stop Loss: {trade.stop_loss}\n"
            f"🎯 Take Profit: {trade.take_profit}\n"
        )

        # Paper trading integration
        if settings.PAPER_TRADING:
            try:
                service = _get_paper_service()

                # Check pre-existing positions for SL/TP breach BEFORE placing
                # any new order. This ensures we never evaluate the order that
                # is about to be opened in this same request.
                sltp_closed = _before_analysis_paper_check(
                    service, trade.symbol, trade.price
                )
                if sltp_closed is not None:
                    message += (
                        f"\n📄 Paper Order: CLOSED (SL/TP)\n"
                        f"   Symbol: {sltp_closed.symbol}\n"
                        f"   Side: {sltp_closed.side.value}\n"
                        f"   Exit Price: ${sltp_closed.exit_price:.2f}\n"
                        f"   Exit Status: {paper_exit_status(sltp_closed)}\n"
                        f"   Realized PnL: ${sltp_closed.realized_pnl:.2f} "
                        f"({_format_pnl_label(sltp_closed)})"
                    )
                    # Sprint 9A: push an SL/TP alert (never raises).
                    exit_value = paper_exit_status(sltp_closed)
                    if exit_value == PaperOrderStatus.STOPPED.value:
                        await _dispatch_alert(
                            "stop_loss_triggered", sltp_closed
                        )
                    elif exit_value == PaperOrderStatus.TAKEN_PROFIT.value:
                        await _dispatch_alert(
                            "take_profit_triggered", sltp_closed
                        )

                # Apply trailing stop to any pre-existing position
                # before placing a new order. This runs after the
                # SL/TP closure check so a just-closed position is not
                # re-evaluated, and before place_trade so the new order
                # being opened in this request is never subject to
                # trailing in the same call.
                trail_result = _apply_trailing_stop(
                    service, trade.symbol, trade.price
                )
                if trail_result is not None:
                    changed, old_stop, new_stop = trail_result
                    message += (
                        f"\n📄 Paper Order: TRAILING STOP UPDATED\n"
                        f"   Symbol: {trade.symbol}\n"
                        f"   Stop Loss: ${old_stop:.2f} → ${new_stop:.2f}"
                    )
                    # Sprint 9A: push a trailing-stop alert (never raises).
                    await _dispatch_alert(
                        "trailing_stop_updated", trade.symbol, old_stop, new_stop
                    )

                # Portfolio risk check (Sprint 7A): run after SL/TP
                # closure and trailing stop update, but before placing
                # any new order. When PORTFOLIO_RISK_LIMIT > 0, reject
                # trades that would exceed the configured limit.
                if settings.PORTFOLIO_RISK_LIMIT > 0:
                    manager = PortfolioRiskManager(
                        service.execution_backend(), settings.PORTFOLIO_RISK_LIMIT
                    )
                    if not manager.can_open_trade(trade):
                        current_risk = total_portfolio_risk(
                            service.execution_backend()
                        )
                        message += (
                            f"\n⚠️ Portfolio Risk Limit Exceeded\n"
                            f"   Current Risk: ${current_risk:,.2f}\n"
                            f"   Risk Limit: ${settings.PORTFOLIO_RISK_LIMIT:,.2f}\n"
                            f"   Trade Risk: "
                            f"${trade_risk_value(trade):,.2f}\n"
                            f"   The trade was not placed.\n"
                            f"   Close existing positions to free up risk budget."
                        )
                        # Sprint 9A: push a guardian/risk-limit rejection alert.
                        await _dispatch_alert(
                            "trade_rejected",
                            trade,
                            "portfolio risk limit exceeded",
                        )
                        await update.message.reply_text(message)
                        return

                # Portfolio exposure check (Sprint 7B): run after the
                # portfolio risk check and before placing any new order.
                # When PORTFOLIO_EXPOSURE_LIMIT > 0, reject trades that
                # would exceed the configured notional exposure limit.
                if settings.PORTFOLIO_EXPOSURE_LIMIT > 0:
                    prospective = trade.price * trade.position_size
                    current_exposure = exposure_metrics(
                        service.execution_backend(), 0.0
                    ).current_exposure
                    if current_exposure + prospective > settings.PORTFOLIO_EXPOSURE_LIMIT:
                        metrics = exposure_metrics(
                            service.execution_backend(), settings.PORTFOLIO_EXPOSURE_LIMIT
                        )
                        message += (
                            f"\n⚠️ Portfolio Exposure Limit Exceeded\n"
                            f"   Current Exposure: ${current_exposure:,.2f}\n"
                            f"   Requested Exposure: ${prospective:,.2f}\n"
                            f"   Exposure Limit: ${settings.PORTFOLIO_EXPOSURE_LIMIT:,.2f}\n"
                            f"   The trade was not placed.\n"
                            f"   Close existing positions to free up exposure budget."
                        )
                        # Sprint 9A: push a guardian/risk-limit rejection alert.
                        await _dispatch_alert(
                            "trade_rejected",
                            trade,
                            "portfolio exposure limit exceeded",
                        )
                        await update.message.reply_text(message)
                        return

                # Daily loss circuit breaker (Sprint 8A): run after the
                # portfolio exposure check and before the symbol
                # concentration check, so portfolio-wide rules are
                # evaluated before the narrower per-symbol rule. When
                # PORTFOLIO_DAILY_LOSS_LIMIT > 0, reject any new trade
                # once the calendar day's net realized PnL reaches
                # -limit (reaching the limit stops trading for the day).
                if settings.PORTFOLIO_DAILY_LOSS_LIMIT > 0:
                    daily_risk_manager = DailyRiskManager(
                        service.execution_backend(),
                        settings.PORTFOLIO_DAILY_LOSS_LIMIT,
                    )
                    if not daily_risk_manager.evaluate().allowed:
                        daily_metrics = daily_risk_manager.metrics()
                        message += (
                            f"\n⛔ Daily Loss Limit Reached\n"
                            f"   Day: {daily_metrics.day}\n"
                            f"   Today's Realized PnL: "
                            f"{_format_pnl_value(daily_metrics.realized_pnl_today)}\n"
                            f"   Daily Loss Limit: "
                            f"${daily_metrics.loss_limit:,.2f}\n"
                            f"   The trade was not placed.\n"
                            f"   New paper trades resume on the next day."
                        )
                        # Sprint 9A: push the daily-loss breaker activation
                        # alert plus the guardian rejection alert (both
                        # never raise).
                        await _dispatch_alert(
                            "daily_loss_breaker", daily_metrics
                        )
                        await _dispatch_alert(
                            "trade_rejected",
                            trade,
                            "daily loss breaker active for today",
                        )
                        await update.message.reply_text(message)
                        return

                # Maximum open portfolio positions (Sprint 8B): run after
                # the daily loss circuit breaker and before the symbol
                # concentration check, so portfolio-wide guards are
                # evaluated before the narrower per-symbol rule. When
                # PORTFOLIO_MAX_OPEN_POSITIONS > 0, reject new paper
                # trades once the current open position count reaches the
                # configured limit.
                if settings.PORTFOLIO_MAX_OPEN_POSITIONS > 0:
                    mgr = MaxOpenPositionsManager(
                        service.execution_backend(),
                        settings.PORTFOLIO_MAX_OPEN_POSITIONS,
                    )
                    mop_result = mgr.evaluate()
                    if not mop_result.allowed:
                        message += (
                            f"\n⛔ Maximum Open Positions Reached\n"
                            f"   Current Open Positions: "
                            f"{mop_result.current_open_positions}\n"
                            f"   Max Open Positions: "
                            f"{mop_result.limit}\n"
                            f"   The trade was not placed.\n"
                            f"   Close existing positions to free up "
                            f"open-position capacity."
                        )
                        # Sprint 9A: push the max-open-positions alert plus
                        # the guardian rejection alert (both never raise).
                        await _dispatch_alert(
                            "max_open_positions_reached", mop_result
                        )
                        await _dispatch_alert(
                            "trade_rejected",
                            trade,
                            "maximum open positions limit reached",
                        )
                        await update.message.reply_text(message)
                        return

                # Correlation / concentration check (Sprint 7C): run after
                # the portfolio exposure check and before placing any new
                # order. When either per-symbol limit is configured, reject
                # trades that would exceed the position-count cap or the
                # per-symbol notional exposure cap. Symbol concentration is
                # the narrowest portfolio constraint, so it is evaluated
                # last and reports the most specific failing rule.
                if (
                    settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL > 0
                    or settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL > 0
                ):
                    concentration_manager = CorrelationRiskManager(
                        service.execution_backend(),
                        settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL,
                        settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL,
                    )
                    conc_check = concentration_manager.evaluate(trade)
                    if not conc_check.allowed:
                        conc_metrics = concentration_manager.metrics(
                            trade.symbol
                        )
                        message += (
                            f"\n⚠️ Symbol Concentration Limit Exceeded\n"
                            f"   Symbol: {trade.symbol}\n"
                        )
                        if conc_check.rule == "positions":
                            message += (
                                f"   Open Positions: "
                                f"{conc_metrics.open_positions}\n"
                                f"   Max Positions: "
                                f"{conc_metrics.max_positions}\n"
                            )
                        else:
                            message += (
                                f"   Current Exposure: "
                                f"${conc_metrics.symbol_exposure:,.2f}\n"
                                f"   Requested Exposure: "
                                f"${trade_exposure(trade):,.2f}\n"
                                f"   Exposure Limit: "
                                f"${conc_metrics.exposure_limit:,.2f}\n"
                            )
                        message += (
                            f"   The trade was not placed.\n"
                            f"   Close existing positions to free up "
                            f"concentration budget."
                        )
                        # Sprint 9A: push a guardian/risk-limit rejection alert.
                        await _dispatch_alert(
                            "trade_rejected",
                            trade,
                            "symbol concentration limit exceeded",
                        )
                        await update.message.reply_text(message)
                        return

                # Now place the new trade (if still approved)
                paper_order = service.place_trade(trade)
                if paper_order is not None:
                    service.save_account(PAPER_STATE_PATH)
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
                    # Sprint 9A: push an order-placed alert (never raises).
                    await _dispatch_alert("order_placed", paper_order)
            except Exception as exc:
                # Paper trading failure should not break the main analysis
                # flow, but it must never fail silently: log it so a bug in
                # the paper-trading branch (e.g. an unexpected mock or a
                # manager error) is visible instead of looking like the
                # paper-trading path was never reached.
                logger.warning(
                    "Paper trading integration failed during /analyze "
                    "for %s: %s",
                    trade.symbol,
                    exc,
                )

    else:
        message += f"🛑 Risk: {trade.risk_verdict}\n"
        if trade.risk_reasons:
            message += "\n".join(f"• {reason}" for reason in trade.risk_reasons)

    await update.message.reply_text(message)


# ---------------------------------------------------------------------------
# Paper Trading Management Commands (Sprint 6B)
# ---------------------------------------------------------------------------

async def paper_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show paper account summary."""
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
        return

    service = _get_paper_service()
    summary = service.summary()
    open_positions = service.open_positions()
    closed_positions = service.closed_positions()

    message = (
        f"📄 Paper Account Summary\n\n"
        f"Balance: ${summary.current_balance:.2f}\n"
        f"Equity: ${summary.equity:.2f}\n"
        f"Open Positions: {len(open_positions)}\n"
        f"Closed Positions: {summary.closed_count}\n"
        f"Realized PnL: ${summary.total_realized_pnl:.2f}\n"
        f"Unrealized PnL: ${summary.open_pnl:.2f}"
    )

    if open_positions:
        message += "\n\n📌 Open Positions:\n"
        for pos in open_positions:
            unrealized = pos.unrealized_pnl_for(pos.entry_price)
            message += (
                f"\n  {pos.symbol} {pos.side.value}\n"
                f"  Entry: ${pos.entry_price:.2f}\n"
                f"  Qty: {pos.quantity}\n"
                f"  Unrealized PnL: ${unrealized:.2f}"
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
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
        return

    service = _get_paper_service()
    open_positions = service.open_positions()

    if not open_positions:
        await update.message.reply_text("📭 No open paper positions.")
        return

    message = "📌 Open Paper Positions:\n\n"
    for pos in open_positions:
        unrealized = pos.unrealized_pnl_for(pos.entry_price)
        message += (
            f"{pos.symbol} {pos.side.value}\n"
            f"  Entry: ${pos.entry_price:.2f}\n"
            f"  Qty: {pos.quantity}\n"
            f"  Unrealized PnL: ${unrealized:.2f}\n\n"
        )

    await update.message.reply_text(message)


async def history_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show recently closed paper trades."""
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
        return

    service = _get_paper_service()
    closed_positions = service.closed_positions()

    if not closed_positions:
        await update.message.reply_text("📭 No closed paper trades yet.")
        return

    message = "📜 Closed Paper Trades (most recent first):\n\n"
    for pos in closed_positions[:10]:
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


async def close_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Close an open paper position. Usage: /close <SYMBOL> <EXIT_PRICE>"""
    if not check_rate_limit(update.effective_chat.id, _rate_limit_config):
        await update.message.reply_text(
            "⏳ Too many requests. Please wait a moment before trying again."
        )
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "❌ Usage: /close <SYMBOL> <EXIT_PRICE>\n\n"
            "Example: /close BTC 110.00"
        )
        return

    symbol = context.args[0].upper()
    try:
        exit_price = float(context.args[1])
    except ValueError:
        await update.message.reply_text(
            f"❌ Invalid exit price: '{context.args[1]}'. "
            "Please provide a valid number."
        )
        return

    service = _get_paper_service()
    order = service.close_position(
        symbol, exit_price, PaperOrderStatus.CLOSED_MANUAL
    )

    if order is None:
        await update.message.reply_text(
            f"❌ No open paper position found for {symbol}."
        )
        return

    service.save_account(PAPER_STATE_PATH)
    new_balance = service.summary().current_balance

    await update.message.reply_text(
        f"✅ Position Closed\n\n"
        f"Symbol: {order.symbol}\n"
        f"Side: {order.side.value}\n"
        f"Exit Price: ${order.exit_price:.2f}\n"
        f"Realized PnL: ${order.realized_pnl:.2f} ({_format_pnl_label(order)})\n"
        f"Updated Balance: ${new_balance:.2f}"
    )


# ---------------------------------------------------------------------------
# Paper Trading Performance Analytics (Sprint 6D)
# ---------------------------------------------------------------------------


def _format_pnl_value(value: float) -> str:
    """Format a PnL value with sign and currency symbol."""
    if value > 0:
        return f"+${value:.2f}"
    if value < 0:
        return f"-${abs(value):.2f}"
    return "$0.00"


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


def trade_risk_value(trade) -> float:
    """Compute the dollar risk of a trade (for display in risk limit messages).

    Mirrors the logic in trader.paper.portfolio_risk.trade_risk without
    importing it here to avoid a circular dependency.
    """
    if trade.position_size <= 0 or not trade.stop_loss or trade.stop_loss <= 0:
        return 0.0
    if trade.side == "LONG":
        return max(0.0, (trade.price - trade.stop_loss) * trade.position_size)
    else:
        return max(0.0, (trade.stop_loss - trade.price) * trade.position_size)


async def stats_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Show paper trading performance dashboard."""
    service = _get_paper_service()
    summary = service.summary()
    closed_positions = service.closed_positions()
    open_positions = service.open_positions()

    # --- Section 1: Account ---
    total_return = (
        ((summary.current_balance - summary.initial_balance)
         / summary.initial_balance * 100.0)
        if summary.initial_balance != 0.0
        else 0.0
    )
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
        f"Open PnL: {_format_pnl_value(summary.open_pnl)}\n"
    )

    if not closed_positions:
        message += "\n📭 No closed trades yet."
        await update.message.reply_text(message)
        return

    # --- Section 2: Overall Performance ---
    overall = compute_overall_performance(closed_positions)
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
    risk = compute_risk_metrics(closed_positions)
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

    # --- Section 3b: Portfolio Risk (Sprint 7A) ---
    if settings.PORTFOLIO_RISK_LIMIT > 0:
        manager = PortfolioRiskManager(service.execution_backend(), settings.PORTFOLIO_RISK_LIMIT)
        prm = manager.metrics()
        message += (
            f"\n\n━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Portfolio Risk\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Current Risk: ${prm.current_risk:,.2f}\n"
            f"Risk Limit: ${prm.risk_limit:,.2f}\n"
            f"Risk Usage: {prm.usage_percent:.1f}%\n"
            f"Remaining: ${prm.remaining:,.2f}\n"
        )

    # --- Section 3c: Portfolio Exposure (Sprint 7B) ---
    if settings.PORTFOLIO_EXPOSURE_LIMIT > 0:
        em = exposure_metrics(service.execution_backend(), settings.PORTFOLIO_EXPOSURE_LIMIT)
        message += (
            f"\n\n━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Portfolio Exposure\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"Current Exposure: ${em.current_exposure:,.2f}\n"
            f"Exposure Limit: ${em.exposure_limit:,.2f}\n"
            f"Exposure Usage: {em.usage_percent:.1f}%\n"
            f"Remaining: ${em.remaining:,.2f}\n"
        )

    # --- Section 3d: Symbol Concentration (Sprint 7C) ---
    if (
        settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL > 0
        or settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL > 0
    ):
        concentration_manager = CorrelationRiskManager(
            service.execution_backend(),
            settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL,
            settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL,
        )
        concentration_report = concentration_manager.concentration_report()
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

    # --- Section 3e: Daily Risk (Sprint 8A) ---
    if settings.PORTFOLIO_DAILY_LOSS_LIMIT > 0:
        daily_risk_manager = DailyRiskManager(
            service.execution_backend(),
            settings.PORTFOLIO_DAILY_LOSS_LIMIT,
        )
        daily_metrics = daily_risk_manager.metrics()
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

    # --- Section 3f: Maximum Open Positions (Sprint 8B) ---
    if settings.PORTFOLIO_MAX_OPEN_POSITIONS > 0:
        mgr = MaxOpenPositionsManager(
            service.execution_backend(),
            settings.PORTFOLIO_MAX_OPEN_POSITIONS,
        )
        mop_metrics = mgr.metrics()
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
    symbol_groups = group_by_symbol(closed_positions)
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
    recent = get_recent_trades(closed_positions, limit=10)
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
