"""
MuzanTradeBot - Telegram Command Handlers
All user-facing command logic lives here.
"""

from datetime import datetime
from pathlib import Path

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
from trader.trade import build_trade
from journal.journal import log_trade

# Path for persisting paper account state
PAPER_STATE_PATH = Path("logs/paper_state.json")

# Module-level paper service (lazy-loaded)
_paper_service: PaperTradingService | None = None


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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
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
        "/close <SYMBOL> <EXIT_PRICE> - Close a paper position"
    )
    await update.message.reply_text(message)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /ping command."""
    await update.message.reply_text("🏓 Pong!")


async def version(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /version command."""
    from main import __version__
    await update.message.reply_text(f"MuzanTradeBot v{__version__}")


async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze a cryptocurrency through the full Guardian Mode pipeline."""
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

        # Paper trading integration: place the approved trade as a paper order
        if settings.PAPER_TRADING:
            try:
                service = _get_paper_service()
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
            except Exception:
                # Paper trading failure should not break the main analysis flow
                pass

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
            f"  Closed: {pos.exit_status.value}\n"
            f"  Date: {pos.exit_at.strftime('%Y-%m-%d %H:%M') if pos.exit_at else 'N/A'}\n\n"
        )

    await update.message.reply_text(message)


async def close_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Close an open paper position. Usage: /close <SYMBOL> <EXIT_PRICE>"""
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
