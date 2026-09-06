"""
MuzanTradeBot - Telegram Command Handlers
All user-facing command logic lives here.
"""

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
        "/analyze <COIN> - Analyze a coin (e.g. /analyze BTC)"
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
