"""
MuzanTradeBot - Bot Initialization
Sets up the Telegram Application and registers handlers.
"""

import logging
from telegram.ext import Application, CommandHandler

from config.settings import settings
from telegram_bot.handlers import start, help_command, ping, version, analyze
logger = logging.getLogger(__name__)


def build_application() -> Application:
    """Build and configure the Telegram Application."""
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("version", version))
    app.add_handler(CommandHandler("analyze", analyze))

    logger.info("Telegram application built successfully.")
    return app


def run_bot() -> None:
    """Start the Telegram bot with polling."""
    app = build_application()
    logger.info("Starting bot polling...")
    app.run_polling(drop_pending_updates=True)
    