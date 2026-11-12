"""
MuzanTradeBot - Bot Initialization
Sets up the Telegram Application and registers handlers.
"""

import logging
from telegram.ext import Application, CommandHandler

from config.settings import settings
from telegram_bot.handlers import (
    start,
    help_command,
    ping,
    version,
    analyze,
    paper_command,
    positions_command,
    history_command,
    close_command,
    stats_command,
)
from trader.paper.monitor import PaperPositionMonitor

logger = logging.getLogger(__name__)

# Module-level monitor so it can be referenced by run_bot() for shutdown.
_paper_monitor: PaperPositionMonitor | None = None


def build_application() -> Application:
    """Build and configure the Telegram Application."""
    app = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("version", version))
    app.add_handler(CommandHandler("analyze", analyze))
    app.add_handler(CommandHandler("paper", paper_command))
    app.add_handler(CommandHandler("positions", positions_command))
    app.add_handler(CommandHandler("history", history_command))
    app.add_handler(CommandHandler("close", close_command))
    app.add_handler(CommandHandler("stats", stats_command))

    logger.info("Telegram application built successfully.")
    return app


def _build_paper_monitor() -> PaperPositionMonitor | None:
    """Build the background paper position monitor when configured.

    Returns None when paper trading is disabled or when the monitor
    interval is not positive.
    """
    if not settings.PAPER_TRADING:
        logger.info("Paper trading disabled; background monitor not started.")
        return None

    interval = settings.PAPER_MONITOR_INTERVAL_SECONDS
    if interval <= 0:
        logger.info(
            "PAPER_MONITOR_INTERVAL_SECONDS <= 0; background monitor disabled."
        )
        return None

    from telegram_bot.handlers import _get_paper_service, _get_alert_manager

    service = _get_paper_service()
    alerts = _get_alert_manager()
    monitor = PaperPositionMonitor(service, alerts, interval_seconds=interval)
    return monitor


async def _shutdown_paper_monitor(application) -> None:
    """Cancel and await the paper monitor from PTB's post_shutdown hook.

    python-telegram-bot calls ``post_shutdown`` while the polling event
    loop is still open and closes the loop only afterwards. Awaiting
    ``aclose()`` here guarantees the monitor task is finished rather than
    left pending for the loop to destroy.
    """
    if _paper_monitor is None:
        return
    try:
        await _paper_monitor.aclose()
    except Exception:
        logger.exception(
            "Paper position monitor shutdown failed; the monitor task may "
            "still be pending."
        )


def run_bot() -> None:
    """Start the Telegram bot with polling."""
    global _paper_monitor

    app = build_application()

    # PTB runs this hook on the still-open polling loop immediately before
    # it closes that loop, so the monitor can be cancelled and awaited.
    app.post_shutdown = _shutdown_paper_monitor

    # Start the background position monitor when configured.
    # The monitor runs as an asyncio task on the polling loop.
    _paper_monitor = _build_paper_monitor()
    if _paper_monitor is not None:
        _paper_monitor.start()
        logger.info("Background paper position monitor started.")

    print("Bot application created.")
    print("Starting polling...")

    try:
        app.run_polling(drop_pending_updates=True)
    finally:
        # Fallback for paths where post_shutdown did not run. This is a
        # no-op when the hook above already stopped the monitor, and it is
        # safe to call after PTB has closed the loop.
        if _paper_monitor is not None:
            _paper_monitor.stop()
            logger.info("Background paper position monitor stopped during shutdown.")

    print("Polling stopped.")