#!/usr/bin/env python3
"""
MuzanTradeBot - Professional Telegram Crypto Trading Assistant
Version: 0.1.0
Mode: Guardian Mode
"""

__version__ = "0.1.0"
__mode__ = "Guardian Mode"

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from config.settings import settings


def main():
    """Entry point for MuzanTradeBot."""
    print("---------------------------------")
    print(f"MuzanTradeBot v{__version__}")
    print("Security Foundation Ready")
    print(f"Guardian Mode: {settings.GUARDIAN_MODE}")
    print(f"Paper Trading: {settings.PAPER_TRADING}")
    print("---------------------------------")

    # Validate execution mode safety (Sprint 10.2)
    # An invalid or inconsistent execution mode must not start the bot.
    # This uses the existing settings safety check; it does not enable
    # any live trading behavior.
    if not settings.execution_mode_is_safe_for_current_intent():
        print("\n[ERROR] Execution mode configuration is not safe to start.")
        print(f"  EXECUTION_MODE: {settings.EXECUTION_MODE}")
        print(f"  PAPER_TRADING: {settings.PAPER_TRADING}")
        print("  Valid modes: PAPER")
        print("  Fix EXECUTION_MODE in your .env file (default is PAPER).")
        sys.exit(1)

    # Validate Telegram token
    if not settings.TELEGRAM_BOT_TOKEN:
        print("\n[ERROR] TELEGRAM_BOT_TOKEN is not set.")
        print("Please add it to your .env file.")
        print("See .env.example for reference.")
        sys.exit(1)

    # Start Telegram bot
    print("\nStarting Telegram bot...")
    from telegram_bot.bot import run_bot
    run_bot()


if __name__ == "__main__":
    main()
