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


if __name__ == "__main__":
    main()
