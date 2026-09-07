"""
MuzanTradeBot - Configuration Settings
Loads environment variables securely using python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment, falling back to ``default`` when
    the variable is missing or not a valid number. Invalid configuration
    must never crash the bot or, worse, trade with corrupted numbers."""
    raw = os.getenv(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


class Settings:
    """Central settings loaded from environment variables."""

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # Binance
    BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET: str = os.getenv("BINANCE_API_SECRET", "")

    # Bybit
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_API_SECRET: str = os.getenv("BYBIT_API_SECRET", "")

    # Kraken
    KRAKEN_API_KEY: str = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET: str = os.getenv("KRAKEN_API_SECRET", "")

    # Bot Modes
    GUARDIAN_MODE: bool = os.getenv("GUARDIAN_MODE", "true").lower() in ("true", "1", "yes")
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() in ("true", "1", "yes")

    # Risk Profile (Guardian Mode parameters)
    ACCOUNT_BALANCE: float = _env_float("ACCOUNT_BALANCE", 1000.0)
    RISK_PERCENT: float = _env_float("RISK_PERCENT", 2.0)
    STOP_LOSS_PERCENT: float = _env_float("STOP_LOSS_PERCENT", 2.0)
    TAKE_PROFIT_PERCENT: float = _env_float("TAKE_PROFIT_PERCENT", 4.0)

    # Portfolio Risk Manager (Sprint 7A)
    # Maximum total dollar risk across all open paper positions.
    # When <= 0, no portfolio risk limit is enforced.
    PORTFOLIO_RISK_LIMIT: float = _env_float("PORTFOLIO_RISK_LIMIT", 0.0)

    # Portfolio Exposure Limit (Sprint 7B)
    # Maximum total notional exposure (entry_price * quantity) across all
    # open paper positions. When <= 0, no exposure limit is enforced.
    PORTFOLIO_EXPOSURE_LIMIT: float = _env_float("PORTFOLIO_EXPOSURE_LIMIT", 0.0)

    @classmethod
    def is_configured(cls) -> bool:
        """Check if at least one exchange API key is set."""
        return bool(
            cls.BINANCE_API_KEY
            or cls.BYBIT_API_KEY
            or cls.KRAKEN_API_KEY
        )

    @classmethod
    def summary(cls) -> str:
        """Return a safe summary (no secrets) of current settings."""
        return (
            f"Guardian Mode: {cls.GUARDIAN_MODE}\n"
            f"Paper Trading: {cls.PAPER_TRADING}\n"
            f"Exchanges Configured: {cls.is_configured()}"
        )


# Singleton instance
settings = Settings()
