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


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to ``default`` when
    the variable is missing or not a valid number. Invalid configuration
    must never crash the bot or, worse, trade with corrupted numbers."""
    raw = os.getenv(name, "")
    try:
        return int(raw)
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

    # Execution mode (Sprint 10.2)
    # Determines which execution backend is active. PAPER is the safe
    # default and the only supported mode for now. The mode is explicit
    # and validated: an unrecognized value is treated as an unsafe config
    # and must not enable live execution.
    EXECUTION_MODE: str = os.getenv("EXECUTION_MODE", "PAPER").strip().upper()

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

    # Correlation Risk Manager (Sprint 7C)
    # Maximum number of open paper positions allowed on a single symbol.
    # When <= 0, the per-symbol position-count rule is disabled.
    PORTFOLIO_MAX_POSITIONS_PER_SYMBOL: int = _env_int(
        "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL", 0
    )

    # Maximum total notional exposure (entry_price * quantity) allowed on a
    # single symbol across all open paper positions. When <= 0, the
    # per-symbol exposure rule is disabled.
    PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL: float = _env_float(
        "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL", 0.0
    )

    # Telegram Alerts (Sprint 9A)
    # Master switch for push notifications (paper order placed, SL/TP
    # triggered, trailing stop updates, guardian/risk-limit rejections,
    # daily loss breaker activation, max open positions reached).
    # When False, no alerts are sent anywhere.
    ALERTS_ENABLED: bool = os.getenv("ALERTS_ENABLED", "false").lower() in (
        "true", "1", "yes"
    )

    # Telegram chat id alerts are sent to. When empty, alerts are
    # disabled for this run even if ALERTS_ENABLED is true.
    TELEGRAM_ALERT_CHAT_ID: str = os.getenv("TELEGRAM_ALERT_CHAT_ID", "")

    # Daily Loss Circuit Breaker (Sprint 8A)
    # Maximum net realized loss allowed per local calendar day across
    # the whole paper account. When the day's realized PnL reaches
    # -limit, new paper trades are halted until the next day.
    # When <= 0, no daily loss limit is enforced.
    PORTFOLIO_DAILY_LOSS_LIMIT: float = _env_float(
        "PORTFOLIO_DAILY_LOSS_LIMIT", 0.0
    )

    # Maximum Open Portfolio Positions (Sprint 8B)
    # Maximum number of open paper positions allowed across the whole
    # paper account. When the current open position count reaches this
    # limit, new paper trades are rejected until existing positions are
    # closed. Applies to all symbols combined. When <= 0, the limit is
    # disabled.
    PORTFOLIO_MAX_OPEN_POSITIONS: int = _env_int(
        "PORTFOLIO_MAX_OPEN_POSITIONS", 0
    )

    # Background Position Monitor (Sprint 9B)
    # Interval in seconds between background monitoring cycles for open
    # paper positions. When <= 0, the background monitor is disabled.
    # The default is 45 seconds, which balances responsiveness with
    # respect for CoinGecko rate limits and Telegram polling.
    PAPER_MONITOR_INTERVAL_SECONDS: float = _env_float(
        "PAPER_MONITOR_INTERVAL_SECONDS", 45.0
    )

    # Telegram Command Rate Limiting (Sprint 10A)
    # Maximum number of Telegram commands allowed per chat in each
    # configured window. When <= 0, command rate limiting is disabled
    # entirely. This is a spam/abuse guard, not a trade execution
    # gate, so it never touches Guardian Mode or the trading flow.
    TELEGRAM_COMMAND_RATE_LIMIT_PER_WINDOW: int = _env_int(
        "TELEGRAM_COMMAND_RATE_LIMIT_PER_WINDOW", 12
    )

    # Telegram Command Rate Limiting (Sprint 10A)
    # Sliding window length in seconds for the per-chat command rate
    # limit. When <= 0, command rate limiting is disabled entirely.
    TELEGRAM_COMMAND_RATE_WINDOW_SECONDS: float = _env_float(
        "TELEGRAM_COMMAND_RATE_WINDOW_SECONDS", 10.0
    )

    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() in ("true", "1", "yes")

    # Valid execution modes (Sprint 10.2)
    _VALID_EXECUTION_MODES = ("PAPER",)

    # Live mode registry (Sprint 10.2)
    # Keep this table small and explicit. Right now it is intentionally empty
    # because no live mode exists yet. When a live mode is introduced it must
    # be added here, not inferred from strings.
    _LIVE_EXECUTION_MODES: tuple[str, ...] = ()

    # Runtime source of truth for paper trading (Sprint 10.2 safety check).
    # PAPER_TRADING remains the existing runtime switch. EXECUTION_MODE is the
    # newer config label. They must not contradict each other while live
    # execution is not implemented.
    @classmethod
    def execution_mode_is_safe_for_current_intent(cls) -> bool:
        """Return True when EXECUTION_MODE is consistent with PAPER_TRADING.

        This is a safety check, not a behavior change. It never enables live
        trading and it does not replace PAPER_TRADING as the runtime source of
        truth.

        Current rules:
        - PAPER_TRADING=True is allowed only with EXECUTION_MODE=PAPER.
        - PAPER_TRADING=False is not a live mode. Since no live backend
          exists, a disabled paper config is still treated as safe to start,
          as long as the execution mode itself is valid.
        - Any unrecognized EXECUTION_MODE is unsafe regardless of
          PAPER_TRADING.
        """
        if cls.EXECUTION_MODE not in cls._VALID_EXECUTION_MODES:
            return False

        if cls.PAPER_TRADING:
            # If paper trading is enabled, the only acceptable execution mode
            # right now is PAPER.
            return cls.EXECUTION_MODE == "PAPER"

        # PAPER_TRADING=False is currently an "off" configuration. Because
        # no live mode is implemented, this is not treated as a valid live
        # execution mode and it never enables live behavior.
        return True

    @classmethod
    def is_execution_mode_valid(cls) -> bool:
        """Return True when the configured execution mode is recognized.

        An invalid or unknown execution mode must never be treated as
        live-enabled. The only valid mode right now is PAPER.
        """
        return cls.EXECUTION_MODE in cls._VALID_EXECUTION_MODES

    @classmethod
    def is_paper_mode(cls) -> bool:
        """Return True when execution is explicitly in PAPER mode.

        This is the safe default. Any future live mode must be enabled
        explicitly and independently of this flag.
        """
        return cls.EXECUTION_MODE == "PAPER"

    @classmethod
    def is_live_mode(cls) -> bool:
        """Return True when execution is explicitly in a live mode.

        There is no live mode implemented yet, so this always returns
        False for valid configurations.
        """
        return cls.EXECUTION_MODE in cls._LIVE_EXECUTION_MODES

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
            f"Execution Mode: {cls.EXECUTION_MODE}\n"
            f"Exchanges Configured: {cls.is_configured()}"
        )


# Singleton instance
settings = Settings()