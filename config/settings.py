"""
MuzanTradeBot - Configuration Settings
Loads environment variables securely using python-dotenv.
"""

import math
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


#: Environment variables that could not be parsed. The variable name is
#: recorded rather than raised: importing the module must stay safe, and
#: ``validate_settings`` turns each entry into a startup error that names
#: the variable.
_INVALID_ENV_VALUES: dict[str, str] = {}

# Conservative portfolio-limit defaults, used only when the matching
# environment variable is unset or blank. They are derived from
# ACCOUNT_BALANCE so a smaller paper account gets proportionally smaller
# dollar limits. An explicit "0" still disables a limit.
_DEFAULT_PORTFOLIO_RISK_PERCENT = 6.0
_DEFAULT_PORTFOLIO_DAILY_LOSS_PERCENT = 3.0
# One default-sized trade risks 2% and uses about 100% of equity as
# notional, so the exposure defaults must leave room for the position
# count they accompany (per-symbol 1, portfolio 3).
_DEFAULT_PORTFOLIO_EXPOSURE_PERCENT = 300.0
_DEFAULT_PORTFOLIO_SYMBOL_EXPOSURE_PERCENT = 200.0
_DEFAULT_PORTFOLIO_MAX_OPEN_POSITIONS = 3
_DEFAULT_PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = 1


def _raw_env(name: str) -> str | None:
    """Return the stripped environment value, or None when unset/blank.

    A blank value (``VAR=`` in ``.env``) is treated as unset so it falls
    back to the conservative default instead of an accidental zero.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return raw.strip()


def _record_invalid(name: str, reason: str) -> None:
    """Remember an unparseable environment value for startup validation."""
    _INVALID_ENV_VALUES.setdefault(name, reason)


def _env_float(name: str, default: float, *, strict: bool = False) -> float:
    """Read a float from the environment, falling back to ``default``.

    With ``strict=True`` an unusable value is recorded so startup
    validation can block the bot and name the variable. Without it the
    old lenient fallback is preserved for settings that are not risk
    limits.
    """
    raw = _raw_env(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        if strict:
            _record_invalid(name, "must be a valid number")
        return default
    if strict and not math.isfinite(value):
        _record_invalid(name, "must be a finite number")
    return value


def _env_int(name: str, default: int, *, strict: bool = False) -> int:
    """Read an int from the environment, falling back to ``default``.

    With ``strict=True`` an unusable value is recorded so startup
    validation can block the bot and name the variable. Without it the
    old lenient fallback is preserved for settings that are not risk
    limits.
    """
    raw = _raw_env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        if strict:
            _record_invalid(name, "must be a whole number")
        return default
    return value


class Settings:
    """Central settings loaded from environment variables."""

    # Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # Telegram access control (Milestone M-A)
    # Comma-separated Telegram user ids and/or chat ids allowed to use
    # the bot. Empty (the default) authorizes nobody: the bot answers
    # unauthorized updates with a denial notice instead of account data.
    TELEGRAM_ALLOWED_CHAT_IDS: str = os.getenv(
        "TELEGRAM_ALLOWED_CHAT_IDS", ""
    )

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
    # These are the inputs every portfolio limit and every position is
    # sized from, so they are parsed strictly: unset/blank keeps the
    # documented default, while a malformed or non-finite value is
    # recorded and blocks startup by name instead of silently reverting
    # to a default that would rescale the whole risk budget.
    ACCOUNT_BALANCE: float = _env_float("ACCOUNT_BALANCE", 1000.0, strict=True)
    RISK_PERCENT: float = _env_float("RISK_PERCENT", 2.0, strict=True)
    STOP_LOSS_PERCENT: float = _env_float("STOP_LOSS_PERCENT", 2.0, strict=True)
    TAKE_PROFIT_PERCENT: float = _env_float(
        "TAKE_PROFIT_PERCENT", 4.0, strict=True
    )

    # Portfolio Risk Manager (Sprint 7A)
    # Maximum total dollar risk across all open paper positions.
    # Unset/blank defaults to 6% of ACCOUNT_BALANCE; an explicit 0
    # disables the gate and a malformed value blocks startup.
    PORTFOLIO_RISK_LIMIT: float = _env_float(
        "PORTFOLIO_RISK_LIMIT",
        round(ACCOUNT_BALANCE * _DEFAULT_PORTFOLIO_RISK_PERCENT / 100, 2),
        strict=True,
    )

    # Portfolio Exposure Limit (Sprint 7B)
    # Maximum total notional exposure (entry_price * quantity) across all
    # open paper positions. Unset/blank defaults to 300% of
    # ACCOUNT_BALANCE; an explicit 0 disables the gate.
    PORTFOLIO_EXPOSURE_LIMIT: float = _env_float(
        "PORTFOLIO_EXPOSURE_LIMIT",
        round(ACCOUNT_BALANCE * _DEFAULT_PORTFOLIO_EXPOSURE_PERCENT / 100, 2),
        strict=True,
    )

    # Correlation Risk Manager (Sprint 7C)
    # Maximum number of open paper positions allowed on a single symbol.
    # Unset/blank defaults to 1; an explicit 0 disables the rule.
    PORTFOLIO_MAX_POSITIONS_PER_SYMBOL: int = _env_int(
        "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
        _DEFAULT_PORTFOLIO_MAX_POSITIONS_PER_SYMBOL,
        strict=True,
    )

    # Maximum total notional exposure (entry_price * quantity) allowed on a
    # single symbol across all open paper positions. Unset/blank defaults
    # to 200% of ACCOUNT_BALANCE; an explicit 0 disables the rule.
    PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL: float = _env_float(
        "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
        round(
            ACCOUNT_BALANCE * _DEFAULT_PORTFOLIO_SYMBOL_EXPOSURE_PERCENT / 100,
            2,
        ),
        strict=True,
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
    # Unset/blank defaults to 3% of ACCOUNT_BALANCE; an explicit 0
    # disables the breaker.
    PORTFOLIO_DAILY_LOSS_LIMIT: float = _env_float(
        "PORTFOLIO_DAILY_LOSS_LIMIT",
        round(ACCOUNT_BALANCE * _DEFAULT_PORTFOLIO_DAILY_LOSS_PERCENT / 100, 2),
        strict=True,
    )

    # Maximum Open Portfolio Positions (Sprint 8B)
    # Maximum number of open paper positions allowed across the whole
    # paper account. When the current open position count reaches this
    # limit, new paper trades are rejected until existing positions are
    # closed. Applies to all symbols combined. Unset/blank defaults to 3;
    # an explicit 0 disables the limit.
    PORTFOLIO_MAX_OPEN_POSITIONS: int = _env_int(
        "PORTFOLIO_MAX_OPEN_POSITIONS",
        _DEFAULT_PORTFOLIO_MAX_OPEN_POSITIONS,
        strict=True,
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


# ---------------------------------------------------------------------------
# Startup configuration validation (Milestone M-A)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SettingsIssue:
    """One configuration problem or visibility notice.

    Attributes:
        severity: ``"error"`` blocks startup; ``"warning"`` is shown
            but does not prevent starting.
        setting: The environment variable / setting name.
        message: A short explanation with no secret values.
    """

    severity: str
    setting: str
    message: str

    def format(self) -> str:
        """Return one printable line naming the setting."""
        return f"{self.setting}: {self.message}"


# Limits documented as "<= 0 disables". A non-positive value keeps its
# documented meaning, but 0 is the supported disable value, so a negative
# limit is reported as a configuration mistake rather than silently
# disabling a safety feature.
_LIMIT_SETTINGS = (
    "PORTFOLIO_RISK_LIMIT",
    "PORTFOLIO_EXPOSURE_LIMIT",
    "PORTFOLIO_DAILY_LOSS_LIMIT",
    "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
)

_COUNT_LIMIT_SETTINGS = (
    "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
    "PORTFOLIO_MAX_OPEN_POSITIONS",
)



def _is_finite_number(value: object) -> bool:
    """Return True for a real, finite int/float (never a bool)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def validate_settings() -> list[SettingsIssue]:
    """Validate risk-related configuration for startup.

    Rejects values that cannot be used safely (negative limits,
    non-finite numbers, a zero risk percentage) while preserving the
    documented ``0 = disabled`` behaviour for portfolio limits.

    Returns:
        Every issue found, in a stable order. Errors block startup;
        warnings (including disabled limits) are reported so an operator
        can see that a safety feature is off.
    """
    issues: list[SettingsIssue] = []

    def error(setting: str, message: str) -> None:
        issues.append(SettingsIssue("error", setting, message))

    def warning(setting: str, message: str) -> None:
        issues.append(SettingsIssue("warning", setting, message))

    balance = settings.ACCOUNT_BALANCE
    if not _is_finite_number(balance) or balance <= 0:
        error(
            "ACCOUNT_BALANCE",
            "must be a finite number greater than 0",
        )

    risk = settings.RISK_PERCENT
    if not _is_finite_number(risk) or risk <= 0:
        error(
            "RISK_PERCENT",
            "must be a finite number greater than 0",
        )
    elif risk > 100:
        error("RISK_PERCENT", "must not exceed 100 percent per trade")

    for name in ("STOP_LOSS_PERCENT", "TAKE_PROFIT_PERCENT"):
        value = getattr(settings, name)
        if not _is_finite_number(value) or value <= 0:
            error(name, "must be a finite number greater than 0")

    for name in _LIMIT_SETTINGS:
        if name in _INVALID_ENV_VALUES:
            continue
        value = getattr(settings, name)
        if not _is_finite_number(value) or value < 0:
            error(
                name,
                "must be a finite number >= 0 (use 0 to disable)",
            )
        elif value == 0:
            warning(name, "disabled (0); no limit is enforced")

    for name in _COUNT_LIMIT_SETTINGS:
        if name in _INVALID_ENV_VALUES:
            continue
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int):
            error(name, "must be an integer >= 0 (use 0 to disable)")
        elif value < 0:
            error(name, "must be an integer >= 0 (use 0 to disable)")
        elif value == 0:
            warning(name, "disabled (0); no limit is enforced")

    # A value that could not be parsed must block startup instead of
    # silently falling back to a default: an operator who mistypes a
    # limit should not run with a limit they did not intend.
    for name, reason in _INVALID_ENV_VALUES.items():
        error(name, f"{reason}; fix or remove the value in .env")

    all_limits = _LIMIT_SETTINGS + _COUNT_LIMIT_SETTINGS
    if not any(
        name in _INVALID_ENV_VALUES for name in all_limits
    ) and all(getattr(settings, name) == 0 for name in all_limits):
        warning(
            "PORTFOLIO_LIMITS",
            "all portfolio limits are disabled (0); Guardian Mode has no "
            "portfolio backstops",
        )

    interval = settings.PAPER_MONITOR_INTERVAL_SECONDS
    if not _is_finite_number(interval):
        error(
            "PAPER_MONITOR_INTERVAL_SECONDS",
            "must be a finite number of seconds",
        )
    elif interval <= 0:
        warning(
            "PAPER_MONITOR_INTERVAL_SECONDS",
            "disabled (<= 0); open positions are not monitored in "
            "the background",
        )

    rate_window = settings.TELEGRAM_COMMAND_RATE_WINDOW_SECONDS
    if not _is_finite_number(rate_window):
        error(
            "TELEGRAM_COMMAND_RATE_WINDOW_SECONDS",
            "must be a finite number of seconds",
        )

    # EXECUTION_MODE is validated separately by the existing startup
    # safety check in main.py (Sprint 10.2); it is not a risk number and
    # is deliberately left out of this validator.

    return issues


def settings_errors(
    issues: list[SettingsIssue] | None = None,
) -> list[SettingsIssue]:
    """Return only the startup-blocking issues."""
    if issues is None:
        issues = validate_settings()
    return [issue for issue in issues if issue.severity == "error"]