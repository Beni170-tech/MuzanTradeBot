"""Historical market-data loading and backtest service.

This module is the network-facing boundary around the pure backtest
runner. It retrieves a supported CoinGecko historical chart through the
existing provider, validates the returned HistoricalBar sequence, and
then delegates entirely to :func:`backtest.runner.run_backtest`.

The existing provider supports hourly (``1h``) USD observations over a
fixed 30-day lookback. Other timeframes and periods are rejected rather
than silently substituted. No Telegram, live trading, paper-account, or
filesystem state is touched.

Example:
    from backtest.runner import BacktestRunConfig
    from backtest.service import run_historical_backtest
    from risk.risk_profile import RiskProfile

    config = BacktestRunConfig(
        symbol="BTC",
        risk_profile=RiskProfile(balance=1000.0),
    )
    result = run_historical_backtest("BTC", config)
    print(result.backtest.report.net_pnl)
    print(result.data.bar_count)
"""

from dataclasses import dataclass
from datetime import datetime
import math
from urllib.error import HTTPError, URLError

from backtest.replay import HistoricalBar
from backtest.runner import (
    BacktestRunConfig,
    BacktestRunResult,
    run_backtest,
)
from market.analyzer import (
    COINGECKO_IDS,
    MARKET_CHART_DAYS,
    fetch_historical_bars,
)


PROVIDER_NAME = "coingecko"
SUPPORTED_TIMEFRAME = "1h"
SUPPORTED_PERIOD = f"{MARKET_CHART_DAYS}d"


class HistoricalDataError(RuntimeError):
    """Base error for unusable or unavailable historical provider data."""


class HistoricalDataUnavailableError(HistoricalDataError):
    """Raised when the historical provider cannot be reached."""


class HistoricalDataRateLimitError(HistoricalDataUnavailableError):
    """Raised when the historical provider reports a rate limit."""


@dataclass(frozen=True, slots=True)
class HistoricalDataMetadata:
    """Metadata describing the historical data loaded for one run.

    Attributes:
        provider: Historical-data provider identifier.
        symbol: Normalized symbol shared by the bars and backtest config.
        timeframe: Provider timeframe, currently always ``1h``.
        period: Provider lookback, currently always ``30d``.
        bar_count: Number of validated bars loaded.
        first_timestamp: Timestamp of the first bar, or None if empty.
        last_timestamp: Timestamp of the last bar, or None if empty.
    """

    provider: str
    symbol: str
    timeframe: str
    period: str
    bar_count: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class HistoricalBacktestResult:
    """One loaded-data backtest and its data-loading metadata."""

    backtest: BacktestRunResult
    data: HistoricalDataMetadata


def _normalize_symbol(symbol: str) -> str:
    if not isinstance(symbol, str):
        raise TypeError("symbol must be a string")
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("symbol must be a non-empty string")
    if normalized not in COINGECKO_IDS:
        raise ValueError(f"Unsupported historical-data symbol: {symbol}")
    return normalized


def _normalize_timeframe(timeframe: str) -> str:
    if not isinstance(timeframe, str):
        raise TypeError("timeframe must be a string")
    normalized = timeframe.strip().lower()
    if normalized != SUPPORTED_TIMEFRAME:
        raise ValueError(
            f"Unsupported historical timeframe: {timeframe!r}; "
            f"supported timeframe is {SUPPORTED_TIMEFRAME!r}"
        )
    return normalized


def _normalize_period(period: str) -> str:
    if not isinstance(period, str):
        raise TypeError("period must be a string")
    normalized = period.strip().lower()
    if normalized != SUPPORTED_PERIOD:
        raise ValueError(
            f"Unsupported historical period: {period!r}; "
            f"supported period is {SUPPORTED_PERIOD!r}"
        )
    return normalized


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_loaded_bars(bars) -> tuple[HistoricalBar, ...]:
    """Validate provider output without sorting or repairing it."""
    if not isinstance(bars, (list, tuple)):
        raise HistoricalDataError(
            "Historical provider must return a list or tuple of bars"
        )
    if not bars:
        raise HistoricalDataError("Historical provider returned no bars")

    validated = []
    previous_timestamp = None
    for index, bar in enumerate(bars):
        if not isinstance(bar, HistoricalBar):
            raise HistoricalDataError(
                f"Historical bar {index} has invalid type "
                f"{type(bar).__name__}"
            )
        if not isinstance(bar.timestamp, datetime):
            raise HistoricalDataError(
                f"Historical bar {index} timestamp must be a datetime"
            )
        if not _is_finite_number(bar.close) or bar.close <= 0:
            raise HistoricalDataError(
                f"Historical bar {index} close must be positive and finite"
            )
        if not _is_finite_number(bar.volume) or bar.volume < 0:
            raise HistoricalDataError(
                f"Historical bar {index} volume must be non-negative and finite"
            )

        if previous_timestamp is not None:
            try:
                in_order = bar.timestamp > previous_timestamp
            except TypeError as exc:
                raise HistoricalDataError(
                    "Historical bar timestamps must be mutually comparable"
                ) from exc
            if not in_order:
                raise HistoricalDataError(
                    "Historical bar timestamps must be strictly increasing "
                    "without duplicates"
                )
        previous_timestamp = bar.timestamp
        validated.append(bar)

    return tuple(validated)


def _load_bars(symbol: str) -> list[HistoricalBar]:
    try:
        return fetch_historical_bars(symbol)
    except HTTPError as exc:
        if exc.code == 429:
            raise HistoricalDataRateLimitError(
                "Historical data provider rate limit reached"
            ) from exc
        raise HistoricalDataUnavailableError(
            f"Historical data provider returned HTTP {exc.code}"
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HistoricalDataUnavailableError(
            "Historical data provider request failed"
        ) from exc
    except ValueError as exc:
        raise HistoricalDataError(
            "Historical data provider returned malformed or empty data"
        ) from exc
    except Exception as exc:
        raise HistoricalDataError(
            "Historical data provider failed"
        ) from exc


def run_historical_backtest(
    symbol: str,
    config: BacktestRunConfig,
    *,
    timeframe: str = SUPPORTED_TIMEFRAME,
    period: str = SUPPORTED_PERIOD,
) -> HistoricalBacktestResult:
    """Load historical bars and run one pure backtest.

    Args:
        symbol: Supported provider symbol (BTC, ETH, or SOL),
            case-insensitive.
        config: Backtest configuration. Its normalized symbol must match
            the requested symbol.
        timeframe: Provider timeframe. The existing provider supports
            only hourly ``"1h"`` data.
        period: Provider lookback. The existing provider supports only
            ``"30d"``.

    Returns:
        An immutable HistoricalBacktestResult containing the normal runner
        result and metadata describing the bars that were loaded.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the symbol, timeframe, period, or symbol/config
            match is unsupported.
        HistoricalDataRateLimitError: If the provider reports HTTP 429.
        HistoricalDataUnavailableError: If the provider cannot be reached
            or returns another HTTP/network failure.
        HistoricalDataError: If provider data is malformed, empty,
            duplicated, unordered, or otherwise unusable.
    """
    if not isinstance(config, BacktestRunConfig):
        raise TypeError("config must be a BacktestRunConfig")

    normalized_symbol = _normalize_symbol(symbol)
    normalized_timeframe = _normalize_timeframe(timeframe)
    normalized_period = _normalize_period(period)
    if config.symbol != normalized_symbol:
        raise ValueError(
            "config.symbol must match the requested historical symbol"
        )

    bars = _load_bars(normalized_symbol)
    validated_bars = _validate_loaded_bars(bars)
    backtest = run_backtest(validated_bars, config)

    return HistoricalBacktestResult(
        backtest=backtest,
        data=HistoricalDataMetadata(
            provider=PROVIDER_NAME,
            symbol=normalized_symbol,
            timeframe=normalized_timeframe,
            period=normalized_period,
            bar_count=len(validated_bars),
            first_timestamp=validated_bars[0].timestamp,
            last_timestamp=validated_bars[-1].timestamp,
        ),
    )
