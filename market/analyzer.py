"""Market analysis utilities for MuzanTradeBot.

Trend, signal, and confidence are now computed from real market data
(EMA/RSI/MACD/volume over the last 30 days of hourly candles fetched
from CoinGecko) instead of simulated values. The analysis dict keeps
the exact schema the rest of the pipeline depends on.
"""

from datetime import datetime, timezone
import json
import math
import urllib.request

from backtest.replay import HistoricalBar
from market.indicators import MIN_CANDLES, compute_analysis


# CoinGecko uses coin IDs, not ticker symbols.
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

# How many days of history to fetch. CoinGecko returns hourly candles
# for ranges up to 90 days, so 30 days yields ~720 bars per coin.
MARKET_CHART_DAYS = 30

# CoinGecko's lightweight simple-price endpoint returns one current price
# per coin. The background paper monitor polls this instead of downloading
# 30 days of chart history on every cycle.
SIMPLE_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"


def get_current_price(symbol: str) -> float | None:
    """Return the latest USD market price for a supported symbol.

    Uses the same CoinGecko coin-id mapping as :func:`analyze_coin`, but
    fetches only the current simple price (no historical chart data), so
    it is suitable for the background paper position monitor.

    Args:
        symbol: The cryptocurrency symbol (e.g. "BTC").

    Returns:
        The current price as a positive finite float, or None when the
        symbol is unsupported, the request fails, or the response does
        not contain a usable price. Callers must treat None as "no
        reliable price" and never fall back to a stored price.
    """
    if not isinstance(symbol, str):
        return None

    coin_id = COINGECKO_IDS.get(symbol.strip().upper())
    if coin_id is None:
        return None

    url = f"{SIMPLE_PRICE_URL}?ids={coin_id}&vs_currencies=usd"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        price = data[coin_id]["usd"]
    except Exception:
        return None

    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return None
    if not math.isfinite(price) or price <= 0:
        return None
    return float(price)


def analyze_coin(symbol: str) -> dict:
    """Analyze a cryptocurrency and return analysis data.

    Args:
        symbol: The cryptocurrency symbol to analyze (e.g. "BTC").

    Returns:
        A dict with analysis data: symbol, price, trend, signal, and
        confidence, plus an "indicators" snapshot for transparency.
        Price, trend, signal, and confidence are computed from live
        CoinGecko market data for supported coins (BTC, ETH, SOL).
        Unsupported coins, API failures, and malformed or insufficient
        data return placeholder data, which downstream risk checks
        block safely.

    Guardian Mode purpose:
        Every failure path resolves to a placeholder (Unknown trend,
        WAIT signal, 0 confidence) so the bot never trades on stale,
        partial, or simulated information.
    """
    symbol = symbol.upper()

    if symbol not in COINGECKO_IDS:
        return _placeholder(symbol)

    try:
        closes, volumes = _fetch_market_chart(symbol)
    except Exception:
        return _placeholder(symbol)

    if len(closes) < MIN_CANDLES:
        return _placeholder(symbol)

    verdict = compute_analysis(closes, volumes)
    if verdict["trend"] == "Unknown":
        return _placeholder(symbol)

    return {
        "symbol": symbol,
        "price": closes[-1],
        "trend": verdict["trend"],
        "signal": verdict["signal"],
        "confidence": verdict["confidence"],
        "indicators": verdict["indicators"],
    }


def _parse_timestamp(value, label: str) -> datetime:
    """Parse a CoinGecko millisecond timestamp into an aware UTC datetime."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} timestamps must be numeric milliseconds")
    try:
        milliseconds = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{label} timestamps must be numeric milliseconds"
        ) from exc
    if not math.isfinite(milliseconds) or milliseconds < 0:
        raise ValueError(
            f"{label} timestamps must be finite and non-negative"
        )
    try:
        return datetime.fromtimestamp(
            milliseconds / 1000.0,
            tz=timezone.utc,
        )
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(
            f"{label} timestamp is outside the supported range"
        ) from exc


def _parse_series(points, label: str, *, positive: bool):
    """Validate one CoinGecko observation series and preserve timestamps."""
    if not isinstance(points, list):
        raise ValueError(f"{label} observations must be a list")

    observations = []
    previous_timestamp = None
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"{label}[{index}] must contain exactly a timestamp and value"
            )

        timestamp = _parse_timestamp(point[0], f"{label}[{index}]")
        value = point[1]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}[{index}] value must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}[{index}] value must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{label}[{index}] value must be finite")
        if positive and number <= 0.0:
            raise ValueError(f"{label}[{index}] value must be positive")
        if not positive and number < 0.0:
            raise ValueError(f"{label}[{index}] value must be non-negative")

        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError(
                f"{label} timestamps must be strictly increasing and unique"
            )
        previous_timestamp = timestamp
        observations.append((timestamp, number))

    if not observations:
        raise ValueError(f"No {label} data returned")
    return observations


def fetch_historical_bars(symbol: str) -> list[HistoricalBar]:
    """Fetch validated historical bars with their original timestamps.

    CoinGecko returns separate price and volume observation arrays. They
    are parsed and paired by timestamp rather than by list position, so
    a missing or extra observation cannot shift prices onto the wrong
    volumes. A price timestamp without matching volume data is rejected.

    Args:
        symbol: Supported coin symbol (BTC, ETH, or SOL), case-insensitive.

    Returns:
        Chronological, immutable HistoricalBar values using aware UTC
        timestamps.

    Raises:
        ValueError: If the symbol is unsupported, the response is
            malformed, observations are empty, duplicated, unordered, or
            invalid, or price/volume timestamps do not align.
    """
    if not isinstance(symbol, str):
        raise ValueError("symbol must be a string")
    coin_id = COINGECKO_IDS.get(symbol.strip().upper())
    if coin_id is None:
        raise ValueError(f"Unsupported CoinGecko symbol: {symbol}")

    url = (
        "https://api.coingecko.com/api/v3/coins/"
        f"{coin_id}/market_chart?vs_currency=usd&days={MARKET_CHART_DAYS}"
    )
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    if not isinstance(data, dict):
        raise ValueError("CoinGecko market chart response must be an object")
    if "prices" not in data or "total_volumes" not in data:
        raise ValueError("CoinGecko market chart is missing prices or volumes")

    prices = _parse_series(data["prices"], "prices", positive=True)
    volumes = _parse_series(
        data["total_volumes"],
        "total_volumes",
        positive=False,
    )
    volume_by_timestamp = dict(volumes)

    bars = []
    for timestamp, price in prices:
        if timestamp not in volume_by_timestamp:
            raise ValueError(
                "price and volume observations do not align by timestamp"
            )
        bars.append(
            HistoricalBar(
                timestamp=timestamp,
                close=price,
                volume=volume_by_timestamp[timestamp],
            )
        )
    return bars


def _fetch_market_chart(symbol: str):
    """Fetch closing prices and volumes for a supported coin from CoinGecko.

    This compatibility helper preserves the existing analysis contract
    while sourcing values from validated, timestamp-aligned bars.

    Returns:
        A tuple (closes, volumes): chronological lists of closing prices
        and trading volumes over the configured lookback window.
    """
    bars = fetch_historical_bars(symbol)
    return (
        [bar.close for bar in bars],
        [bar.volume for bar in bars],
    )


def _placeholder(symbol: str) -> dict:
    """Return the placeholder response for unsupported or failed analyses."""
    return {
        "symbol": symbol,
        "price": "Unknown",
        "trend": "Unknown",
        "signal": "WAIT",
        "confidence": 0,
        "indicators": None,
    }
