"""Market analysis utilities for MuzanTradeBot.

Trend, signal, and confidence are now computed from real market data
(EMA/RSI/MACD/volume over the last 30 days of hourly candles fetched
from CoinGecko) instead of simulated values. The analysis dict keeps
the exact schema the rest of the pipeline depends on.
"""

import json
import urllib.request

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


def _fetch_market_chart(symbol: str):
    """Fetch closing prices and volumes for a supported coin from CoinGecko.

    Returns:
        A tuple (closes, volumes): chronological lists of closing prices
        and trading volumes over the configured lookback window.
    """
    coin_id = COINGECKO_IDS[symbol]
    url = (
        "https://api.coingecko.com/api/v3/coins/"
        f"{coin_id}/market_chart?vs_currency=usd&days={MARKET_CHART_DAYS}"
    )
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    closes = [float(point[1]) for point in data["prices"]]
    volumes = [float(point[1]) for point in data.get("total_volumes", [])]

    if not closes:
        raise ValueError(f"No price data returned for {symbol}")

    return closes, volumes


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
