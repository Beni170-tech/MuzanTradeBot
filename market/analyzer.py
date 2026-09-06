"""Market analysis utilities for MuzanTradeBot."""

import json
import urllib.request


# CoinGecko uses coin IDs, not ticker symbols.
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

# Trend, signal, and confidence stay simulated for now; only price
# is fetched live.
SIMULATED = {
    "BTC": {"trend": "Bullish", "signal": "BUY", "confidence": 85},
    "ETH": {"trend": "Bullish", "signal": "BUY", "confidence": 78},
    "SOL": {"trend": "Bearish", "signal": "SELL", "confidence": 74},
}


def analyze_coin(symbol: str) -> dict:
    """Analyze a cryptocurrency and return analysis data.

    Args:
        symbol: The cryptocurrency symbol to analyze (e.g. "BTC").

    Returns:
        A dict with analysis data: symbol, price, trend, signal, and
        confidence. Price is fetched live from the CoinGecko API for
        supported coins (BTC, ETH, SOL); trend, signal, and confidence
        remain simulated. Unsupported coins, API failures, and any
        other exception return placeholder data.
    """
    symbol = symbol.upper()

    if symbol not in COINGECKO_IDS:
        return _placeholder(symbol)

    try:
        price = _fetch_price(symbol)
    except Exception:
        return _placeholder(symbol)

    simulated = SIMULATED[symbol]
    return {
        "symbol": symbol,
        "price": price,
        "trend": simulated["trend"],
        "signal": simulated["signal"],
        "confidence": simulated["confidence"],
    }


def _fetch_price(symbol: str) -> float:
    """Fetch the current USD price of a supported coin from CoinGecko."""
    coin_id = COINGECKO_IDS[symbol]
    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd"
    )
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data[coin_id]["usd"]


def _placeholder(symbol: str) -> dict:
    """Return the placeholder response for unsupported or failed analyses."""
    return {
        "symbol": symbol,
        "price": "Unknown",
        "trend": "Unknown",
        "signal": "WAIT",
        "confidence": 0,
    }