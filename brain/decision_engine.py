"""
MuzanTradeBot - Decision Engine

Core decision-making logic for the bot. In Guardian Mode, the
decision engine decides whether a trade signal is strong enough to
execute, prioritizing capital preservation above all else.
"""


def make_decision(analysis: dict) -> str:
    """Decide what action to take based on market analysis.

    Args:
        analysis: A dict of market analysis data (e.g. from
            market.analyzer.analyze_coin) containing price, trend,
            signal, and confidence.

    Returns:
        A decision string: "BUY", "SELL", or "WAIT". A trade is
        only recommended when the signal, trend, and confidence all
        agree strongly; anything malformed or weak returns "WAIT".
    """
    signal = analysis.get("signal")
    trend = analysis.get("trend")
    confidence = analysis.get("confidence")

    if not isinstance(signal, str) or not isinstance(trend, str):
        return "WAIT"
    if not isinstance(confidence, (int, float)):
        return "WAIT"

    if (
        signal == "BUY"
        and trend == "Bullish"
        and confidence >= 70
    ):
        return "BUY"

    if (
        signal == "SELL"
        and trend == "Bearish"
        and confidence >= 70
    ):
        return "SELL"

    return "WAIT"