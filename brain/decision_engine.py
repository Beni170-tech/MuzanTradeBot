"""
MuzanTradeBot - Decision Engine

Core decision-making logic for the bot. In Guardian Mode, the
decision engine decides whether a trade signal is strong enough to
execute, prioritizing capital preservation above all else.
"""

import math

from strategy.parameters import (
    DEFAULT_STRATEGY_PARAMETERS,
    StrategyParameters,
)


def make_decision(analysis: dict, *, parameters=None) -> str:
    """Decide what action to take based on market analysis.

    Args:
        analysis: A dict of market analysis data (e.g. from
            market.analyzer.analyze_coin) containing price, trend,
            signal, and confidence.
        parameters: Optional immutable strategy configuration. ``None``
            uses the existing defaults exactly.

    Returns:
        A decision string: "BUY", "SELL", or "WAIT". A trade is
        only recommended when the signal, trend, and confidence all
        agree strongly; anything malformed or weak returns "WAIT".

    Raises:
        TypeError: If ``parameters`` is not a StrategyParameters or None.
    """
    if parameters is None:
        parameters = DEFAULT_STRATEGY_PARAMETERS
    elif not isinstance(parameters, StrategyParameters):
        raise TypeError("parameters must be a StrategyParameters or None")

    signal = analysis.get("signal")
    trend = analysis.get("trend")
    confidence = analysis.get("confidence")

    if not isinstance(signal, str) or not isinstance(trend, str):
        return "WAIT"
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        return "WAIT"
    # Compare against the valid range before math.isfinite(): very large
    # integers are out of range and would otherwise raise OverflowError.
    if confidence < 0 or confidence > 100:
        return "WAIT"
    if not math.isfinite(confidence):
        return "WAIT"

    if (
        signal == "BUY"
        and trend == "Bullish"
        and confidence >= parameters.min_trade_confidence
    ):
        return "BUY"

    if (
        signal == "SELL"
        and trend == "Bearish"
        and confidence >= parameters.min_trade_confidence
    ):
        return "SELL"

    return "WAIT"