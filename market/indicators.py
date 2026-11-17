"""
MuzanTradeBot - Technical Indicators

Pure-Python implementations of the indicators used by the market
analyzer (EMA, RSI, MACD, and volume ratio) plus a multi-confirmation
synthesizer that turns a series of closes into the trend, signal, and
confidence values the rest of the pipeline expects.

Guardian Mode purpose:
    No single indicator decides a trade. A BUY or SELL is only
    emitted when several independent measures agree with the trend,
    and anything ambiguous or lacking data resolves to WAIT so the
    existing risk gates decide the outcome.
"""

from strategy.parameters import (
    DEFAULT_STRATEGY_PARAMETERS,
    StrategyParameters,
)

# Lookback / sensitivity constants. They mirror the immutable default
# strategy configuration so existing imports keep working.
FAST_EMA = DEFAULT_STRATEGY_PARAMETERS.fast_ema
SLOW_EMA = DEFAULT_STRATEGY_PARAMETERS.slow_ema
RSI_PERIOD = DEFAULT_STRATEGY_PARAMETERS.rsi_period
MACD_FAST = DEFAULT_STRATEGY_PARAMETERS.macd_fast
MACD_SLOW = DEFAULT_STRATEGY_PARAMETERS.macd_slow
MACD_SIGNAL = DEFAULT_STRATEGY_PARAMETERS.macd_signal
VOLUME_PERIOD = DEFAULT_STRATEGY_PARAMETERS.volume_period

# Data sufficiency: enough bars to warm up the slowest indicators
# (EMA 50, MACD 26/9) with room to spare.
MIN_CANDLES = DEFAULT_STRATEGY_PARAMETERS.min_candles

# Signal thresholds.
VOLUME_CONFIRM_RATIO = DEFAULT_STRATEGY_PARAMETERS.volume_confirm_ratio
VOLUME_CONFIRM_BOOST = DEFAULT_STRATEGY_PARAMETERS.volume_confirm_boost
CONFIDENCE_STRONG = DEFAULT_STRATEGY_PARAMETERS.confidence_strong
CONFIDENCE_MID = DEFAULT_STRATEGY_PARAMETERS.confidence_mid
CONFIDENCE_WAIT = DEFAULT_STRATEGY_PARAMETERS.confidence_wait


def _sma(values) -> float:
    """Simple moving average of a non-empty sequence of numbers."""
    return sum(values) / len(values)


def _ema_series(values, period: int) -> list:
    """Return an EMA series, defined from the first ``period`` bars on.

    The series is seeded with the SMA of the first ``period`` values,
    so each point uses only past data. Returns [] if there are not
    enough values.
    """
    n = len(values)
    if n < period or period <= 0:
        return []
    alpha = 2 / (period + 1)
    series = [_sma(values[:period])]
    for value in values[period:]:
        series.append((value - series[-1]) * alpha + series[-1])
    return series


def ema(values, period: int):
    """Return the latest EMA value, or None if there is insufficient data."""
    series = _ema_series(values, period)
    return series[-1] if series else None


def rsi(closes, period: int = RSI_PERIOD):
    """Return the latest Wilder's RSI value, or None if data is short.

    A perfectly flat series resolves to 50; a series with only gains
    resolves to 100 (no division by zero).
    """
    n = len(closes)
    if n < period + 1 or period <= 0:
        return None

    gains = []
    losses = []
    for i in range(1, n):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = _sma(gains[:period])
    avg_loss = _sma(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    closes,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
):
    """Return the latest MACD line, signal line, and histogram.

    Returns None if the data is too short to warm up both EMAs and the
    signal line.
    """
    if fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow:
        return None

    fast_series = _ema_series(closes, fast)
    slow_series = _ema_series(closes, slow)
    if not slow_series:
        return None

    # The fast EMA series is longer (defined earlier); align both to the
    # slow EMA's absolute start so macd = ema_fast - ema_slow per bar.
    offset = slow - fast
    macd_line = [
        fast_series[i + offset] - slow_value
        for i, slow_value in enumerate(slow_series)
    ]

    if len(macd_line) < signal:
        return None
    signal_series = _ema_series(macd_line, signal)
    if not signal_series:
        return None

    macd_last = macd_line[-1]
    signal_last = signal_series[-1]
    return {
        "macd_line": macd_last,
        "signal_line": signal_last,
        "histogram": macd_last - signal_last,
    }


def volume_ratio(volumes, period: int = VOLUME_PERIOD):
    """Return the latest volume divided by the average of the prior bars.

    Returns None if there are not enough volume bars or the prior
    average is zero. A value >= 1 means volume is expanding relative to
    recent activity.
    """
    n = len(volumes)
    if n < period + 1 or period <= 0:
        return None
    prior = volumes[-(period + 1):-1]
    avg_prior = _sma(prior)
    if avg_prior <= 0:
        return None
    return volumes[-1] / avg_prior


def _neutral_result():
    """Result used when there is not enough data to analyze."""
    return {
        "trend": "Unknown",
        "signal": "WAIT",
        "confidence": 0,
        "indicators": None,
    }


def compute_analysis(closes, volumes=None, *, parameters=None) -> dict:
    """Turn a series of closes (and optional volumes) into a verdict.

    Args:
        closes: Chronological list of closing prices.
        volumes: Optional chronological list of trading volumes, aligned
            with ``closes``.
        parameters: Optional immutable strategy configuration. ``None``
            uses the existing defaults exactly.

    Returns:
        A dict with:
          - trend: "Bullish" (close above the slow EMA) or "Bearish",
            or "Unknown" when data is insufficient.
          - signal: "BUY", "SELL", or "WAIT". BUY/SELL require the
            directional votes (EMA vs fast EMA, MACD histogram, RSI
            momentum) to agree with the trend.
          - confidence: 0-100. At least ``min_trade_confidence`` only for
            BUY/SELL with enough configured directional votes aligned.
          - indicators: A snapshot of the raw indicator values for
            transparency and debugging (or None if unusable).

    Guardian Mode purpose:
        Multiple confirmation signals must agree before a BUY/SELL is
        emitted. Votes are: price vs fast EMA, MACD histogram sign, and
        RSI momentum (RSI is neutral when overbought/oversold). Volume
        expansion raises confidence only, never a vote.

    Raises:
        TypeError: If ``parameters`` is not a StrategyParameters or None.
    """
    if parameters is None:
        parameters = DEFAULT_STRATEGY_PARAMETERS
    elif not isinstance(parameters, StrategyParameters):
        raise TypeError("parameters must be a StrategyParameters or None")

    if (
        not isinstance(closes, (list, tuple))
        or len(closes) < parameters.min_candles
    ):
        return _neutral_result()

    fast = ema(closes, parameters.fast_ema)
    slow = ema(closes, parameters.slow_ema)
    macd_result = macd(
        closes,
        fast=parameters.macd_fast,
        slow=parameters.macd_slow,
        signal=parameters.macd_signal,
    )
    rsi_value = rsi(closes, period=parameters.rsi_period)
    ratio = (
        volume_ratio(volumes, period=parameters.volume_period)
        if volumes
        else None
    )

    if fast is None or slow is None or macd_result is None:
        return _neutral_result()

    last_price = closes[-1]
    # Equality is treated as no edge: a flat market must never be
    # classified as trending down (or up) by a rounding artifact.
    trend = "Bullish" if last_price >= slow else "Bearish"

    # --- Directional votes ------------------------------------------
    vote_ema = 1 if last_price >= fast else -1

    vote_macd = 0
    if macd_result is not None:
        histogram = macd_result["histogram"]
        vote_macd = 1 if histogram > 0 else (-1 if histogram < 0 else 0)

    vote_rsi = 0
    if rsi_value is not None and not (
        rsi_value >= parameters.rsi_overbought
        or rsi_value <= parameters.rsi_oversold
    ):
        # The configured neutral value has no momentum edge.
        if rsi_value > parameters.rsi_neutral:
            vote_rsi = 1
        elif rsi_value < parameters.rsi_neutral:
            vote_rsi = -1

    net_votes = vote_ema + vote_macd + vote_rsi

    # --- Decision and confidence ------------------------------------
    signal = "WAIT"
    confidence = parameters.confidence_wait
    if (
        trend == "Bullish"
        and net_votes >= parameters.min_directional_votes
    ):
        signal = "BUY"
        confidence = (
            parameters.confidence_strong
            if net_votes >= 3
            else parameters.confidence_mid
        )
    elif (
        trend == "Bearish"
        and net_votes <= -parameters.min_directional_votes
    ):
        signal = "SELL"
        confidence = (
            parameters.confidence_strong
            if net_votes <= -3
            else parameters.confidence_mid
        )

    # Volume never votes, but it can confirm (or cast doubt on) a move.
    if signal != "WAIT" and ratio is not None:
        if ratio >= parameters.volume_confirm_ratio:
            confidence += parameters.volume_confirm_boost

    confidence = max(0, min(parameters.max_confidence, confidence))

    return {
        "trend": trend,
        "signal": signal,
        "confidence": confidence,
        "indicators": {
            "ema_fast": round(fast, 2) if fast is not None else None,
            "ema_slow": round(slow, 2) if slow is not None else None,
            "rsi": round(rsi_value, 2) if rsi_value is not None else None,
            "macd_histogram": (
                round(macd_result["histogram"], 4)
                if macd_result
                else None
            ),
            "volume_ratio": round(ratio, 2) if ratio is not None else None,
        },
    }
