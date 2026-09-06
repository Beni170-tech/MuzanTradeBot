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

# Lookback / sensitivity constants.
FAST_EMA = 20
SLOW_EMA = 50
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_PERIOD = 20

# Data sufficiency: enough bars to warm up the slowest indicators
# (EMA 50, MACD 26/9) with room to spare.
MIN_CANDLES = 60

# Signal thresholds.
VOLUME_CONFIRM_RATIO = 1.5  # Current volume >= this multiple of average confirms a move.
VOLUME_CONFIRM_BOOST = 6    # Added to confidence when volume confirms.
CONFIDENCE_STRONG = 88      # All available votes agree with the trend.
CONFIDENCE_MID = 74         # Two of three votes agree with the trend.
CONFIDENCE_WAIT = 45        # Neutral value for WAIT (always below the trade threshold).


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


def macd(closes, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
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


def compute_analysis(closes, volumes=None) -> dict:
    """Turn a series of closes (and optional volumes) into a verdict.

    Args:
        closes: Chronological list of closing prices.
        volumes: Optional chronological list of trading volumes, aligned
            with ``closes``.

    Returns:
        A dict with:
          - trend: "Bullish" (close above the slow EMA) or "Bearish",
            or "Unknown" when data is insufficient.
          - signal: "BUY", "SELL", or "WAIT". BUY/SELL require the
            directional votes (EMA vs fast EMA, MACD histogram, RSI
            momentum) to agree with the trend.
          - confidence: 0-100. >= 70 only for BUY/SELL with at least
            two votes aligned.
          - indicators: A snapshot of the raw indicator values for
            transparency and debugging (or None if unusable).

    Guardian Mode purpose:
        Multiple confirmation signals must agree before a BUY/SELL is
        emitted. Votes are: price vs fast EMA, MACD histogram sign, and
        RSI momentum (RSI is neutral when overbought/oversold). Volume
        expansion raises confidence only, never a vote.
    """
    if not isinstance(closes, (list, tuple)) or len(closes) < MIN_CANDLES:
        return _neutral_result()

    fast = ema(closes, FAST_EMA)
    slow = ema(closes, SLOW_EMA)
    macd_result = macd(closes)
    rsi_value = rsi(closes)
    ratio = volume_ratio(volumes) if volumes else None

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
    if rsi_value is not None and not (rsi_value >= 70 or rsi_value <= 30):
        # 50 is exactly neutral; only real momentum (either way) votes.
        vote_rsi = 1 if rsi_value > 50 else (-1 if rsi_value < 50 else 0)

    net_votes = vote_ema + vote_macd + vote_rsi

    # --- Decision and confidence ------------------------------------
    signal = "WAIT"
    confidence = CONFIDENCE_WAIT
    if trend == "Bullish" and net_votes >= 2:
        signal = "BUY"
        confidence = CONFIDENCE_STRONG if net_votes >= 3 else CONFIDENCE_MID
    elif trend == "Bearish" and net_votes <= -2:
        signal = "SELL"
        confidence = CONFIDENCE_STRONG if net_votes <= -3 else CONFIDENCE_MID

    # Volume never votes, but it can confirm (or cast doubt on) a move.
    if signal != "WAIT" and ratio is not None:
        if ratio >= VOLUME_CONFIRM_RATIO:
            confidence += VOLUME_CONFIRM_BOOST

    confidence = max(0, min(95, confidence))

    return {
        "trend": trend,
        "signal": signal,
        "confidence": confidence,
        "indicators": {
            "ema_fast": round(fast, 2) if fast is not None else None,
            "ema_slow": round(slow, 2) if slow is not None else None,
            "rsi": round(rsi_value, 2) if rsi_value is not None else None,
            "macd_histogram": round(macd_result["histogram"], 4) if macd_result else None,
            "volume_ratio": round(ratio, 2) if ratio is not None else None,
        },
    }
