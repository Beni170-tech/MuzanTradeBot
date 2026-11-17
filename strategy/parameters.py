"""Validated strategy parameters for the market and decision pipeline.

The defaults here exactly match the constants and literals previously
used by :mod:`market.indicators` and :mod:`brain.decision_engine`. The
module is intentionally dependency-free so both layers can share one
immutable configuration without a circular import.
"""

from dataclasses import dataclass
import math


_INTEGER_FIELDS = (
    "fast_ema",
    "slow_ema",
    "rsi_period",
    "macd_fast",
    "macd_slow",
    "macd_signal",
    "volume_period",
    "min_candles",
    "min_directional_votes",
)

_NUMERIC_FIELDS = (
    "rsi_overbought",
    "rsi_oversold",
    "rsi_neutral",
    "volume_confirm_ratio",
    "volume_confirm_boost",
    "confidence_strong",
    "confidence_mid",
    "confidence_wait",
    "max_confidence",
    "min_trade_confidence",
)


def _finite_number(value, name: str) -> float:
    """Return ``value`` as a finite number or raise a controlled error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not finite:
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class StrategyParameters:
    """Immutable technical and decision parameters for one strategy run.

    Every field is a strategy input, not a backtest-only setting. The
    default instance reproduces the existing behavior exactly.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a field is non-finite, out of range, or violates a
            required logical relationship.
    """

    fast_ema: int = 20
    slow_ema: int = 50
    rsi_period: int = 14
    rsi_overbought: int | float = 70
    rsi_oversold: int | float = 30
    rsi_neutral: int | float = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_period: int = 20
    volume_confirm_ratio: int | float = 1.5
    volume_confirm_boost: int | float = 6
    confidence_strong: int | float = 88
    confidence_mid: int | float = 74
    confidence_wait: int | float = 45
    max_confidence: int | float = 95
    min_trade_confidence: int | float = 70
    min_candles: int = 60
    min_directional_votes: int = 2

    def __post_init__(self) -> None:
        for name in _INTEGER_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        for name in _NUMERIC_FIELDS:
            _finite_number(getattr(self, name), name)

        if self.fast_ema >= self.slow_ema:
            raise ValueError("fast_ema must be less than slow_ema")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be less than macd_slow")

        if not (
            0
            < self.rsi_oversold
            < self.rsi_neutral
            < self.rsi_overbought
            <= 100
        ):
            raise ValueError(
                "RSI thresholds must satisfy 0 < rsi_oversold < "
                "rsi_neutral < rsi_overbought <= 100"
            )

        if self.volume_confirm_ratio < 0:
            raise ValueError("volume_confirm_ratio must be non-negative")
        if self.volume_confirm_boost < 0:
            raise ValueError("volume_confirm_boost must be non-negative")

        confidence_values = (
            self.confidence_wait,
            self.confidence_mid,
            self.confidence_strong,
            self.max_confidence,
        )
        if any(value < 0 or value > 100 for value in confidence_values):
            raise ValueError("confidence values must be in the range [0, 100]")
        if not (
            self.confidence_wait
            <= self.confidence_mid
            <= self.confidence_strong
            <= self.max_confidence
        ):
            raise ValueError(
                "confidence thresholds must satisfy confidence_wait <= "
                "confidence_mid <= confidence_strong <= max_confidence"
            )
        if not 0 <= self.min_trade_confidence <= 100:
            raise ValueError(
                "min_trade_confidence must be in the range [0, 100]"
            )

        if not 1 <= self.min_directional_votes <= 3:
            raise ValueError(
                "min_directional_votes must be in the range [1, 3]"
            )

        required_warmup = max(
            self.slow_ema,
            self.macd_slow + self.macd_signal - 1,
            self.rsi_period + 1,
            self.volume_period + 1,
        )
        if self.min_candles < required_warmup:
            raise ValueError(
                "min_candles must be at least the configured indicator "
                f"warmup requirement ({required_warmup})"
            )


DEFAULT_STRATEGY_PARAMETERS = StrategyParameters()

__all__ = [
    "DEFAULT_STRATEGY_PARAMETERS",
    "StrategyParameters",
]
