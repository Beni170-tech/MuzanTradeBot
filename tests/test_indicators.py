"""Tests for market.indicators (pure, offline)."""

import pytest

from market.indicators import (
    compute_analysis,
    ema,
    macd,
    rsi,
    volume_ratio,
)
from strategy.parameters import DEFAULT_STRATEGY_PARAMETERS, StrategyParameters


def _uptrend(n=250, growth=0.001, start=100.0):
    """Compounding rise (keeps MACD momentum positive in the tail)."""
    values = [start]
    for _ in range(1, n):
        values.append(values[-1] * (1 + growth))
    return values


def _downtrend(n=220, curvature=0.0015, start=100.0):
    """Accelerating decline (keeps MACD momentum negative in the tail)."""
    return [start - curvature * i * i for i in range(n)]


# --- ema ------------------------------------------------------------


def test_ema_of_constant_series_is_constant():
    closes = [5.0] * 40
    assert ema(closes, 10) == pytest.approx(5.0)


def test_ema_lags_a_rising_series():
    closes = _uptrend()
    value = ema(closes, 20)
    assert closes[-1] > value > closes[0]


def test_ema_returns_none_when_data_is_short():
    assert ema([1.0, 2.0, 3.0], 20) is None


# --- rsi ------------------------------------------------------------


def test_rsi_of_constant_series_is_50():
    assert rsi([100.0] * 30) == pytest.approx(50.0)


def test_rsi_of_strict_uptrend_is_100():
    assert rsi(_uptrend()) == pytest.approx(100.0)


def test_rsi_of_strict_downtrend_is_0():
    assert rsi(_downtrend()) == pytest.approx(0.0)


def test_rsi_is_between_0_and_100_for_mixed_series():
    closes = [100 + (10 if i % 2 == 0 else -8) * (1 + i / 100) for i in range(80)]
    value = rsi(closes)
    assert 0 <= value <= 100


def test_rsi_returns_none_when_data_is_short():
    assert rsi([1.0, 2.0, 3.0]) is None


# --- macd -----------------------------------------------------------


def test_macd_returns_components_with_histogram_identity():
    result = macd(_uptrend())
    assert set(result) == {"macd_line", "signal_line", "histogram"}
    assert result["histogram"] == pytest.approx(
        result["macd_line"] - result["signal_line"]
    )


def test_macd_histogram_positive_in_uptrend():
    assert macd(_uptrend())["histogram"] > 0


def test_macd_histogram_negative_in_downtrend():
    assert macd(_downtrend())["histogram"] < 0


def test_macd_returns_none_when_data_is_short():
    assert macd([100.0] * 25) is None


# --- volume_ratio ---------------------------------------------------


def test_volume_ratio_compares_last_bar_to_prior_average():
    volumes = [10.0] * 20 + [30.0]
    assert volume_ratio(volumes, period=20) == pytest.approx(3.0)


def test_volume_ratio_is_one_when_flat():
    volumes = [10.0] * 25
    assert volume_ratio(volumes, period=20) == pytest.approx(1.0)


def test_volume_ratio_returns_none_when_short():
    assert volume_ratio([10.0] * 5) is None


# --- compute_analysis ------------------------------------------------


def test_insufficient_data_is_neutral():
    result = compute_analysis(_uptrend(n=40))
    assert result["trend"] == "Unknown"
    assert result["signal"] == "WAIT"
    assert result["confidence"] == 0


def test_steady_uptrend_yields_buy():
    result = compute_analysis(_uptrend())
    assert result["trend"] == "Bullish"
    assert result["signal"] == "BUY"
    assert result["confidence"] >= 70


def test_steady_downtrend_yields_sell():
    result = compute_analysis(_downtrend())
    assert result["trend"] == "Bearish"
    assert result["signal"] == "SELL"
    assert result["confidence"] >= 70


def test_flat_market_waits():
    result = compute_analysis([100.0] * 200)
    assert result["signal"] == "WAIT"
    assert result["confidence"] < 70


def test_volume_confirmation_raises_confidence():
    closes = _uptrend()
    volumes = [1_000_000.0] * (len(closes) - 1) + [2_000_000.0]  # ratio 2.0 >= 1.5
    boosted = compute_analysis(closes, volumes)
    plain = compute_analysis(closes)
    assert boosted["signal"] == "BUY"
    assert boosted["confidence"] == plain["confidence"] + 6


def test_confidence_stays_within_bounds():
    closes = _uptrend()
    volumes = [1_000_000.0] * (len(closes) - 1) + [5_000_000.0]  # extreme volume
    result = compute_analysis(closes, volumes)
    assert 0 <= result["confidence"] <= 100


def test_result_includes_indicator_snapshot():
    result = compute_analysis(_uptrend())
    assert result["indicators"] is not None
    for key in ("ema_fast", "ema_slow", "rsi", "macd_histogram", "volume_ratio"):
        assert key in result["indicators"]


def test_default_parameters_match_omitted_behavior():
    closes = _uptrend()
    volumes = [1_000.0] * (len(closes) - 1) + [2_000.0]

    omitted = compute_analysis(closes, volumes)
    explicit = compute_analysis(
        closes,
        volumes,
        parameters=DEFAULT_STRATEGY_PARAMETERS,
    )

    assert explicit == omitted
    assert explicit["confidence"] == omitted["confidence"]


def test_custom_indicator_parameters_change_snapshot():
    custom = StrategyParameters(
        fast_ema=5,
        slow_ema=10,
        rsi_period=5,
        macd_fast=5,
        macd_slow=10,
        macd_signal=3,
        volume_period=5,
        min_candles=12,
    )

    default = compute_analysis(_uptrend(n=80))
    changed = compute_analysis(_uptrend(n=20), parameters=custom)

    assert changed["indicators"]["ema_fast"] != default["indicators"]["ema_fast"]
    assert changed["indicators"]["ema_slow"] != default["indicators"]["ema_slow"]


def test_custom_min_candles_controls_data_sufficiency():
    custom = StrategyParameters(
        fast_ema=5,
        slow_ema=10,
        rsi_period=5,
        macd_fast=5,
        macd_slow=10,
        macd_signal=3,
        volume_period=5,
        min_candles=20,
    )

    result = compute_analysis(_uptrend(n=19), parameters=custom)

    assert result["trend"] == "Unknown"
    assert result["signal"] == "WAIT"
    assert result["confidence"] == 0


def test_custom_confidence_cap_limits_volume_boost():
    custom = StrategyParameters(
        confidence_mid=90,
        confidence_strong=90,
        max_confidence=90,
        volume_confirm_boost=10,
    )
    closes = _uptrend()
    volumes = [1_000.0] * (len(closes) - 1) + [3_000.0]

    result = compute_analysis(closes, volumes, parameters=custom)

    assert result["signal"] == "BUY"
    assert result["confidence"] == 90


def test_invalid_parameters_type_is_rejected():
    with pytest.raises(TypeError, match="parameters"):
        compute_analysis(_uptrend(), parameters=object())
