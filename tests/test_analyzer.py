"""Tests for market.analyzer.analyze_coin (network replaced by fixtures)."""

import pytest

import market.analyzer as analyzer


def _uptrend(n=250, growth=0.001, start=100.0):
    """Compounding rise that produces a genuine BUY setup."""
    values = [start]
    for _ in range(1, n):
        values.append(values[-1] * (1 + growth))
    return values


def test_supported_coin_returns_full_schema(monkeypatch):
    closes = _uptrend()
    volumes = [1_000_000.0] * len(closes)

    def fake_fetch(symbol):
        return closes, volumes

    monkeypatch.setattr(analyzer, "_fetch_market_chart", fake_fetch)

    result = analyzer.analyze_coin("BTC")
    for key in ("symbol", "price", "trend", "signal", "confidence"):
        assert key in result
    assert result["symbol"] == "BTC"
    assert result["price"] == pytest.approx(closes[-1])
    assert result["trend"] == "Bullish"
    assert result["signal"] == "BUY"
    assert 0 <= result["confidence"] <= 100


def test_symbol_is_case_insensitive(monkeypatch):
    closes = _uptrend()

    def fake_fetch(symbol):
        return closes, [1_000_000.0] * len(closes)

    monkeypatch.setattr(analyzer, "_fetch_market_chart", fake_fetch)
    assert analyzer.analyze_coin("eth")["symbol"] == "ETH"


def test_fetch_failure_returns_placeholder(monkeypatch):
    def fake_fetch(symbol):
        raise RuntimeError("CoinGecko unavailable")

    monkeypatch.setattr(analyzer, "_fetch_market_chart", fake_fetch)

    result = analyzer.analyze_coin("BTC")
    assert result["price"] == "Unknown"
    assert result["trend"] == "Unknown"
    assert result["signal"] == "WAIT"
    assert result["confidence"] == 0


def test_insufficient_candles_returns_placeholder(monkeypatch):
    def fake_fetch(symbol):
        return _uptrend(n=10), [1_000_000.0] * 10

    monkeypatch.setattr(analyzer, "_fetch_market_chart", fake_fetch)

    result = analyzer.analyze_coin("BTC")
    assert result["price"] == "Unknown"
    assert result["trend"] == "Unknown"


def test_unsupported_coin_returns_placeholder_without_fetching(monkeypatch):
    def fake_fetch(symbol):
        raise AssertionError("fetch must not be called for unsupported coins")

    monkeypatch.setattr(analyzer, "_fetch_market_chart", fake_fetch)

    result = analyzer.analyze_coin("XRP")
    assert result["symbol"] == "XRP"
    assert result["price"] == "Unknown"
    assert result["signal"] == "WAIT"
