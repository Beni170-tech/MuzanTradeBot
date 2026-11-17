"""Offline tests for the historical data and backtest service."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from urllib.error import HTTPError, URLError

import pytest

import backtest.service as service
import backtest.trades as trades_module
import market.analyzer as analyzer
from backtest.costs import TradingCostModel
from backtest.replay import HistoricalBar, SignalRecord
from backtest.runner import BacktestRunConfig
from backtest.service import (
    HistoricalBacktestResult,
    HistoricalData,
    HistoricalDataError,
    HistoricalDataRateLimitError,
    HistoricalDataUnavailableError,
    load_historical_bars,
    run_historical_backtest,
)
from market.indicators import MIN_CANDLES
from risk.risk_profile import RiskProfile

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROFILE = RiskProfile(
    balance=1000.0,
    risk_percent=2.0,
    stop_loss_percent=2.0,
    take_profit_percent=4.0,
)
CONFIG = BacktestRunConfig("BTC", PROFILE)


def _bars(count, start_price=100.0):
    return [
        HistoricalBar(
            timestamp=START + timedelta(hours=index),
            close=start_price + index,
            volume=1_000.0,
        )
        for index in range(count)
    ]


def _flat_bars(count, price=100.0):
    return [
        HistoricalBar(
            timestamp=START + timedelta(hours=index),
            close=price,
            volume=1_000.0,
        )
        for index in range(count)
    ]


def _record(bar, decision):
    if decision == "BUY":
        signal = "BUY"
        trend = "Bullish"
    elif decision == "SELL":
        signal = "SELL"
        trend = "Bearish"
    else:
        signal = "WAIT"
        trend = "Bullish"
    return SignalRecord(
        timestamp=bar.timestamp,
        signal=signal,
        trend=trend,
        confidence=80,
        decision=decision,
    )


def _patch_loader(monkeypatch, bars):
    calls = []

    def fake_fetch(symbol):
        calls.append(symbol)
        return bars

    monkeypatch.setattr(service, "fetch_historical_bars", fake_fetch)
    return calls


def _patch_replay(monkeypatch, bars):
    records = [_record(bars[0], "BUY")]
    records.extend(_record(bar, "WAIT") for bar in bars[1:])
    monkeypatch.setattr(
        trades_module,
        "replay_signals",
        lambda received_bars, *, parameters=None: records,
    )


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._payload


def _millis(hours):
    return int((START + timedelta(hours=hours)).timestamp() * 1000)


def _payload(prices, volumes):
    return {"prices": prices, "total_volumes": volumes}


def test_service_loads_bars_and_executes_real_backtest(monkeypatch):
    bars = _bars(MIN_CANDLES + 1)
    calls = _patch_loader(monkeypatch, bars)
    _patch_replay(monkeypatch, bars)

    result = run_historical_backtest(
        "btc",
        CONFIG,
        timeframe=" 1H ",
        period=" 30D ",
    )

    assert isinstance(result, HistoricalBacktestResult)
    assert calls == ["BTC"]
    assert result.data.provider == "coingecko"
    assert result.data.symbol == "BTC"
    assert result.data.timeframe == "1h"
    assert result.data.period == service.SUPPORTED_PERIOD
    assert result.data.bar_count == len(bars)
    assert result.data.first_timestamp == START
    assert result.data.last_timestamp == bars[-1].timestamp
    assert result.backtest.report.total_trades == 1
    assert result.backtest.completed_trades == 1
    assert result.backtest.config is CONFIG
    assert result.backtest.history_sufficient


def test_load_historical_bars_returns_validated_dataset(monkeypatch):
    bars = _bars(3)
    calls = _patch_loader(monkeypatch, bars)

    data = load_historical_bars(
        "btc",
        timeframe=" 1H ",
        period=" 30D ",
    )

    assert isinstance(data, HistoricalData)
    assert calls == ["BTC"]
    assert data.bars == tuple(bars)
    assert isinstance(data.bars, tuple)
    assert data.metadata.provider == "coingecko"
    assert data.metadata.symbol == "BTC"
    assert data.metadata.timeframe == "1h"
    assert data.metadata.period == service.SUPPORTED_PERIOD
    assert data.metadata.bar_count == 3
    assert data.metadata.first_timestamp == START
    assert data.metadata.last_timestamp == bars[-1].timestamp


def test_run_historical_backtest_delegates_to_public_loader(monkeypatch):
    bars = _bars(3)
    _patch_loader(monkeypatch, bars)
    calls = []
    original_load = service.load_historical_bars

    def spy_load(symbol, *, timeframe, period):
        calls.append((symbol, timeframe, period))
        return original_load(
            symbol,
            timeframe=timeframe,
            period=period,
        )

    monkeypatch.setattr(service, "load_historical_bars", spy_load)

    result = run_historical_backtest(
        "btc",
        CONFIG,
        timeframe="1H",
        period="30D",
    )

    assert calls == [("BTC", "1h", "30d")]
    assert result.data.bar_count == 3
    assert result.data.symbol == "BTC"


def test_existing_provider_converts_response_into_historical_bars(monkeypatch):
    captured = []
    payload = _payload(
        [[_millis(0), 100.0], [_millis(1), 101.0], [_millis(2), 102.0]],
        [[_millis(0), 10.0], [_millis(1), 11.0], [_millis(2), 12.0]],
    )
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return _FakeResponse(payload)

    original_fetch = analyzer.fetch_historical_bars

    def capture_bars(symbol):
        bars = original_fetch(symbol)
        captured.extend(bars)
        return bars

    monkeypatch.setattr(analyzer.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(service, "fetch_historical_bars", capture_bars)

    result = run_historical_backtest("BTC", CONFIG)

    assert len(calls) == 1
    assert "/bitcoin/market_chart" in calls[0][0]
    assert f"days={analyzer.MARKET_CHART_DAYS}" in calls[0][0]
    assert calls[0][1] == 10
    assert captured == [
        HistoricalBar(START, 100.0, 10.0),
        HistoricalBar(START + timedelta(hours=1), 101.0, 11.0),
        HistoricalBar(START + timedelta(hours=2), 102.0, 12.0),
    ]
    assert all(isinstance(bar, HistoricalBar) for bar in captured)
    assert result.data.bar_count == 3
    assert result.data.first_timestamp == START
    assert result.data.last_timestamp == START + timedelta(hours=2)
    assert not result.backtest.history_sufficient
    assert result.backtest.report.total_trades == 0


def test_zero_trade_service_run_is_safe(monkeypatch):
    _patch_loader(monkeypatch, _flat_bars(120))

    result = run_historical_backtest("BTC", CONFIG)

    assert result.backtest.report.total_trades == 0
    assert result.backtest.report.ending_equity == PROFILE.balance
    assert result.backtest.history_sufficient
    assert any(
        "No trades were generated" in warning
        for warning in result.backtest.report.warnings
    )


def test_insufficient_history_is_returned_safely(monkeypatch):
    bars = _bars(MIN_CANDLES - 1)
    _patch_loader(monkeypatch, bars)

    result = run_historical_backtest("BTC", CONFIG)

    assert result.data.bar_count == MIN_CANDLES - 1
    assert not result.backtest.history_sufficient
    assert result.backtest.report.total_trades == 0
    assert result.backtest.report.ending_equity == PROFILE.balance
    assert result.backtest.trades == ()


def test_empty_provider_data_is_rejected(monkeypatch):
    calls = _patch_loader(monkeypatch, [])

    with pytest.raises(HistoricalDataError, match="no bars"):
        run_historical_backtest("BTC", CONFIG)

    assert calls == ["BTC"]


@pytest.mark.parametrize(
    "bars",
    [
        None,
        {},
        "not-bars",
        [object()],
        [HistoricalBar(START, 0.0, 1_000.0)],
        [HistoricalBar(START, -1.0, 1_000.0)],
        [HistoricalBar(START, float("nan"), 1_000.0)],
        [HistoricalBar(START, 100.0, -1.0)],
        [HistoricalBar(START, 100.0, float("inf"))],
        [HistoricalBar("2024-01-01", 100.0, 1_000.0)],
    ],
)
def test_malformed_provider_bars_are_rejected(monkeypatch, bars):
    _patch_loader(monkeypatch, bars)

    with pytest.raises(HistoricalDataError):
        run_historical_backtest("BTC", CONFIG)


def test_duplicate_timestamps_are_rejected(monkeypatch):
    bar = HistoricalBar(START, 100.0, 1_000.0)
    _patch_loader(monkeypatch, [bar, replace(bar, close=101.0)])

    with pytest.raises(HistoricalDataError, match="strictly increasing"):
        run_historical_backtest("BTC", CONFIG)


def test_out_of_order_timestamps_are_rejected(monkeypatch):
    bars = _bars(2)
    _patch_loader(monkeypatch, list(reversed(bars)))

    with pytest.raises(HistoricalDataError, match="strictly increasing"):
        run_historical_backtest("BTC", CONFIG)


@pytest.mark.parametrize(
    ("exception", "error_type"),
    [
        (URLError("provider down"), HistoricalDataUnavailableError),
        (TimeoutError("request timed out"), HistoricalDataUnavailableError),
        (ValueError("malformed response"), HistoricalDataError),
        (RuntimeError("provider failed"), HistoricalDataError),
    ],
)
def test_provider_failures_are_wrapped(monkeypatch, exception, error_type):
    def fail(symbol):
        raise exception

    monkeypatch.setattr(service, "fetch_historical_bars", fail)

    with pytest.raises(error_type):
        run_historical_backtest("BTC", CONFIG)


def test_rate_limit_response_is_mapped(monkeypatch):
    def rate_limited(symbol):
        raise HTTPError(
            "https://api.coingecko.com/test",
            429,
            "Too Many Requests",
            None,
            None,
        )

    monkeypatch.setattr(service, "fetch_historical_bars", rate_limited)

    with pytest.raises(HistoricalDataRateLimitError, match="rate limit"):
        run_historical_backtest("BTC", CONFIG)


def test_other_http_failure_is_mapped(monkeypatch):
    def unavailable(symbol):
        raise HTTPError(
            "https://api.coingecko.com/test",
            503,
            "Service Unavailable",
            None,
            None,
        )

    monkeypatch.setattr(service, "fetch_historical_bars", unavailable)

    with pytest.raises(HistoricalDataUnavailableError, match="HTTP 503"):
        run_historical_backtest("BTC", CONFIG)


@pytest.mark.parametrize(
    ("symbol", "timeframe", "period", "error"),
    [
        ("XRP", "1h", "30d", ValueError),
        ("", "1h", "30d", ValueError),
        (None, "1h", "30d", TypeError),
        ("BTC", "4h", "30d", ValueError),
        ("BTC", "60m", "30d", ValueError),
        ("BTC", None, "30d", TypeError),
        ("BTC", "1h", "7d", ValueError),
        ("BTC", "1h", "90d", ValueError),
        ("BTC", "1h", None, TypeError),
    ],
)
def test_unsupported_or_invalid_parameters_are_rejected_without_fetch(
    monkeypatch,
    symbol,
    timeframe,
    period,
    error,
):
    def must_not_fetch(received_symbol):
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(service, "fetch_historical_bars", must_not_fetch)

    with pytest.raises(error):
        run_historical_backtest(
            symbol,
            CONFIG,
            timeframe=timeframe,
            period=period,
        )


def test_config_symbol_must_match_requested_symbol(monkeypatch):
    _patch_loader(monkeypatch, _bars(3))

    with pytest.raises(ValueError, match="config.symbol"):
        run_historical_backtest("ETH", CONFIG)


def test_invalid_config_type_is_rejected(monkeypatch):
    _patch_loader(monkeypatch, _bars(3))

    with pytest.raises(TypeError, match="config"):
        run_historical_backtest("BTC", object())


def test_risk_profile_and_cost_model_propagate(monkeypatch):
    bars = _bars(MIN_CANDLES + 1)
    _patch_loader(monkeypatch, bars)
    _patch_replay(monkeypatch, bars)
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)
    profile = RiskProfile(
        balance=5_000.0,
        risk_percent=1.0,
        stop_loss_percent=3.0,
        take_profit_percent=6.0,
    )
    config = BacktestRunConfig("BTC", profile, model)
    captured = {}
    original_run = service.run_backtest

    def spy_run(received_bars, received_config):
        captured["bars"] = received_bars
        captured["config"] = received_config
        return original_run(received_bars, received_config)

    monkeypatch.setattr(service, "run_backtest", spy_run)

    result = run_historical_backtest("BTC", config)

    assert captured["config"] is config
    assert captured["bars"] == tuple(bars)
    assert result.backtest.config is config
    assert result.backtest.report.cost_model is model
    assert result.backtest.report.total_costs > 0.0
    assert result.backtest.trades[0].quantity > 0.0


def test_identical_data_produces_deterministic_results(monkeypatch):
    bars = _bars(MIN_CANDLES + 1)
    _patch_loader(monkeypatch, bars)
    _patch_replay(monkeypatch, bars)
    model = TradingCostModel(fee_rate=0.001)
    config = BacktestRunConfig("BTC", PROFILE, model)

    first = run_historical_backtest("BTC", config)
    second = run_historical_backtest("BTC", config)

    assert first == second
    assert first.data == second.data
    assert first.backtest.report.trades == second.backtest.report.trades
    assert first.backtest.trades == second.backtest.trades
