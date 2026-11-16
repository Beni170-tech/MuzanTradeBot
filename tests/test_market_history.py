"""Offline tests for timestamp-preserving CoinGecko historical bars."""

from datetime import datetime, timedelta, timezone
import json

import pytest

import market.analyzer as analyzer
from backtest.replay import HistoricalBar

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


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


def _install_response(monkeypatch, payload):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        return _FakeResponse(payload)

    monkeypatch.setattr(analyzer.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_historical_timestamps_are_preserved(monkeypatch):
    payload = _payload(
        [[_millis(0), 100.0], [_millis(1), 101.0], [_millis(2), 102.0]],
        [[_millis(0), 10.0], [_millis(1), 11.0], [_millis(2), 12.0]],
    )
    _install_response(monkeypatch, payload)

    bars = analyzer.fetch_historical_bars("btc")

    assert bars == [
        HistoricalBar(START, 100.0, 10.0),
        HistoricalBar(START + timedelta(hours=1), 101.0, 11.0),
        HistoricalBar(START + timedelta(hours=2), 102.0, 12.0),
    ]
    assert all(bar.timestamp.tzinfo is timezone.utc for bar in bars)


def test_price_and_volume_are_aligned_by_timestamp(monkeypatch):
    payload = _payload(
        [[_millis(0), 100.0], [_millis(1), 101.0], [_millis(2), 102.0]],
        [
            [_millis(0), 10.0],
            [_millis(1), 11.0],
            [_millis(2), 12.0],
            [_millis(3), 13.0],  # extra volume-only observation
        ],
    )
    _install_response(monkeypatch, payload)

    bars = analyzer.fetch_historical_bars("BTC")

    assert [bar.close for bar in bars] == [100.0, 101.0, 102.0]
    assert [bar.volume for bar in bars] == [10.0, 11.0, 12.0]


def test_compatibility_chart_helper_uses_timestamp_order(monkeypatch):
    payload = _payload(
        [[_millis(0), 100.0], [_millis(1), 101.0]],
        [[_millis(0), 10.0], [_millis(1), 11.0]],
    )
    _install_response(monkeypatch, payload)

    closes, volumes = analyzer._fetch_market_chart("ETH")

    assert closes == [100.0, 101.0]
    assert volumes == [10.0, 11.0]


def test_supported_coin_request_uses_existing_endpoint(monkeypatch):
    payload = _payload(
        [[_millis(0), 100.0]],
        [[_millis(0), 10.0]],
    )
    calls = _install_response(monkeypatch, payload)

    analyzer.fetch_historical_bars("SOL")

    assert len(calls) == 1
    assert "/solana/market_chart" in calls[0][0]
    assert calls[0][1] == 10


@pytest.mark.parametrize("symbol", ["XRP", "", "   ", None, 42])
def test_unsupported_or_invalid_symbol_is_rejected(monkeypatch, symbol):
    _install_response(monkeypatch, {})
    with pytest.raises(ValueError):
        analyzer.fetch_historical_bars(symbol)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"prices": [[_millis(0), 100.0]]},
        {"total_volumes": [[_millis(0), 10.0]]},
        {"prices": [], "total_volumes": []},
        {"prices": [[_millis(0), 100.0]], "total_volumes": []},
        {"prices": [], "total_volumes": [[_millis(0), 10.0]]},
    ],
)
def test_missing_or_empty_market_data_is_rejected(monkeypatch, payload):
    _install_response(monkeypatch, payload)
    with pytest.raises(ValueError):
        analyzer.fetch_historical_bars("BTC")


@pytest.mark.parametrize(
    "point",
    [
        [],
        [_millis(0)],
        [_millis(0), 100.0, "extra"],
        ["bad-timestamp", 100.0],
        [True, 100.0],
        [float("nan"), 100.0],
        [float("inf"), 100.0],
        [-1, 100.0],
        [_millis(0), "100"],
        [_millis(0), True],
        [_millis(0), float("nan")],
        [_millis(0), float("inf")],
        [_millis(0), 0],
        [_millis(0), -1],
    ],
)
def test_invalid_price_observation_is_rejected(monkeypatch, point):
    _install_response(
        monkeypatch,
        _payload([point], [[_millis(0), 10.0]]),
    )
    with pytest.raises(ValueError):
        analyzer.fetch_historical_bars("BTC")


@pytest.mark.parametrize(
    "point",
    [
        [_millis(0)],
        [_millis(0), "10"],
        [_millis(0), True],
        [_millis(0), float("nan")],
        [_millis(0), float("inf")],
        [_millis(0), -1],
    ],
)
def test_invalid_volume_observation_is_rejected(monkeypatch, point):
    _install_response(
        monkeypatch,
        _payload([[_millis(0), 100.0]], [point]),
    )
    with pytest.raises(ValueError):
        analyzer.fetch_historical_bars("BTC")


@pytest.mark.parametrize(
    ("prices", "volumes"),
    [
        (
            [[_millis(0), 100.0], [_millis(0), 101.0]],
            [[_millis(0), 10.0], [_millis(0), 11.0]],
        ),
        (
            [[_millis(1), 100.0], [_millis(0), 101.0]],
            [[_millis(0), 10.0], [_millis(1), 11.0]],
        ),
        (
            [[_millis(0), 100.0]],
            [[_millis(0), 10.0], [_millis(0), 11.0]],
        ),
        (
            [[_millis(0), 100.0]],
            [[_millis(1), 10.0], [_millis(0), 11.0]],
        ),
    ],
)
def test_duplicate_or_unordered_observations_are_rejected(
    monkeypatch, prices, volumes
):
    _install_response(monkeypatch, _payload(prices, volumes))
    with pytest.raises(ValueError, match="strictly increasing"):
        analyzer.fetch_historical_bars("BTC")


def test_missing_matching_volume_timestamp_is_rejected(monkeypatch):
    payload = _payload(
        [[_millis(0), 100.0], [_millis(1), 101.0]],
        [[_millis(0), 10.0], [_millis(2), 12.0]],
    )
    _install_response(monkeypatch, payload)

    with pytest.raises(ValueError, match="align by timestamp"):
        analyzer.fetch_historical_bars("BTC")


def test_malformed_json_shape_is_rejected(monkeypatch):
    class BadResponse(_FakeResponse):
        def __init__(self):
            self._payload = b'"not-an-object"'

    monkeypatch.setattr(
        analyzer.urllib.request,
        "urlopen",
        lambda url, timeout: BadResponse(),
    )

    with pytest.raises(ValueError, match="must be an object"):
        analyzer.fetch_historical_bars("BTC")
