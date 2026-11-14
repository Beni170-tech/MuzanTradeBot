"""Tests for market-price-based paper position monitoring (Task 1).

These tests use real PaperTradingService, PaperBroker, PaperAccount and
PaperOrder objects plus the real handler SL/TP and trailing-stop
functions. Only the market-price provider and the alert sink are
substituted, so no network access is required.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest

from market import analyzer
from market.analyzer import get_current_price
from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.monitor import PaperPositionMonitor
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.paper.service import PaperTradingService
from trader.trade import Trade


class _RecordingAlerts:
    """Minimal async alert sink that records monitor-driven alerts."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def stop_loss_triggered(self, order) -> None:
        self.events.append(("stop_loss", order.symbol, order.status.value))

    async def take_profit_triggered(self, order) -> None:
        self.events.append(("take_profit", order.symbol, order.status.value))

    async def trailing_stop_updated(self, symbol, old_stop, new_stop) -> None:
        self.events.append(("trailing", symbol, old_stop, new_stop))


def _make_trade(
    symbol: str = "BTC",
    price: float = 100.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    quantity: float = 1.0,
) -> Trade:
    if stop_loss is None:
        stop_loss = price * 0.90
    if take_profit is None:
        take_profit = price * 1.20
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        side="LONG",
        approved=True,
    )


@pytest.fixture
def paper_state(tmp_path, monkeypatch):
    """Point the handler layer's state file at a temporary path."""
    path = tmp_path / "paper_state.json"
    monkeypatch.setattr("telegram_bot.handlers.PAPER_STATE_PATH", path)
    return path


def _service_with(*trades: Trade):
    service = PaperTradingService.create_new_account(1000.0)
    orders = [service.place_trade(trade) for trade in trades]
    assert all(order is not None for order in orders)
    return service, orders


def _service_with_legacy_long(
    price: float = 100.0,
    stop_loss: float = 101.0,
    take_profit: float = 130.0,
):
    """Insert a position directly to exercise the monitor's price guard.

    The wrong-sided stop makes an entry-price substitution visible. The
    broker now rejects such a trade at placement, so this legacy position
    is inserted directly into the account for these monitor-only tests.
    """
    account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
    order = PaperOrder(
        order_id="LEGACY-1",
        symbol="BTC",
        side=PaperOrderSide.LONG,
        quantity=1.0,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        status=PaperOrderStatus.OPEN,
    )
    account.open_position(order)
    return PaperTradingService(account, PaperBroker(account)), (order,)


def _monitor(service, price_provider) -> PaperPositionMonitor:
    return PaperPositionMonitor(
        service, _RecordingAlerts(), price_provider=price_provider
    )


# ---------------------------------------------------------------------------
# Price pass-through to the real handler functions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fresh_price_is_passed_to_sltp_and_trailing(paper_state):
    """The monitor must hand a fresh market price to both helpers."""
    from telegram_bot.handlers import (
        _apply_trailing_stop_to_position,
        _paper_position_check,
    )

    service, (order,) = _service_with(_make_trade())
    seen: dict[str, float] = {}

    def sltp(svc, position, current_price):
        seen["sltp"] = current_price
        return _paper_position_check(svc, position, current_price)

    def trailing(svc, position, current_price):
        seen["trailing"] = current_price
        return _apply_trailing_stop_to_position(svc, position, current_price)

    monitor = PaperPositionMonitor(
        service,
        _RecordingAlerts(),
        sltp_check=sltp,
        trailing_fn=trailing,
        price_provider=lambda symbol: 105.0,
    )
    await monitor.monitor_open_positions()

    assert seen["sltp"] == 105.0
    assert seen["trailing"] == 105.0
    assert order.status == PaperOrderStatus.OPEN


# ---------------------------------------------------------------------------
# Real stop-loss / take-profit / no-action behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_price_below_stop_loss_triggers_stopped_exit(paper_state):
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=120.0)
    )
    alerts = _RecordingAlerts()
    monitor = PaperPositionMonitor(
        service, alerts, price_provider=lambda symbol: 89.5
    )

    await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.STOPPED
    assert order.exit_price == 90.0
    assert order.realized_pnl == pytest.approx(-10.0)
    assert service.summary().current_balance == pytest.approx(990.0)
    assert alerts.events == [("stop_loss", "BTC", "STOPPED")]


@pytest.mark.asyncio
async def test_price_reaching_take_profit_triggers_taken_profit_exit(
    paper_state,
):
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=120.0)
    )
    alerts = _RecordingAlerts()
    monitor = PaperPositionMonitor(
        service, alerts, price_provider=lambda symbol: 120.0
    )

    await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.TAKEN_PROFIT
    assert order.exit_price == 120.0
    assert order.realized_pnl == pytest.approx(20.0)
    assert service.summary().current_balance == pytest.approx(1020.0)
    assert alerts.events == [("take_profit", "BTC", "TAKEN_PROFIT")]


@pytest.mark.asyncio
async def test_price_between_levels_does_not_close_position(paper_state):
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=130.0)
    )
    alerts = _RecordingAlerts()
    monitor = PaperPositionMonitor(
        service, alerts, price_provider=lambda symbol: 105.0
    )

    await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.OPEN
    assert order.stop_loss == 90.0
    assert alerts.events == []


@pytest.mark.asyncio
async def test_trailing_stop_activates_at_one_r(paper_state):
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=130.0)
    )
    alerts = _RecordingAlerts()
    monitor = PaperPositionMonitor(
        service, alerts, price_provider=lambda symbol: 111.0
    )

    await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.OPEN
    assert order.stop_loss == 100.0
    assert alerts.events == [("trailing", "BTC", 90.0, 100.0)]


# ---------------------------------------------------------------------------
# Price retrieval failures must never fall back to the entry price
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_price_request_does_not_close_position(
    paper_state, caplog
):
    """A wrong-sided stop would trigger if entry_price were substituted."""
    service, (order,) = _service_with_legacy_long()
    monitor = _monitor(service, lambda symbol: None)

    with caplog.at_level(logging.WARNING, logger="trader.paper.monitor"):
        await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.OPEN
    assert order.exit_price is None
    assert "no reliable market price" in caplog.text


@pytest.mark.parametrize(
    "bad_price",
    [0, -5.0, float("nan"), float("inf"), "100", True, None],
)
@pytest.mark.asyncio
async def test_invalid_or_missing_price_does_not_use_entry_price(
    paper_state, bad_price
):
    service, (order,) = _service_with_legacy_long()
    monitor = _monitor(service, lambda symbol: bad_price)

    await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.OPEN
    assert order.stop_loss == 101.0


@pytest.mark.asyncio
async def test_one_failed_position_does_not_block_other_positions(
    paper_state, caplog
):
    service, (btc_order, eth_order) = _service_with(
        _make_trade("BTC", stop_loss=90.0, take_profit=120.0),
        _make_trade("ETH", stop_loss=90.0, take_profit=120.0),
    )

    def provider(symbol):
        if symbol == "BTC":
            raise RuntimeError("price provider down")
        return 89.0

    monitor = _monitor(service, provider)

    with caplog.at_level(logging.WARNING, logger="trader.paper.monitor"):
        await monitor.monitor_open_positions()

    assert btc_order.status == PaperOrderStatus.OPEN
    assert eth_order.status == PaperOrderStatus.STOPPED
    assert "Market price lookup failed for BTC" in caplog.text


@pytest.mark.asyncio
async def test_unsupported_symbol_is_skipped_without_network(
    paper_state, monkeypatch, caplog
):
    service, (order,) = _service_with(
        _make_trade("DOGE", stop_loss=90.0, take_profit=120.0)
    )

    def no_network(*args, **kwargs):
        raise AssertionError("network access must not be attempted")

    monkeypatch.setattr(analyzer.urllib.request, "urlopen", no_network)
    monitor = PaperPositionMonitor(service, _RecordingAlerts())

    with caplog.at_level(logging.WARNING, logger="trader.paper.monitor"):
        await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.OPEN
    assert "no reliable market price" in caplog.text


# ---------------------------------------------------------------------------
# Current-price provider (reuses the existing CoinGecko integration)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_get_current_price_uses_simple_price_endpoint(monkeypatch):
    calls: dict[str, object] = {}

    def fake_urlopen(url, timeout=None):
        calls["url"] = url
        calls["timeout"] = timeout
        return _FakeResponse({"bitcoin": {"usd": 61234.5}})

    monkeypatch.setattr(analyzer.urllib.request, "urlopen", fake_urlopen)

    assert get_current_price("btc") == 61234.5
    assert "ids=bitcoin" in str(calls["url"])
    assert "vs_currencies=usd" in str(calls["url"])
    assert calls["timeout"] == 10


def test_get_current_price_returns_none_on_request_failure(monkeypatch):
    def boom(url, timeout=None):
        raise OSError("offline")

    monkeypatch.setattr(analyzer.urllib.request, "urlopen", boom)

    assert get_current_price("BTC") is None


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"bitcoin": {}},
        {"bitcoin": {"usd": "n/a"}},
        {"bitcoin": {"usd": 0}},
        {"bitcoin": {"usd": -1}},
    ],
)
def test_get_current_price_rejects_unusable_payloads(monkeypatch, payload):
    monkeypatch.setattr(
        analyzer.urllib.request,
        "urlopen",
        lambda url, timeout=None: _FakeResponse(payload),
    )

    assert get_current_price("BTC") is None


def test_get_current_price_returns_none_for_unsupported_symbol(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network access must not be attempted")

    monkeypatch.setattr(analyzer.urllib.request, "urlopen", no_network)

    assert get_current_price("DOGE") is None


# ---------------------------------------------------------------------------
# The synchronous lookup must not block the event loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_price_lookup_does_not_block_event_loop(paper_state):
    """A slow synchronous provider must run off the event-loop thread."""
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=120.0)
    )

    def slow_provider(symbol):
        time.sleep(0.2)
        return None

    monitor = _monitor(service, slow_provider)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker_task = asyncio.create_task(ticker())
    monitor_task = asyncio.create_task(monitor.monitor_open_positions())

    # The provider is sleeping for 0.2s. If the event loop were blocked,
    # this coroutine could not resume until the lookup returned and the
    # ticker would have made no progress.
    await asyncio.sleep(0.05)
    assert ticks >= 1

    await monitor_task
    ticker_task.cancel()
    try:
        await ticker_task
    except asyncio.CancelledError:
        pass

    assert order.status == PaperOrderStatus.OPEN
