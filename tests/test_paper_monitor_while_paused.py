"""Tests that the background monitor keeps protecting paper positions while
global paper trading is paused.

A pause only blocks *new* entries. The background monitor is pause-agnostic by
design, so stop loss and take profit must keep closing existing positions for
as long as the operator has paper trading paused.

These tests use the real ``PaperTradingService``, ``PaperBroker``,
``PaperAccount`` and ``PaperOrder`` objects plus the monitor's default SL/TP
and trailing-stop functions. Only the market-price provider and the alert sink
are substituted, so no network access is required. The pause control and paper
state are real files under ``tmp_path``.
"""

from __future__ import annotations

import pytest

from trader.paper.monitor import PaperPositionMonitor
from trader.paper.order import PaperOrderStatus
from trader.paper.pause import read_pause_state, request_pause
from trader.paper.service import PaperTradingService
from trader.trade import Trade

PAUSE_REASON = "maintenance window"


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
    stop_loss: float = 90.0,
    take_profit: float = 130.0,
    quantity: float = 1.0,
) -> Trade:
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
    """Point the handler layer's paper state file at a temporary path."""
    path = tmp_path / "paper_state.json"
    monkeypatch.setattr("telegram_bot.handlers.PAPER_STATE_PATH", path)
    return path


@pytest.fixture
def paused_control(tmp_path, monkeypatch):
    """Pause paper trading through the real control file under ``tmp_path``."""
    path = tmp_path / "paper_pause.json"
    monkeypatch.setattr("telegram_bot.handlers.PAUSE_STATE_PATH", path)
    state = request_pause(PAUSE_REASON, path)
    assert state is not None and state.paused is True
    assert read_pause_state(path).paused is True
    return path


def _service_with(*trades: Trade):
    service = PaperTradingService.create_new_account(1000.0)
    orders = [service.place_trade(trade) for trade in trades]
    assert all(order is not None for order in orders)
    return service, orders


def _monitor(service, alerts, prices: dict[str, float]):
    """Build a monitor with deterministic prices per symbol."""
    return PaperPositionMonitor(
        service,
        alerts,
        price_provider=lambda symbol: prices[symbol],
    )


# ---------------------------------------------------------------------------
# Stop loss while paused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_loss_closes_position_while_paused(
    paper_state, paused_control
):
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=130.0)
    )
    alerts = _RecordingAlerts()

    assert read_pause_state(paused_control).paused is True

    await _monitor(service, alerts, {"BTC": 89.5}).monitor_open_positions()

    assert order.status == PaperOrderStatus.STOPPED
    assert order.exit_price == 90.0
    assert order.realized_pnl == pytest.approx(-10.0)
    assert service.summary().current_balance == pytest.approx(990.0)
    assert service.open_positions() == []
    assert alerts.events == [("stop_loss", "BTC", "STOPPED")]


# ---------------------------------------------------------------------------
# Take profit while paused
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_take_profit_closes_position_while_paused(
    paper_state, paused_control
):
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=120.0)
    )
    alerts = _RecordingAlerts()

    assert read_pause_state(paused_control).paused is True

    await _monitor(service, alerts, {"BTC": 120.0}).monitor_open_positions()

    assert order.status == PaperOrderStatus.TAKEN_PROFIT
    assert order.exit_price == 120.0
    assert order.realized_pnl == pytest.approx(20.0)
    assert service.summary().current_balance == pytest.approx(1020.0)
    assert service.open_positions() == []
    assert alerts.events == [("take_profit", "BTC", "TAKEN_PROFIT")]


# ---------------------------------------------------------------------------
# Only the breached position is closed, and nothing new is opened
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_closes_only_the_breached_position_while_paused(
    paper_state, paused_control, monkeypatch
):
    service, (btc, eth) = _service_with(
        _make_trade(symbol="BTC", stop_loss=90.0, take_profit=130.0),
        _make_trade(symbol="ETH", stop_loss=80.0, take_profit=150.0),
    )
    alerts = _RecordingAlerts()

    def _forbidden(*args, **kwargs):
        raise AssertionError("the monitor must never open a new position")

    monkeypatch.setattr(service, "place_trade", _forbidden)

    # BTC breaches its stop, ETH stays well inside its levels.
    await _monitor(
        service, alerts, {"BTC": 89.5, "ETH": 105.0}
    ).monitor_open_positions()

    assert btc.status == PaperOrderStatus.STOPPED
    assert btc.exit_price == 90.0
    assert eth.status == PaperOrderStatus.OPEN
    assert eth.exit_price is None
    assert eth.stop_loss == 80.0
    assert [position.order_id for position in service.open_positions()] == [
        eth.order_id
    ]
    assert len(service.closed_positions()) == 1
    assert alerts.events == [("stop_loss", "BTC", "STOPPED")]


# ---------------------------------------------------------------------------
# The pause itself is never modified by monitoring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_state_is_unchanged_across_monitoring_cycles(
    paper_state, paused_control
):
    service, (order,) = _service_with(
        _make_trade(stop_loss=90.0, take_profit=130.0)
    )
    alerts = _RecordingAlerts()
    monitor = _monitor(service, alerts, {"BTC": 89.5})

    before_bytes = paused_control.read_bytes()
    before_state = read_pause_state(paused_control)
    assert before_state.paused is True
    assert before_state.reason == PAUSE_REASON

    await monitor.monitor_open_positions()
    # A second cycle runs with the position already closed.
    remaining = await monitor.monitor_open_positions()

    assert order.status == PaperOrderStatus.STOPPED
    assert remaining == 0

    after_state = read_pause_state(paused_control)
    assert after_state.paused is True
    assert after_state.reason == PAUSE_REASON
    assert after_state.degraded is False
    assert paused_control.read_bytes() == before_bytes
