"""Regression tests for the monitor's state-path consistency (D2).

The background monitor accepts an injected ``state_path``. Every paper
account write it causes must go there: recorded market marks,
stop-loss / take-profit closes, and trailing-stop changes.

Before this fix only the market-mark save honoured the injected path;
the exit helpers resolved the path registered by
``telegram_bot.handlers.bind_state_path`` instead. A monitor pointed at a
test file could therefore still save the real paper account.

Each test binds the handler layer to a *different* temporary file, so a
save leaking through the handler binding fails an assertion instead of
reaching ``logs/``. That binding is also what the monitor must fall back
to when no custom path is supplied.
"""

from __future__ import annotations

import pytest

import telegram_bot.handlers as handlers
from trader.paper.monitor import PaperPositionMonitor
from trader.paper.order import PaperOrderStatus
from trader.paper.service import PaperTradingService
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trade(
    symbol: str = "BTC",
    price: float = 100.0,
    quantity: float = 1.0,
    side: str = "LONG",
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> Trade:
    is_short = side.strip().upper() == "SHORT"
    if stop_loss is None:
        stop_loss = price * 1.10 if is_short else price * 0.90
    if take_profit is None:
        take_profit = price * 0.90 if is_short else price * 1.30
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish" if side == "LONG" else "Bearish",
        signal="BUY" if side == "LONG" else "SELL",
        confidence=80,
        decision="BUY" if side == "LONG" else "SELL",
        risk_verdict="APPROVED",
        position_size=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        side=side,
        approved=True,
        risk_reasons=[],
        timestamp="2026-01-01 00:00:00",
    )


def _service_with(*trades: Trade) -> PaperTradingService:
    service = PaperTradingService.create_new_account(1000.0)
    for trade in trades:
        assert service.place_trade(trade) is not None
    return service


class _RecordingAlerts:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def stop_loss_triggered(self, order) -> None:
        self.events.append(("stop_loss", order.symbol, order.order_id))

    async def take_profit_triggered(self, order) -> None:
        self.events.append(("take_profit", order.symbol, order.order_id))

    async def trailing_stop_updated(self, symbol, old_stop, new_stop) -> None:
        self.events.append(("trailing", symbol, old_stop, new_stop))


def _monitor(
    service: PaperTradingService,
    price: float,
    state_path=None,
    alerts=None,
) -> PaperPositionMonitor:
    """Build a monitor over the real lifecycle helpers (no overrides)."""
    kwargs = {} if state_path is None else {"state_path": state_path}
    return PaperPositionMonitor(
        service,
        alerts or _RecordingAlerts(),
        price_provider=lambda symbol: price,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Market-mark persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_market_mark_is_saved_to_the_monitor_state_path(
    tmp_path, monkeypatch
) -> None:
    service = _service_with(_trade(stop_loss=90.0, take_profit=130.0))
    handler_path = tmp_path / "handler.json"
    monitor_path = tmp_path / "monitor.json"
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", handler_path)

    # 105 is between the levels and below 1R, so only the mark is written.
    await _monitor(service, 105.0, monitor_path).monitor_open_positions()

    reloaded = PaperTradingService.load_account(monitor_path)
    assert reloaded.get_position("BTC").mark_price == pytest.approx(105.0)
    assert not handler_path.exists()


# ---------------------------------------------------------------------------
# 2. Stop-loss / take-profit saves
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stop_loss_close_is_saved_to_the_monitor_state_path(
    tmp_path, monkeypatch
) -> None:
    service = _service_with(_trade(stop_loss=99.5, take_profit=130.0))
    (position,) = service.open_positions()
    handler_path = tmp_path / "handler.json"
    monitor_path = tmp_path / "monitor.json"
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", handler_path)

    await _monitor(service, 99.0, monitor_path).monitor_open_positions()

    assert position.status == PaperOrderStatus.STOPPED
    assert "STOPPED" in monitor_path.read_text(encoding="utf-8")
    assert not handler_path.exists()

    reloaded = PaperTradingService.load_account(monitor_path)
    assert reloaded.open_positions() == []
    assert len(reloaded.closed_positions()) == 1


@pytest.mark.asyncio
async def test_take_profit_close_is_saved_to_the_monitor_state_path(
    tmp_path, monkeypatch
) -> None:
    service = _service_with(_trade(stop_loss=90.0, take_profit=104.0))
    (position,) = service.open_positions()
    handler_path = tmp_path / "handler.json"
    monitor_path = tmp_path / "monitor.json"
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", handler_path)

    await _monitor(service, 105.0, monitor_path).monitor_open_positions()

    assert position.status == PaperOrderStatus.TAKEN_PROFIT
    assert "TAKEN_PROFIT" in monitor_path.read_text(encoding="utf-8")
    assert not handler_path.exists()


# ---------------------------------------------------------------------------
# 3. Trailing-stop saves
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trailing_stop_change_is_saved_to_the_monitor_state_path(
    tmp_path, monkeypatch
) -> None:
    # Entry 100, stop 90 → 1R = 10; 115 clears the threshold and trails
    # the stop to break-even (100).
    service = _service_with(_trade(stop_loss=90.0, take_profit=200.0))
    (position,) = service.open_positions()
    handler_path = tmp_path / "handler.json"
    monitor_path = tmp_path / "monitor.json"
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", handler_path)

    alerts = _RecordingAlerts()
    await _monitor(service, 115.0, monitor_path, alerts).monitor_open_positions()

    assert position.stop_loss == pytest.approx(100.0)
    assert alerts.events == [("trailing", "BTC", 90.0, 100.0)]
    assert not handler_path.exists()

    reloaded = PaperTradingService.load_account(monitor_path)
    saved = reloaded.get_position("BTC")
    assert saved.stop_loss == pytest.approx(100.0)
    assert "trailing_active" in saved.notes


# ---------------------------------------------------------------------------
# Preserved behaviour: no custom path still uses the shared lifecycle path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_injected_path_still_uses_the_shared_lifecycle_path(
    tmp_path, monkeypatch
) -> None:
    """Without a custom path the handler binding still decides the file."""
    service = _service_with(_trade(stop_loss=99.5, take_profit=130.0))
    (position,) = service.open_positions()
    handler_path = tmp_path / "handler.json"
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", handler_path)

    monitor = _monitor(service, 99.0)
    assert monitor._state_path is None

    await monitor.monitor_open_positions()

    assert position.status == PaperOrderStatus.STOPPED
    assert handler_path.exists()
    assert "STOPPED" in handler_path.read_text(encoding="utf-8")
