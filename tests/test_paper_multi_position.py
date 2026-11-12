"""Regression tests for multiple open paper positions on one symbol.

Covers:
    - The background monitor marks, checks SL/TP, and trails each exact
      position independently (no first-position-by-symbol lookups).
    - One position closing or trailing never affects another position on
      the same symbol.
    - /close refuses to guess when several positions share a symbol, and
      closes exactly one position when an order id is supplied.
    - The original single-position /close behaviour still works.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

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
    stop_loss: float = 90.0,
    take_profit: float = 130.0,
) -> Trade:
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
    *,
    prices,
    state_path,
    alerts=None,
) -> PaperPositionMonitor:
    """Build a monitor using the real handler SL/TP and trailing helpers."""
    if callable(prices):
        provider = prices
    else:
        provider = lambda symbol: prices[symbol]
    return PaperPositionMonitor(
        service,
        alerts or _RecordingAlerts(),
        price_provider=provider,
        state_path=state_path,
    )


def _update() -> MagicMock:
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = 4242
    return update


def _run_close(monkeypatch, tmp_path, service, args) -> str:
    update = _update()
    context = MagicMock()
    context.args = args
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", tmp_path / "state.json")
    with patch("telegram_bot.handlers._get_paper_service", return_value=service):
        asyncio.run(handlers.close_command(update, context))
    return update.message.reply_text.await_args[0][0]


# ---------------------------------------------------------------------------
# Monitoring: marks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_btc_positions_are_marked_independently(tmp_path) -> None:
    service = _service_with(
        _trade(quantity=1.0, stop_loss=90.0, take_profit=130.0),
        _trade(quantity=2.0, stop_loss=80.0, take_profit=140.0),
    )
    first, second = service.open_positions()

    monitor = _monitor(service, prices={"BTC": 105.0}, state_path=tmp_path / "s.json")
    await monitor.monitor_open_positions()

    assert first is not second
    assert first.mark_price == pytest.approx(105.0)
    assert second.mark_price == pytest.approx(105.0)
    assert first.mark_at is not None
    assert second.mark_at is not None


@pytest.mark.asyncio
async def test_each_position_gets_its_own_symbol_price(tmp_path) -> None:
    service = _service_with(
        _trade(symbol="BTC", price=100.0, quantity=1.0),
        _trade(symbol="ETH", price=2000.0, quantity=1.0),
    )
    btc, eth = service.open_positions()

    monitor = _monitor(
        service,
        prices={"BTC": 105.0, "ETH": 2100.0},
        state_path=tmp_path / "s.json",
    )
    await monitor.monitor_open_positions()

    assert btc.mark_price == pytest.approx(105.0)
    assert eth.mark_price == pytest.approx(2100.0)


# ---------------------------------------------------------------------------
# Monitoring: stop-loss independence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_position_stop_loss_does_not_close_first(tmp_path) -> None:
    """The first position is safe; the second is stopped out.

    A first-position-by-symbol lookup would have closed the wrong one.
    """
    service = _service_with(
        _trade(quantity=1.0, stop_loss=90.0, take_profit=130.0),
        _trade(quantity=2.0, stop_loss=99.5, take_profit=130.0),
    )
    safe, stopped = service.open_positions()
    alerts = _RecordingAlerts()

    monitor = _monitor(
        service,
        prices={"BTC": 99.0},
        state_path=tmp_path / "s.json",
        alerts=alerts,
    )
    await monitor.monitor_open_positions()

    assert stopped.status == PaperOrderStatus.STOPPED
    assert stopped.exit_price == pytest.approx(99.5)
    assert safe.status == PaperOrderStatus.OPEN
    assert safe.exit_price is None
    assert safe.stop_loss == pytest.approx(90.0)
    assert service.open_positions() == [safe]
    assert alerts.events == [
        ("stop_loss", "BTC", stopped.order_id)
    ]


@pytest.mark.asyncio
async def test_second_position_take_profit_does_not_close_first(tmp_path) -> None:
    service = _service_with(
        _trade(quantity=1.0, stop_loss=90.0, take_profit=130.0),
        _trade(quantity=2.0, stop_loss=90.0, take_profit=104.0),
    )
    safe, taken = service.open_positions()

    monitor = _monitor(service, prices={"BTC": 105.0}, state_path=tmp_path / "s.json")
    await monitor.monitor_open_positions()

    assert taken.status == PaperOrderStatus.TAKEN_PROFIT
    assert taken.exit_price == pytest.approx(104.0)
    assert safe.status == PaperOrderStatus.OPEN
    assert safe.exit_price is None
    assert service.open_positions() == [safe]


# ---------------------------------------------------------------------------
# Monitoring: trailing-stop independence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trailing_stop_updates_only_the_intended_position(tmp_path) -> None:
    # First: entry 100, stop 90 → 1R = 10, price 115 activates trailing
    # and moves the stop to break-even (100).
    # Second: entry 100, stop 80 → 1R = 20, price 115 is below 120, so it
    # must not be touched.
    service = _service_with(
        _trade(quantity=1.0, stop_loss=90.0, take_profit=200.0),
        _trade(quantity=2.0, stop_loss=80.0, take_profit=200.0),
    )
    trailed, untouched = service.open_positions()
    alerts = _RecordingAlerts()

    monitor = _monitor(
        service,
        prices={"BTC": 115.0},
        state_path=tmp_path / "s.json",
        alerts=alerts,
    )
    await monitor.monitor_open_positions()

    assert trailed.stop_loss == pytest.approx(100.0)
    assert "trailing_active" in trailed.notes
    assert untouched.stop_loss == pytest.approx(80.0)
    assert "trailing_active" not in untouched.notes
    assert alerts.events == [("trailing", "BTC", 90.0, 100.0)]


@pytest.mark.asyncio
async def test_trailing_stop_trigger_closes_only_that_position(tmp_path) -> None:
    """One position's trailed stop is hit; the other (same symbol) remains."""
    service = _service_with(
        _trade(quantity=1.0, stop_loss=90.0, take_profit=200.0),
        _trade(quantity=2.0, stop_loss=80.0, take_profit=200.0),
    )
    trailed, untouched = service.open_positions()

    state = tmp_path / "s.json"
    # First cycle at 115 trails `trailed` to break-even and leaves the
    # other position alone.
    await _monitor(service, prices={"BTC": 115.0}, state_path=state).monitor_open_positions()
    assert trailed.stop_loss == pytest.approx(100.0)
    assert untouched.stop_loss == pytest.approx(80.0)

    # Second cycle at 100 hits the trailed position's stop.
    await _monitor(service, prices={"BTC": 100.0}, state_path=state).monitor_open_positions()

    assert trailed.status == PaperOrderStatus.STOPPED
    assert trailed.exit_price == pytest.approx(100.0)
    assert untouched.status == PaperOrderStatus.OPEN
    assert untouched.stop_loss == pytest.approx(80.0)
    assert service.open_positions() == [untouched]


# ---------------------------------------------------------------------------
# Manual close
# ---------------------------------------------------------------------------


def test_manual_close_without_id_closes_nothing_when_ambiguous(
    monkeypatch, tmp_path
) -> None:
    service = _service_with(
        _trade(quantity=1.0, stop_loss=90.0, take_profit=130.0),
        _trade(quantity=2.0, stop_loss=80.0, take_profit=140.0),
    )
    first, second = service.open_positions()

    message = _run_close(monkeypatch, tmp_path, service, ["BTC", "110"])

    assert "Multiple open positions for BTC" in message
    assert "nothing was closed" in message
    assert first.order_id in message
    assert second.order_id in message
    assert first.status == PaperOrderStatus.OPEN
    assert second.status == PaperOrderStatus.OPEN
    assert len(service.open_positions()) == 2
    assert service.summary().current_balance == pytest.approx(1000.0)


def test_manual_close_with_valid_id_closes_only_that_position(
    monkeypatch, tmp_path
) -> None:
    service = _service_with(
        _trade(quantity=1.0, stop_loss=90.0, take_profit=130.0),
        _trade(quantity=2.0, stop_loss=80.0, take_profit=140.0),
    )
    first, second = service.open_positions()

    message = _run_close(
        monkeypatch, tmp_path, service, ["BTC", "110", second.order_id]
    )

    assert "Position Closed" in message
    assert f"Order ID: {second.order_id}" in message
    assert second.status == PaperOrderStatus.CLOSED_MANUAL
    assert second.exit_price == pytest.approx(110.0)
    assert second.realized_pnl == pytest.approx(20.0)
    assert first.status == PaperOrderStatus.OPEN
    assert service.open_positions() == [first]
    assert service.summary().current_balance == pytest.approx(1020.0)


def test_manual_close_accepts_lowercase_order_id(monkeypatch, tmp_path) -> None:
    service = _service_with(_trade(quantity=1.0), _trade(quantity=2.0))
    first, second = service.open_positions()

    _run_close(
        monkeypatch, tmp_path, service, ["BTC", "110", second.order_id.lower()]
    )

    assert second.status == PaperOrderStatus.CLOSED_MANUAL
    assert first.status == PaperOrderStatus.OPEN


def test_manual_close_with_unknown_id_closes_nothing(monkeypatch, tmp_path) -> None:
    service = _service_with(_trade(quantity=1.0), _trade(quantity=2.0))
    first, second = service.open_positions()

    message = _run_close(
        monkeypatch, tmp_path, service, ["BTC", "110", "PAPE-NOT-A-REAL-ID"]
    )

    assert "nothing was closed" in message
    assert first.order_id in message
    assert second.order_id in message
    assert first.status == PaperOrderStatus.OPEN
    assert second.status == PaperOrderStatus.OPEN


def test_manual_close_with_id_from_other_symbol_closes_nothing(
    monkeypatch, tmp_path
) -> None:
    service = _service_with(
        _trade(symbol="BTC", price=100.0, quantity=1.0),
        _trade(symbol="ETH", price=2000.0, quantity=1.0),
    )
    btc, eth = service.open_positions()

    message = _run_close(monkeypatch, tmp_path, service, ["BTC", "110", eth.order_id])

    assert "nothing was closed" in message
    assert btc.status == PaperOrderStatus.OPEN
    assert eth.status == PaperOrderStatus.OPEN


def test_manual_close_single_position_without_id_still_works(
    monkeypatch, tmp_path
) -> None:
    service = _service_with(_trade(quantity=1.0))
    (only,) = service.open_positions()

    message = _run_close(monkeypatch, tmp_path, service, ["BTC", "110"])

    assert "Position Closed" in message
    assert only.status == PaperOrderStatus.CLOSED_MANUAL
    assert service.open_positions() == []
    assert service.summary().current_balance == pytest.approx(1010.0)


def test_manual_close_missing_position_reports_nothing(monkeypatch, tmp_path) -> None:
    service = _service_with(_trade(symbol="ETH", price=2000.0, quantity=1.0))

    message = _run_close(monkeypatch, tmp_path, service, ["BTC", "110"])

    assert "No open paper position found for BTC" in message
    assert len(service.open_positions()) == 1


# ---------------------------------------------------------------------------
# Exact-id close primitives
# ---------------------------------------------------------------------------


def test_close_position_by_id_rejects_unknown_id() -> None:
    service = _service_with(_trade(quantity=1.0), _trade(quantity=2.0))

    assert service.get_position_by_id("missing") is None
    assert (
        service.close_position_by_id(
            "missing", 110.0, PaperOrderStatus.CLOSED_MANUAL
        )
        is None
    )
    assert len(service.open_positions()) == 2


def test_close_position_by_id_closes_exact_position() -> None:
    service = _service_with(_trade(quantity=1.0), _trade(quantity=2.0))
    first, second = service.open_positions()

    closed = service.close_position_by_id(
        second.order_id, 110.0, PaperOrderStatus.CLOSED_MANUAL
    )

    assert closed is second
    assert second.status == PaperOrderStatus.CLOSED_MANUAL
    assert first.status == PaperOrderStatus.OPEN
