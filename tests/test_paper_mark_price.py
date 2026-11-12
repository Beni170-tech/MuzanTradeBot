"""Tests for recorded paper market marks (Task 3, Part 2).

Covers:
    - PaperOrder.mark_to_market validation and LONG/SHORT unrealized PnL.
    - Missing marks contribute zero; the entry price is never substituted.
    - PaperAccount equity and PaperBroker open PnL/equity use marks.
    - Mark persistence through the JSON storage layer.
    - The background monitor records the validated price for the exact
      position being monitored and persists it.
    - The analysis flow records an already-fetched price without any
      additional market request.
    - Telegram summaries/position displays label marked values and call
      out missing marks instead of showing a misleading zero.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import telegram_bot.handlers as handlers
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.paper.monitor import PaperPositionMonitor
from trader.paper.service import PaperTradingService
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trade(
    symbol: str = "BTC",
    price: float = 100.0,
    quantity: float = 10.0,
    side: str = "LONG",
    stop_loss: float = 95.0,
    take_profit: float = 120.0,
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


def _service(*trades: Trade) -> PaperTradingService:
    service = PaperTradingService.create_new_account(1000.0)
    for trade in trades:
        order = service.place_trade(trade)
        assert order is not None
    return service


def _order(
    side: PaperOrderSide = PaperOrderSide.LONG,
    entry_price: float = 100.0,
    quantity: float = 2.0,
) -> PaperOrder:
    return PaperOrder(
        order_id="ORD-1",
        symbol="BTC",
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=90.0,
        take_profit=120.0,
        status=PaperOrderStatus.OPEN,
    )


class _RecordingAlerts:
    """Minimal async alert sink; monitor alerts are not under test here."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def stop_loss_triggered(self, order) -> None:
        self.events.append(("stop_loss", order.symbol))

    async def take_profit_triggered(self, order) -> None:
        self.events.append(("take_profit", order.symbol))

    async def trailing_stop_updated(self, symbol, old_stop, new_stop) -> None:
        self.events.append(("trailing", symbol, old_stop, new_stop))


def _update() -> MagicMock:
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


# ---------------------------------------------------------------------------
# PaperOrder mark mechanism
# ---------------------------------------------------------------------------


def test_mark_to_market_long_profit_and_loss() -> None:
    order = _order(PaperOrderSide.LONG, entry_price=100.0, quantity=2.0)

    assert order.mark_price is None
    assert order.mark_to_market(110.0) is True
    assert order.mark_price == 110.0
    assert order.unrealized_pnl == pytest.approx(20.0)

    assert order.mark_to_market(90.0) is True
    assert order.mark_price == 90.0
    assert order.unrealized_pnl == pytest.approx(-20.0)


def test_mark_to_market_short_profit_and_loss() -> None:
    order = _order(PaperOrderSide.SHORT, entry_price=100.0, quantity=2.0)

    assert order.mark_to_market(90.0) is True
    assert order.unrealized_pnl == pytest.approx(20.0)

    assert order.mark_to_market(110.0) is True
    assert order.unrealized_pnl == pytest.approx(-20.0)


@pytest.mark.parametrize(
    "bad_price",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan"), "100.0", None, True],
)
def test_mark_to_market_rejects_unusable_prices(bad_price) -> None:
    order = _order()

    assert order.mark_to_market(bad_price) is False
    assert order.mark_price is None
    assert order.unrealized_pnl == 0.0


def test_bad_mark_does_not_overwrite_last_good_mark() -> None:
    order = _order()
    assert order.mark_to_market(110.0) is True

    assert order.mark_to_market(float("nan")) is False
    assert order.mark_price == 110.0
    assert order.unrealized_pnl == pytest.approx(20.0)


def test_unrealized_pnl_for_rejects_unusable_prices() -> None:
    order = _order()
    assert order.unrealized_pnl_for(0.0) == 0.0
    assert order.unrealized_pnl_for(None) == 0.0
    assert order.unrealized_pnl_for(float("nan")) == 0.0


# ---------------------------------------------------------------------------
# Account equity / broker open PnL
# ---------------------------------------------------------------------------


def test_account_equity_uses_recorded_marks() -> None:
    service = _service(_trade(quantity=10.0))
    order = service.open_positions()[0]
    assert order.mark_to_market(110.0) is True

    account = service._account
    assert account.current_balance == 1000.0
    assert account.equity() == pytest.approx(1100.0)


def test_account_equity_excludes_missing_marks() -> None:
    service = _service(
        _trade(symbol="BTC", price=100.0, quantity=10.0),
        _trade(symbol="ETH", price=2000.0, quantity=1.0),
    )
    btc, eth = service.open_positions()
    assert btc.mark_to_market(110.0) is True
    assert eth.mark_price is None

    # Only the marked BTC position contributes: +100. ETH is excluded even
    # though a mark of 2000 would be its entry price.
    assert service._account.equity() == pytest.approx(1100.0)
    assert service.execution_backend().get_open_pnl() == pytest.approx(100.0)


def test_broker_open_pnl_uses_marks_for_long_and_short() -> None:
    service = _service(
        _trade(symbol="BTC", price=100.0, quantity=10.0, side="LONG"),
        _trade(
            symbol="ETH",
            price=2000.0,
            quantity=1.0,
            side="SHORT",
            stop_loss=2100.0,
            take_profit=1800.0,
        ),
    )
    btc, eth = service.open_positions()
    btc.mark_to_market(110.0)  # +100
    eth.mark_to_market(1900.0)  # +100 short

    broker = service.execution_backend()
    assert broker.get_open_pnl() == pytest.approx(200.0)
    assert broker.get_equity() == pytest.approx(1200.0)


def test_missing_mark_never_falls_back_to_entry_price() -> None:
    service = _service(_trade(quantity=10.0))
    broker = service.execution_backend()

    assert service.open_positions()[0].mark_price is None
    assert broker.get_open_pnl() == 0.0
    assert broker.get_equity() == pytest.approx(1000.0)


def test_summary_reports_marked_unrealized_pnl() -> None:
    service = _service(_trade(quantity=10.0))
    service.open_positions()[0].mark_to_market(105.0)

    summary = service.summary()
    assert summary.open_pnl == pytest.approx(50.0)
    assert summary.equity == pytest.approx(1050.0)
    assert summary.current_balance == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_mark_price_round_trips_through_storage(tmp_path) -> None:
    path = tmp_path / "paper_state.json"
    service = _service(
        _trade(symbol="BTC", price=100.0, quantity=10.0),
        _trade(symbol="ETH", price=2000.0, quantity=1.0),
    )
    btc, eth = service.open_positions()
    btc.mark_to_market(110.0)
    service.save_account(path)

    loaded = PaperTradingService.load_account(path)
    loaded_btc, loaded_eth = loaded.open_positions()

    assert loaded_btc.mark_price == pytest.approx(110.0)
    assert loaded_btc.unrealized_pnl == pytest.approx(100.0)
    assert loaded_eth.mark_price is None
    assert loaded.summary().open_pnl == pytest.approx(100.0)
    assert loaded.summary().equity == pytest.approx(1100.0)


def test_unmarked_position_round_trips_as_none(tmp_path) -> None:
    path = tmp_path / "paper_state.json"
    service = _service(_trade(quantity=10.0))
    service.save_account(path)

    loaded = PaperTradingService.load_account(path)
    assert loaded.open_positions()[0].mark_price is None
    assert loaded.execution_backend().get_open_pnl() == 0.0


# ---------------------------------------------------------------------------
# Background monitor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_records_mark_for_the_exact_position(tmp_path) -> None:
    service = _service(
        _trade(symbol="BTC", price=100.0, quantity=10.0, stop_loss=95.0, take_profit=120.0),
        _trade(symbol="ETH", price=2000.0, quantity=1.0, stop_loss=1900.0, take_profit=2400.0),
    )
    state_path = tmp_path / "paper_state.json"
    prices = {"BTC": 110.0, "ETH": 2100.0}

    monitor = PaperPositionMonitor(
        service,
        _RecordingAlerts(),
        sltp_check=lambda svc, symbol, price: None,
        trailing_fn=lambda svc, symbol, price: None,
        price_provider=lambda symbol: prices[symbol],
        state_path=state_path,
    )
    await monitor.monitor_open_positions()

    btc = service.get_position("BTC")
    eth = service.get_position("ETH")
    assert btc.mark_price == pytest.approx(110.0)
    assert eth.mark_price == pytest.approx(2100.0)
    # BTC +100, ETH +100
    assert service.execution_backend().get_open_pnl() == pytest.approx(200.0)

    assert state_path.exists()
    loaded = PaperTradingService.load_account(state_path)
    assert loaded.get_position("BTC").mark_price == pytest.approx(110.0)
    assert loaded.get_position("ETH").mark_price == pytest.approx(2100.0)


@pytest.mark.asyncio
async def test_monitor_does_not_record_mark_when_price_is_unavailable(tmp_path) -> None:
    service = _service(_trade(quantity=10.0))
    monitor = PaperPositionMonitor(
        service,
        _RecordingAlerts(),
        sltp_check=lambda svc, symbol, price: None,
        trailing_fn=lambda svc, symbol, price: None,
        price_provider=lambda symbol: None,
        state_path=tmp_path / "paper_state.json",
    )
    await monitor.monitor_open_positions()

    assert service.get_position("BTC").mark_price is None
    assert service.execution_backend().get_open_pnl() == 0.0


@pytest.mark.asyncio
async def test_monitor_bad_price_does_not_overwrite_good_mark(tmp_path) -> None:
    service = _service(_trade(quantity=10.0))
    prices = {"BTC": 105.0}

    monitor = PaperPositionMonitor(
        service,
        _RecordingAlerts(),
        sltp_check=lambda svc, symbol, price: None,
        trailing_fn=lambda svc, symbol, price: None,
        price_provider=lambda symbol: prices[symbol],
        state_path=tmp_path / "paper_state.json",
    )
    await monitor.monitor_open_positions()
    assert service.get_position("BTC").mark_price == pytest.approx(105.0)

    prices["BTC"] = float("nan")
    await monitor.monitor_open_positions()
    assert service.get_position("BTC").mark_price == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Analysis flow
# ---------------------------------------------------------------------------


def test_analysis_flow_records_fetched_price_without_market_call(monkeypatch) -> None:
    service = _service(_trade(quantity=10.0))

    def _fail(symbol):
        raise AssertionError("analysis flow must not fetch a price for marking")

    monkeypatch.setattr("market.analyzer.get_current_price", _fail)

    result = handlers._before_analysis_paper_check(service, "BTC", 105.0)

    assert result is None
    assert service.get_position("BTC").mark_price == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# Telegram display
# ---------------------------------------------------------------------------


def _run_command(command, service) -> str:
    update = _update()
    with patch("telegram_bot.handlers._get_paper_service", return_value=service):
        asyncio.run(command(update, MagicMock()))
    return update.message.reply_text.await_args[0][0]


def _run_paper_command(service) -> str:
    return _run_command(handlers.paper_command, service)


def _run_positions_command(service) -> str:
    return _run_command(handlers.positions_command, service)


def _run_stats_command(service) -> str:
    return _run_command(handlers.stats_command, service)


def test_paper_command_labels_marked_equity() -> None:
    service = _service(_trade(quantity=10.0))
    service.open_positions()[0].mark_to_market(110.0)

    msg = _run_paper_command(service)

    assert "Equity: $1100.00 (based on last recorded marks, not live prices)" in msg
    assert "Unrealized PnL: $100.00 (last recorded marks)" in msg
    assert "Mark: $110.00 (last recorded)" in msg
    assert "Unrealized PnL: $100.00" in msg


def test_paper_command_reports_missing_marks_as_unavailable() -> None:
    service = _service(_trade(quantity=10.0))

    msg = _run_paper_command(service)

    assert (
        "Equity: $1000.00 (settled balance; no market marks recorded yet)"
        in msg
    )
    assert "Unrealized PnL: unavailable (no market mark recorded yet)" in msg
    assert "Unrealized PnL: $0.00" not in msg


def test_paper_command_labels_partial_marks() -> None:
    service = _service(
        _trade(symbol="BTC", price=100.0, quantity=10.0),
        _trade(symbol="ETH", price=2000.0, quantity=1.0),
    )
    service.get_position("BTC").mark_to_market(110.0)

    msg = _run_paper_command(service)

    assert "1 position(s) without a mark excluded" in msg
    assert "Unrealized PnL: $100.00 (last recorded marks)" in msg


def test_positions_command_shows_recorded_mark() -> None:
    service = _service(_trade(quantity=10.0))
    service.open_positions()[0].mark_to_market(90.0)

    msg = _run_positions_command(service)

    assert "Mark: $90.00 (last recorded)" in msg
    assert "Unrealized PnL: $-100.00" in msg


def test_positions_command_reports_missing_mark() -> None:
    service = _service(_trade(quantity=10.0))

    msg = _run_positions_command(service)

    assert "Unrealized PnL: unavailable (no market mark recorded yet)" in msg
    assert "Unrealized PnL: $0.00" not in msg


def test_stats_command_labels_open_pnl_marks() -> None:
    service = _service(_trade(quantity=10.0))
    service.open_positions()[0].mark_to_market(110.0)

    msg = _run_stats_command(service)

    assert "Open PnL: $100.00 (last recorded marks)" in msg
