"""Regression tests for manual paper-trade exit price validation.

A manual /close (and any direct close call) must accept only a finite,
strictly positive numeric exit price. A rejected price must leave the
order, account balance, realized PnL, persistence and alerts untouched.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import telegram_bot.handlers as handlers
from trader.paper.executor import (
    close_paper_position,
    close_paper_position_by_id,
)
from trader.paper.order import PaperOrderStatus
from trader.paper.rules import is_valid_exit_price
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


def _run_close(monkeypatch, tmp_path, service, args):
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = 4242
    context = MagicMock()
    context.args = args

    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(
        handlers, "PAPER_STATE_PATH", str(tmp_path / "paper_state.json")
    )
    factory = MagicMock(return_value=service)
    with patch.object(handlers, "_get_paper_service", factory):
        asyncio.run(handlers.close_command(update, context))
    return update.message.reply_text.await_args[0][0], factory


def _snapshot(service: PaperTradingService) -> tuple:
    summary = service.summary()
    return (
        summary.current_balance,
        summary.total_realized_pnl,
        summary.closed_count,
        tuple((p.order_id, p.status, p.exit_price) for p in service.open_positions()),
    )


BAD_PRICES = [
    "0",
    "0.0",
    "-1",
    "-0.01",
    "nan",
    "NaN",
    "inf",
    "Infinity",
    "-inf",
    "abc",
    "1e999",
]


# ---------------------------------------------------------------------------
# Command-level validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_price", BAD_PRICES)
def test_invalid_close_price_rejected_before_any_change(
    monkeypatch, tmp_path, bad_price
) -> None:
    service = _service_with(_trade())
    before = _snapshot(service)
    state_path = tmp_path / "paper_state.json"

    message, factory = _run_close(
        monkeypatch, tmp_path, service, ["BTC", bad_price]
    )

    assert "❌ Invalid exit price:" in message
    # The service is not even loaded for a rejected price.
    factory.assert_not_called()
    assert _snapshot(service) == before
    assert service.open_positions()[0].status == PaperOrderStatus.OPEN
    assert not state_path.exists()


@pytest.mark.parametrize("bad_price", ["0", "-5", "nan", "inf"])
def test_invalid_price_does_not_close_ambiguous_positions(
    monkeypatch, tmp_path, bad_price
) -> None:
    service = _service_with(_trade(quantity=1.0), _trade(quantity=2.0))
    before = _snapshot(service)

    message, factory = _run_close(
        monkeypatch, tmp_path, service, ["BTC", bad_price]
    )

    assert "❌ Invalid exit price:" in message
    factory.assert_not_called()
    assert _snapshot(service) == before
    assert all(
        position.status == PaperOrderStatus.OPEN
        for position in service.open_positions()
    )


@pytest.mark.parametrize("bad_price", ["0", "-5", "nan", "inf"])
def test_invalid_price_with_order_id_closes_nothing(
    monkeypatch, tmp_path, bad_price
) -> None:
    service = _service_with(_trade(quantity=1.0), _trade(quantity=2.0))
    first, second = service.open_positions()
    before = _snapshot(service)

    message, factory = _run_close(
        monkeypatch,
        tmp_path,
        service,
        ["BTC", bad_price, first.order_id],
    )

    assert "❌ Invalid exit price:" in message
    factory.assert_not_called()
    assert _snapshot(service) == before
    assert first.status == PaperOrderStatus.OPEN
    assert second.status == PaperOrderStatus.OPEN


def test_valid_close_updates_balance_and_persists(
    monkeypatch, tmp_path
) -> None:
    service = _service_with(_trade(quantity=2.0))
    order = service.open_positions()[0]
    state_path = tmp_path / "paper_state.json"

    message, factory = _run_close(monkeypatch, tmp_path, service, ["BTC", "110"])

    factory.assert_called_once()
    assert "✅ Position Closed" in message
    assert "Exit Price: $110.00" in message
    assert order.status == PaperOrderStatus.CLOSED_MANUAL
    assert order.exit_price == pytest.approx(110.0)
    assert order.realized_pnl == pytest.approx(20.0)
    assert service.summary().current_balance == pytest.approx(1020.0)
    assert service.open_positions() == []
    assert state_path.exists()


def test_valid_close_accepts_exponent_notation(monkeypatch, tmp_path) -> None:
    service = _service_with(_trade(quantity=1.0))

    message, _ = _run_close(monkeypatch, tmp_path, service, ["BTC", "1e2"])

    assert "✅ Position Closed" in message
    assert service.open_positions() == []
    assert service.summary().current_balance == pytest.approx(1000.0)


def test_valid_id_close_only_closes_target_position(
    monkeypatch, tmp_path
) -> None:
    service = _service_with(_trade(quantity=1.0), _trade(quantity=2.0))
    first, second = service.open_positions()

    message, _ = _run_close(
        monkeypatch, tmp_path, service, ["BTC", "110", first.order_id]
    )

    assert "✅ Position Closed" in message
    assert first.status == PaperOrderStatus.CLOSED_MANUAL
    assert first.exit_price == pytest.approx(110.0)
    assert second.status == PaperOrderStatus.OPEN
    assert [position.order_id for position in service.open_positions()] == [
        second.order_id
    ]
    assert service.summary().current_balance == pytest.approx(1010.0)


# ---------------------------------------------------------------------------
# Lower-level guards (direct callers cannot bypass validation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_price",
    [0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf"), True, "110", None],
)
def test_broker_close_order_rejects_invalid_price(bad_price) -> None:
    service = _service_with(_trade(quantity=1.0))
    order = service.open_positions()[0]
    broker = service._broker
    before = _snapshot(service)

    with pytest.raises(ValueError):
        broker.close_order(
            order, exit_price=bad_price, status=PaperOrderStatus.CLOSED_MANUAL
        )

    assert _snapshot(service) == before
    assert order.status == PaperOrderStatus.OPEN
    assert order.exit_price is None
    assert order.realized_pnl == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bad_price", [0, -1, float("nan"), float("inf"), True, "110", None]
)
def test_order_record_exit_rejects_invalid_price(bad_price) -> None:
    service = _service_with(_trade(quantity=1.0))
    order = service.open_positions()[0]

    with pytest.raises(ValueError):
        order.record_exit(bad_price, PaperOrderStatus.CLOSED_MANUAL)

    assert order.status == PaperOrderStatus.OPEN
    assert order.exit_price is None
    assert order.exit_at is None
    assert order.realized_pnl == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bad_price", [0, -1, float("nan"), float("inf"), True, "110"]
)
def test_executor_close_by_symbol_rejects_invalid_price(bad_price) -> None:
    service = _service_with(_trade(quantity=1.0))
    order = service.open_positions()[0]
    before = _snapshot(service)

    assert (
        close_paper_position(
            service._broker, "BTC", bad_price, PaperOrderStatus.CLOSED_MANUAL
        )
        is None
    )
    assert _snapshot(service) == before
    assert order.status == PaperOrderStatus.OPEN


@pytest.mark.parametrize(
    "bad_price", [0, -1, float("nan"), float("inf"), True, "110"]
)
def test_executor_close_by_id_rejects_invalid_price(bad_price) -> None:
    service = _service_with(_trade(quantity=1.0))
    order = service.open_positions()[0]
    before = _snapshot(service)

    assert (
        close_paper_position_by_id(
            service._broker,
            order.order_id,
            bad_price,
            PaperOrderStatus.CLOSED_MANUAL,
        )
        is None
    )
    assert _snapshot(service) == before
    assert order.status == PaperOrderStatus.OPEN


def test_service_close_rejects_invalid_price() -> None:
    service = _service_with(_trade(quantity=1.0))
    order = service.open_positions()[0]
    before = _snapshot(service)

    assert (
        service.close_position_by_id(
            order.order_id, 0.0, PaperOrderStatus.CLOSED_MANUAL
        )
        is None
    )
    assert _snapshot(service) == before


# ---------------------------------------------------------------------------
# Validator unit tests
# ---------------------------------------------------------------------------


def test_is_valid_exit_price_accepts_finite_positive_numbers() -> None:
    assert is_valid_exit_price(1) is True
    assert is_valid_exit_price(0.01) is True
    assert is_valid_exit_price(110.5) is True


@pytest.mark.parametrize(
    "bad_price",
    [0, 0.0, -1, -0.01, float("nan"), float("inf"), float("-inf"), True, False, "1", None],
)
def test_is_valid_exit_price_rejects_unusable_values(bad_price) -> None:
    assert is_valid_exit_price(bad_price) is False
