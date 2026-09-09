"""
Tests for Sprint 9A - AlertManager.

AlertManager composes the pure message builders with a notifier and
exposes one semantic method per alert event. Contract under test:

    - Each event method builds the right message and hands it to the
      notifier exactly once.
    - The manager never raises: if the notifier fails, the failure is
      logged and swallowed (Guardian Mode: a broken notifier must not
      break trading).
    - A disabled manager does nothing (no messages built, nothing
      sent).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alerts.manager import AlertManager
from alerts.messages import (
    daily_loss_breaker_message,
    max_open_positions_message,
    order_placed_message,
    stop_loss_triggered_message,
    take_profit_triggered_message,
    trailing_stop_updated_message,
    trade_rejected_message,
)
from alerts.notifier import NullNotifier, TelegramNotifier
from trader.paper.daily_risk import DailyRiskMetrics
from trader.paper.max_open_positions import MaxOpenPositionsResult
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_order():
    return PaperOrder(
        order_id="PAPE-BTC-LONG-001",
        symbol="BTC",
        side=PaperOrderSide.LONG,
        quantity=1.0,
        entry_price=100.0,
        stop_loss=98.0,
        take_profit=104.0,
        status=PaperOrderStatus.OPEN,
    )


def _make_closed_order(status, exit_price, realized_pnl):
    order = _make_open_order()
    order.exit_price = exit_price
    order.realized_pnl = realized_pnl
    order.exit_at = "2026-09-08 10:00:00"
    order.status = status
    return order


def _make_trade():
    return Trade(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=1.0,
        stop_loss=98.0,
        take_profit=104.0,
        side="LONG",
        approved=True,
        risk_reasons=["Portfolio risk limit exceeded"],
    )


def _make_daily_metrics():
    return DailyRiskMetrics(
        day="2026-09-08",
        realized_pnl_today=-120.0,
        loss_limit=100.0,
        loss_usage_percent=120.0,
        remaining_budget=0.0,
        breached=True,
    )


def _make_mop_result():
    return MaxOpenPositionsResult(
        allowed=False,
        current_open_positions=5,
        limit=5,
        remaining=0,
        reason="portfolio already has the maximum number of open positions",
    )


def _make_manager(notifier=None):
    notifier = notifier or MagicMock()
    notifier.send = AsyncMock()
    return AlertManager(notifier), notifier


# ---------------------------------------------------------------------------
# Event methods send exactly one message
# ---------------------------------------------------------------------------


class TestAlertManagerEvents:
    @pytest.mark.asyncio
    async def test_order_placed_sends_placed_message(self):
        manager, notifier = _make_manager()
        order = _make_open_order()
        await manager.order_placed(order)
        (text,), _ = notifier.send.await_args
        assert order_placed_message(order) == text
        notifier.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_loss_triggered_sends_sl_message(self):
        manager, notifier = _make_manager()
        order = _make_closed_order(PaperOrderStatus.STOPPED, 98.0, -2.0)
        await manager.stop_loss_triggered(order)
        (text,), _ = notifier.send.await_args
        assert stop_loss_triggered_message(order) == text
        notifier.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_take_profit_triggered_sends_tp_message(self):
        manager, notifier = _make_manager()
        order = _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        await manager.take_profit_triggered(order)
        (text,), _ = notifier.send.await_args
        assert take_profit_triggered_message(order) == text
        notifier.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trailing_stop_updated_sends_trail_message(self):
        manager, notifier = _make_manager()
        await manager.trailing_stop_updated("BTC", 98.0, 100.0)
        (text,), _ = notifier.send.await_args
        assert trailing_stop_updated_message("BTC", 98.0, 100.0) == text
        notifier.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_trade_rejected_sends_rejection_message(self):
        manager, notifier = _make_manager()
        trade = _make_trade()
        await manager.trade_rejected(trade, "Portfolio risk limit exceeded")
        (text,), _ = notifier.send.await_args
        assert trade_rejected_message(trade, "Portfolio risk limit exceeded") == text
        notifier.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_daily_loss_breaker_sends_breaker_message(self):
        manager, notifier = _make_manager()
        metrics = _make_daily_metrics()
        await manager.daily_loss_breaker(metrics)
        (text,), _ = notifier.send.await_args
        assert daily_loss_breaker_message(metrics) == text
        notifier.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_max_open_positions_sends_mop_message(self):
        manager, notifier = _make_manager()
        result = _make_mop_result()
        await manager.max_open_positions_reached(result)
        (text,), _ = notifier.send.await_args
        assert max_open_positions_message(result) == text
        notifier.send.assert_awaited_once()


# ---------------------------------------------------------------------------
# Disabled manager does nothing
# ---------------------------------------------------------------------------


class TestAlertManagerDisabled:
    @pytest.mark.asyncio
    async def test_disabled_manager_never_sends(self):
        notifier = MagicMock()
        notifier.send = AsyncMock()
        manager = AlertManager(notifier, enabled=False)
        await manager.order_placed(_make_open_order())
        await manager.stop_loss_triggered(
            _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        )
        await manager.trade_rejected(_make_trade(), "reason")
        await manager.trailing_stop_updated("BTC", 98.0, 100.0)
        await manager.daily_loss_breaker(_make_daily_metrics())
        await manager.max_open_positions_reached(_make_mop_result())
        notifier.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Never-raise contract
# ---------------------------------------------------------------------------


class TestAlertManagerNeverRaises:
    @pytest.mark.asyncio
    async def test_notifier_failure_is_swallowed(self):
        notifier = MagicMock()
        notifier.send = AsyncMock(side_effect=RuntimeError("boom"))
        manager = AlertManager(notifier)

        # None of these may raise.
        await manager.order_placed(_make_open_order())
        await manager.stop_loss_triggered(
            _make_closed_order(PaperOrderStatus.STOPPED, 98.0, -2.0)
        )
        await manager.take_profit_triggered(
            _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        )
        await manager.trailing_stop_updated("BTC", 98.0, 100.0)
        await manager.trade_rejected(_make_trade(), "reason")
        await manager.daily_loss_breaker(_make_daily_metrics())
        await manager.max_open_positions_reached(_make_mop_result())

    @pytest.mark.asyncio
    async def test_swallowed_error_is_logged(self):
        notifier = MagicMock()
        notifier.send = AsyncMock(side_effect=RuntimeError("boom"))
        manager = AlertManager(notifier)
        with patch("alerts.manager.logger.exception") as mock_log:
            await manager.order_placed(_make_open_order())
        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_builder_failure_is_swallowed(self):
        """Even a broken domain object must not break the caller."""
        notifier = MagicMock()
        notifier.send = AsyncMock()
        manager = AlertManager(notifier)
        broken_order = object()  # no attributes at all
        await manager.order_placed(broken_order)
        notifier.send.assert_not_awaited()


# ---------------------------------------------------------------------------
# Message routing (right builder for the right event)
# ---------------------------------------------------------------------------


class TestAlertManagerRouting:
    @pytest.mark.asyncio
    async def test_max_open_positions_uses_mop_builder(self):
        manager, notifier = _make_manager()
        result = _make_mop_result()
        await manager.max_open_positions_reached(result)
        (text,), _ = notifier.send.await_args
        assert "Maximum Open Positions Reached" in text

    @pytest.mark.asyncio
    async def test_daily_breaker_uses_daily_builder(self):
        manager, notifier = _make_manager()
        metrics = _make_daily_metrics()
        await manager.daily_loss_breaker(metrics)
        (text,), _ = notifier.send.await_args
        assert "Daily Loss Breaker" in text

    @pytest.mark.asyncio
    async def test_take_profit_pnl_is_positive(self):
        manager, notifier = _make_manager()
        order = _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        await manager.take_profit_triggered(order)
        (text,), _ = notifier.send.await_args
        assert "+$4.00" in text

    @pytest.mark.asyncio
    async def test_stop_loss_pnl_is_negative(self):
        manager, notifier = _make_manager()
        order = _make_closed_order(PaperOrderStatus.STOPPED, 98.0, -2.0)
        await manager.stop_loss_triggered(order)
        (text,), _ = notifier.send.await_args
        assert "-$2.00" in text


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


class TestAlertManagerFactory:
    def test_null_manager_when_disabled(self):
        from alerts.manager import build_alert_manager

        manager = build_alert_manager(alerts_enabled=False, chat_id="1", token="t")
        assert isinstance(manager, AlertManager)
        assert manager.enabled is False
        assert isinstance(manager._notifier, NullNotifier)

    def test_manager_when_enabled(self, monkeypatch):
        from alerts.manager import build_alert_manager

        class _FakeBot:
            def __init__(self, token):
                self.send_message = AsyncMock()

        monkeypatch.setattr("alerts.notifier._build_bot", _FakeBot)
        manager = build_alert_manager(
            alerts_enabled=True, chat_id="12345", token="tok"
        )
        assert isinstance(manager, AlertManager)
        assert manager.enabled is True
        assert isinstance(manager._notifier, TelegramNotifier)
