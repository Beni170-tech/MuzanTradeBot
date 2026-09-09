"""
Tests for automatic stop-loss / take-profit paper position closure (Sprint 6C).

The _before_analysis_paper_check helper runs BEFORE a new paper order is placed
during /analyze and only evaluates positions that already existed before the
current request.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from telegram_bot.handlers import _before_analysis_paper_check
from trader.paper.order import PaperOrderStatus


def _make_order(
    symbol="BTC",
    side="LONG",
    entry_price=100.0,
    quantity=1.0,
    stop_loss=90.0,
    take_profit=110.0,
    status="OPEN",
    order_id="ORD-001",
):
    order = MagicMock()
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side
    order.entry_price = entry_price
    order.quantity = quantity
    order.stop_loss = stop_loss
    order.take_profit = take_profit
    order.status = MagicMock()
    order.status.value = status
    order.order_id = order_id
    order.realized_pnl = 0.0
    order.exit_price = 0.0
    # Legacy/mock shape kept alongside the real ``status`` attribute
    # (paper_exit_status prefers exit_status, falls back to status).
    order.exit_status = MagicMock()
    order.exit_status.value = "OPEN"
    order.exit_at = None
    return order


def _make_service(open_position=None, closed_positions=None, close_returns=None):
    service = MagicMock()
    service.get_position.return_value = open_position
    # close_position returns the closed order on success, or None if no
    # position exists. By default we return whatever was passed, or the
    # open_position if given (so a successful close returns the order).
    if close_returns is None:
        close_returns = open_position
    service.close_position.return_value = close_returns
    service.save_account = MagicMock()
    service.open_positions.return_value = [open_position] if open_position else []
    service.closed_positions.return_value = closed_positions or []
    service.summary.return_value = MagicMock(
        current_balance=10000.0,
        initial_balance=10000.0,
        equity=10000.0,
        total_realized_pnl=0.0,
        open_pnl=0.0,
        closed_count=0,
    )
    return service


class TestBeforeAnalysisPaperCheck:

    def test_no_open_position_returns_none(self):
        service = _make_service(open_position=None)
        result = _before_analysis_paper_check(service, "BTC", 50000.0)
        assert result is None
        service.close_position.assert_not_called()
        service.save_account.assert_not_called()

    def test_no_sl_tp_levels_returns_none(self):
        order = _make_order(stop_loss=0.0, take_profit=0.0)
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "BTC", 50000.0)
        assert result is None
        service.close_position.assert_not_called()

    def test_only_sl_no_tp(self):
        order = _make_order(stop_loss=90.0, take_profit=0.0)
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "BTC", 85.0)
        assert result is not None
        service.close_position.assert_called_once_with("BTC", 90.0, PaperOrderStatus.STOPPED)
        service.save_account.assert_called_once()

    def test_only_tp_no_sl(self):
        order = _make_order(stop_loss=0.0, take_profit=110.0)
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "BTC", 115.0)
        assert result is not None
        service.close_position.assert_called_once_with("BTC", 110.0, PaperOrderStatus.TAKEN_PROFIT)
        service.save_account.assert_called_once()

    def test_stop_loss_hit_long(self):
        order = _make_order(
            symbol="BTC", side="LONG", entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "BTC", 47000.0)
        assert result is not None
        assert result.symbol == "BTC"
        service.close_position.assert_called_once_with("BTC", 48000.0, PaperOrderStatus.STOPPED)
        service.save_account.assert_called_once()

    def test_stop_loss_at_exact_price_long(self):
        order = _make_order(
            symbol="ETH", side="LONG", entry_price=2000.0,
            stop_loss=1800.0, take_profit=2200.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "ETH", 1800.0)
        assert result is not None
        service.close_position.assert_called_once_with("ETH", 1800.0, PaperOrderStatus.STOPPED)

    def test_take_profit_hit_long(self):
        order = _make_order(
            symbol="BTC", side="LONG", entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "BTC", 56000.0)
        assert result is not None
        service.close_position.assert_called_once_with("BTC", 55000.0, PaperOrderStatus.TAKEN_PROFIT)
        service.save_account.assert_called_once()

    def test_take_profit_at_exact_price_long(self):
        order = _make_order(
            symbol="ETH", side="LONG", entry_price=2000.0,
            stop_loss=1800.0, take_profit=2200.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "ETH", 2200.0)
        assert result is not None
        service.close_position.assert_called_once_with("ETH", 2200.0, PaperOrderStatus.TAKEN_PROFIT)

    def test_within_range_no_closure(self):
        order = _make_order(
            symbol="BTC", side="LONG", entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "BTC", 52000.0)
        assert result is None
        service.close_position.assert_not_called()
        service.save_account.assert_not_called()

    def test_sl_priority_over_tp_long(self):
        order = _make_order(
            symbol="BTC", side="LONG", entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "BTC", 45000.0)
        assert result is not None
        service.close_position.assert_called_once_with("BTC", 48000.0, PaperOrderStatus.STOPPED)

    def test_short_stop_loss_hit(self):
        order = _make_order(
            symbol="ETH", side="SHORT", entry_price=2000.0,
            stop_loss=2200.0, take_profit=1800.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "ETH", 2300.0)
        assert result is not None
        service.close_position.assert_called_once_with("ETH", 2200.0, PaperOrderStatus.STOPPED)

    def test_short_take_profit_hit(self):
        order = _make_order(
            symbol="SOL", side="SHORT", entry_price=100.0,
            stop_loss=120.0, take_profit=80.0,
        )
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "SOL", 75.0)
        assert result is not None
        service.close_position.assert_called_once_with("SOL", 80.0, PaperOrderStatus.TAKEN_PROFIT)

    def test_symbol_passed_through_unchanged(self):
        # Order has both SL and TP; price (85) is below SL (90) so SL triggers.
        # Symbol is passed through unchanged (lowercase 'btc' -> 'btc').
        order = _make_order(symbol="BTC", stop_loss=90.0, take_profit=110.0)
        service = _make_service(open_position=order)
        result = _before_analysis_paper_check(service, "btc", 85.0)
        service.get_position.assert_called_once_with("btc")
        service.close_position.assert_called_once_with(
            "btc", 90.0, PaperOrderStatus.STOPPED
        )

    def test_get_position_called_always(self):
        service = _make_service(open_position=None)
        result = _before_analysis_paper_check(service, "NONEXISTENT", 100.0)
        assert result is None
        service.get_position.assert_called_once_with("NONEXISTENT")

    def test_idempotent_after_close(self):
        order = _make_order(
            symbol="BTC", side="LONG", entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
        )
        closed_order = _make_order(
            symbol="BTC", side="LONG", entry_price=50000.0,
            stop_loss=48000.0, take_profit=55000.0,
            status="CLOSED_MANUAL",
        )
        service = _make_service(open_position=order)
        service.close_position.return_value = closed_order
        result1 = _before_analysis_paper_check(service, "BTC", 47000.0)
        assert result1 is not None
        assert service.close_position.call_count == 1
        service.get_position.r
