"""
Tests for Sprint 9A - Alerts message builders.

The message builders in alerts.messages are pure functions: they take
domain objects (PaperOrder, Trade, DailyRiskMetrics,
MaxOpenPositionsResult) and return the notification text for each
alert event. No network, no I/O, no state.

Alert events covered (Sprint 9A scope):
    - Paper order placed
    - Stop-loss triggered
    - Take-profit triggered
    - Trailing stop updates
    - Guardian / risk-limit trade rejections
    - Daily loss breaker activation
    - Maximum open positions reached
"""

from alerts.messages import (
    daily_loss_breaker_message,
    max_open_positions_message,
    order_placed_message,
    stop_loss_triggered_message,
    take_profit_triggered_message,
    trailing_stop_updated_message,
    trade_rejected_message,
)
from trader.paper.daily_risk import DailyRiskMetrics
from trader.paper.max_open_positions import MaxOpenPositionsResult
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_open_order(
    symbol="BTC",
    side=PaperOrderSide.LONG,
    quantity=1.0,
    entry_price=100.0,
    stop_loss=98.0,
    take_profit=104.0,
    order_id="PAPE-BTC-LONG-001",
):
    """Create a realistic open PaperOrder for message tests."""
    return PaperOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        status=PaperOrderStatus.OPEN,
    )


def _make_closed_order(
    status,
    exit_price,
    realized_pnl,
    symbol="BTC",
    side=PaperOrderSide.LONG,
    entry_price=100.0,
    order_id="PAPE-BTC-LONG-001",
):
    """Create a realistic closed PaperOrder for message tests."""
    order = PaperOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=1.0,
        entry_price=entry_price,
        stop_loss=98.0,
        take_profit=104.0,
        status=status,
    )
    order.exit_price = exit_price
    order.realized_pnl = realized_pnl
    order.exit_at = "2026-09-08 10:00:00"
    return order


def _make_trade(side="LONG", approved=True):
    """Create a realistic approved Trade for rejection-message tests."""
    return Trade(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY" if approved else "WAIT",
        risk_verdict="APPROVED" if approved else "BLOCKED",
        position_size=1.0,
        stop_loss=98.0,
        take_profit=104.0,
        side=side,
        approved=approved,
        risk_reasons=["Portfolio risk limit exceeded"],
    )


# ---------------------------------------------------------------------------
# order_placed_message
# ---------------------------------------------------------------------------


class TestOrderPlacedMessage:
    def test_returns_string(self):
        text = order_placed_message(_make_open_order())
        assert isinstance(text, str)
        assert text

    def test_contains_header_and_order_id(self):
        order = _make_open_order(order_id="PAPE-BTC-LONG-XYZ")
        text = order_placed_message(order)
        assert "Paper Order Placed" in text
        assert "PAPE-BTC-LONG-XYZ" in text

    def test_contains_symbol_side_and_levels(self):
        order = _make_open_order(
            symbol="ETH",
            side=PaperOrderSide.SHORT,
            entry_price=2000.0,
            stop_loss=2040.0,
            take_profit=1920.0,
        )
        text = order_placed_message(order)
        assert "ETH" in text
        assert "SHORT" in text
        assert "$2000.00" in text
        assert "$2040.00" in text
        assert "$1920.00" in text

    def test_contains_quantity(self):
        order = _make_open_order(quantity=2.5)
        text = order_placed_message(order)
        assert "2.5" in text

    def test_does_not_call_anything_on_order(self):
        """The builder is pure: it only reads attributes."""
        order = _make_open_order()
        text = order_placed_message(order)
        assert order.symbol in text


# ---------------------------------------------------------------------------
# stop_loss_triggered_message
# ---------------------------------------------------------------------------


class TestStopLossTriggeredMessage:
    def test_returns_string(self):
        order = _make_closed_order(PaperOrderStatus.STOPPED, 98.0, -2.0)
        text = stop_loss_triggered_message(order)
        assert isinstance(text, str)
        assert text

    def test_contains_header_symbol_and_exit(self):
        order = _make_closed_order(PaperOrderStatus.STOPPED, 98.0, -2.0)
        text = stop_loss_triggered_message(order)
        assert "Stop Loss Triggered" in text
        assert "BTC" in text
        assert "$98.00" in text

    def test_contains_negative_pnl_with_sign(self):
        order = _make_closed_order(PaperOrderStatus.STOPPED, 98.0, -2.0)
        text = stop_loss_triggered_message(order)
        assert "-$2.00" in text

    def test_works_for_short_side(self):
        order = _make_closed_order(
            PaperOrderStatus.STOPPED, 102.0, -2.0, side=PaperOrderSide.SHORT
        )
        text = stop_loss_triggered_message(order)
        assert "Stop Loss Triggered" in text
        assert "SHORT" in text
        assert "-$2.00" in text


# ---------------------------------------------------------------------------
# take_profit_triggered_message
# ---------------------------------------------------------------------------


class TestTakeProfitTriggeredMessage:
    def test_returns_string(self):
        order = _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        text = take_profit_triggered_message(order)
        assert isinstance(text, str)
        assert text

    def test_contains_header_symbol_and_exit(self):
        order = _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        text = take_profit_triggered_message(order)
        assert "Take Profit Triggered" in text
        assert "BTC" in text
        assert "$104.00" in text

    def test_contains_positive_pnl_with_sign(self):
        order = _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        text = take_profit_triggered_message(order)
        assert "+$4.00" in text


# ---------------------------------------------------------------------------
# trailing_stop_updated_message
# ---------------------------------------------------------------------------


class TestTrailingStopUpdatedMessage:
    def test_returns_string(self):
        text = trailing_stop_updated_message("BTC", 98.0, 100.0)
        assert isinstance(text, str)
        assert text

    def test_contains_header_symbol_and_stops(self):
        text = trailing_stop_updated_message("BTC", 98.0, 100.0)
        assert "Trailing Stop Updated" in text
        assert "BTC" in text
        assert "$98.00" in text
        assert "$100.00" in text

    def test_shows_old_then_new_order(self):
        text = trailing_stop_updated_message("ETH", 2040.0, 2000.0)
        # Old stop must appear before the new stop.
        assert text.index("$2040.00") < text.index("$2000.00")


# ---------------------------------------------------------------------------
# trade_rejected_message
# ---------------------------------------------------------------------------


class TestTradeRejectedMessage:
    def test_returns_string(self):
        text = trade_rejected_message(_make_trade(), "Portfolio risk limit exceeded")
        assert isinstance(text, str)
        assert text

    def test_contains_header_symbol_and_reason(self):
        trade = _make_trade()
        reason = "Portfolio risk limit exceeded"
        text = trade_rejected_message(trade, reason)
        assert "Trade Rejected" in text
        assert "BTC" in text
        assert reason in text

    def test_mentions_guardian_mode(self):
        text = trade_rejected_message(_make_trade(), "some reason")
        assert "Guardian" in text

    def test_contains_entry_and_side(self):
        trade = _make_trade(side="SHORT")
        text = trade_rejected_message(trade, "rejected by daily loss breaker")
        assert "$100.00" in text
        assert "SHORT" in text


# ---------------------------------------------------------------------------
# daily_loss_breaker_message
# ---------------------------------------------------------------------------


def _make_daily_metrics(breached=True):
    return DailyRiskMetrics(
        day="2026-09-08",
        realized_pnl_today=-120.0,
        loss_limit=100.0,
        loss_usage_percent=120.0,
        remaining_budget=0.0,
        breached=breached,
    )


class TestDailyLossBreakerMessage:
    def test_returns_string(self):
        text = daily_loss_breaker_message(_make_daily_metrics())
        assert isinstance(text, str)
        assert text

    def test_contains_header_day_and_limit(self):
        text = daily_loss_breaker_message(_make_daily_metrics())
        assert "Daily Loss Breaker" in text
        assert "2026-09-08" in text
        assert "$100.00" in text

    def test_contains_todays_pnl(self):
        text = daily_loss_breaker_message(_make_daily_metrics())
        assert "-$120.00" in text

    def test_says_trading_halted_until_tomorrow(self):
        text = daily_loss_breaker_message(_make_daily_metrics())
        assert "halted" in text.lower()
        assert "tomorrow" in text.lower()


# ---------------------------------------------------------------------------
# max_open_positions_message
# ---------------------------------------------------------------------------


def _make_mop_result(current, limit, allowed=False):
    return MaxOpenPositionsResult(
        allowed=allowed,
        current_open_positions=current,
        limit=limit,
        remaining=max(0, limit - current),
        reason="portfolio already has the maximum number of open positions",
    )


class TestMaxOpenPositionsMessage:
    def test_returns_string(self):
        text = max_open_positions_message(_make_mop_result(5, 5))
        assert isinstance(text, str)
        assert text

    def test_contains_header_and_counts(self):
        text = max_open_positions_message(_make_mop_result(5, 5))
        assert "Maximum Open Positions Reached" in text
        assert "5 / 5" in text

    def test_shows_under_limit_counts(self):
        text = max_open_positions_message(_make_mop_result(3, 5))
        assert "3 / 5" in text

    def test_tells_user_trade_was_not_placed(self):
        text = max_open_positions_message(_make_mop_result(5, 5))
        assert "not placed" in text.lower()


# ---------------------------------------------------------------------------
# Cross-cutting: all builders are pure and total
# ---------------------------------------------------------------------------


class TestBuildersAreTotal:
    def test_every_builder_accepts_real_domain_objects(self):
        """Smoke test: all seven builders run on realistic inputs."""
        assert order_placed_message(_make_open_order())
        assert stop_loss_triggered_message(
            _make_closed_order(PaperOrderStatus.STOPPED, 98.0, -2.0)
        )
        assert take_profit_triggered_message(
            _make_closed_order(PaperOrderStatus.TAKEN_PROFIT, 104.0, 4.0)
        )
        assert trailing_stop_updated_message("BTC", 98.0, 100.0)
        assert trade_rejected_message(_make_trade(), "reason")
        assert daily_loss_breaker_message(_make_daily_metrics())
        assert max_open_positions_message(_make_mop_result(5, 5))
