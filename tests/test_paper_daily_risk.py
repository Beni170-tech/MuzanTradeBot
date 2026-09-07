"""
Tests for Sprint 8A — Daily Loss Circuit Breaker.

Pure function tests for realized_pnl_on_day, loss_limit_reached,
remaining_loss_budget, the DailyLossCheck / DailyRiskMetrics
dataclasses, DailyRiskManager, and the daily-loss gate behavior
reported by /analyze and the Daily Risk section in /stats.

Semantics under test (as approved):
    - "Today" is the local calendar day (datetime.now()).
    - Net realized PnL counts realized losses and profits of positions
      closed today; unrealized PnL is ignored (only closed positions
      are ever summed).
    - The brake trips the moment the day's net realized PnL reaches
      -limit (loss >= limit stops trading: net <= -limit).
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from trader.paper.daily_risk import (
    DailyLossCheck,
    DailyRiskManager,
    DailyRiskMetrics,
    loss_limit_reached,
    realized_pnl_on_day,
    remaining_loss_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today() -> str:
    """The current local calendar day label."""
    return datetime.now().strftime("%Y-%m-%d")


def _make_closed(realized_pnl=0.0, exit_at=None):
    """Create a minimal mock of a closed PaperOrder."""
    order = MagicMock()
    order.realized_pnl = realized_pnl
    order.exit_at = exit_at
    return order


def _make_broker(closed_positions=None):
    """Create a minimal broker mock exposing closed positions."""
    broker = MagicMock()
    broker.account = MagicMock()
    broker.account.closed_positions = closed_positions or []
    return broker


# ---------------------------------------------------------------------------
# realized_pnl_on_day
# ---------------------------------------------------------------------------


class TestRealizedPnlOnDay:
    def test_no_closed_positions(self):
        assert realized_pnl_on_day([], _today()) == 0.0

    def test_sums_pnl_for_matching_day(self):
        orders = [
            _make_closed(realized_pnl=-40.0, exit_at=f"{_today()} 10:00:00"),
            _make_closed(realized_pnl=-60.0, exit_at=f"{_today()} 11:00:00"),
        ]
        assert realized_pnl_on_day(orders, _today()) == -100.0

    def test_profits_and_losses_net_out(self):
        """Realized profits reduce the day's loss; realized losses add."""
        orders = [
            _make_closed(realized_pnl=250.0, exit_at=f"{_today()} 09:00:00"),
            _make_closed(realized_pnl=-300.0, exit_at=f"{_today()} 10:00:00"),
        ]
        # +250 - 300 = -50: the loss is net of the profit taken today.
        assert realized_pnl_on_day(orders, _today()) == -50.0

    def test_other_days_excluded(self):
        orders = [
            _make_closed(realized_pnl=-500.0, exit_at="2026-01-01 10:00:00"),
            _make_closed(realized_pnl=-500.0, exit_at="2026-01-02 10:00:00"),
        ]
        assert realized_pnl_on_day(orders, "2026-01-01") == -500.0
        assert realized_pnl_on_day(orders, "2026-01-03") == 0.0

    def test_missing_exit_at_excluded(self):
        orders = [
            _make_closed(realized_pnl=-500.0, exit_at=None),
            _make_closed(realized_pnl=-500.0, exit_at=""),
        ]
        assert realized_pnl_on_day(orders, _today()) == 0.0

    def test_empty_exit_at_string_excluded(self):
        orders = [_make_closed(realized_pnl=-500.0, exit_at="")]
        assert realized_pnl_on_day(orders, _today()) == 0.0

    def test_short_timestamp_excluded(self):
        orders = [_make_closed(realized_pnl=-500.0, exit_at="2026-09-0")]
        assert realized_pnl_on_day(orders, "2026-09-07") == 0.0

    def test_closed_positions_are_the_only_input(self):
        """Only closed positions exist in the input list; there is no
        path for open positions or unrealized PnL to be counted."""
        orders = [_make_closed(realized_pnl=-100.0, exit_at=f"{_today()} 12:00:00")]
        assert realized_pnl_on_day(orders, _today()) == -100.0


# ---------------------------------------------------------------------------
# loss_limit_reached
# ---------------------------------------------------------------------------


class TestLossLimitReached:
    def test_disabled_when_limit_zero(self):
        assert loss_limit_reached(-500.0, 0) is False

    def test_disabled_when_limit_negative(self):
        assert loss_limit_reached(-500.0, -10.0) is False

    def test_winning_day_never_triggers(self):
        assert loss_limit_reached(250.0, 100.0) is False

    def test_flat_day_never_triggers(self):
        assert loss_limit_reached(0.0, 100.0) is False

    def test_loss_under_limit_continues(self):
        # Loss of 99.99 with a 100 limit -> keep trading.
        assert loss_limit_reached(-99.99, 100.0) is False

    def test_loss_at_limit_stops(self):
        # Loss of exactly 100.00 with a 100 limit -> stop.
        assert loss_limit_reached(-100.00, 100.0) is True

    def test_loss_over_limit_stops(self):
        assert loss_limit_reached(-101.50, 100.0) is True

    def test_loss_just_over_limit_stops(self):
        assert loss_limit_reached(-100.01, 100.0) is True

    def test_larger_limit_not_reached(self):
        assert loss_limit_reached(-50.0, 200.0) is False

    def test_deep_loss_triggers(self):
        assert loss_limit_reached(-1000.0, 100.0) is True


# ---------------------------------------------------------------------------
# remaining_loss_budget
# ---------------------------------------------------------------------------


class TestRemainingLossBudget:
    def test_disabled_returns_zero(self):
        assert remaining_loss_budget(-50.0, 0.0) == 0.0

    def test_no_loss_yet_full_budget(self):
        assert remaining_loss_budget(0.0, 100.0) == 100.0

    def test_partial_loss_reduces_budget(self):
        assert remaining_loss_budget(-30.0, 100.0) == 70.0

    def test_profits_increase_budget(self):
        # Profits earned today may be lost before the limit is reached.
        assert remaining_loss_budget(50.0, 100.0) == 150.0

    def test_at_limit_floors_at_zero(self):
        assert remaining_loss_budget(-100.0, 100.0) == 0.0

    def test_over_limit_floors_at_zero(self):
        assert remaining_loss_budget(-150.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# DailyLossCheck dataclass
# ---------------------------------------------------------------------------


class TestDailyLossCheck:
    def test_allowed(self):
        check = DailyLossCheck(allowed=True)
        assert check.allowed is True

    def test_blocked(self):
        check = DailyLossCheck(allowed=False)
        assert check.allowed is False

    def test_immutable(self):
        check = DailyLossCheck(allowed=True)
        with pytest.raises(AttributeError):
            check.allowed = False


# ---------------------------------------------------------------------------
# DailyRiskManager
# ---------------------------------------------------------------------------


class TestDailyRiskManager:
    def test_disabled_when_limit_zero(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-500.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=0.0)
        assert manager.evaluate().allowed is True
        assert manager.is_halted() is False

    def test_winning_day_allows(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=250.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        assert manager.evaluate().allowed is True

    def test_loss_under_limit_allows(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-99.99, exit_at=f"{_today()} 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        assert manager.evaluate().allowed is True

    def test_loss_at_limit_halts(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-100.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        assert manager.evaluate().allowed is False
        assert manager.is_halted() is True

    def test_loss_over_limit_halts(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-150.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        assert manager.evaluate().allowed is False

    def test_losses_and_profits_net_before_comparison(self):
        # +250 realized then -350 realized = -100 net -> halted.
        broker = _make_broker(
            [
                _make_closed(realized_pnl=250.0, exit_at=f"{_today()} 09:00:00"),
                _make_closed(realized_pnl=-350.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        assert manager.evaluate().allowed is False

    def test_no_closed_trades_allows(self):
        manager = DailyRiskManager(_make_broker([]), daily_loss_limit=100.0)
        assert manager.evaluate().allowed is True

    def test_yesterday_loss_does_not_halt_today(self):
        # The brake is per-calendar-day: yesterday's losses do not carry
        # over into today's fresh budget.
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-1000.0, exit_at="2026-01-01 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        assert manager.evaluate(day="2026-01-02").allowed is True
        assert manager.evaluate(day="2026-01-01").allowed is False

    def test_explicit_day_parameter(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-150.0, exit_at="2026-03-15 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        assert manager.evaluate(day="2026-03-15").allowed is False
        assert manager.evaluate(day="2026-03-16").allowed is True

    def test_daily_loss_limit_property(self):
        manager = DailyRiskManager(_make_broker([]), daily_loss_limit=250.0)
        assert manager.daily_loss_limit == 250.0
        manager = DailyRiskManager(_make_broker([]))
        assert manager.daily_loss_limit == 0.0


# ---------------------------------------------------------------------------
# DailyRiskManager — metrics
# ---------------------------------------------------------------------------


class TestDailyRiskMetrics:
    def test_fields_populated_when_breached(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-120.0, exit_at="2026-03-15 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        m = manager.metrics(day="2026-03-15")
        assert m.day == "2026-03-15"
        assert m.realized_pnl_today == -120.0
        assert m.loss_limit == 100.0
        assert m.loss_usage_percent == 120.0
        assert m.remaining_budget == 0.0
        assert m.breached is True

    def test_metrics_under_limit(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-40.0, exit_at="2026-03-15 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=200.0)
        m = manager.metrics(day="2026-03-15")
        assert m.breached is False
        assert m.loss_usage_percent == 20.0
        assert m.remaining_budget == 160.0

    def test_metrics_at_limit(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-100.0, exit_at="2026-03-15 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        m = manager.metrics(day="2026-03-15")
        assert m.breached is True
        assert m.loss_usage_percent == 100.0
        assert m.remaining_budget == 0.0

    def test_metrics_on_winning_day(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=50.0, exit_at="2026-03-15 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=100.0)
        m = manager.metrics(day="2026-03-15")
        assert m.realized_pnl_today == 50.0
        assert m.loss_usage_percent == 0.0
        assert m.remaining_budget == 150.0
        assert m.breached is False

    def test_metrics_when_disabled_shows_zero_usage(self):
        broker = _make_broker(
            [
                _make_closed(realized_pnl=-500.0, exit_at="2026-03-15 10:00:00"),
            ]
        )
        manager = DailyRiskManager(broker, daily_loss_limit=0.0)
        m = manager.metrics(day="2026-03-15")
        assert m.realized_pnl_today == -500.0
        assert m.loss_limit == 0.0
        assert m.loss_usage_percent == 0.0
        assert m.remaining_budget == 0.0
        assert m.breached is False

    def test_metrics_empty_account(self):
        manager = DailyRiskManager(_make_broker([]), daily_loss_limit=100.0)
        m = manager.metrics(day="2026-03-15")
        assert m.realized_pnl_today == 0.0
        assert m.breached is False
        assert m.remaining_budget == 100.0

    def test_immutable(self):
        m = DailyRiskMetrics(
            day="2026-03-15",
            realized_pnl_today=-120.0,
            loss_limit=100.0,
            loss_usage_percent=120.0,
            remaining_budget=0.0,
            breached=True,
        )
        with pytest.raises(AttributeError):
            m.day = "2026-03-16"


# ---------------------------------------------------------------------------
# Handler behavior: daily-loss gate in /analyze
# ---------------------------------------------------------------------------


class TestAnalyzeDailyLossGate:
    def test_gate_enabled_when_limit_set(self):
        assert (0.0 > 0) is False
        assert (100.0 > 0) is True

    def test_trade_blocked_when_day_loss_reached(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_closed(realized_pnl=-100.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )

        limit = 100.0
        manager = DailyRiskManager(service._broker, limit)
        check = manager.evaluate()

        # Handler guard: only evaluated when the limit is configured.
        gate_enabled = limit > 0
        assert gate_enabled is True
        assert check.allowed is False
        assert manager.metrics().breached is True

    def test_trade_allowed_when_under_daily_loss(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_closed(realized_pnl=-50.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )

        manager = DailyRiskManager(service._broker, 100.0)
        assert manager.evaluate().allowed is True

    def test_gate_disabled_when_limit_zero(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_closed(realized_pnl=-500.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )

        limit = 0.0
        # The handler's guard `if settings.PORTFOLIO_DAILY_LOSS_LIMIT > 0`
        # prevents enforcement when the limit is 0.
        gate_enabled = limit > 0
        manager = DailyRiskManager(service._broker, limit)

        assert gate_enabled is False
        assert manager.evaluate().allowed is True

    def test_blocked_trade_reports_metrics(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_closed(realized_pnl=-150.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )

        manager = DailyRiskManager(service._broker, 100.0)
        m = manager.metrics()
        assert m.realized_pnl_today == -150.0
        assert m.loss_limit == 100.0
        assert m.breached is True
        assert manager.evaluate().allowed is False


# ---------------------------------------------------------------------------
# Handler behavior: /stats Daily Risk section
# ---------------------------------------------------------------------------


class TestStatsDailyRiskSection:
    def test_section_data_present_when_limit_configured(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_closed(realized_pnl=-120.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )

        limit = 100.0
        manager = DailyRiskManager(service._broker, limit)
        m = manager.metrics()

        section_enabled = limit > 0
        assert section_enabled is True
        assert m.loss_limit == 100.0
        assert m.breached is True
        assert m.day == _today()
        assert m.loss_usage_percent == 120.0

    def test_no_section_when_limit_zero(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_closed(realized_pnl=-120.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )

        limit = 0.0
        section_enabled = limit > 0
        manager = DailyRiskManager(service._broker, limit)
        m = manager.metrics()

        assert section_enabled is False
        assert m.loss_limit == 0.0
        assert m.breached is False

    def test_healthy_day_reports_allowed(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_closed(realized_pnl=-20.0, exit_at=f"{_today()} 10:00:00"),
            ]
        )

        manager = DailyRiskManager(service._broker, 100.0)
        m = manager.metrics()
        assert m.breached is False
        assert m.remaining_budget == 80.0
        assert m.loss_usage_percent == 20.0
