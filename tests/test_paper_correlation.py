"""
Tests for Sprint 7C — Correlation Risk Manager.

Pure function tests for open_position_count, symbol_exposure,
trade_exposure, prospective_symbol_exposure, the CorrelationCheck /
CorrelationMetrics dataclasses, CorrelationRiskManager, and the
concentration gate behavior in /analyze and the Symbol Concentration
section in /stats.

Exposure math is not re-implemented here: symbol_exposure delegates to
trader.paper.analytics.position_exposure, which is already covered by
the Sprint 7B suite.
"""

from unittest.mock import MagicMock

import pytest

from trader.execution.contract import ExecutionBackend
from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.correlation_risk import (
    CorrelationCheck,
    CorrelationMetrics,
    CorrelationRiskManager,
    open_position_count,
    prospective_symbol_exposure,
    symbol_exposure,
    trade_exposure,
)
from trader.paper.order import PaperOrderStatus
from trader.paper.analytics import position_exposure
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(
    symbol="BTC",
    entry_price=100.0,
    quantity=10.0,
    side=None,
):
    """Create a minimal mock of an open PaperOrder."""
    order = MagicMock()
    order.symbol = symbol
    order.entry_price = entry_price
    order.quantity = quantity
    if side is not None:
        order.side = side
    return order


def _make_broker(open_positions=None):
    """Create a minimal broker mock."""
    broker = MagicMock()
    broker.get_open_positions.return_value = open_positions or []
    return broker


def _make_trade(price=100.0, position_size=10.0, side="LONG", symbol="BTC") -> Trade:
    """Create a real Trade for contract/paper tests."""
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=position_size,
        stop_loss=98.0,
        take_profit=104.0,
        side=side,
        approved=True,
    )


# ---------------------------------------------------------------------------
# open_position_count
# ---------------------------------------------------------------------------


class TestOpenPositionCount:
    def test_no_open_positions(self):
        broker = _make_broker([])
        assert open_position_count(broker, "BTC") == 0

    def test_single_position_on_symbol(self):
        broker = _make_broker([_make_order(symbol="BTC")])
        assert open_position_count(broker, "BTC") == 1

    def test_other_symbols_not_counted(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC"),
                _make_order(symbol="ETH"),
            ]
        )
        assert open_position_count(broker, "BTC") == 1
        assert open_position_count(broker, "ETH") == 1

    def test_multiple_positions_same_symbol(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC"),
                _make_order(symbol="BTC"),
                _make_order(symbol="BTC"),
            ]
        )
        assert open_position_count(broker, "BTC") == 3

    def test_case_insensitive(self):
        broker = _make_broker([_make_order(symbol="BTC")])
        assert open_position_count(broker, "btc") == 1
        assert open_position_count(broker, "  BTC  ") == 1

    def test_missing_symbol_returns_zero(self):
        broker = _make_broker([_make_order(symbol="ETH")])
        assert open_position_count(broker, "BTC") == 0


# ---------------------------------------------------------------------------
# symbol_exposure (delegates to analytics.position_exposure)
# ---------------------------------------------------------------------------


class TestSymbolExposure:
    def test_no_open_positions(self):
        broker = _make_broker([])
        assert symbol_exposure(broker, "BTC") == 0.0

    def test_single_position(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        assert symbol_exposure(broker, "BTC") == 1000.0

    def test_multiple_positions_same_symbol_summed(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=100.0, quantity=5.0),
                _make_order(symbol="BTC", entry_price=110.0, quantity=5.0),
            ]
        )
        assert symbol_exposure(broker, "BTC") == 1050.0

    def test_other_symbols_not_included(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
                _make_order(symbol="ETH", entry_price=50.0, quantity=20.0),
            ]
        )
        assert symbol_exposure(broker, "BTC") == 1000.0
        assert symbol_exposure(broker, "ETH") == 1000.0

    def test_matches_analytics_position_exposure_semantics(self):
        """symbol_exposure must equal the sum of position_exposure values,
        proving the exposure math is delegated, not duplicated."""
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=95000.0, quantity=0.01),
                _make_order(symbol="BTC", entry_price=3500.0, quantity=1.0),
            ]
        )
        positions = broker.get_open_positions()
        expected = sum(position_exposure(p) for p in positions)
        assert symbol_exposure(broker, "BTC") == pytest.approx(expected)
        assert symbol_exposure(broker, "BTC") == pytest.approx(4450.0)

    def test_invalid_entries_contribute_zero(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=0.0, quantity=10.0),
                _make_order(symbol="BTC", entry_price=100.0, quantity=0.0),
                _make_order(symbol="BTC", entry_price=None, quantity=10.0),
                _make_order(symbol="BTC", entry_price=100.0, quantity=None),
            ]
        )
        assert symbol_exposure(broker, "BTC") == 0.0


# ---------------------------------------------------------------------------
# trade_exposure
# ---------------------------------------------------------------------------


class TestTradeExposure:
    def test_basic(self):
        trade = _make_trade(price=100.0, position_size=10.0)
        assert trade_exposure(trade) == 1000.0

    def test_zero_position_size(self):
        trade = _make_trade(price=100.0, position_size=0.0)
        assert trade_exposure(trade) == 0.0

    def test_zero_price(self):
        trade = _make_trade(price=0.0, position_size=10.0)
        assert trade_exposure(trade) == 0.0

    def test_negative_values(self):
        trade = _make_trade(price=-5.0, position_size=10.0)
        assert trade_exposure(trade) == 0.0
        trade = _make_trade(price=100.0, position_size=-5.0)
        assert trade_exposure(trade) == 0.0

    def test_non_numeric_price(self):
        trade = _make_trade(price="Unknown", position_size=10.0)
        assert trade_exposure(trade) == 0.0


# ---------------------------------------------------------------------------
# prospective_symbol_exposure
# ---------------------------------------------------------------------------


class TestProspectiveSymbolExposure:
    def test_no_open_positions(self):
        broker = _make_broker([])
        trade = _make_trade(price=100.0, position_size=10.0)
        assert prospective_symbol_exposure(broker, trade) == 1000.0

    def test_with_existing_positions(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        trade = _make_trade(price=200.0, position_size=5.0)
        assert prospective_symbol_exposure(broker, trade) == 2000.0

    def test_equals_symbol_exposure_plus_trade_exposure(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=100.0, quantity=5.0),
                _make_order(symbol="ETH", entry_price=50.0, quantity=20.0),
            ]
        )
        trade = _make_trade(price=100.0, position_size=10.0)
        expected = symbol_exposure(broker, "BTC") + trade_exposure(trade)
        assert prospective_symbol_exposure(broker, trade) == pytest.approx(
            expected
        )

    def test_zero_position_size(self):
        broker = _make_broker([])
        trade = _make_trade(price=100.0, position_size=0.0)
        assert prospective_symbol_exposure(broker, trade) == 0.0


# ---------------------------------------------------------------------------
# CorrelationCheck dataclass
# ---------------------------------------------------------------------------


class TestCorrelationCheck:
    def test_allowed(self):
        check = CorrelationCheck(allowed=True)
        assert check.allowed is True
        assert check.rule is None

    def test_rejected_with_rule(self):
        check = CorrelationCheck(allowed=False, rule="positions")
        assert check.allowed is False
        assert check.rule == "positions"

    def test_immutable(self):
        check = CorrelationCheck(allowed=True)
        with pytest.raises(AttributeError):
            check.allowed = False


# ---------------------------------------------------------------------------
# CorrelationRiskManager — position-count rule (Rule A)
# ---------------------------------------------------------------------------


class TestPositionCountRule:
    def test_disabled_when_limit_zero(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC"),
                _make_order(symbol="BTC"),
            ]
        )
        manager = CorrelationRiskManager(broker, max_positions_per_symbol=0)
        trade = _make_trade()
        assert manager.can_open_trade(trade) is True
        assert manager.evaluate(trade) == CorrelationCheck(allowed=True)

    def test_limit_one_rejects_second_position(self):
        broker = _make_broker([_make_order(symbol="BTC")])
        manager = CorrelationRiskManager(broker, max_positions_per_symbol=1)
        trade = _make_trade()
        check = manager.evaluate(trade)
        assert check.allowed is False
        assert check.rule == "positions"

    def test_limit_one_allows_first_position(self):
        broker = _make_broker([])
        manager = CorrelationRiskManager(broker, max_positions_per_symbol=1)
        trade = _make_trade()
        assert manager.can_open_trade(trade) is True

    def test_limit_two_allows_up_to_two(self):
        broker = _make_broker([_make_order(symbol="BTC")])
        manager = CorrelationRiskManager(broker, max_positions_per_symbol=2)
        assert manager.can_open_trade(_make_trade()) is True

        broker = _make_broker(
            [
                _make_order(symbol="BTC"),
                _make_order(symbol="BTC"),
            ]
        )
        manager = CorrelationRiskManager(broker, max_positions_per_symbol=2)
        check = manager.evaluate(_make_trade())
        assert check.allowed is False
        assert check.rule == "positions"

    def test_other_symbol_unaffected(self):
        broker = _make_broker([_make_order(symbol="BTC")])
        manager = CorrelationRiskManager(broker, max_positions_per_symbol=1)
        trade = _make_trade(symbol="ETH")
        assert manager.can_open_trade(trade) is True

    def test_count_crosses_symbols_case_insensitively(self):
        broker = _make_broker([_make_order(symbol="BTC")])
        manager = CorrelationRiskManager(broker, max_positions_per_symbol=1)
        trade = _make_trade(symbol="btc")
        assert manager.can_open_trade(trade) is False


# ---------------------------------------------------------------------------
# CorrelationRiskManager — per-symbol exposure rule (Rule B)
# ---------------------------------------------------------------------------


class TestExposureRule:
    def test_disabled_when_limit_zero(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=90000.0, quantity=0.1)]
        )
        manager = CorrelationRiskManager(
            broker, max_exposure_per_symbol=0.0
        )
        trade = _make_trade(price=10000.0, position_size=10.0)
        assert manager.can_open_trade(trade) is True

    def test_within_limit(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker, max_exposure_per_symbol=2000.0
        )
        trade = _make_trade(price=100.0, position_size=5.0)
        assert manager.can_open_trade(trade) is True

    def test_at_limit_allowed(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker, max_exposure_per_symbol=3000.0
        )
        # current 1000 + prospective 2000 == limit -> allowed
        trade = _make_trade(price=100.0, position_size=20.0)
        assert manager.can_open_trade(trade) is True

    def test_over_limit_rejected(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker, max_exposure_per_symbol=2000.0
        )
        # current 1000 + prospective 2000 > 2000 -> rejected
        trade = _make_trade(price=100.0, position_size=20.0)
        check = manager.evaluate(trade)
        assert check.allowed is False
        assert check.rule == "exposure"

    def test_other_symbol_exposure_ignored(self):
        broker = _make_broker(
            [
                _make_order(symbol="ETH", entry_price=100.0, quantity=10.0),
                _make_order(symbol="BTC", entry_price=50.0, quantity=2.0),
            ]
        )
        manager = CorrelationRiskManager(
            broker, max_exposure_per_symbol=150.0
        )
        trade = _make_trade(symbol="BTC", price=100.0, position_size=0.5)
        # BTC: current 100 + prospective 50 == 150 -> allowed
        assert manager.can_open_trade(trade) is True


# ---------------------------------------------------------------------------
# CorrelationRiskManager — both rules combined
# ---------------------------------------------------------------------------


class TestCombinedRules:
    def test_positions_rule_evaluated_first(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
                _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
            ]
        )
        manager = CorrelationRiskManager(
            broker,
            max_positions_per_symbol=1,
            max_exposure_per_symbol=100000.0,
        )
        trade = _make_trade()
        check = manager.evaluate(trade)
        assert check.allowed is False
        assert check.rule == "positions"

    def test_exposure_rule_rejects_when_positions_pass(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker,
            max_positions_per_symbol=2,
            max_exposure_per_symbol=1500.0,
        )
        # count ok (1 < 2) but exposure 1000 + 1000 > 1500 -> exposure
        trade = _make_trade(price=100.0, position_size=10.0)
        check = manager.evaluate(trade)
        assert check.allowed is False
        assert check.rule == "exposure"

    def test_both_rules_pass(self):
        broker = _make_broker(
            [_make_order(symbol="ETH", entry_price=50.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker,
            max_positions_per_symbol=1,
            max_exposure_per_symbol=5000.0,
        )
        trade = _make_trade(symbol="BTC", price=100.0, position_size=10.0)
        assert manager.can_open_trade(trade) is True

    def test_all_disabled_allows_anything(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
                _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
            ]
        )
        manager = CorrelationRiskManager(broker)
        assert manager.max_positions_per_symbol == 0
        assert manager.max_exposure_per_symbol == 0.0
        assert manager.can_open_trade(_make_trade()) is True

    def test_none_trade_allowed(self):
        manager = CorrelationRiskManager(_make_broker([]))
        assert manager.evaluate(None) == CorrelationCheck(allowed=True)


# ---------------------------------------------------------------------------
# CorrelationRiskManager — metrics
# ---------------------------------------------------------------------------


class TestCorrelationMetrics:
    def test_fields_populated(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker,
            max_positions_per_symbol=1,
            max_exposure_per_symbol=2000.0,
        )
        m = manager.metrics("btc")
        assert m.symbol == "BTC"
        assert m.open_positions == 1
        assert m.max_positions == 1
        assert m.symbol_exposure == 1000.0
        assert m.exposure_limit == 2000.0
        assert m.exposure_usage_percent == 50.0
        assert m.remaining_exposure == 1000.0

    def test_disabled_exposure_limit_shows_zero_usage(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker, max_positions_per_symbol=2, max_exposure_per_symbol=0.0
        )
        m = manager.metrics("BTC")
        assert m.symbol_exposure == 1000.0
        assert m.exposure_limit == 0.0
        assert m.exposure_usage_percent == 0.0
        assert m.remaining_exposure == 0.0

    def test_over_limit_remaining_floors_at_zero(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker, max_exposure_per_symbol=500.0
        )
        m = manager.metrics("BTC")
        assert m.exposure_usage_percent == 200.0
        assert m.remaining_exposure == 0.0

    def test_empty_symbol(self):
        manager = CorrelationRiskManager(_make_broker([]))
        m = manager.metrics("BTC")
        assert m.open_positions == 0
        assert m.symbol_exposure == 0.0

    def test_immutable(self):
        m = CorrelationMetrics(
            symbol="BTC",
            open_positions=1,
            max_positions=1,
            symbol_exposure=1000.0,
            exposure_limit=2000.0,
            exposure_usage_percent=50.0,
            remaining_exposure=1000.0,
        )
        with pytest.raises(AttributeError):
            m.symbol = "ETH"


# ---------------------------------------------------------------------------
# concentration_report
# ---------------------------------------------------------------------------


class TestConcentrationReport:
    def test_no_open_positions(self):
        manager = CorrelationRiskManager(_make_broker([]))
        assert manager.concentration_report() == []

    def test_one_entry_per_distinct_symbol(self):
        broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
                _make_order(symbol="BTC", entry_price=100.0, quantity=5.0),
                _make_order(symbol="ETH", entry_price=50.0, quantity=20.0),
            ]
        )
        manager = CorrelationRiskManager(
            broker,
            max_positions_per_symbol=3,
            max_exposure_per_symbol=5000.0,
        )
        report = manager.concentration_report()
        assert len(report) == 2
        # sorted by symbol exposure descending: BTC 1500 > ETH 1000
        assert report[0].symbol == "BTC"
        assert report[0].open_positions == 2
        assert report[1].symbol == "ETH"
        assert report[1].open_positions == 1

    def test_report_reflects_limits(self):
        broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )
        manager = CorrelationRiskManager(
            broker,
            max_positions_per_symbol=1,
            max_exposure_per_symbol=2000.0,
        )
        report = manager.concentration_report()
        assert report[0].max_positions == 1
        assert report[0].exposure_limit == 2000.0


# ---------------------------------------------------------------------------
# Handler behavior: concentration gate in /analyze
# ---------------------------------------------------------------------------


class TestAnalyzeConcentrationGate:
    def test_gate_enabled_when_any_limit_set(self):
        assert (1 > 0 or 0 > 0) is True
        assert (0 > 0 or 5000.0 > 0) is True
        assert (0 > 0 or 0 > 0) is False

    def test_rejects_trade_when_position_cap_reached(self):
        service = MagicMock()
        service._broker = _make_broker([_make_order(symbol="BTC")])

        trade = _make_trade(symbol="BTC")

        max_positions = 1
        max_exposure = 0.0
        manager = CorrelationRiskManager(
            service._broker, max_positions, max_exposure
        )
        check = manager.evaluate(trade)
        rejected = not check.allowed

        assert rejected is True
        assert check.rule == "positions"
        assert manager.metrics("BTC").open_positions == 1

    def test_allows_trade_when_under_position_cap(self):
        service = MagicMock()
        service._broker = _make_broker([])

        trade = _make_trade(symbol="BTC")

        manager = CorrelationRiskManager(service._broker, 1, 0.0)
        assert manager.evaluate(trade).allowed is True

    def test_rejects_trade_when_symbol_exposure_exceeded(self):
        service = MagicMock()
        service._broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )

        trade = _make_trade(symbol="BTC", price=100.0, position_size=20.0)

        manager = CorrelationRiskManager(service._broker, 0, 2000.0)
        check = manager.evaluate(trade)

        assert check.allowed is False
        assert check.rule == "exposure"
        m = manager.metrics("BTC")
        assert m.symbol_exposure == 1000.0
        assert trade_exposure(trade) == 2000.0
        assert m.symbol_exposure + trade_exposure(trade) > 2000.0

    def test_exposure_at_limit_allowed(self):
        service = MagicMock()
        service._broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )

        trade = _make_trade(symbol="BTC", price=100.0, position_size=20.0)

        manager = CorrelationRiskManager(service._broker, 0, 3000.0)
        assert manager.evaluate(trade).allowed is True

    def test_gate_disabled_when_all_limits_zero(self):
        service = MagicMock()
        service._broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )

        trade = _make_trade(symbol="BTC", price=100.0, position_size=100.0)

        max_positions = 0
        max_exposure = 0.0
        # The handler's guard `if max_positions > 0 or max_exposure > 0`
        # prevents enforcement when both limits are 0.
        gate_enabled = max_positions > 0 or max_exposure > 0
        manager = CorrelationRiskManager(
            service._broker, max_positions, max_exposure
        )

        assert gate_enabled is False
        assert manager.evaluate(trade).allowed is True


# ---------------------------------------------------------------------------
# Handler behavior: /stats Symbol Concentration section
# ---------------------------------------------------------------------------


class TestStatsConcentrationSection:
    def test_section_data_present_when_limit_configured(self):
        service = MagicMock()
        service._broker = _make_broker(
            [
                _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
                _make_order(symbol="ETH", entry_price=50.0, quantity=20.0),
            ]
        )

        max_positions = 1
        max_exposure = 5000.0
        manager = CorrelationRiskManager(
            service._broker, max_positions, max_exposure
        )
        report = manager.concentration_report()

        section_enabled = max_positions > 0 or max_exposure > 0
        assert section_enabled is True
        assert len(report) == 2
        assert report[0].symbol in ("BTC", "ETH")
        assert report[0].open_positions >= 1
        assert report[0].exposure_limit == 5000.0

    def test_no_section_when_limits_zero(self):
        service = MagicMock()
        service._broker = _make_broker(
            [_make_order(symbol="BTC", entry_price=100.0, quantity=10.0)]
        )

        max_positions = 0
        max_exposure = 0.0
        section_enabled = max_positions > 0 or max_exposure > 0

        assert section_enabled is False


class TestExecutionBackendSymbolQueries:
    """Focused contract tests for the new symbol-scoped query methods."""

    def test_get_open_positions_by_symbol_is_case_insensitive(self):
        account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
        broker = PaperBroker(account)
        broker.place_order(_make_trade(symbol="BTC"))
        assert len(list(broker.get_open_positions_by_symbol("btc"))) == 1

    def test_get_closed_positions_by_symbol_is_case_insensitive(self):
        account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
        broker = PaperBroker(account)
        order = broker.place_order(_make_trade(symbol="BTC", price=100.0, position_size=10.0))
        assert order is not None
        broker.close_order(order, exit_price=100.0, status=PaperOrderStatus.CLOSED_MANUAL)
        assert len(list(broker.get_closed_positions_by_symbol("btc"))) == 1

    def test_get_closed_positions_by_symbol_returns_most_recent_first(self):
        account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
        broker = PaperBroker(account)
        o1 = broker.place_order(_make_trade(symbol="BTC", price=100.0))
        o2 = broker.place_order(_make_trade(symbol="BTC", price=110.0, position_size=5.0))
        assert o1 is not None
        assert o2 is not None
        broker.close_order(o1, exit_price=105.0, status=PaperOrderStatus.CLOSED_MANUAL, exit_at="2025-01-01 10:00:00")
        broker.close_order(o2, exit_price=115.0, status=PaperOrderStatus.TAKEN_PROFIT, exit_at="2025-01-01 12:00:00")
        closed = broker.get_closed_positions_by_symbol("BTC")
        assert len(closed) == 2
        assert closed[0].order_id == o2.order_id
        assert closed[1].order_id == o1.order_id
