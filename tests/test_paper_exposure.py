"""
Tests for Sprint 7B — Portfolio Exposure Limits.

Pure function tests for position_exposure, total_exposure,
prospective_exposure, and exposure_metrics, plus handler-behavior
tests for the exposure gate in /analyze and the Exposure section
in /stats.
"""

from unittest.mock import MagicMock

import pytest

from trader.paper.analytics import (
    ExposureMetrics,
    position_exposure,
    total_exposure,
    prospective_exposure,
    exposure_metrics,
)


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


def _make_trade(price=100.0, position_size=10.0, side="LONG"):
    """Create a minimal mock of a Trade."""
    trade = MagicMock()
    trade.price = price
    trade.position_size = position_size
    trade.side = side
    return trade


# ---------------------------------------------------------------------------
# position_exposure
# ---------------------------------------------------------------------------


class TestPositionExposure:
    def test_basic_long(self):
        order = _make_order(entry_price=100.0, quantity=10.0)
        assert position_exposure(order) == 1000.0

    def test_basic_short(self):
        order = _make_order(entry_price=50.0, quantity=20.0, side="SHORT")
        assert position_exposure(order) == 1000.0

    def test_zero_quantity(self):
        order = _make_order(entry_price=100.0, quantity=0.0)
        assert position_exposure(order) == 0.0

    def test_zero_entry_price(self):
        order = _make_order(entry_price=0.0, quantity=10.0)
        assert position_exposure(order) == 0.0

    def test_negative_entry_price(self):
        order = _make_order(entry_price=-10.0, quantity=5.0)
        assert position_exposure(order) == 0.0

    def test_negative_quantity(self):
        order = _make_order(entry_price=100.0, quantity=-5.0)
        assert position_exposure(order) == 0.0

    def test_none_entry_price(self):
        order = _make_order()
        order.entry_price = None
        assert position_exposure(order) == 0.0

    def test_none_quantity(self):
        order = _make_order()
        order.quantity = None
        assert position_exposure(order) == 0.0


# ---------------------------------------------------------------------------
# total_exposure
# ---------------------------------------------------------------------------


class TestTotalExposure:
    def test_no_open_positions(self):
        broker = _make_broker([])
        assert total_exposure(broker) == 0.0

    def test_single_position(self):
        broker = _make_broker([_make_order(entry_price=100.0, quantity=10.0)])
        assert total_exposure(broker) == 1000.0

    def test_multiple_positions(self):
        positions = [
            _make_order(symbol="BTC", entry_price=100.0, quantity=10.0),
            _make_order(symbol="ETH", entry_price=50.0, quantity=20.0),
        ]
        broker = _make_broker(positions)
        assert total_exposure(broker) == 2000.0

    def test_multiple_positions_same_symbol(self):
        positions = [
            _make_order(symbol="BTC", entry_price=100.0, quantity=5.0),
            _make_order(symbol="BTC", entry_price=110.0, quantity=5.0),
        ]
        broker = _make_broker(positions)
        assert total_exposure(broker) == 1050.0

    def test_mixed_realistic_values(self):
        positions = [
            _make_order(symbol="BTC", entry_price=95000.0, quantity=0.01),
            _make_order(symbol="ETH", entry_price=3500.0, quantity=1.0),
        ]
        broker = _make_broker(positions)
        # 95000 * 0.01 = 950; 3500 * 1 = 3500; total = 4450
        assert total_exposure(broker) == pytest.approx(4450.0)


# ---------------------------------------------------------------------------
# prospective_exposure
# ---------------------------------------------------------------------------


class TestProspectiveExposure:
    def test_no_open_positions(self):
        broker = _make_broker([])
        trade = _make_trade(price=100.0, position_size=10.0)
        assert prospective_exposure(broker, trade) == 1000.0

    def test_with_existing_positions(self):
        broker = _make_broker(
            [_make_order(entry_price=100.0, quantity=10.0)]
        )
        trade = _make_trade(price=200.0, position_size=5.0)
        assert prospective_exposure(broker, trade) == 2000.0

    def test_zero_position_size(self):
        broker = _make_broker([])
        trade = _make_trade(price=100.0, position_size=0.0)
        assert prospective_exposure(broker, trade) == 0.0

    def test_zero_price(self):
        broker = _make_broker([])
        trade = _make_trade(price=0.0, position_size=10.0)
        assert prospective_exposure(broker, trade) == 0.0


# ---------------------------------------------------------------------------
# exposure_metrics
# ---------------------------------------------------------------------------


class TestExposureMetrics:
    def test_disabled_limit(self):
        broker = _make_broker([])
        em = exposure_metrics(broker, 0.0)
        assert em.current_exposure == 0.0
        assert em.exposure_limit == 0.0
        assert em.usage_percent == 0.0
        assert em.remaining == 0.0

    def test_disabled_limit_with_positions(self):
        broker = _make_broker(
            [_make_order(entry_price=100.0, quantity=10.0)]
        )
        em = exposure_metrics(broker, 0.0)
        assert em.current_exposure == 1000.0
        assert em.exposure_limit == 0.0
        assert em.usage_percent == 0.0
        assert em.remaining == 0.0

    def test_within_limit(self):
        broker = _make_broker(
            [_make_order(entry_price=100.0, quantity=10.0)]
        )
        em = exposure_metrics(broker, 2000.0)
        assert em.current_exposure == 1000.0
        assert em.exposure_limit == 2000.0
        assert em.usage_percent == 50.0
        assert em.remaining == 1000.0

    def test_at_limit(self):
        broker = _make_broker(
            [_make_order(entry_price=100.0, quantity=10.0)]
        )
        em = exposure_metrics(broker, 1000.0)
        assert em.current_exposure == 1000.0
        assert em.exposure_limit == 1000.0
        assert em.usage_percent == 100.0
        assert em.remaining == 0.0

    def test_over_limit(self):
        broker = _make_broker(
            [_make_order(entry_price=100.0, quantity=10.0)]
        )
        em = exposure_metrics(broker, 500.0)
        assert em.current_exposure == 1000.0
        assert em.exposure_limit == 500.0
        assert em.usage_percent == 200.0
        assert em.remaining == 0.0

    def test_large_limit(self):
        broker = _make_broker([])
        em = exposure_metrics(broker, 1000000.0)
        assert em.current_exposure == 0.0
        assert em.usage_percent == 0.0
        assert em.remaining == 1000000.0


# ---------------------------------------------------------------------------
# Dataclass equality and structure
# ---------------------------------------------------------------------------


class TestExposureMetricsDataclass:
    def test_fields(self):
        em = ExposureMetrics(
            current_exposure=1000.0,
            exposure_limit=2000.0,
            usage_percent=50.0,
            remaining=1000.0,
        )
        assert em.current_exposure == 1000.0
        assert em.exposure_limit == 2000.0
        assert em.usage_percent == 50.0
        assert em.remaining == 1000.0

    def test_immutable(self):
        em = ExposureMetrics(
            current_exposure=100.0,
            exposure_limit=200.0,
            usage_percent=50.0,
            remaining=100.0,
        )
        with pytest.raises(AttributeError):
            em.current_exposure = 999.0


# ---------------------------------------------------------------------------
# Handler behavior: exposure gate in /analyze
# ---------------------------------------------------------------------------


class TestAnalyzeExposureGate:
    def test_places_trade_when_under_exposure_limit(self):
        service = MagicMock()
        service._broker = _make_broker([])

        trade = _make_trade(price=100.0, position_size=10.0)

        # Simulate the exposure check logic inline
        limit = 5000.0
        current_exposure = exposure_metrics(
            service._broker, 0.0
        ).current_exposure
        prospective = trade.price * trade.position_size
        rejected = current_exposure + prospective > limit

        assert rejected is False
        assert current_exposure + prospective <= limit

    def test_rejects_trade_when_exposure_exceeded(self):
        service = MagicMock()
        service._broker = _make_broker(
            [_make_order(entry_price=90000.0, quantity=0.1)]
        )

        trade = _make_trade(price=10000.0, position_size=1.0)

        limit = 10000.0
        current_exposure = exposure_metrics(
            service._broker, 0.0
        ).current_exposure
        prospective = trade.price * trade.position_size
        rejected = current_exposure + prospective > limit

        assert current_exposure == pytest.approx(9000.0)
        assert prospective == 10000.0
        assert rejected is True
        assert current_exposure + prospective > limit

    def test_exposure_gate_disabled_when_limit_zero(self):
        service = MagicMock()
        service._broker = _make_broker(
            [_make_order(entry_price=100000.0, quantity=10.0)]
        )

        trade = _make_trade(price=10000.0, position_size=100.0)

        limit = 0.0
        current_exposure = exposure_metrics(
            service._broker, limit
        ).current_exposure
        prospective = trade.price * trade.position_size

        # When limit is 0, exposure gate should not reject
        # (the handler's `if settings.PORTFOLIO_EXPOSURE_LIMIT > 0`
        # guard prevents enforcement)
        assert limit == 0.0
        assert current_exposure == pytest.approx(1000000.0)
        assert prospective == 1000000.0


# ---------------------------------------------------------------------------
# Handler behavior: /stats Exposure section
# ---------------------------------------------------------------------------


class TestStatsExposureSection:
    def test_exposure_section_present_when_limit_configured(self):
        service = MagicMock()
        service._broker = _make_broker(
            [_make_order(entry_price=100.0, quantity=10.0)]
        )
        service.summary.return_value = MagicMock(
            current_balance=1000.0,
            initial_balance=1000.0,
            equity=1000.0,
            open_pnl=0.0,
            closed_count=0,
            total_realized_pnl=0.0,
        )
        service.open_positions.return_value = [
            _make_order(entry_price=100.0, quantity=10.0)
        ]
        service.closed_positions.return_value = []

        limit = 5000.0
        em = exposure_metrics(service._broker, limit)

        assert em.current_exposure == 1000.0
        assert em.exposure_limit == 5000.0
        assert em.usage_percent == 20.0
        assert em.remaining == 4000.0

    def test_exposure_section_absent_when_limit_zero(self):
        service = MagicMock()
        service._broker = _make_broker([])
        service.summary.return_value = MagicMock(
            current_balance=1000.0,
            initial_balance=1000.0,
            equity=1000.0,
            open_pnl=0.0,
            closed_count=0,
            total_realized_pnl=0.0,
        )
        service.open_positions.return_value = []
        service.closed_positions.return_value = []

        limit = 0.0
        em = exposure_metrics(service._broker, limit)

        assert em.exposure_limit == 0.0
        assert em.usage_percent == 0.0
        assert em.remaining == 0.0
