"""Focused tests for the paper-trading entry gates module.

These tests exercise ``trader.paper.entry_gates.evaluate_entry_gates``
in isolation: no Telegram, no settings, no handler imports. The module is
expected to be a pure, read-only evaluator over an execution backend.
"""

import copy
import inspect
from types import SimpleNamespace

import pytest

from trader.paper.analytics import ExposureMetrics
from trader.paper.daily_risk import DailyRiskMetrics
from trader.paper.max_open_positions import MaxOpenPositionsMetrics
from trader.paper.order import PaperOrderSide
from trader.paper.portfolio_risk import PortfolioRiskMetrics
from trader.paper.entry_gates import (
    RULE_DAILY_LOSS,
    RULE_INVALID_TRADE,
    RULE_MAX_OPEN_POSITIONS,
    RULE_PORTFOLIO_EXPOSURE,
    RULE_PORTFOLIO_RISK,
    RULE_SYMBOL_EXPOSURE,
    RULE_SYMBOL_POSITIONS,
    EntryGateMetrics,
    EntryGateOutcome,
    collect_entry_gate_metrics,
    evaluate_entry_gates,
)
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Backend:
    """Minimal read-only execution backend for gate evaluation.

    Tracks reads so tests can prove disabled gates are not consulted and
    that evaluation stops at the first rejection.
    """

    def __init__(self, positions=None, closed_positions=None):
        self.positions = list(positions or [])
        self.closed_positions = list(closed_positions or [])
        self.open_position_reads = 0
        self.closed_position_reads = 0

    @property
    def account(self):
        self.closed_position_reads += 1
        return SimpleNamespace(closed_positions=self.closed_positions)

    def get_open_positions(self):
        self.open_position_reads += 1
        return list(self.positions)


def _make_position(
    symbol="BTC",
    entry_price=100.0,
    quantity=10.0,
    stop_loss=98.0,
    side="LONG",
    mark_price=None,
):
    """Create a minimal open-position shape.

    A position is marked at its entry price unless a mark is supplied, so
    it carries a usable mark with zero floating PnL: the shape the daily
    brake requires before it will measure the floating side of a day.
    """
    mark = entry_price if mark_price is None else mark_price
    position = SimpleNamespace(
        symbol=symbol,
        entry_price=entry_price,
        quantity=quantity,
        stop_loss=stop_loss,
        side=PaperOrderSide(side),
        mark_price=mark,
    )
    sign = 1.0 if side == "LONG" else -1.0
    position.unrealized_pnl_for = (
        lambda price: (price - entry_price) * quantity * sign
    )
    return position


def _make_trade(
    symbol="BTC",
    price=100.0,
    position_size=10.0,
    stop_loss=98.0,
    side="LONG",
):
    """Create a real Trade with a valid stop for risk/exposure math."""
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=price * 1.04 if side == "LONG" else price * 0.96,
        side=side,
        approved=True,
    )


def _limits(**overrides):
    """Every gate limit defaulted to disabled."""
    config = {
        "portfolio_risk_limit": 0.0,
        "portfolio_exposure_limit": 0.0,
        "daily_loss_limit": 0.0,
        "max_open_positions": 0,
        "max_positions_per_symbol": 0,
        "max_exposure_per_symbol": 0.0,
    }
    config.update(overrides)
    return config


def _evaluate(backend, trade, **limits):
    """Evaluate gates with every limit defaulted to disabled."""
    return evaluate_entry_gates(backend, trade, **_limits(**limits))


def _collect(backend, **limits):
    """Collect gate metrics with every limit defaulted to disabled."""
    return collect_entry_gate_metrics(backend, **_limits(**limits))


def _closed_today(realized_pnl, day="2026-01-15"):
    return SimpleNamespace(
        realized_pnl=realized_pnl, exit_at=f"{day} 10:00:00"
    )


# ---------------------------------------------------------------------------
# Module purity
# ---------------------------------------------------------------------------


class TestModulePurity:
    def test_no_telegram_or_settings_imports(self):
        import trader.paper.entry_gates as module

        source = inspect.getsource(module)
        assert "telegram" not in source
        assert "config.settings" not in source
        assert "from config" not in source

    def test_no_io_imports(self):
        import trader.paper.entry_gates as module

        source = inspect.getsource(module)
        for forbidden in ("import os", "import io", "requests", "urllib"):
            assert forbidden not in source

    def test_outcome_is_frozen(self):
        outcome = EntryGateOutcome(allowed=True)
        with pytest.raises(Exception):
            outcome.allowed = False


# ---------------------------------------------------------------------------
# All gates disabled
# ---------------------------------------------------------------------------


class TestAllGatesDisabled:
    def test_allows_and_skips_account_reads(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(backend, _make_trade())

        assert outcome.allowed is True
        assert outcome.rule is None
        assert outcome.detail == ""
        assert outcome.metrics is None
        assert backend.open_position_reads == 0
        assert backend.closed_position_reads == 0

    def test_no_limits_with_many_positions_still_allows(self):
        backend = _Backend([_make_position() for _ in range(25)])
        assert _evaluate(backend, _make_trade()).allowed is True

    @staticmethod
    def _position_state(positions):
        return [
            (
                p.symbol,
                p.entry_price,
                p.quantity,
                p.stop_loss,
                p.side,
            )
            for p in positions
        ]

    def test_does_not_mutate_backend_or_positions(self):
        positions = [_make_position(), _make_position(symbol="ETH")]
        backend = _Backend(positions, [_closed_today(-10.0)])
        trade = _make_trade()
        positions_snapshot = self._position_state(positions)
        trade_snapshot = copy.deepcopy(trade.__dict__)

        _evaluate(backend, trade)

        assert self._position_state(backend.positions) == positions_snapshot
        assert vars(trade) == trade_snapshot


# ---------------------------------------------------------------------------
# Portfolio risk gate
# ---------------------------------------------------------------------------


class TestPortfolioRiskGate:
    def test_rejects_when_prospective_exceeds_limit(self):
        # current risk 20 (100-98)*10, trade risk 20 -> 40 > 30.
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), portfolio_risk_limit=30.0
        )

        assert outcome.allowed is False
        assert outcome.rule == RULE_PORTFOLIO_RISK
        assert "exceeds portfolio risk limit" in outcome.detail
        assert outcome.metrics is not None

    def test_allows_at_exact_limit(self):
        # Prospective risk exactly 40.
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), portfolio_risk_limit=40.0
        )
        assert outcome.allowed is True

    def test_rejects_just_above_limit(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), portfolio_risk_limit=39.99
        )
        assert outcome.allowed is False
        assert outcome.rule == RULE_PORTFOLIO_RISK


# ---------------------------------------------------------------------------
# Portfolio exposure gate
# ---------------------------------------------------------------------------


class TestPortfolioExposureGate:
    def test_rejects_when_prospective_exceeds_limit(self):
        # Current 1000, requested 1000 -> 2000 > 1500.
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), portfolio_exposure_limit=1500.0
        )

        assert outcome.allowed is False
        assert outcome.rule == RULE_PORTFOLIO_EXPOSURE
        assert "exceeds portfolio exposure limit" in outcome.detail
        assert outcome.metrics is not None

    def test_allows_at_exact_limit(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), portfolio_exposure_limit=2000.0
        )
        assert outcome.allowed is True

    def test_rejection_reads_positions_once(self):
        # The current exposure and the requested notional must come from
        # the same broker snapshot (one read), matching the original
        # check's arithmetic exactly.
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), portfolio_exposure_limit=1500.0
        )

        assert outcome.rule == RULE_PORTFOLIO_EXPOSURE
        assert backend.open_position_reads == 1

    def test_allows_when_only_current_exposure_matches_limit(self):
        # Requested notional is returned as zero-sized trade risk budget.
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend,
            _make_trade(position_size=0.0),
            portfolio_exposure_limit=1500.0,
        )
        assert outcome.allowed is True


# ---------------------------------------------------------------------------
# Daily loss gate
# ---------------------------------------------------------------------------


class TestDailyLossGate:
    def test_rejects_when_loss_limit_reached_exactly(self):
        backend = _Backend(
            closed_positions=[_closed_today(-100.0)]
        )
        outcome = _evaluate(
            backend,
            _make_trade(),
            daily_loss_limit=100.0,
            day="2026-01-15",
        )

        assert outcome.allowed is False
        assert outcome.rule == RULE_DAILY_LOSS
        assert "reached the daily loss limit" in outcome.detail
        assert outcome.metrics is not None

    def test_allows_just_inside_limit(self):
        backend = _Backend(
            closed_positions=[_closed_today(-99.99)]
        )
        outcome = _evaluate(
            backend,
            _make_trade(),
            daily_loss_limit=100.0,
            day="2026-01-15",
        )
        assert outcome.allowed is True

    def test_profitable_day_increases_budget(self):
        backend = _Backend(
            closed_positions=[_closed_today(-150.0), _closed_today(75.0)]
        )
        outcome = _evaluate(
            backend,
            _make_trade(),
            daily_loss_limit=100.0,
            day="2026-01-15",
        )
        assert outcome.allowed is True

    def test_other_day_does_not_trip(self):
        backend = _Backend(
            closed_positions=[_closed_today(-500.0, day="2026-01-14")]
        )
        outcome = _evaluate(
            backend,
            _make_trade(),
            daily_loss_limit=100.0,
            day="2026-01-15",
        )
        assert outcome.allowed is True


# ---------------------------------------------------------------------------
# Maximum open positions gate
# ---------------------------------------------------------------------------


class TestMaxOpenPositionsGate:
    def test_rejects_at_limit(self):
        backend = _Backend([_make_position(), _make_position(symbol="ETH")])
        outcome = _evaluate(
            backend, _make_trade(), max_open_positions=2
        )

        assert outcome.allowed is False
        assert outcome.rule == RULE_MAX_OPEN_POSITIONS
        assert "reached the maximum of 2" in outcome.detail
        assert outcome.metrics is not None

    def test_allows_below_limit(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), max_open_positions=2
        )
        assert outcome.allowed is True

    def test_rejects_above_limit(self):
        backend = _Backend([_make_position() for _ in range(4)])
        outcome = _evaluate(
            backend, _make_trade(), max_open_positions=3
        )
        assert outcome.allowed is False
        assert outcome.rule == RULE_MAX_OPEN_POSITIONS


# ---------------------------------------------------------------------------
# Symbol concentration gate
# ---------------------------------------------------------------------------


class TestSymbolConcentrationGate:
    def test_count_cap_rejects_at_limit(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), max_positions_per_symbol=1
        )

        assert outcome.allowed is False
        assert outcome.rule == RULE_SYMBOL_POSITIONS
        assert "reached the maximum of 1" in outcome.detail
        assert outcome.metrics is not None

    def test_count_cap_allows_below_limit(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), max_positions_per_symbol=2
        )
        assert outcome.allowed is True

    def test_count_cap_ignores_other_symbols(self):
        backend = _Backend([_make_position(symbol="ETH")])
        outcome = _evaluate(
            backend, _make_trade(symbol="BTC"), max_positions_per_symbol=1
        )
        assert outcome.allowed is True

    def test_exposure_cap_rejects_above_limit(self):
        # Current 1000 + requested 1000 = 2000 > 1500.
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), max_exposure_per_symbol=1500.0
        )

        assert outcome.allowed is False
        assert outcome.rule == RULE_SYMBOL_EXPOSURE
        assert "exceeds the per-symbol exposure limit" in outcome.detail

    def test_exposure_cap_allows_at_exact_limit(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend, _make_trade(), max_exposure_per_symbol=2000.0
        )
        assert outcome.allowed is True

    def test_count_cap_takes_precedence_over_exposure(self):
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend,
            _make_trade(),
            max_positions_per_symbol=1,
            max_exposure_per_symbol=10.0,
        )
        assert outcome.rule == RULE_SYMBOL_POSITIONS


# ---------------------------------------------------------------------------
# Daily loss: floating PnL and unmarked positions
# ---------------------------------------------------------------------------


class TestDailyLossFloatingPnl:
    def test_floating_loss_alone_rejects(self):
        # -150 floating, no settled PnL, 100 daily budget.
        backend = _Backend([_make_position(mark_price=85.0)])
        outcome = _evaluate(backend, _make_trade(), daily_loss_limit=100.0)

        assert outcome.allowed is False
        assert outcome.rule == RULE_DAILY_LOSS
        assert "floating" in outcome.detail
        assert outcome.metrics.day_loss == -150.0

    def test_floating_loss_under_limit_allows(self):
        backend = _Backend([_make_position(mark_price=99.0)])
        outcome = _evaluate(backend, _make_trade(), daily_loss_limit=100.0)

        assert outcome.allowed is True
        assert outcome.rule is None

    def test_unpriced_position_rejects_and_names_it(self):
        position = _make_position(symbol="BTC")
        position.mark_price = None
        backend = _Backend([position])

        outcome = _evaluate(backend, _make_trade(), daily_loss_limit=100.0)

        assert outcome.allowed is False
        assert outcome.rule == RULE_DAILY_LOSS
        assert "no usable market mark" in outcome.detail
        assert "BTC" in outcome.detail
        assert outcome.metrics.unpriced_symbols == ("BTC",)

    def test_unpriced_position_never_masks_an_earlier_gate(self):
        position = _make_position()
        position.mark_price = None
        backend = _Backend([position])

        outcome = _evaluate(
            backend,
            _make_trade(),
            portfolio_risk_limit=1.0,
            daily_loss_limit=100.0,
        )

        assert outcome.rule == RULE_PORTFOLIO_RISK


# ---------------------------------------------------------------------------
# Gate precedence and stop-at-first-rejection
# ---------------------------------------------------------------------------


ALL_REJECTING = {
    "portfolio_risk_limit": 1.0,
    "portfolio_exposure_limit": 1.0,
    "daily_loss_limit": 1.0,
    "max_open_positions": 1,
    "max_positions_per_symbol": 1,
    "max_exposure_per_symbol": 1.0,
    # Pin the daily brake to the day the losing position closed on.
    "day": "2026-01-15",
}


def _rejecting_backend():
    return _Backend(
        [_make_position()],
        [_closed_today(-100.0)],
    )


class TestGatePrecedence:
    def test_portfolio_risk_wins_when_all_reject(self):
        outcome = _evaluate(
            _rejecting_backend(), _make_trade(), **ALL_REJECTING
        )
        assert outcome.rule == RULE_PORTFOLIO_RISK

    def test_exposure_next(self):
        outcome = _evaluate(
            _rejecting_backend(),
            _make_trade(),
            **{**ALL_REJECTING, "portfolio_risk_limit": 0.0},
        )
        assert outcome.rule == RULE_PORTFOLIO_EXPOSURE

    def test_daily_loss_next(self):
        outcome = _evaluate(
            _rejecting_backend(),
            _make_trade(),
            **{
                **ALL_REJECTING,
                "portfolio_risk_limit": 0.0,
                "portfolio_exposure_limit": 0.0,
            },
        )
        assert outcome.rule == RULE_DAILY_LOSS

    def test_max_open_next(self):
        outcome = _evaluate(
            _rejecting_backend(),
            _make_trade(),
            **{
                **ALL_REJECTING,
                "portfolio_risk_limit": 0.0,
                "portfolio_exposure_limit": 0.0,
                "daily_loss_limit": 0.0,
            },
        )
        assert outcome.rule == RULE_MAX_OPEN_POSITIONS

    def test_symbol_concentration_last(self):
        outcome = _evaluate(
            _rejecting_backend(),
            _make_trade(),
            **{
                **ALL_REJECTING,
                "portfolio_risk_limit": 0.0,
                "portfolio_exposure_limit": 0.0,
                "daily_loss_limit": 0.0,
                "max_open_positions": 0,
            },
        )
        assert outcome.rule == RULE_SYMBOL_POSITIONS

    def test_stops_at_first_rejection(self):
        # The risk gate always rejects here; neither the account's
        # closed positions (daily gate) nor the per-symbol metrics are
        # needed, so those reads must never happen.
        backend = _Backend([_make_position()])
        outcome = _evaluate(
            backend,
            _make_trade(),
            **{**ALL_REJECTING, "daily_loss_limit": 1.0},
        )

        assert outcome.rule == RULE_PORTFOLIO_RISK
        assert backend.closed_position_reads == 0

    def test_reports_one_outcome_only(self):
        outcome = _evaluate(
            _rejecting_backend(), _make_trade(), **ALL_REJECTING
        )
        assert isinstance(outcome, EntryGateOutcome)
        assert outcome.allowed is False


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


class TestInvalidTrade:
    def test_none_trade_fails_closed(self):
        backend = _Backend()
        outcome = _evaluate(backend, None)

        assert outcome.allowed is False
        assert outcome.rule == RULE_INVALID_TRADE
        assert backend.open_position_reads == 0
        assert backend.closed_position_reads == 0


# ---------------------------------------------------------------------------
# Collector (/stats path)
# ---------------------------------------------------------------------------


ALL_ENABLED = _limits(
    portfolio_risk_limit=500.0,
    portfolio_exposure_limit=5000.0,
    daily_loss_limit=100.0,
    max_open_positions=5,
    max_positions_per_symbol=2,
    max_exposure_per_symbol=5000.0,
    day="2026-01-15",
)


class TestCollectEntryGateMetrics:
    def test_returns_frozen_container(self):
        metrics = _collect(_Backend())

        assert isinstance(metrics, EntryGateMetrics)
        with pytest.raises(Exception):
            metrics.daily_loss = None

    def test_disabled_gates_are_none_and_read_nothing(self):
        backend = _Backend(
            [_make_position()], [_closed_today(-10.0)]
        )
        metrics = _collect(backend)

        assert metrics.portfolio_risk is None
        assert metrics.portfolio_exposure is None
        assert metrics.daily_loss is None
        assert metrics.max_open_positions is None
        assert metrics.symbol_concentration is None
        assert backend.open_position_reads == 0
        assert backend.closed_position_reads == 0

    def test_reports_every_enabled_gate(self):
        backend = _Backend([_make_position()], [_closed_today(-10.0)])
        metrics = _collect(backend, **ALL_ENABLED)

        assert isinstance(metrics.portfolio_risk, PortfolioRiskMetrics)
        assert isinstance(metrics.portfolio_exposure, ExposureMetrics)
        assert isinstance(metrics.daily_loss, DailyRiskMetrics)
        assert isinstance(metrics.max_open_positions, MaxOpenPositionsMetrics)
        assert metrics.symbol_concentration is not None

    def test_no_short_circuit_when_first_gate_would_reject(self):
        # The risk gate would reject this trade, yet every other gate
        # still reports its own snapshot.
        backend = _Backend([_make_position()], [_closed_today(-10.0)])
        outcome = _evaluate(
            backend,
            _make_trade(),
            **{**ALL_ENABLED, "portfolio_risk_limit": 1.0},
        )
        assert outcome.allowed is False

        metrics = _collect(
            backend, **{**ALL_ENABLED, "portfolio_risk_limit": 1.0}
        )

        assert metrics.portfolio_risk is not None
        assert metrics.portfolio_risk.risk_limit == 1.0
        assert metrics.portfolio_exposure is not None
        assert metrics.daily_loss is not None
        assert metrics.max_open_positions is not None
        assert metrics.symbol_concentration is not None

    def test_only_configured_section_is_populated(self):
        metrics = _collect(_Backend([_make_position()]), max_open_positions=3)

        assert metrics.max_open_positions is not None
        assert metrics.max_open_positions.limit == 3
        assert metrics.max_open_positions.status == "ok"
        assert metrics.portfolio_risk is None
        assert metrics.portfolio_exposure is None
        assert metrics.daily_loss is None
        assert metrics.symbol_concentration is None

    def test_daily_day_is_passed_through(self):
        backend = _Backend(
            [],
            [
                _closed_today(-100.0, day="2026-01-15"),
                _closed_today(-50.0, day="2026-01-16"),
            ],
        )
        metrics = _collect(
            backend, daily_loss_limit=100.0, day="2026-01-15"
        )

        assert metrics.daily_loss is not None
        assert metrics.daily_loss.day == "2026-01-15"
        assert metrics.daily_loss.realized_pnl_today == -100.0
        assert metrics.daily_loss.breached is True

    def test_symbol_concentration_covers_every_open_symbol(self):
        backend = _Backend(
            [
                _make_position(symbol="BTC", entry_price=100.0, quantity=10.0),
                _make_position(symbol="ETH", entry_price=200.0, quantity=10.0),
            ]
        )
        metrics = _collect(
            backend,
            max_positions_per_symbol=2,
            max_exposure_per_symbol=10_000.0,
        )

        assert metrics.symbol_concentration is not None
        assert [cm.symbol for cm in metrics.symbol_concentration] == [
            "ETH",
            "BTC",
        ]
        assert metrics.symbol_concentration[0].symbol_exposure == 2000.0

    def test_does_not_mutate_backend_or_positions(self):
        positions = [_make_position(), _make_position(symbol="ETH")]
        backend = _Backend(positions, [_closed_today(-10.0)])
        snapshot = [
            (p.symbol, p.entry_price, p.quantity, p.stop_loss, p.side)
            for p in positions
        ]

        _collect(backend, **ALL_ENABLED)

        assert [
            (p.symbol, p.entry_price, p.quantity, p.stop_loss, p.side)
            for p in backend.positions
        ] == snapshot
