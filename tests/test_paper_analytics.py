"""
Tests for trader.paper.analytics (Sprint 6D).

Pure function tests for compute_overall_performance,
compute_risk_metrics, compute_symbol_performance,
group_by_symbol, and get_recent_trades.
"""

from unittest.mock import MagicMock

import pytest

from trader.paper.analytics import (
    OverallPerformance,
    RiskMetrics,
    SymbolPerformance,
    compute_overall_performance,
    compute_risk_metrics,
    compute_symbol_performance,
    get_recent_trades,
    group_by_symbol,
)


def _make_order(
    symbol="BTC",
    realized_pnl=0.0,
    exit_status="CLOSED_MANUAL",
    exit_at=None,
):
    """Create a minimal mocked PaperOrder with PnL data."""
    order = MagicMock()
    order.symbol = symbol
    order.realized_pnl = realized_pnl
    order.exit_status = MagicMock()
    order.exit_status.value = exit_status
    order.exit_at = exit_at
    return order


# ---------------------------------------------------------------------------
# compute_overall_performance
# ---------------------------------------------------------------------------


class TestComputeOverallPerformance:
    def test_empty_list_returns_zeros(self):
        result = compute_overall_performance([])
        assert result == OverallPerformance(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            total_realized_pnl=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            expectancy=0.0,
        )

    def test_single_win(self):
        orders = [_make_order(symbol="BTC", realized_pnl=500.0)]
        result = compute_overall_performance(orders)
        assert result.total_trades == 1
        assert result.winning_trades == 1
        assert result.losing_trades == 0
        assert result.breakeven_trades == 0
        assert result.win_rate == 100.0
        assert result.total_realized_pnl == 500.0
        assert result.average_win == 500.0
        assert result.average_loss == 0.0
        assert result.largest_win == 500.0
        assert result.largest_loss == 0.0
        assert result.expectancy == 500.0

    def test_single_loss(self):
        orders = [_make_order(symbol="ETH", realized_pnl=-300.0)]
        result = compute_overall_performance(orders)
        assert result.total_trades == 1
        assert result.winning_trades == 0
        assert result.losing_trades == 1
        assert result.breakeven_trades == 0
        assert result.win_rate == 0.0
        assert result.total_realized_pnl == -300.0
        assert result.average_win == 0.0
        assert result.average_loss == 300.0
        assert result.largest_win == 0.0
        assert result.largest_loss == -300.0
        assert result.expectancy == -300.0

    def test_single_breakeven(self):
        orders = [_make_order(symbol="SOL", realized_pnl=0.0)]
        result = compute_overall_performance(orders)
        assert result.total_trades == 1
        assert result.winning_trades == 0
        assert result.losing_trades == 0
        assert result.breakeven_trades == 1
        assert result.win_rate == 0.0
        assert result.total_realized_pnl == 0.0
        assert result.average_win == 0.0
        assert result.average_loss == 0.0
        assert result.largest_win == 0.0
        assert result.largest_loss == 0.0
        assert result.expectancy == 0.0

    def test_mixed_trades(self):
        orders = [
            _make_order(symbol="BTC", realized_pnl=500.0),
            _make_order(symbol="BTC", realized_pnl=-200.0),
            _make_order(symbol="ETH", realized_pnl=300.0),
            _make_order(symbol="ETH", realized_pnl=-100.0),
            _make_order(symbol="SOL", realized_pnl=0.0),
        ]
        result = compute_overall_performance(orders)
        assert result.total_trades == 5
        assert result.winning_trades == 2
        assert result.losing_trades == 2
        assert result.breakeven_trades == 1
        assert result.win_rate == 40.0
        assert result.total_realized_pnl == 500.0
        assert result.average_win == 400.0
        assert result.average_loss == 150.0
        assert result.largest_win == 500.0
        assert result.largest_loss == -200.0
        # Expectancy = (40/100)*400 - (40/100)*150 = 160 - 60 = 100
        assert result.expectancy == 100.0

    def test_multiple_wins_no_losses(self):
        orders = [
            _make_order(symbol="BTC", realized_pnl=100.0),
            _make_order(symbol="ETH", realized_pnl=200.0),
        ]
        result = compute_overall_performance(orders)
        assert result.winning_trades == 2
        assert result.losing_trades == 0
        assert result.win_rate == 100.0
        assert result.average_win == 150.0
        assert result.largest_win == 200.0
        assert result.largest_loss == 0.0
        assert result.expectancy == 150.0

    def test_multiple_losses_no_wins(self):
        orders = [
            _make_order(symbol="BTC", realized_pnl=-100.0),
            _make_order(symbol="ETH", realized_pnl=-200.0),
        ]
        result = compute_overall_performance(orders)
        assert result.winning_trades == 0
        assert result.losing_trades == 2
        assert result.win_rate == 0.0
        assert result.average_loss == 150.0
        assert result.largest_win == 0.0
        assert result.largest_loss == -200.0
        assert result.expectancy == -150.0

    def test_total_pnl_is_sum(self):
        orders = [
            _make_order(symbol="BTC", realized_pnl=1000.0),
            _make_order(symbol="BTC", realized_pnl=-500.0),
        ]
        result = compute_overall_performance(orders)
        assert result.total_realized_pnl == 500.0


# ---------------------------------------------------------------------------
# compute_risk_metrics
# ---------------------------------------------------------------------------


class TestComputeRiskMetrics:
    def test_empty_list(self):
        result = compute_risk_metrics([])
        assert result.profit_factor == 0.0
        assert result.win_loss_ratio is None
        assert result.payoff_ratio is None
        assert result.average_trade_pnl == 0.0
        assert result.current_streak == 0
        assert result.max_consecutive_losses == 0

    def test_single_trade_no_streak_stats(self):
        orders = [_make_order(symbol="BTC", realized_pnl=500.0)]
        result = compute_risk_metrics(orders)
        assert result.profit_factor == float("inf")
        assert result.win_loss_ratio is None
        assert result.payoff_ratio is None
        assert result.average_trade_pnl == 500.0
        assert result.current_streak == 1
        assert result.max_consecutive_losses == 0

    def test_profit_factor_inf_when_no_losses(self):
        orders = [
            _make_order(symbol="BTC", realized_pnl=100.0),
            _make_order(symbol="ETH", realized_pnl=200.0),
        ]
        result = compute_risk_metrics(orders)
        assert result.profit_factor == float("inf")
        assert result.win_loss_ratio is None
        assert result.payoff_ratio is None

    def test_profit_factor_zero_when_no_wins(self):
        orders = [
            _make_order(symbol="BTC", realized_pnl=-100.0),
            _make_order(symbol="ETH", realized_pnl=-200.0),
        ]
        result = compute_risk_metrics(orders)
        assert result.profit_factor == 0.0
        assert result.win_loss_ratio == 0.0

    def test_profit_factor_normal(self):
        orders = [
            _make_order(symbol="BTC", realized_pnl=500.0),
            _make_order(symbol="BTC", realized_pnl=-200.0),
            _make_order(symbol="ETH", realized_pnl=300.0),
            _make_order(symbol="ETH", realized_pnl=-100.0),
        ]
        # sum_wins = 800, abs(sum_losses) = 300, PF = 800/300 = 2.666...
        result = compute_risk_metrics(orders)
        assert result.profit_factor == pytest.approx(2.6667, rel=1e-4)
        assert result.win_loss_ratio == pytest.approx(1.0)
        assert result.payoff_ratio == pytest.approx(2.6667, rel=1e-4)

   
