"""Integration tests for costs in the historical trade pipeline."""

from datetime import datetime, timedelta, timezone

import pytest

import backtest.trades as trades_module
from backtest.analytics import compute_backtest_metrics
from backtest.costs import TradingCostModel, ZERO_COSTS
from backtest.replay import HistoricalBar, SignalRecord
from backtest.trades import simulate_trades
from risk.risk_profile import RiskProfile

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PROFILE = RiskProfile(
    balance=1000.0,
    risk_percent=2.0,
    stop_loss_percent=2.0,
    take_profit_percent=4.0,
)


def _bars(closes):
    return [
        HistoricalBar(
            timestamp=START + timedelta(hours=index),
            close=close,
            volume=1_000.0,
        )
        for index, close in enumerate(closes)
    ]


def _patch_replay(monkeypatch, bars, decisions):
    records = []
    for bar, decision in zip(bars, decisions):
        if decision == "BUY":
            signal, trend = "BUY", "Bullish"
        else:
            signal, trend = "WAIT", "Bullish"
        records.append(
            SignalRecord(
                timestamp=bar.timestamp,
                signal=signal,
                trend=trend,
                confidence=80,
                decision=decision,
            )
        )
    monkeypatch.setattr(
        trades_module,
        "replay_signals",
        lambda received_bars, *, parameters=None: records,
    )


def test_completed_trade_carries_breakdown_and_analytics_infers_it(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    _patch_replay(monkeypatch, bars, ["BUY", "WAIT", "WAIT"])
    model = TradingCostModel(fee_rate=0.001, slippage_rate=0.01)

    trades = simulate_trades("BTC", bars, PROFILE, cost_model=model)
    trade = trades[0]
    breakdown = trade.cost_breakdown

    assert breakdown is not None
    assert breakdown.cost_model == model
    assert breakdown.gross_pnl > breakdown.gross_pnl_after_slippage
    assert breakdown.total_fees > 0.0
    assert breakdown.net_pnl < breakdown.gross_pnl

    metrics = compute_backtest_metrics(trades, PROFILE.balance)

    assert metrics.cost_model == model
    assert metrics.total_realized_pnl == pytest.approx(breakdown.net_pnl)
    assert metrics.total_fees == pytest.approx(breakdown.total_fees)
    assert metrics.total_slippage_cost == pytest.approx(
        breakdown.slippage_cost
    )
    assert metrics.ending_equity == pytest.approx(
        PROFILE.balance + breakdown.net_pnl
    )


def test_explicit_zero_cost_pipeline_preserves_gross_results(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    _patch_replay(monkeypatch, bars, ["BUY", "WAIT", "WAIT"])

    trades = simulate_trades("BTC", bars, PROFILE, cost_model=ZERO_COSTS)
    breakdown = trades[0].cost_breakdown

    assert breakdown is not None
    assert breakdown.net_pnl == breakdown.gross_pnl
    assert breakdown.total_costs == 0.0

    metrics = compute_backtest_metrics(trades, PROFILE.balance)

    assert metrics.cost_model == ZERO_COSTS
    assert metrics.total_realized_pnl == pytest.approx(breakdown.gross_pnl)
    assert metrics.total_costs == 0.0


def test_default_pipeline_output_does_not_attach_costs(monkeypatch):
    bars = _bars([100.0, 110.0, 115.0])
    _patch_replay(monkeypatch, bars, ["BUY", "WAIT", "WAIT"])

    trades = simulate_trades("BTC", bars, PROFILE)

    assert trades[0].cost_breakdown is None


def test_open_trade_has_no_cost_breakdown(monkeypatch):
    bars = _bars([100.0, 110.0, 111.0])
    _patch_replay(monkeypatch, bars, ["BUY", "WAIT", "WAIT"])
    model = TradingCostModel(fee_rate=0.001)

    trades = simulate_trades("BTC", bars, PROFILE, cost_model=model)

    assert trades[0].exit_status is None
    assert trades[0].cost_breakdown is None
