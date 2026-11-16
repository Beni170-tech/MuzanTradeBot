"""Tests for pure historical backtest performance analytics."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from backtest.analytics import BacktestMetrics, compute_backtest_metrics
from backtest.trades import HistoricalTrade
from trader.paper.order import PaperOrderStatus

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
INITIAL_BALANCE = 1000.0


def _closed_trade(
    *,
    side="LONG",
    entry_price=100.0,
    exit_price=110.0,
    quantity=1.0,
    exit_at=None,
    status=PaperOrderStatus.TAKEN_PROFIT,
):
    if exit_at is None:
        exit_at = START + timedelta(hours=2)
    if side.upper() == "LONG":
        stop_loss = 90.0
        take_profit = 120.0
    else:
        stop_loss = 110.0
        take_profit = 80.0
    entry_at = START + timedelta(hours=1)
    return HistoricalTrade(
        symbol="BTC",
        signal_timestamp=START,
        entry_timestamp=entry_at,
        entry_price=entry_price,
        side=side,
        quantity=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        trade_timestamp=entry_at.strftime("%Y-%m-%d %H:%M:%S"),
        exit_timestamp=exit_at,
        exit_price=exit_price,
        exit_status=status,
    )


def _open_trade(**overrides):
    values = {
        "side": "LONG",
        "entry_price": 100.0,
        "quantity": 1.0,
    }
    values.update(overrides)
    trade = _closed_trade(**values)
    return replace(
        trade,
        exit_timestamp=None,
        exit_price=None,
        exit_status=None,
    )


def test_long_profitable_trade_metrics():
    result = compute_backtest_metrics(
        [_closed_trade(entry_price=100.0, exit_price=110.0, quantity=2.0)],
        INITIAL_BALANCE,
    )

    assert result == BacktestMetrics(
        total_trades=1,
        open_trades=0,
        wins=1,
        losses=0,
        breakeven=0,
        win_rate=100.0,
        gross_profit=20.0,
        gross_loss=0.0,
        total_realized_pnl=20.0,
        profit_factor=None,
        ending_equity=1020.0,
        max_drawdown=0.0,
        max_drawdown_percent=0.0,
        total_gross_pnl=20.0,
        net_profit=20.0,
        net_loss=0.0,
        net_profit_factor=None,
        total_fees=0.0,
        total_slippage_cost=0.0,
        total_costs=0.0,
    )


def test_short_profitable_trade_metrics():
    result = compute_backtest_metrics(
        [
            _closed_trade(
                side="SHORT",
                entry_price=110.0,
                exit_price=100.0,
                quantity=2.0,
            )
        ],
        INITIAL_BALANCE,
    )

    assert result.wins == 1
    assert result.gross_profit == 20.0
    assert result.total_realized_pnl == 20.0
    assert result.ending_equity == 1020.0


def test_long_losing_trade_metrics():
    result = compute_backtest_metrics(
        [_closed_trade(entry_price=100.0, exit_price=95.0, quantity=2.0)],
        INITIAL_BALANCE,
    )

    assert result.losses == 1
    assert result.win_rate == 0.0
    assert result.gross_profit == 0.0
    assert result.gross_loss == 10.0
    assert result.total_realized_pnl == -10.0
    assert result.profit_factor == 0.0
    assert result.ending_equity == 990.0
    assert result.max_drawdown == 10.0
    assert result.max_drawdown_percent == pytest.approx(1.0)


def test_short_losing_trade_metrics():
    result = compute_backtest_metrics(
        [
            _closed_trade(
                side="SHORT",
                entry_price=100.0,
                exit_price=105.0,
                quantity=2.0,
            )
        ],
        INITIAL_BALANCE,
    )

    assert result.losses == 1
    assert result.total_realized_pnl == -10.0
    assert result.gross_loss == 10.0
    assert result.ending_equity == 990.0


def test_breakeven_trade_metrics():
    result = compute_backtest_metrics(
        [_closed_trade(entry_price=100.0, exit_price=100.0)],
        INITIAL_BALANCE,
    )

    assert result.breakeven == 1
    assert result.wins == 0
    assert result.losses == 0
    assert result.win_rate == 0.0
    assert result.total_realized_pnl == 0.0
    assert result.profit_factor is None
    assert result.ending_equity == INITIAL_BALANCE
    assert result.max_drawdown == 0.0


def test_open_trades_are_excluded_from_realized_metrics():
    result = compute_backtest_metrics(
        [
            _open_trade(entry_price=80.0),
            _closed_trade(entry_price=100.0, exit_price=110.0),
        ],
        INITIAL_BALANCE,
    )

    assert result.total_trades == 2
    assert result.open_trades == 1
    assert result.wins == 1
    assert result.losses == 0
    assert result.win_rate == 100.0
    assert result.total_realized_pnl == 10.0
    assert result.ending_equity == 1010.0


def test_equity_and_drawdown_use_chronological_exit_order():
    trades = [
        _closed_trade(
            entry_price=100.0,
            exit_price=95.0,
            quantity=10.0,
            exit_at=START + timedelta(hours=3),
        ),
        _closed_trade(
            entry_price=200.0,
            exit_price=100.0,
            exit_at=START + timedelta(hours=2),
        ),
        _closed_trade(
            entry_price=100.0,
            exit_price=200.0,
            exit_at=START + timedelta(hours=1),
        ),
    ]

    result = compute_backtest_metrics(trades, INITIAL_BALANCE)

    # 1000 -> 1100 -> 1000 -> 950
    assert result.ending_equity == 950.0
    assert result.total_realized_pnl == -50.0
    assert result.max_drawdown == 150.0
    assert result.max_drawdown_percent == pytest.approx(150.0 / 1100.0 * 100.0)


def test_equal_exit_timestamps_preserve_input_order():
    same_time = START + timedelta(hours=2)
    ordered = [
        _closed_trade(
            entry_price=100.0,
            exit_price=200.0,
            exit_at=same_time,
        ),
        _closed_trade(
            entry_price=100.0,
            exit_price=70.0,
            exit_at=same_time,
        ),
        _closed_trade(
            entry_price=200.0,
            exit_price=100.0,
            exit_at=same_time,
        ),
        _closed_trade(
            entry_price=100.0,
            exit_price=150.0,
            exit_at=same_time,
        ),
    ]
    reordered = [ordered[1], ordered[0], ordered[2], ordered[3]]

    first = compute_backtest_metrics(ordered, INITIAL_BALANCE)
    second = compute_backtest_metrics(reordered, INITIAL_BALANCE)

    assert first.ending_equity == 1020.0
    assert first.max_drawdown == 130.0
    assert second.ending_equity == 1020.0
    assert second.max_drawdown == 100.0


def test_profit_factor_is_none_without_losses():
    result = compute_backtest_metrics(
        [
            _closed_trade(entry_price=100.0, exit_price=110.0),
            _closed_trade(entry_price=100.0, exit_price=120.0),
        ],
        INITIAL_BALANCE,
    )

    assert result.gross_loss == 0.0
    assert result.profit_factor is None


def test_empty_trade_list_returns_zero_metrics():
    result = compute_backtest_metrics([], INITIAL_BALANCE)

    assert result == BacktestMetrics(
        total_trades=0,
        open_trades=0,
        wins=0,
        losses=0,
        breakeven=0,
        win_rate=0.0,
        gross_profit=0.0,
        gross_loss=0.0,
        total_realized_pnl=0.0,
        profit_factor=None,
        ending_equity=INITIAL_BALANCE,
        max_drawdown=0.0,
        max_drawdown_percent=0.0,
        total_gross_pnl=0.0,
        net_profit=0.0,
        net_loss=0.0,
        net_profit_factor=None,
        total_fees=0.0,
        total_slippage_cost=0.0,
        total_costs=0.0,
    )


@pytest.mark.parametrize(
    "initial_balance",
    [0, -1, float("nan"), float("inf"), float("-inf"), True, "1000", 10**400],
)
def test_invalid_initial_balance_is_rejected(initial_balance):
    with pytest.raises(ValueError):
        compute_backtest_metrics([], initial_balance)


@pytest.mark.parametrize(
    "changes",
    [
        {"exit_timestamp": START},
        {"exit_price": 110.0},
        {"exit_status": PaperOrderStatus.TAKEN_PROFIT},
        {"exit_timestamp": START, "exit_price": 110.0},
        {"exit_timestamp": START, "exit_status": PaperOrderStatus.TAKEN_PROFIT},
        {"exit_price": 110.0, "exit_status": PaperOrderStatus.TAKEN_PROFIT},
    ],
)
def test_inconsistent_exit_fields_are_rejected(changes):
    trade = replace(_open_trade(), **changes)

    with pytest.raises(ValueError, match="all None or all populated"):
        compute_backtest_metrics([trade], INITIAL_BALANCE)


@pytest.mark.parametrize(
    "trade",
    [
        _closed_trade(side="SIDEWAYS"),
        replace(_closed_trade(), side=None),
        _closed_trade(entry_price=0),
        _closed_trade(entry_price=-1.0),
        _closed_trade(entry_price=float("nan")),
        _closed_trade(entry_price=float("inf")),
        _closed_trade(entry_price=True),
        _closed_trade(entry_price="100"),
        _closed_trade(entry_price=10**400),
        _closed_trade(exit_price=0),
        _closed_trade(exit_price=float("nan")),
        _closed_trade(quantity=0),
        _closed_trade(quantity=-1.0),
        _closed_trade(quantity=float("inf")),
        _closed_trade(quantity=True),
        replace(_closed_trade(), stop_loss=0),
        replace(_closed_trade(), take_profit=float("nan")),
    ],
)
def test_invalid_trade_values_are_rejected(trade):
    with pytest.raises((TypeError, ValueError)):
        compute_backtest_metrics([trade], INITIAL_BALANCE)


@pytest.mark.parametrize(
    "trade",
    [
        replace(_closed_trade(), signal_timestamp="2024-01-01"),
        replace(_closed_trade(), entry_timestamp=None),
        replace(_closed_trade(), exit_timestamp="2024-01-01"),
        replace(_closed_trade(), exit_status="TAKEN_PROFIT"),
        replace(
            _closed_trade(),
            exit_status=PaperOrderStatus.PENDING,
        ),
        replace(
            _closed_trade(),
            exit_status=PaperOrderStatus.OPEN,
        ),
    ],
)
def test_invalid_timestamps_and_statuses_are_rejected(trade):
    with pytest.raises((TypeError, ValueError)):
        compute_backtest_metrics([trade], INITIAL_BALANCE)


def test_incomparable_exit_timestamps_are_rejected():
    aware = _closed_trade(exit_at=START + timedelta(hours=1))
    naive = _closed_trade(
        exit_at=datetime(2024, 1, 1, 3),
    )

    with pytest.raises(ValueError, match="mutually comparable"):
        compute_backtest_metrics([aware, naive], INITIAL_BALANCE)


@pytest.mark.parametrize("trade_timestamp", ["", "   ", None, 123])
def test_invalid_trade_timestamp_is_rejected(trade_timestamp):
    trade = replace(
        _closed_trade(),
        trade_timestamp=trade_timestamp,
    )

    with pytest.raises((TypeError, ValueError)):
        compute_backtest_metrics([trade], INITIAL_BALANCE)


def test_exit_before_entry_is_rejected():
    trade = replace(
        _closed_trade(),
        exit_timestamp=START - timedelta(hours=1),
    )

    with pytest.raises(ValueError, match="earlier than entry_timestamp"):
        compute_backtest_metrics([trade], INITIAL_BALANCE)


def test_incomparable_entry_and_exit_timestamps_are_rejected():
    trade = replace(
        _closed_trade(),
        entry_timestamp=START,
        exit_timestamp=datetime(2024, 1, 1, 3),
    )

    with pytest.raises(ValueError, match="mutually comparable"):
        compute_backtest_metrics([trade], INITIAL_BALANCE)


def test_total_pnl_overflow_is_rejected_safely():
    huge_profit = _closed_trade(
        entry_price=1.0,
        exit_price=2.0,
        quantity=1e308,
    )

    with pytest.raises(ValueError, match="aggregate PnL overflowed"):
        compute_backtest_metrics([huge_profit, huge_profit], 1000.0)


def test_drawdown_percentage_overflow_is_rejected_safely():
    huge_loss = _closed_trade(
        side="SHORT",
        entry_price=1e-308,
        exit_price=1.0,
        quantity=1e308,
    )

    with pytest.raises(ValueError, match="drawdown percentage is not finite"):
        compute_backtest_metrics([huge_loss], 1e-308)


def test_profit_factor_overflow_is_rejected_safely():
    huge_profit = _closed_trade(
        entry_price=1.0,
        exit_price=2.0,
        quantity=1e308,
        exit_at=START + timedelta(hours=1),
    )
    tiny_loss = _closed_trade(
        side="SHORT",
        entry_price=1e-308,
        exit_price=1.5e-308,
        quantity=1.0,
        exit_at=START + timedelta(hours=2),
    )

    with pytest.raises(ValueError, match="profit factor"):
        compute_backtest_metrics([huge_profit, tiny_loss], INITIAL_BALANCE)


def test_non_trade_and_non_sequence_inputs_are_rejected():
    with pytest.raises(TypeError):
        compute_backtest_metrics(None, INITIAL_BALANCE)
    with pytest.raises(TypeError):
        compute_backtest_metrics([object()], INITIAL_BALANCE)
