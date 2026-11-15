"""Tests for the deterministic historical exit engine."""

from datetime import datetime, timedelta, timezone

import pytest

from backtest.execution import (
    BacktestPosition,
    ExitEvent,
    ExitSimulation,
    check_exit,
    simulate_exit,
)
from backtest.replay import HistoricalBar
from trader.paper.order import PaperOrderStatus

START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _bar(hours, close):
    return HistoricalBar(
        timestamp=START + timedelta(hours=hours),
        close=close,
        volume=1_000.0,
    )


def _long_position(**overrides):
    values = {
        "side": "LONG",
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "take_profit": 120.0,
    }
    values.update(overrides)
    return BacktestPosition(**values)


def _short_position(**overrides):
    values = {
        "side": "SHORT",
        "entry_price": 100.0,
        "stop_loss": 110.0,
        "take_profit": 80.0,
    }
    values.update(overrides)
    return BacktestPosition(**values)


def test_long_stop_loss_exits_at_stored_stop():
    result = simulate_exit(_long_position(), [_bar(0, 89.0)])

    assert result.exit_event == ExitEvent(
        timestamp=START,
        price=90.0,
        status=PaperOrderStatus.STOPPED,
    )
    assert result.exit_event.status.value == "STOPPED"


def test_short_stop_loss_exits_at_stored_stop():
    result = simulate_exit(_short_position(), [_bar(0, 111.0)])

    assert result.exit_event == ExitEvent(
        timestamp=START,
        price=110.0,
        status=PaperOrderStatus.STOPPED,
    )


def test_long_take_profit_exits_at_stored_target():
    result = simulate_exit(_long_position(), [_bar(0, 125.0)])

    assert result.exit_event == ExitEvent(
        timestamp=START,
        price=120.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )
    assert result.exit_event.status.value == "TAKEN_PROFIT"


def test_short_take_profit_exits_at_stored_target():
    result = simulate_exit(_short_position(), [_bar(0, 75.0)])

    assert result.exit_event == ExitEvent(
        timestamp=START,
        price=80.0,
        status=PaperOrderStatus.TAKEN_PROFIT,
    )


def test_stop_loss_takes_precedence_when_both_levels_trigger():
    status = check_exit(
        side="LONG",
        current_price=100.0,
        stop_loss=105.0,
        take_profit=95.0,
    )

    assert status is PaperOrderStatus.STOPPED


def test_long_trailing_activates_strictly_beyond_one_r():
    position = simulate_exit(_long_position(), [_bar(0, 111.0)])

    assert position.exit_event is None
    assert position.trailing_active is True
    assert position.final_stop_loss == 100.0
    assert position.initial_risk_distance == 10.0


def test_short_trailing_activates_strictly_beyond_one_r():
    position = simulate_exit(_short_position(), [_bar(0, 89.0)])

    assert position.exit_event is None
    assert position.trailing_active is True
    assert position.final_stop_loss == 100.0
    assert position.initial_risk_distance == 10.0


@pytest.mark.parametrize(
    ("position", "close", "expected_stop"),
    [
        (_long_position(), 110.0, 90.0),
        (_short_position(), 90.0, 110.0),
    ],
)
def test_exact_one_r_does_not_activate_trailing(
    position, close, expected_stop
):
    result = simulate_exit(position, [_bar(0, close)])

    assert result.exit_event is None
    assert result.trailing_active is False
    assert result.final_stop_loss == expected_stop


def test_long_trailing_stop_moves_up_using_original_risk_distance():
    result = simulate_exit(
        _long_position(),
        [_bar(0, 111.0), _bar(1, 112.0), _bar(2, 105.0)],
    )

    assert result.exit_event is None
    assert result.trailing_active is True
    assert result.final_stop_loss == 102.0
    assert result.initial_risk_distance == 10.0
    assert result.final_take_profit == 120.0


def test_short_trailing_stop_moves_down_using_original_risk_distance():
    result = simulate_exit(
        _short_position(),
        [_bar(0, 89.0), _bar(1, 88.0), _bar(2, 95.0)],
    )

    assert result.exit_event is None
    assert result.trailing_active is True
    assert result.final_stop_loss == 98.0
    assert result.initial_risk_distance == 10.0


def test_long_trailing_stop_never_moves_backward():
    result = simulate_exit(
        _long_position(),
        [_bar(0, 111.0), _bar(1, 105.0), _bar(2, 99.0)],
    )

    assert result.final_stop_loss == 100.0
    assert result.exit_event.price == 100.0
    assert result.exit_event.status is PaperOrderStatus.STOPPED


def test_short_trailing_stop_never_moves_backward():
    result = simulate_exit(
        _short_position(),
        [_bar(0, 89.0), _bar(1, 95.0), _bar(2, 101.0)],
    )

    assert result.final_stop_loss == 100.0
    assert result.exit_event.price == 100.0
    assert result.exit_event.status is PaperOrderStatus.STOPPED


def test_updated_trailing_stop_can_trigger_on_next_bar():
    result = simulate_exit(
        _long_position(),
        [_bar(0, 111.0), _bar(1, 100.0)],
    )

    assert result.bars_processed == 2
    assert result.exit_event.timestamp == START + timedelta(hours=1)
    assert result.exit_event == ExitEvent(
        timestamp=START + timedelta(hours=1),
        price=100.0,
        status=PaperOrderStatus.STOPPED,
    )


def test_exit_uses_historical_bar_timestamp():
    result = simulate_exit(
        _long_position(),
        [_bar(0, 110.0), _bar(1, 109.0), _bar(2, 89.0)],
    )

    expected_timestamp = START + timedelta(hours=2)
    assert result.exit_event.timestamp == expected_timestamp
    assert result.bars_processed == 3


def test_no_exit_returns_final_state():
    result = simulate_exit(_long_position(), [_bar(0, 100.0), _bar(1, 105.0)])

    assert result.exit_event is None
    assert result.final_stop_loss == 90.0
    assert result.final_take_profit == 120.0
    assert result.trailing_active is False
    assert result.bars_processed == 2


def test_identical_inputs_produce_identical_results():
    position = _long_position()
    bars = [_bar(0, 111.0), _bar(1, 112.0), _bar(2, 105.0)]

    first = simulate_exit(position, bars)
    second = simulate_exit(position, bars)

    assert isinstance(first, ExitSimulation)
    assert first == second


@pytest.mark.parametrize(
    "position",
    [
        _long_position(side="SIDEWAYS"),
        _long_position(side=1),
        _long_position(entry_price=0),
        _long_position(entry_price=float("nan")),
        _long_position(entry_price=float("inf")),
        _long_position(entry_price=True),
        _long_position(stop_loss=0),
        _long_position(stop_loss=float("inf")),
        _long_position(take_profit=-1),
        _long_position(take_profit=float("nan")),
        _long_position(initial_risk_distance=0),
        _long_position(initial_risk_distance=-5),
        _long_position(initial_risk_distance=float("nan")),
        _long_position(stop_loss=105.0),
        _long_position(take_profit=95.0),
        _short_position(stop_loss=95.0),
        _short_position(take_profit=105.0),
    ],
)
def test_invalid_position_parameters_are_rejected(position):
    with pytest.raises(ValueError):
        simulate_exit(position, [_bar(0, 100.0)])


@pytest.mark.parametrize(
    "close",
    [0, -1.0, float("nan"), float("inf"), True, "100"],
)
def test_invalid_bar_prices_are_rejected(close):
    with pytest.raises(ValueError):
        simulate_exit(_long_position(), [_bar(0, close)])


def test_non_bar_and_non_sequence_inputs_are_rejected():
    with pytest.raises(TypeError):
        simulate_exit(_long_position(), None)
    with pytest.raises(TypeError):
        simulate_exit(_long_position(), [object()])


def test_invalid_timestamp_and_ordering_are_rejected():
    with pytest.raises(TypeError):
        simulate_exit(
            _long_position(),
            [HistoricalBar(timestamp="2024-01-01", close=100.0, volume=1.0)],
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        simulate_exit(
            _long_position(),
            [_bar(1, 100.0), _bar(0, 100.0)],
        )
