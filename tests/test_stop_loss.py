"""Tests for risk.stop_loss.calculate_exit_levels."""

import pytest

from risk.stop_loss import calculate_exit_levels


def test_long_levels_stop_below_take_profit_above():
    levels = calculate_exit_levels(100, 2, 4, side="LONG")
    assert levels == {"stop_loss": 98.0, "take_profit": 104.0}


def test_short_levels_stop_above_take_profit_below():
    levels = calculate_exit_levels(100, 2, 4, side="SHORT")
    assert levels == {"stop_loss": 102.0, "take_profit": 96.0}


def test_long_is_default_side():
    assert calculate_exit_levels(100, 2, 4) == {"stop_loss": 98.0, "take_profit": 104.0}


def test_side_is_case_insensitive():
    assert calculate_exit_levels(100, 2, 4, side="short")["stop_loss"] == 102.0


def test_cheap_asset_never_gets_zero_stop():
    """Rounding must not collapse sub-$1 stops to 0.0."""
    levels = calculate_exit_levels(0.001, 2, 4)
    assert levels["stop_loss"] > 0
    assert levels["take_profit"] > 0
    assert levels["stop_loss"] < 0.001
    assert levels["take_profit"] > 0.001


def test_large_price_keeps_two_decimals():
    levels = calculate_exit_levels(65000.123, 2, 4)
    assert levels["stop_loss"] == round(65000.123 * 0.98, 2)


@pytest.mark.parametrize(
    "entry,stop_percent,take_percent,side",
    [
        (0, 2, 4, "LONG"),
        (100, 0, 4, "LONG"),
        (100, 2, 0, "SHORT"),
        (-5, 2, 4, "LONG"),
    ],
)
def test_non_positive_inputs_return_zeros(entry, stop_percent, take_percent, side):
    assert calculate_exit_levels(entry, stop_percent, take_percent, side) == {
        "stop_loss": 0,
        "take_profit": 0,
    }


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        calculate_exit_levels(100, 2, 4, side="sideways")
