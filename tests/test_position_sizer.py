"""Tests for risk.position_sizer.calculate_position_size."""

import pytest

from risk.position_sizer import calculate_position_size


def test_position_size_risks_exact_budget_on_stop_out():
    """A stop-out should lose exactly balance * risk_percent."""
    # balance 1000, risk 2% => risk amount 20; stop 2% below 100 => distance 2
    qty = calculate_position_size(
        balance=1000,
        risk_percent=2,
        entry_price=100,
        stop_loss_percent=2,
    )
    assert qty == pytest.approx(10.0)

    stop_distance = 100 * (2 / 100)
    assert qty * stop_distance == pytest.approx(1000 * (2 / 100))


def test_position_size_scales_with_risk_budget():
    """Doubling the risk budget doubles the quantity."""
    qty_a = calculate_position_size(1000, 1, 100, 2)
    qty_b = calculate_position_size(1000, 2, 100, 2)
    assert qty_b == pytest.approx(qty_a * 2)


def test_position_size_scales_with_stop_distance():
    """A tighter stop needs a larger quantity for the same risk amount."""
    tight = calculate_position_size(1000, 2, 100, 1)
    wide = calculate_position_size(1000, 2, 100, 2)
    assert tight == pytest.approx(wide * 2)


def test_returns_float_rounded():
    result = calculate_position_size(1000, 2, 100, 2)
    assert isinstance(result, float)


@pytest.mark.parametrize(
    "balance,risk_percent,entry_price,stop_loss_percent",
    [
        (0, 2, 100, 2),
        (1000, 0, 100, 2),
        (1000, 2, 0, 2),
        (1000, 2, 100, 0),
        (-100, 2, 100, 2),
    ],
)
def test_non_positive_inputs_return_zero(
    balance, risk_percent, entry_price, stop_loss_percent
):
    assert (
        calculate_position_size(balance, risk_percent, entry_price, stop_loss_percent)
        == 0
    )
