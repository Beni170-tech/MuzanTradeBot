"""Tests for RiskProfile domain validation."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from risk.risk_profile import RiskProfile, validate_risk_configuration


NUMERIC_FIELDS = (
    "balance",
    "risk_percent",
    "stop_loss_percent",
    "take_profit_percent",
)


def _settings(**overrides) -> SimpleNamespace:
    values = {
        "ACCOUNT_BALANCE": 1500.0,
        "RISK_PERCENT": 1.0,
        "STOP_LOSS_PERCENT": 1.5,
        "TAKE_PROFIT_PERCENT": 3.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_values_are_preserved():
    profile = RiskProfile()

    assert profile.balance == 1000.0
    assert profile.risk_percent == 2.0
    assert profile.stop_loss_percent == 2.0
    assert profile.take_profit_percent == 4.0
    assert profile.risk_amount == pytest.approx(20.0)


def test_profile_is_immutable():
    profile = RiskProfile()

    with pytest.raises(FrozenInstanceError):
        profile.balance = 2000.0


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_values_are_rejected(field, value):
    with pytest.raises(ValueError, match="finite"):
        RiskProfile(**{field: value})


@pytest.mark.parametrize("field", NUMERIC_FIELDS)
@pytest.mark.parametrize("value", [0, -1, -0.5])
def test_non_positive_values_are_rejected(field, value):
    with pytest.raises(ValueError, match="positive"):
        RiskProfile(**{field: value})


@pytest.mark.parametrize("value", [100, 100.0, 150.0])
def test_stop_loss_at_or_above_100_percent_is_rejected(value):
    with pytest.raises(ValueError, match="less than 100"):
        RiskProfile(stop_loss_percent=value)


def test_stop_loss_just_below_100_percent_is_accepted():
    profile = RiskProfile(stop_loss_percent=99.99)

    assert profile.stop_loss_percent == pytest.approx(99.99)


@pytest.mark.parametrize("value", [100, 100.0, 100.01, 1_000.0])
def test_risk_percent_at_or_above_100_is_rejected(value):
    with pytest.raises(ValueError, match="risk_percent must be less than"):
        RiskProfile(risk_percent=value)

    with pytest.raises(ValueError, match="risk_percent must be less than"):
        validate_risk_configuration(
            balance=1000.0,
            risk_percent=value,
            stop_loss_percent=2.0,
            take_profit_percent=4.0,
        )


def test_risk_percent_just_below_100_is_accepted():
    profile = RiskProfile(risk_percent=99.99)

    assert profile.risk_percent == pytest.approx(99.99)

    validate_risk_configuration(
        balance=1000.0,
        risk_percent=99.99,
        stop_loss_percent=2.0,
        take_profit_percent=4.0,
    )


def test_integer_values_are_accepted():
    profile = RiskProfile(
        balance=500,
        risk_percent=1,
        stop_loss_percent=2,
        take_profit_percent=3,
    )

    assert profile.balance == 500
    assert profile.risk_percent == 1


@pytest.mark.parametrize("value", [True, False])
def test_bool_values_are_rejected(value):
    with pytest.raises(TypeError, match="balance"):
        RiskProfile(balance=value)


@pytest.mark.parametrize("value", ["1000", None, [], {}, object()])
def test_non_numeric_values_are_rejected(value):
    with pytest.raises(TypeError):
        RiskProfile(balance=value)


def test_oversized_integer_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        RiskProfile(balance=10**400)


def test_from_settings_uses_configured_values():
    profile = RiskProfile.from_settings(_settings())

    assert profile.balance == 1500.0
    assert profile.risk_percent == 1.0
    assert profile.stop_loss_percent == 1.5
    assert profile.take_profit_percent == 3.0


def test_from_settings_accepts_balance_override():
    profile = RiskProfile.from_settings(_settings(), balance=842.5)

    assert profile.balance == 842.5
    assert profile.risk_percent == 1.0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("RISK_PERCENT", float("nan"), ValueError),
        ("RISK_PERCENT", 100.0, ValueError),
        ("STOP_LOSS_PERCENT", 0.0, ValueError),
        ("TAKE_PROFIT_PERCENT", float("inf"), ValueError),
        ("STOP_LOSS_PERCENT", 100.0, ValueError),
        ("ACCOUNT_BALANCE", "1000", TypeError),
    ],
)
def test_from_settings_rejects_invalid_configuration(field, value, error):
    with pytest.raises(error):
        RiskProfile.from_settings(_settings(**{field: value}))


def test_validate_risk_configuration_matches_profile_rules():
    validate_risk_configuration(
        balance=1000.0,
        risk_percent=2.0,
        stop_loss_percent=2.0,
        take_profit_percent=4.0,
    )

    with pytest.raises(ValueError, match="positive"):
        validate_risk_configuration(
            balance=0.0,
            risk_percent=2.0,
            stop_loss_percent=2.0,
            take_profit_percent=4.0,
        )
    with pytest.raises(ValueError, match="less than 100"):
        validate_risk_configuration(
            balance=1000.0,
            risk_percent=2.0,
            stop_loss_percent=100.0,
            take_profit_percent=4.0,
        )
