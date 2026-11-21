"""Tests for conservative Guardian portfolio defaults and strict parsing.

Environment-derived settings are computed at import time, so these tests
run small subprocesses with a controlled environment instead of reloading
the module. That keeps the rest of the suite's imported ``settings``
object untouched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from trader.paper.entry_gates import (
    RULE_MAX_OPEN_POSITIONS,
    evaluate_entry_gates,
)
from trader.paper.order import PaperOrderSide
from trader.trade import Trade

ROOT = Path(__file__).resolve().parents[1]

LIMIT_NAMES = (
    "PORTFOLIO_RISK_LIMIT",
    "PORTFOLIO_EXPOSURE_LIMIT",
    "PORTFOLIO_DAILY_LOSS_LIMIT",
    "PORTFOLIO_MAX_OPEN_POSITIONS",
    "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
    "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
)

#: Core Guardian risk inputs. They are not portfolio limits, but every
#: limit and every position size is derived from them, so they are parsed
#: strictly too.
CORE_RISK_NAMES = (
    "ACCOUNT_BALANCE",
    "RISK_PERCENT",
    "STOP_LOSS_PERCENT",
    "TAKE_PROFIT_PERCENT",
)

DEFAULT_CORE_VALUES = {
    "ACCOUNT_BALANCE": 1000.0,
    "RISK_PERCENT": 2.0,
    "STOP_LOSS_PERCENT": 2.0,
    "TAKE_PROFIT_PERCENT": 4.0,
}

DEFAULT_VALUES = {
    "PORTFOLIO_RISK_LIMIT": 60.0,
    "PORTFOLIO_EXPOSURE_LIMIT": 3000.0,
    "PORTFOLIO_DAILY_LOSS_LIMIT": 30.0,
    "PORTFOLIO_MAX_OPEN_POSITIONS": 3,
    "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL": 1,
    "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL": 2000.0,
}

#: ``evaluate_entry_gates`` keyword for each limit variable.
GATE_PARAMETERS = {
    "PORTFOLIO_RISK_LIMIT": "portfolio_risk_limit",
    "PORTFOLIO_EXPOSURE_LIMIT": "portfolio_exposure_limit",
    "PORTFOLIO_DAILY_LOSS_LIMIT": "daily_loss_limit",
    "PORTFOLIO_MAX_OPEN_POSITIONS": "max_open_positions",
    "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL": "max_positions_per_symbol",
    "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL": "max_exposure_per_symbol",
}

_READ_LIMITS = (
    "import json\n"
    "from config.settings import settings as s\n"
    f"names = {LIMIT_NAMES!r}\n"
    "print(json.dumps({name: getattr(s, name) for name in names}))\n"
)

_READ_CORE = (
    "import json\n"
    "from config.settings import settings as s, validate_settings\n"
    f"names = {CORE_RISK_NAMES!r}\n"
    "print(json.dumps({\n"
    "    'values': {name: getattr(s, name) for name in names},\n"
    "    'errors': [\n"
    "        [issue.severity, issue.setting, issue.message]\n"
    "        for issue in validate_settings()\n"
    "        if issue.severity == 'error'\n"
    "    ],\n"
    "}))\n"
)

_READ_ERRORS = (
    "import json\n"
    "from config.settings import validate_settings\n"
    "print(json.dumps([\n"
    "    [issue.severity, issue.setting, issue.message]\n"
    "    for issue in validate_settings()\n"
    "]))\n"
)


def _environment(**overrides: str) -> dict[str, str]:
    """Return a clean environment with no inherited risk configuration."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in LIMIT_NAMES
        and key not in CORE_RISK_NAMES
    }
    env.update(overrides)
    return env


def _run(code: str, **overrides: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=_environment(**overrides),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _errors(overrides: dict[str, str]) -> list[list[str]]:
    payload = json.loads(_run(_READ_ERRORS, **overrides))
    return [item for item in payload if item[0] == "error"]


def _core(overrides: dict[str, str]) -> dict:
    """Read the core risk values and startup errors in one subprocess."""
    return json.loads(_run(_READ_CORE, **overrides))


class TestConservativeDefaults:
    def test_unset_limits_use_conservative_defaults(self):
        assert json.loads(_run(_READ_LIMITS)) == DEFAULT_VALUES

    def test_defaults_scale_with_account_balance(self):
        values = json.loads(
            _run(_READ_LIMITS, ACCOUNT_BALANCE="2000")
        )
        assert values["PORTFOLIO_RISK_LIMIT"] == 120.0
        assert values["PORTFOLIO_EXPOSURE_LIMIT"] == 6000.0
        assert values["PORTFOLIO_DAILY_LOSS_LIMIT"] == 60.0
        assert values["PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL"] == 4000.0

    def test_explicit_zero_still_disables_every_limit(self):
        values = json.loads(
            _run(_READ_LIMITS, **{name: "0" for name in LIMIT_NAMES})
        )
        assert values == {name: 0 for name in LIMIT_NAMES}

    def test_blank_value_uses_the_default_not_zero(self):
        values = json.loads(_run(_READ_LIMITS, PORTFOLIO_RISK_LIMIT=""))
        assert values["PORTFOLIO_RISK_LIMIT"] == 60.0

    def test_partial_configuration_keeps_other_defaults(self):
        values = json.loads(
            _run(_READ_LIMITS, PORTFOLIO_MAX_OPEN_POSITIONS="5")
        )
        assert values["PORTFOLIO_MAX_OPEN_POSITIONS"] == 5
        assert values["PORTFOLIO_RISK_LIMIT"] == 60.0


class TestMalformedLimitValues:
    def test_malformed_float_is_a_startup_error_naming_the_variable(self):
        errors = _errors({"PORTFOLIO_RISK_LIMIT": "abc"})
        assert [(severity, name) for severity, name, _ in errors] == [
            ("error", "PORTFOLIO_RISK_LIMIT")
        ]
        assert "valid number" in errors[0][2]

    def test_malformed_int_is_a_startup_error_naming_the_variable(self):
        errors = _errors({"PORTFOLIO_MAX_OPEN_POSITIONS": "3.5"})
        assert [(severity, name) for severity, name, _ in errors] == [
            ("error", "PORTFOLIO_MAX_OPEN_POSITIONS")
        ]
        assert "whole number" in errors[0][2]

    def test_non_finite_value_is_a_startup_error(self):
        errors = _errors({"PORTFOLIO_EXPOSURE_LIMIT": "nan"})
        assert [(severity, name) for severity, name, _ in errors] == [
            ("error", "PORTFOLIO_EXPOSURE_LIMIT")
        ]

    def test_main_refuses_to_start_and_names_the_variable(self):
        env = _environment(PORTFOLIO_RISK_LIMIT="not-a-number")
        env.pop("TELEGRAM_BOT_TOKEN", None)
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 1
        assert "Invalid configuration; refusing to start" in completed.stdout
        assert "PORTFOLIO_RISK_LIMIT" in completed.stdout


class TestMalformedCoreRiskValues:
    """The four core risk inputs must never fall back silently.

    Task 30 made the six portfolio limits strict. These inputs are not
    limits, but every limit and every position size is derived from
    them, so a mistyped value there rescales the whole risk budget. It
    must block startup and name the variable instead.
    """

    def test_unset_core_values_use_the_documented_defaults(self):
        payload = _core({})
        assert payload["values"] == DEFAULT_CORE_VALUES
        assert payload["errors"] == []

    def test_blank_core_values_use_the_defaults(self):
        payload = _core({name: "" for name in CORE_RISK_NAMES})
        assert payload["values"] == DEFAULT_CORE_VALUES
        assert payload["errors"] == []

    def test_valid_core_values_are_used_verbatim(self):
        payload = _core(
            {
                "ACCOUNT_BALANCE": "2500",
                "RISK_PERCENT": "1.5",
                "STOP_LOSS_PERCENT": "1.0",
                "TAKE_PROFIT_PERCENT": "3.0",
            }
        )
        assert payload["values"] == {
            "ACCOUNT_BALANCE": 2500.0,
            "RISK_PERCENT": 1.5,
            "STOP_LOSS_PERCENT": 1.0,
            "TAKE_PROFIT_PERCENT": 3.0,
        }
        assert payload["errors"] == []

        # The derived portfolio defaults follow the configured balance
        # rather than a fallback value.
        limits = json.loads(_run(_READ_LIMITS, ACCOUNT_BALANCE="2500"))
        assert limits["PORTFOLIO_RISK_LIMIT"] == 150.0
        assert limits["PORTFOLIO_EXPOSURE_LIMIT"] == 7500.0
        assert limits["PORTFOLIO_DAILY_LOSS_LIMIT"] == 75.0
        assert limits["PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL"] == 5000.0

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("ACCOUNT_BALANCE", "10O0"),
            ("RISK_PERCENT", "2%"),
            ("STOP_LOSS_PERCENT", "2 percent"),
            ("TAKE_PROFIT_PERCENT", "nan"),
        ],
    )
    def test_malformed_core_value_is_a_startup_error(self, name, value):
        payload = _core({name: value})
        assert payload["errors"], f"{name}={value!r} was accepted silently"
        # Every error must name this variable and nothing else, so a
        # mistyped input can never be attributed to another setting.
        assert {setting for _, setting, _ in payload["errors"]} == {name}
        assert any(
            "fix or remove the value in .env" in message
            for _, _, message in payload["errors"]
        )

    def test_main_refuses_to_start_on_malformed_account_balance(self):
        env = _environment(ACCOUNT_BALANCE="10O0")
        env.pop("TELEGRAM_BOT_TOKEN", None)
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert completed.returncode == 1
        assert "Invalid configuration; refusing to start" in completed.stdout
        assert "ACCOUNT_BALANCE" in completed.stdout
        # Configuration errors are checked before the Telegram token, so
        # the run is stopped by the malformed value, not by the token.
        assert "TELEGRAM_BOT_TOKEN is not set" not in completed.stdout


class TestDefaultGateIntegration:
    def _backend(self, open_positions):
        return SimpleNamespace(
            account=SimpleNamespace(closed_positions=[]),
            get_open_positions=lambda: list(open_positions),
        )

    def _position(self, symbol: str):
        # A real open position carries a market mark; mark it at entry so
        # the daily-loss brake can measure the floating side of the day
        # (an unmarked position fails closed and would reject earlier).
        position = SimpleNamespace(
            symbol=symbol,
            entry_price=100.0,
            quantity=0.01,
            stop_loss=98.0,
            side=PaperOrderSide("LONG"),
            mark_price=100.0,
        )
        position.unrealized_pnl_for = lambda price: (price - 100.0) * 0.01
        return position

    def _trade(self):
        return Trade(
            symbol="BTC",
            price=100.0,
            trend="Bullish",
            signal="BUY",
            confidence=80,
            decision="BUY",
            risk_verdict="APPROVED",
            position_size=10.0,
            stop_loss=98.0,
            take_profit=104.0,
            side="LONG",
            approved=True,
        )

    def test_defaults_enable_the_gates(self):
        backend = self._backend(
            [self._position("BTC"), self._position("ETH"), self._position("SOL")]
        )
        outcome = evaluate_entry_gates(
            backend,
            self._trade(),
            **{
                GATE_PARAMETERS[name]: DEFAULT_VALUES[name]
                for name in LIMIT_NAMES
            },
        )
        assert outcome.allowed is False
        assert outcome.rule == RULE_MAX_OPEN_POSITIONS

    def test_explicit_zero_disables_the_gates(self):
        backend = self._backend(
            [self._position("BTC"), self._position("ETH"), self._position("SOL")]
        )
        outcome = evaluate_entry_gates(
            backend,
            self._trade(),
            **{GATE_PARAMETERS[name]: 0 for name in LIMIT_NAMES},
        )
        assert outcome.allowed is True
        assert outcome.rule is None
