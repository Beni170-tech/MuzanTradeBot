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

_READ_ERRORS = (
    "import json\n"
    "from config.settings import validate_settings\n"
    "print(json.dumps([\n"
    "    [issue.severity, issue.setting, issue.message]\n"
    "    for issue in validate_settings()\n"
    "]))\n"
)


def _environment(**overrides: str) -> dict[str, str]:
    """Return a clean environment with no inherited limit configuration."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in LIMIT_NAMES and key != "ACCOUNT_BALANCE"
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


class TestDefaultGateIntegration:
    def _backend(self, open_positions):
        return SimpleNamespace(
            account=SimpleNamespace(closed_positions=[]),
            get_open_positions=lambda: list(open_positions),
        )

    def _position(self, symbol: str):
        return SimpleNamespace(
            symbol=symbol,
            entry_price=100.0,
            quantity=0.01,
            stop_loss=98.0,
            side=PaperOrderSide("LONG"),
        )

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
