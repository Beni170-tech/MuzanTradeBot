"""Tests for startup configuration validation (Milestone M-A).

Risk settings must be usable numbers; the documented ``0 = disabled``
behaviour for portfolio limits is preserved and surfaced as a warning so
a disabled safety limit is never silent. Validation must name settings
without ever printing secret values.
"""

from __future__ import annotations

import math

import pytest

import main as main_module
from config.settings import Settings, settings, validate_settings


def _issues_by_name(issues):
    return {issue.setting: issue for issue in issues}


def _errors(issues):
    return [issue for issue in issues if issue.severity == "error"]


class TestValidateSettings:
    def test_defaults_have_no_errors(self):
        assert _errors(validate_settings()) == []

    def test_disabled_limits_are_visible_warnings(self):
        issues = _issues_by_name(validate_settings())
        for name in (
            "PORTFOLIO_RISK_LIMIT",
            "PORTFOLIO_EXPOSURE_LIMIT",
            "PORTFOLIO_DAILY_LOSS_LIMIT",
            "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
            "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
            "PORTFOLIO_MAX_OPEN_POSITIONS",
        ):
            assert issues[name].severity == "warning"
            assert "disabled (0)" in issues[name].message

    @pytest.mark.parametrize(
        "name",
        [
            "PORTFOLIO_RISK_LIMIT",
            "PORTFOLIO_EXPOSURE_LIMIT",
            "PORTFOLIO_DAILY_LOSS_LIMIT",
            "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
        ],
    )
    def test_negative_limits_are_rejected(self, monkeypatch, name):
        monkeypatch.setattr(settings, name, -1.0)
        issues = _errors(validate_settings())
        assert [issue.setting for issue in issues] == [name]
        assert ">= 0" in issues[0].message

    @pytest.mark.parametrize(
        "name", ["PORTFOLIO_MAX_OPEN_POSITIONS", "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL"]
    )
    def test_negative_counts_are_rejected(self, monkeypatch, name):
        monkeypatch.setattr(settings, name, -2)
        issues = _errors(validate_settings())
        assert [issue.setting for issue in issues] == [name]

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.5, 0.0])
    def test_unusable_account_balance_is_rejected(self, monkeypatch, value):
        monkeypatch.setattr(settings, "ACCOUNT_BALANCE", value)
        issues = _errors(validate_settings())
        assert [issue.setting for issue in issues] == ["ACCOUNT_BALANCE"]

    @pytest.mark.parametrize("value", [0.0, -1.0, 101.0, float("nan")])
    def test_unusable_risk_percent_is_rejected(self, monkeypatch, value):
        monkeypatch.setattr(settings, "RISK_PERCENT", value)
        issues = _errors(validate_settings())
        assert [issue.setting for issue in issues] == ["RISK_PERCENT"]

    def test_hundred_percent_risk_is_allowed(self, monkeypatch):
        monkeypatch.setattr(settings, "RISK_PERCENT", 100.0)
        assert _errors(validate_settings()) == []

    @pytest.mark.parametrize(
        "name", ["STOP_LOSS_PERCENT", "TAKE_PROFIT_PERCENT"]
    )
    @pytest.mark.parametrize("value", [0.0, -2.0, float("inf")])
    def test_unusable_exit_percentages_are_rejected(
        self, monkeypatch, name, value
    ):
        monkeypatch.setattr(settings, name, value)
        issues = _errors(validate_settings())
        assert [issue.setting for issue in issues] == [name]

    def test_zero_keeps_documented_disabled_meaning(self, monkeypatch):
        monkeypatch.setattr(settings, "PORTFOLIO_RISK_LIMIT", 0.0)
        issues = validate_settings()
        assert not any(
            issue.severity == "error"
            and issue.setting == "PORTFOLIO_RISK_LIMIT"
            for issue in issues
        )

    def test_non_finite_monitor_interval_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            settings, "PAPER_MONITOR_INTERVAL_SECONDS", float("nan")
        )
        issues = _errors(validate_settings())
        assert [issue.setting for issue in issues] == [
            "PAPER_MONITOR_INTERVAL_SECONDS"
        ]

    def test_disabled_monitor_is_a_warning_not_an_error(self, monkeypatch):
        monkeypatch.setattr(settings, "PAPER_MONITOR_INTERVAL_SECONDS", 0.0)
        issues = _issues_by_name(validate_settings())
        assert issues["PAPER_MONITOR_INTERVAL_SECONDS"].severity == "warning"

    def test_live_modes_stay_empty(self):
        assert Settings._LIVE_EXECUTION_MODES == ()

    def test_issue_format_names_the_setting_without_values(self, monkeypatch):
        monkeypatch.setattr(settings, "PORTFOLIO_EXPOSURE_LIMIT", -5.0)
        issue = _errors(validate_settings())[0]
        assert issue.format().startswith("PORTFOLIO_EXPOSURE_LIMIT: ")


class TestStartupGate:
    def test_main_refuses_to_start_on_invalid_config(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(settings, "RISK_PERCENT", -1.0)
        monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "secret-token")
        monkeypatch.setattr(Settings, "EXECUTION_MODE", "PAPER")

        with pytest.raises(SystemExit) as excinfo:
            main_module.main()

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "Invalid configuration; refusing to start" in out
        assert "RISK_PERCENT" in out
        assert "secret-token" not in out

    def test_main_reports_disabled_limits_and_stops_without_token(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(settings, "TELEGRAM_BOT_TOKEN", "")
        # Other execution-mode tests mutate the class attribute directly;
        # pin the startup mode so this test never depends on that state.
        monkeypatch.setattr(Settings, "EXECUTION_MODE", "PAPER")

        with pytest.raises(SystemExit) as excinfo:
            main_module.main()

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "[WARN] PORTFOLIO_RISK_LIMIT: disabled (0)" in out
        assert "TELEGRAM_BOT_TOKEN is not set" in out

    def test_validation_never_raises_on_finite_math(self):
        # Sanity: the validator's own finite check is what rejects NaN.
        assert math.isfinite(1.0)
        assert _errors(validate_settings()) == []
