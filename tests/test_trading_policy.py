"""Tests for the validated TradingPolicy and its use in /analyze."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import strategy.trading_policy as trading_policy_module
import telegram_bot.handlers as handlers
from config import settings as cfg_settings
from risk.risk_profile import RiskProfile
from strategy.parameters import (
    DEFAULT_STRATEGY_PARAMETERS,
    StrategyParameters,
)
from strategy.trading_policy import TradingPolicy
from trader.paper.service import PaperTradingService


CUSTOM_STRATEGY = StrategyParameters(min_candles=80)


def _unvalidated_profile(**overrides) -> RiskProfile:
    """Build a RiskProfile while bypassing its constructor validation."""
    profile = object.__new__(RiskProfile)
    values = {
        "balance": 1000.0,
        "risk_percent": 2.0,
        "stop_loss_percent": 2.0,
        "take_profit_percent": 4.0,
    }
    values.update(overrides)
    for name, value in values.items():
        object.__setattr__(profile, name, value)
    return profile


def _analysis() -> dict:
    return {
        "symbol": "BTC",
        "price": 105.0,
        "trend": "Bullish",
        "signal": "BUY",
        "confidence": 80,
        "indicators": None,
    }


def _update_and_context():
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.chat.id = 123
    context = MagicMock()
    context.args = ["BTC"]
    return update, context


# ---------------------------------------------------------------------------
# Construction and parity
# ---------------------------------------------------------------------------

def test_settings_policy_matches_existing_defaults():
    policy = TradingPolicy.from_settings(cfg_settings.settings)

    assert policy.strategy is DEFAULT_STRATEGY_PARAMETERS
    assert policy.risk_profile == RiskProfile.from_settings(
        cfg_settings.settings
    )
    assert policy.risk_profile.balance == pytest.approx(
        cfg_settings.settings.ACCOUNT_BALANCE
    )
    assert policy.risk_profile.risk_percent == pytest.approx(
        cfg_settings.settings.RISK_PERCENT
    )
    assert policy.risk_profile.stop_loss_percent == pytest.approx(
        cfg_settings.settings.STOP_LOSS_PERCENT
    )
    assert policy.risk_profile.take_profit_percent == pytest.approx(
        cfg_settings.settings.TAKE_PROFIT_PERCENT
    )


def test_settings_policy_accepts_balance_and_strategy():
    policy = TradingPolicy.from_settings(
        cfg_settings.settings,
        balance=842.5,
        strategy=CUSTOM_STRATEGY,
    )

    assert policy.strategy is CUSTOM_STRATEGY
    assert policy.risk_profile.balance == pytest.approx(842.5)


def test_policy_is_immutable_and_keeps_exact_objects():
    profile = RiskProfile(balance=500.0)
    policy = TradingPolicy(
        strategy=CUSTOM_STRATEGY,
        risk_profile=profile,
    )

    assert policy.strategy is CUSTOM_STRATEGY
    assert policy.risk_profile is profile
    with pytest.raises(FrozenInstanceError):
        policy.strategy = DEFAULT_STRATEGY_PARAMETERS


@pytest.mark.parametrize(
    ("strategy", "risk_profile"),
    [
        (object(), RiskProfile()),
        (DEFAULT_STRATEGY_PARAMETERS, object()),
        (None, RiskProfile()),
        (DEFAULT_STRATEGY_PARAMETERS, None),
    ],
)
def test_invalid_object_types_are_rejected(strategy, risk_profile):
    with pytest.raises(TypeError):
        TradingPolicy(strategy=strategy, risk_profile=risk_profile)


def test_from_settings_rejects_invalid_strategy_type():
    with pytest.raises(TypeError, match="strategy"):
        TradingPolicy.from_settings(
            cfg_settings.settings,
            strategy=object(),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("balance", float("nan"), ValueError),
        ("risk_percent", float("inf"), ValueError),
        ("stop_loss_percent", float("-inf"), ValueError),
        ("take_profit_percent", float("nan"), ValueError),
        ("balance", 0.0, ValueError),
        ("risk_percent", -1.0, ValueError),
        ("stop_loss_percent", 0.0, ValueError),
        ("take_profit_percent", -2.0, ValueError),
        ("stop_loss_percent", 100.0, ValueError),
        ("risk_percent", 100.0, ValueError),
        ("balance", True, TypeError),
        ("risk_percent", "2", TypeError),
    ],
)
def test_invalid_configuration_cannot_be_carried(field, value, error):
    with pytest.raises(error):
        TradingPolicy(
            strategy=DEFAULT_STRATEGY_PARAMETERS,
            risk_profile=RiskProfile(**{field: value}),
        )


def test_policy_rechecks_a_profile_that_bypassed_validation():
    with pytest.raises(ValueError, match="finite"):
        TradingPolicy(
            strategy=DEFAULT_STRATEGY_PARAMETERS,
            risk_profile=_unvalidated_profile(balance=float("nan")),
        )

    with pytest.raises(ValueError, match="less than 100"):
        TradingPolicy(
            strategy=DEFAULT_STRATEGY_PARAMETERS,
            risk_profile=_unvalidated_profile(stop_loss_percent=100.0),
        )

    with pytest.raises(ValueError, match="less than 100"):
        TradingPolicy(
            strategy=DEFAULT_STRATEGY_PARAMETERS,
            risk_profile=_unvalidated_profile(risk_percent=100.0),
        )


def test_from_settings_rejects_invalid_settings(monkeypatch):
    monkeypatch.setattr(cfg_settings.settings, "RISK_PERCENT", float("nan"))

    with pytest.raises(ValueError, match="finite"):
        TradingPolicy.from_settings(cfg_settings.settings)


# ---------------------------------------------------------------------------
# /analyze integration
# ---------------------------------------------------------------------------

def _spy_policy_resolution(monkeypatch, captured: dict) -> None:
    """Record the strategy/balance handed to the policy constructor."""
    real_from_settings = (
        trading_policy_module.TradingPolicy.from_settings.__func__
    )

    def spy_from_settings(cls, settings, *, balance=None, strategy=None):
        captured["strategy"] = strategy
        captured["balance"] = balance
        return real_from_settings(
            cls,
            settings,
            balance=balance,
            strategy=strategy,
        )

    monkeypatch.setattr(
        trading_policy_module.TradingPolicy,
        "from_settings",
        classmethod(spy_from_settings),
    )


def test_equity_resolution_follows_quote_and_freshness(monkeypatch, tmp_path):
    events = []
    captured = {}
    profiles = {}
    service = PaperTradingService.create_new_account(1000.0)

    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", True)
    monkeypatch.setattr(cfg_settings.settings, "ALERTS_ENABLED", False)
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(
        handlers, "PAPER_STATE_PATH", tmp_path / "paper_state.json"
    )
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)
    monkeypatch.setattr(handlers, "log_trade", MagicMock())

    def spy_analyze(symbol, *, parameters=None):
        events.append(("analyze", parameters))
        return _analysis()

    monkeypatch.setattr(handlers, "analyze_coin", spy_analyze)

    def spy_quote(symbol):
        events.append(("quote", None))
        return 105.0

    monkeypatch.setattr(handlers, "get_current_price", spy_quote)

    real_mark_block = handlers._paper_mark_block

    def spy_mark_block():
        events.append(("freshness", None))
        return real_mark_block()

    monkeypatch.setattr(handlers, "_paper_mark_block", spy_mark_block)

    def spy_equity():
        events.append(("equity", None))
        return 1000.0

    monkeypatch.setattr(
        handlers, "_paper_equity_for_sizing", spy_equity
    )

    real_validate = handlers.validate_trade

    def spy_validate(decision, result, profile):
        events.append(("validate", None))
        profiles["validate"] = profile
        return real_validate(decision, result, profile)

    monkeypatch.setattr(handlers, "validate_trade", spy_validate)

    real_build = handlers.build_trade

    def spy_build(result, verdict, profile):
        events.append(("build", None))
        profiles["build"] = profile
        return real_build(result, verdict, profile)

    monkeypatch.setattr(handlers, "build_trade", spy_build)
    _spy_policy_resolution(monkeypatch, captured)

    update, context = _update_and_context()
    asyncio.run(handlers.analyze(update, context))

    order = [name for name, _ in events]
    assert order.index("analyze") < order.index("quote")
    assert order.index("quote") < order.index("freshness")
    assert order.index("freshness") < order.index("equity")
    assert order.index("equity") < order.index("validate")
    assert order.index("validate") < order.index("build")

    # The exact strategy instance from analysis is what the policy carries.
    analyzed_strategy = events[0][1]
    assert captured["strategy"] is analyzed_strategy
    assert analyzed_strategy is DEFAULT_STRATEGY_PARAMETERS
    assert captured["balance"] == pytest.approx(1000.0)

    # One profile instance is shared by validation and construction.
    assert profiles["validate"] is profiles["build"]
    assert profiles["validate"].balance == pytest.approx(1000.0)
    assert profiles["validate"] == RiskProfile.from_settings(
        cfg_settings.settings, balance=1000.0
    )

    message = update.message.reply_text.await_args[0][0]
    assert "Analysis for BTC" in message
    assert "Decision: BUY" in message


def _run_equity_failure(monkeypatch, tmp_path, summary):
    """Run /analyze against a paper service that yields unusable equity."""
    service = MagicMock()
    service.open_positions.return_value = []
    service.summary.side_effect = summary

    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", True)
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(
        handlers, "PAPER_STATE_PATH", tmp_path / "paper_state.json"
    )
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)
    monkeypatch.setattr(handlers, "log_trade", MagicMock())

    analyze_mock = MagicMock(return_value=_analysis())
    monkeypatch.setattr(handlers, "analyze_coin", analyze_mock)
    monkeypatch.setattr(handlers, "get_current_price", lambda _: 105.0)

    validate_mock = MagicMock()
    monkeypatch.setattr(handlers, "validate_trade", validate_mock)
    build_mock = MagicMock()
    monkeypatch.setattr(handlers, "build_trade", build_mock)

    update, context = _update_and_context()
    asyncio.run(handlers.analyze(update, context))

    message = update.message.reply_text.await_args[0][0]
    assert "paper account equity unavailable" in message
    assert "no fallback balance was used" in message
    assert analyze_mock.call_count == 1
    validate_mock.assert_not_called()
    build_mock.assert_not_called()
    service.place_trade.assert_not_called()


def test_unreadable_equity_prevents_trade_placement(monkeypatch, tmp_path):
    _run_equity_failure(
        monkeypatch,
        tmp_path,
        RuntimeError("state file corrupt"),
    )


@pytest.mark.parametrize("equity", [0.0, -5.0, float("nan")])
def test_invalid_equity_prevents_trade_placement(
    monkeypatch, tmp_path, equity
):
    service = MagicMock()
    service.open_positions.return_value = []
    service.summary.return_value = SimpleNamespace(equity=equity)

    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", True)
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(
        handlers, "PAPER_STATE_PATH", tmp_path / "paper_state.json"
    )
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)
    monkeypatch.setattr(handlers, "log_trade", MagicMock())
    monkeypatch.setattr(
        handlers, "analyze_coin", MagicMock(return_value=_analysis())
    )
    monkeypatch.setattr(handlers, "get_current_price", lambda _: 105.0)
    validate_mock = MagicMock()
    monkeypatch.setattr(handlers, "validate_trade", validate_mock)
    build_mock = MagicMock()
    monkeypatch.setattr(handlers, "build_trade", build_mock)

    update, context = _update_and_context()
    asyncio.run(handlers.analyze(update, context))

    message = update.message.reply_text.await_args[0][0]
    assert "paper account equity unavailable" in message
    validate_mock.assert_not_called()
    build_mock.assert_not_called()
    service.place_trade.assert_not_called()


def test_paper_disabled_uses_configured_balance(monkeypatch):
    captured = {}
    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", False)
    monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
    monkeypatch.setattr(handlers, "log_trade", MagicMock())
    monkeypatch.setattr(
        handlers, "analyze_coin", lambda _s, *, parameters=None: _analysis()
    )
    monkeypatch.setattr(handlers, "get_current_price", lambda _: 105.0)
    monkeypatch.setattr(
        handlers,
        "_paper_equity_for_sizing",
        lambda: (_ for _ in ()).throw(AssertionError("equity read")),
    )
    _spy_policy_resolution(monkeypatch, captured)

    update, context = _update_and_context()
    asyncio.run(handlers.analyze(update, context))

    assert captured["balance"] is None
    assert captured["strategy"] is DEFAULT_STRATEGY_PARAMETERS
    message = update.message.reply_text.await_args[0][0]
    assert "Analysis for BTC" in message
