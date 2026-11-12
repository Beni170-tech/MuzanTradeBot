"""Tests for equity-based paper position sizing.

Covers:
    - RiskProfile.from_settings keeps the configured balance by default and
      accepts an explicit balance override.
    - The /analyze pipeline sizes from the paper account equity, including
      realized results and unrealized PnL at recorded marks.
    - Sizing falls back to the configured balance only when paper trading is
      disabled, and fails closed (no trade) when equity cannot be read.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import telegram_bot.handlers as handlers
from config import settings as cfg_settings
from risk.risk_profile import RiskProfile
from trader.paper.order import PaperOrderStatus
from trader.paper.service import PaperTradingService
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trade(
    symbol: str = "BTC",
    price: float = 100.0,
    quantity: float = 10.0,
    stop_loss: float = 95.0,
    take_profit: float = 120.0,
) -> Trade:
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        side="LONG",
        approved=True,
        risk_reasons=[],
        timestamp="2026-01-01 00:00:00",
    )


def _service(*trades: Trade) -> PaperTradingService:
    service = PaperTradingService.create_new_account(1000.0)
    for trade in trades:
        assert service.place_trade(trade) is not None
    return service


def _analysis(symbol: str = "ETH", price: float = 100.0) -> dict:
    return {
        "symbol": symbol,
        "price": price,
        "trend": "Bullish",
        "signal": "BUY",
        "confidence": 80,
    }


def _expected_quantity(equity: float, price: float = 100.0) -> float:
    """Mirror calculate_position_size using the live configured percentages."""
    risk_amount = equity * (cfg_settings.settings.RISK_PERCENT / 100.0)
    stop_distance = price * (cfg_settings.settings.STOP_LOSS_PERCENT / 100.0)
    return round(risk_amount / stop_distance, 8)


_SETTING_NAMES = (
    "PAPER_TRADING",
    "PORTFOLIO_MAX_OPEN_POSITIONS",
    "PORTFOLIO_DAILY_LOSS_LIMIT",
    "PORTFOLIO_RISK_LIMIT",
    "PORTFOLIO_EXPOSURE_LIMIT",
    "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
    "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
    "ALERTS_ENABLED",
)


def _run_analyze(monkeypatch, tmp_path, *, service, analysis=None) -> str:
    """Run the real /analyze pipeline (real risk/sizing) for one coin."""
    analysis = analysis or _analysis()
    originals = {
        name: getattr(cfg_settings.settings, name) for name in _SETTING_NAMES
    }
    try:
        cfg_settings.settings.PAPER_TRADING = True
        cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS = 0
        cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_RISK_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = 0
        cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL = 0.0
        cfg_settings.settings.ALERTS_ENABLED = False

        monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
        monkeypatch.setattr(handlers, "analyze_coin", lambda _: dict(analysis))
        monkeypatch.setattr(handlers, "make_decision", lambda r: "BUY")
        monkeypatch.setattr(handlers, "log_trade", MagicMock())
        monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)
        monkeypatch.setattr(
            handlers, "PAPER_STATE_PATH", tmp_path / "paper_state.json"
        )

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        update.message.chat.id = 123
        asyncio.run(handlers.analyze(update, MagicMock()))
        return update.message.reply_text.await_args[0][0]
    finally:
        for name, value in originals.items():
            setattr(cfg_settings.settings, name, value)


# ---------------------------------------------------------------------------
# RiskProfile.from_settings
# ---------------------------------------------------------------------------


def test_from_settings_uses_configured_balance_by_default() -> None:
    profile = RiskProfile.from_settings(cfg_settings.settings)
    assert profile.balance == pytest.approx(cfg_settings.settings.ACCOUNT_BALANCE)


def test_from_settings_accepts_balance_override() -> None:
    profile = RiskProfile.from_settings(cfg_settings.settings, balance=842.5)
    assert profile.balance == pytest.approx(842.5)
    assert profile.risk_percent == pytest.approx(
        cfg_settings.settings.RISK_PERCENT
    )


# ---------------------------------------------------------------------------
# Sizing profile
# ---------------------------------------------------------------------------


def test_sizing_profile_uses_paper_equity(monkeypatch) -> None:
    service = _service(_trade(quantity=10.0))
    service._account.add_realized_pnl(-150.0)  # equity 850

    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", True)
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)

    profile = handlers._sizing_profile()

    assert profile.balance == pytest.approx(850.0)


def test_sizing_profile_includes_unrealized_marks(monkeypatch) -> None:
    service = _service(_trade(quantity=10.0))
    service.get_position("BTC").mark_to_market(110.0)  # equity 1100

    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", True)
    monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)

    assert handlers._sizing_profile().balance == pytest.approx(1100.0)


def test_sizing_profile_uses_configured_balance_when_paper_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", False)
    monkeypatch.setattr(
        handlers,
        "_get_paper_service",
        lambda: (_ for _ in ()).throw(AssertionError("paper service used")),
    )

    profile = handlers._sizing_profile()

    assert profile.balance == pytest.approx(cfg_settings.settings.ACCOUNT_BALANCE)


def test_sizing_profile_fails_closed_when_account_unreadable(monkeypatch) -> None:
    monkeypatch.setattr(cfg_settings.settings, "PAPER_TRADING", True)
    monkeypatch.setattr(
        handlers,
        "_get_paper_service",
        lambda: (_ for _ in ()).throw(RuntimeError("state file corrupt")),
    )

    assert handlers._sizing_profile().balance == 0.0


# ---------------------------------------------------------------------------
# /analyze integration
# ---------------------------------------------------------------------------


def test_analyze_sizes_from_equity_after_realized_loss(
    monkeypatch, tmp_path
) -> None:
    service = _service(_trade(symbol="BTC", price=100.0, quantity=10.0))
    # Close the BTC position at 90 for -100, leaving equity 900.
    service.close_position("BTC", 90.0, PaperOrderStatus.STOPPED)
    assert service.summary().equity == pytest.approx(900.0)

    text = _run_analyze(monkeypatch, tmp_path, service=service)

    order = service.get_position("ETH")
    assert order is not None
    assert order.quantity == pytest.approx(_expected_quantity(900.0))
    assert order.quantity != pytest.approx(
        _expected_quantity(cfg_settings.settings.ACCOUNT_BALANCE)
    )
    assert "Paper Order: PLACED" in text


def test_analyze_sizes_from_equity_including_unrealized_marks(
    monkeypatch, tmp_path
) -> None:
    service = _service(_trade(symbol="BTC", price=100.0, quantity=10.0))
    service.get_position("BTC").mark_to_market(110.0)
    assert service.summary().equity == pytest.approx(1100.0)

    _run_analyze(monkeypatch, tmp_path, service=service)

    order = service.get_position("ETH")
    assert order is not None
    assert order.quantity == pytest.approx(_expected_quantity(1100.0))


def test_analyze_does_not_size_from_static_balance_in_paper_mode(
    monkeypatch, tmp_path
) -> None:
    service = _service()
    service._account.current_balance = 400.0

    _run_analyze(monkeypatch, tmp_path, service=service)

    order = service.get_position("ETH")
    assert order is not None
    assert order.quantity == pytest.approx(_expected_quantity(400.0))
    assert order.quantity == pytest.approx(4.0)


def test_analyze_blocks_new_trade_when_equity_is_zero(
    monkeypatch, tmp_path
) -> None:
    service = _service()
    service._account.current_balance = 0.0

    text = _run_analyze(monkeypatch, tmp_path, service=service)

    assert "Paper Order: PLACED" not in text
    assert "Account balance must be positive." in text
    assert service.open_positions() == []


def test_analyze_blocks_new_trade_when_equity_cannot_be_read(
    monkeypatch, tmp_path
) -> None:
    service = MagicMock()
    service.summary.side_effect = RuntimeError("state file corrupt")
    service.open_positions.return_value = []

    text = _run_analyze(monkeypatch, tmp_path, service=service)

    assert "Paper Order: PLACED" not in text
    assert "Account balance must be positive." in text
    service.place_trade.assert_not_called()
