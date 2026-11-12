"""Tests for using a fresh market quote in the /analyze trading flow.

Historical candles still drive trend/signal/confidence, but the paper
trade must be priced from a current quote. These tests prove the quote
is used for entry, sizing, stop-loss/take-profit, and current-price
checks, and that an unusable quote blocks placement without falling
back to the last historical close.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import telegram_bot.handlers as handlers
from config import settings as cfg_settings
from trader.paper.order import PaperOrderStatus
from trader.paper.service import PaperTradingService
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _historical_analysis(
    symbol: str = "BTC",
    close: float = 100.0,
    trend: str = "Bullish",
    signal: str = "BUY",
    confidence: int = 80,
) -> dict:
    """Analysis dict as returned by analyze_coin (price = last close)."""
    return {
        "symbol": symbol,
        "price": close,
        "trend": trend,
        "signal": signal,
        "confidence": confidence,
        "indicators": {"ema_fast": 1.0, "rsi": 55.0},
    }


def _trade(
    symbol: str = "BTC",
    price: float = 100.0,
    quantity: float = 1.0,
    side: str = "LONG",
    stop_loss: float = 90.0,
    take_profit: float = 130.0,
) -> Trade:
    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish" if side == "LONG" else "Bearish",
        signal="BUY" if side == "LONG" else "SELL",
        confidence=80,
        decision="BUY" if side == "LONG" else "SELL",
        risk_verdict="APPROVED",
        position_size=quantity,
        stop_loss=stop_loss,
        take_profit=take_profit,
        side=side,
        approved=True,
        risk_reasons=[],
        timestamp="2026-01-01 00:00:00",
    )


def _service(*trades: Trade) -> PaperTradingService:
    service = PaperTradingService.create_new_account(1000.0)
    for trade in trades:
        assert service.place_trade(trade) is not None
    return service


def _expected_quantity(equity: float, price: float) -> float:
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


def _run_analyze(
    monkeypatch,
    tmp_path,
    *,
    service,
    analysis=None,
    quote=105.0,
    quote_side_effect=None,
):
    """Run the real /analyze pipeline with a controlled quote."""
    analysis = analysis or _historical_analysis()
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

        analyze_mock = MagicMock(return_value=dict(analysis))
        if quote_side_effect is not None:
            quote_mock = MagicMock(side_effect=quote_side_effect)
        else:
            quote_mock = MagicMock(return_value=quote)

        monkeypatch.setattr(handlers, "check_rate_limit", lambda *a, **k: True)
        monkeypatch.setattr(handlers, "analyze_coin", analyze_mock)
        monkeypatch.setattr(handlers, "get_current_price", quote_mock)
        monkeypatch.setattr(handlers, "log_trade", MagicMock())
        monkeypatch.setattr(handlers, "_get_paper_service", lambda: service)
        monkeypatch.setattr(
            handlers, "PAPER_STATE_PATH", tmp_path / "paper_state.json"
        )

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        update.message.chat.id = 123
        asyncio.run(handlers.analyze(update, MagicMock()))
        return update.message.reply_text.await_args[0][0], analyze_mock, quote_mock
    finally:
        for name, value in originals.items():
            setattr(cfg_settings.settings, name, value)


# ---------------------------------------------------------------------------
# Fresh quote drives the trade
# ---------------------------------------------------------------------------


def test_historical_analysis_drives_signal_but_quote_prices_trade(
    monkeypatch, tmp_path
) -> None:
    service = _service()
    analysis = _historical_analysis(close=100.0)

    message, analyze_mock, quote_mock = _run_analyze(
        monkeypatch, tmp_path, service=service, analysis=analysis, quote=105.0
    )

    order = service.get_position("BTC")
    assert order is not None
    # Entry is the fresh quote, not the historical close.
    assert order.entry_price == pytest.approx(105.0)
    assert order.entry_price != pytest.approx(100.0)
    # Sizing and exit levels are derived from the quote.
    assert order.quantity == pytest.approx(_expected_quantity(1000.0, 105.0))
    assert order.stop_loss == pytest.approx(round(105.0 * 0.98, 2))
    assert order.take_profit == pytest.approx(round(105.0 * 1.04, 2))
    # Historical analysis still drives trend/signal/decision.
    assert "Trend: Bullish" in message
    assert "Signal: BUY" in message
    assert "Decision: BUY" in message
    assert "Price: 105.0" in message
    # Exactly one historical analysis and one quote request.
    assert analyze_mock.call_count == 1
    assert quote_mock.call_count == 1


def test_quote_prices_short_trade_and_exit_levels(monkeypatch, tmp_path) -> None:
    service = _service()
    analysis = _historical_analysis(
        close=100.0, trend="Bearish", signal="SELL"
    )

    _run_analyze(
        monkeypatch, tmp_path, service=service, analysis=analysis, quote=95.0
    )

    order = service.get_position("BTC")
    assert order is not None
    assert order.side.value == "SHORT"
    assert order.entry_price == pytest.approx(95.0)
    assert order.quantity == pytest.approx(_expected_quantity(1000.0, 95.0))
    assert order.stop_loss == pytest.approx(round(95.0 * 1.02, 2))
    assert order.take_profit == pytest.approx(round(95.0 * 0.96, 2))


def test_quote_is_used_for_existing_position_sltp_check(
    monkeypatch, tmp_path
) -> None:
    """A position's TP is hit by the fresh quote, not the historical close."""
    service = _service(
        _trade(price=100.0, quantity=1.0, stop_loss=90.0, take_profit=104.0)
    )
    position = service.get_position("BTC")
    assert position.mark_to_market(100.0) is True  # fresh mark for sizing gate

    message, _, _ = _run_analyze(
        monkeypatch,
        tmp_path,
        service=service,
        analysis=_historical_analysis(close=100.0),
        quote=105.0,
    )

    assert position.status == PaperOrderStatus.TAKEN_PROFIT
    assert position.exit_price == pytest.approx(104.0)
    assert "Paper Order: CLOSED (SL/TP)" in message


# ---------------------------------------------------------------------------
# Unusable quotes block placement
# ---------------------------------------------------------------------------


def test_unavailable_quote_blocks_paper_order(monkeypatch, tmp_path) -> None:
    service = _service()

    message, _, _ = _run_analyze(
        monkeypatch, tmp_path, service=service, quote=None
    )

    assert "current market quote unavailable" in message
    assert "No trade was placed" in message
    assert "Trend: Bullish" in message
    assert "Signal: BUY" in message
    assert service.open_positions() == []


@pytest.mark.parametrize(
    "bad_quote",
    [0, 0.0, -1, -0.5, float("nan"), float("inf"), float("-inf"), "105.0", True],
)
def test_invalid_quote_blocks_paper_order(
    monkeypatch, tmp_path, bad_quote
) -> None:
    service = _service()

    message, _, _ = _run_analyze(
        monkeypatch, tmp_path, service=service, quote=bad_quote
    )

    assert "current market quote unavailable" in message
    assert service.open_positions() == []


def test_quote_lookup_exception_blocks_paper_order(monkeypatch, tmp_path) -> None:
    service = _service()

    message, _, _ = _run_analyze(
        monkeypatch,
        tmp_path,
        service=service,
        quote_side_effect=RuntimeError("CoinGecko unavailable"),
    )

    assert "current market quote unavailable" in message
    assert service.open_positions() == []


def test_quote_failure_never_falls_back_to_historical_close(
    monkeypatch, tmp_path
) -> None:
    service = _service()

    message, _, _ = _run_analyze(
        monkeypatch,
        tmp_path,
        service=service,
        analysis=_historical_analysis(close=100.0),
        quote=None,
    )

    assert "Price: 100.0" not in message
    assert "Paper Order: PLACED" not in message
    assert service.open_positions() == []
