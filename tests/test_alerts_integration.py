"""
Tests for Sprint 9A - Alert wiring in the /analyze handler.

Each of the seven alert events must fire from the right place in the
paper-trading flow, without changing any existing reply text or
behavior:

    1. Paper order placed            -> AlertManager.order_placed
    2. Stop-loss triggered           -> AlertManager.stop_loss_triggered
    3. Take-profit triggered         -> AlertManager.take_profit_triggered
    4. Trailing stop updates         -> AlertManager.trailing_stop_updated
    5. Guardian/risk-limit rejection -> AlertManager.trade_rejected
    6. Daily loss breaker activation -> AlertManager.daily_loss_breaker
    7. Max open positions reached    -> AlertManager.max_open_positions_reached

The tests reuse the run-analyze harness pattern from
tests/test_paper_max_open_positions.py and assert that AlertManager
methods are awaited with the right domain objects at the right points,
while every existing reply-text assertion still holds.
"""

from __future__ import annotations

import asyncio
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import telegram_bot.handlers as _handlers
from alerts.manager import AlertManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(trade_price=1.0):
    trade = MagicMock()
    trade.symbol = "X"
    trade.price = trade_price
    trade.position_size = 1.0
    trade.stop_loss = 0.5
    trade.take_profit = 2.0
    trade.side = "LONG"
    trade.approved = True
    trade.risk_verdict = "APPROVED"
    trade.risk_reasons = []
    return trade


def _make_open_position(
    symbol="BTC", quantity=1.0, entry_price=100.0, stop_loss=90.0
):
    """An open-position mock that satisfies the risk-manager math."""
    position = MagicMock()
    position.symbol = symbol
    position.quantity = quantity
    position.entry_price = entry_price
    position.stop_loss = stop_loss
    position.side = MagicMock()
    position.side.value = "LONG"
    position.side.name = "LONG"
    return position


def _make_broker(open_positions=None, closed_positions=None):
    broker = types.SimpleNamespace()
    broker.open_positions = open_positions or []
    broker.get_open_positions = lambda: tuple(broker.open_positions)
    broker.account = types.SimpleNamespace()
    broker.account.closed_positions = closed_positions or []
    return broker


def _make_service(open_positions=None, closed_positions=None):
    service = MagicMock()
    service._open_positions_fixture = open_positions or []
    service._closed_positions_fixture = closed_positions or []
    service.get_position.return_value = None
    service.place_trade.return_value = None
    return service


def _make_placed_order():
    order = MagicMock()
    order.order_id = "PAPE-X-PLACE-001"
    order.status = MagicMock()
    order.status.value = "OPEN"
    order.quantity = 1.0
    order.entry_price = 1.0
    order.stop_loss = 0.5
    order.take_profit = 2.0
    order.side = MagicMock()
    order.side.value = "LONG"
    order.symbol = "X"
    return order


def _make_preexisting_position(
    side="LONG", entry_price=100.0, stop_loss=90.0, take_profit=110.0
):
    order = MagicMock()
    order.symbol = "X"
    order.side = MagicMock()
    order.side.value = side
    order.entry_price = entry_price
    order.quantity = 1.0
    order.stop_loss = stop_loss
    order.take_profit = take_profit
    order.status = MagicMock()
    order.status.value = "OPEN"
    order.exit_price = 0.0
    order.realized_pnl = 0.0
    order.order_id = "PAPE-X-LONG-001"
    order.notes = []
    return order


def _make_closed_order(status="STOPPED", exit_price=90.0, realized_pnl=-10.0):
    """A closed order in the real PaperOrder shape: ``status`` only.

    Real PaperOrder objects have no ``exit_status`` attribute;
    record_exit sets ``status`` to the terminal value. Mocks must
    mirror that so the wiring is tested against production data.
    """
    order = _make_preexisting_position()
    order.status.value = status
    order.exit_price = exit_price
    order.realized_pnl = realized_pnl
    return order


def _run_analyze(monkeypatch, *, alerts, service, settings_overrides=None, trade_price=1.0):
    """Run the analyze handler with mocked pipeline pieces.

    The service is expected to be pre-configured by the caller
    (get_position / place_trade return values). Returns the joined text
    of every reply sent through reply_text.
    """
    from config import settings as cfg_settings
    import telegram_bot.handlers as h

    saved = {}
    keys = (
        "PAPER_TRADING",
        "PORTFOLIO_MAX_OPEN_POSITIONS",
        "PORTFOLIO_DAILY_LOSS_LIMIT",
        "PORTFOLIO_RISK_LIMIT",
        "PORTFOLIO_EXPOSURE_LIMIT",
        "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
        "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
        "ALERTS_ENABLED",
    )
    for key in keys:
        saved[key] = getattr(cfg_settings.settings, key)
    try:
        cfg_settings.settings.PAPER_TRADING = True
        cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS = 0
        cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_RISK_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = 0
        cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL = 0.0
        cfg_settings.settings.ALERTS_ENABLED = True
        for key, value in (settings_overrides or {}).items():
            setattr(cfg_settings.settings, key, value)

        fake_trade = _make_trade(trade_price=trade_price)
        service._broker = _make_broker(
            open_positions=service._open_positions_fixture,
            closed_positions=service._closed_positions_fixture,
        )
        service.execution_backend.return_value = service._broker

        monkeypatch.setattr(h, "analyze_coin", lambda _: {"symbol": "X", "price": trade_price})
        monkeypatch.setattr(h, "make_decision", lambda r: MagicMock(approved=True))
        monkeypatch.setattr(
            h, "validate_trade", lambda d, r, p: MagicMock(approved=True)
        )
        monkeypatch.setattr(h, "build_trade", lambda r, v, p: fake_trade)
        monkeypatch.setattr(h, "log_trade", MagicMock())
        monkeypatch.setattr(h, "_get_paper_service", lambda: service)
        monkeypatch.setattr(h, "_alert_manager", alerts)
        monkeypatch.setattr(h, "PAPER_STATE_PATH", "/tmp/ignore.json")

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        asyncio.run(h.analyze(update, MagicMock()))
        replies = update.message.reply_text.await_args_list
        assert replies, "Expected analyze() to call reply_text at least once"
        all_text = "\n".join(r[0][0] for r in replies)
        return all_text, replies
    finally:
        for key, value in saved.items():
            setattr(cfg_settings.settings, key, value)


def _make_alerts(*methods):
    alerts = MagicMock()
    for method in (
        "order_placed",
        "stop_loss_triggered",
        "take_profit_triggered",
        "trailing_stop_updated",
        "trade_rejected",
        "daily_loss_breaker",
        "max_open_positions_reached",
    ):
        setattr(alerts, method, AsyncMock())
    for method in methods:
        getattr(alerts, method).assert_not_awaited()  # sanity: fresh mocks
    return alerts


# ---------------------------------------------------------------------------
# 1. Paper order placed
# ---------------------------------------------------------------------------


def test_alert_fires_when_paper_order_placed(monkeypatch):
    service = _make_service()
    placed = _make_placed_order()
    service.place_trade.return_value = placed
    alerts = _make_alerts()

    all_text, _ = _run_analyze(monkeypatch, alerts=alerts, service=service)

    alerts.order_placed.assert_awaited_once()
    assert alerts.order_placed.await_args.args[0] is placed
    # Existing behavior intact:
    assert "Paper Order: PLACED" in all_text
    assert "PAPE-X-PLACE-001" in all_text


def test_no_alert_when_no_order_placed(monkeypatch):
    service = _make_service()
    service.place_trade.return_value = None
    alerts = _make_alerts()

    _run_analyze(monkeypatch, alerts=alerts, service=service)

    alerts.order_placed.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2/3. Stop-loss / take-profit triggered
# ---------------------------------------------------------------------------


def test_alert_fires_on_stop_loss_closure(monkeypatch):
    position = _make_preexisting_position(stop_loss=90.0)
    service = _make_service()
    closed = _make_closed_order("STOPPED", 90.0, -10.0)
    service.get_position.side_effect = [position, None]
    service.close_position.return_value = closed
    alerts = _make_alerts()

    all_text, _ = _run_analyze(monkeypatch, alerts=alerts, service=service)

    alerts.stop_loss_triggered.assert_awaited_once()
    assert alerts.stop_loss_triggered.await_args.args[0] is closed
    alerts.take_profit_triggered.assert_not_awaited()
    # Existing behavior intact:
    assert "Paper Order: CLOSED (SL/TP)" in all_text
    service.close_position.assert_called_once_with("X", 90.0, PaperOrderStatus_STOPPED())


def PaperOrderStatus_STOPPED():
    from trader.paper.order import PaperOrderStatus

    return PaperOrderStatus.STOPPED


def test_alert_fires_on_take_profit_closure(monkeypatch):
    position = _make_preexisting_position(take_profit=110.0)
    service = _make_service()
    closed = _make_closed_order("TAKEN_PROFIT", 110.0, 10.0)
    service.get_position.side_effect = [position, None]
    service.close_position.return_value = closed
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch, alerts=alerts, service=service, trade_price=120.0
    )

    alerts.take_profit_triggered.assert_awaited_once()
    assert alerts.take_profit_triggered.await_args.args[0] is closed
    alerts.stop_loss_triggered.assert_not_awaited()
    assert "Paper Order: CLOSED (SL/TP)" in all_text


def test_no_alert_when_no_sl_tp_closure(monkeypatch):
    service = _make_service()
    service.get_position.return_value = None
    alerts = _make_alerts()

    _run_analyze(monkeypatch, alerts=alerts, service=service)

    alerts.stop_loss_triggered.assert_not_awaited()
    alerts.take_profit_triggered.assert_not_awaited()


# ---------------------------------------------------------------------------
# 4. Trailing stop updated
# ---------------------------------------------------------------------------


def test_alert_fires_on_trailing_stop_update(monkeypatch):
    # Entry 1.0, stop 0.9, price 1.2 -> 1R profit, stop trails to
    # breakeven (1.0). The same position is returned by both
    # get_position calls (SL/TP check, then trailing check).
    position = _make_preexisting_position(entry_price=1.0, stop_loss=0.9, take_profit=2.0)
    service = _make_service()
    service.get_position.return_value = position
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch, alerts=alerts, service=service, trade_price=1.2
    )

    alerts.trailing_stop_updated.assert_awaited_once()
    args = alerts.trailing_stop_updated.await_args.args
    assert args[0] == "X"
    assert args[1] == 0.9  # old stop
    assert args[2] == 1.0  # new stop (breakeven)
    # Existing behavior intact:
    assert "TRAILING STOP UPDATED" in all_text


def test_no_trailing_alert_when_no_update(monkeypatch):
    service = _make_service()
    service.get_position.return_value = None
    alerts = _make_alerts()

    _run_analyze(monkeypatch, alerts=alerts, service=service)

    alerts.trailing_stop_updated.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Guardian / risk-limit trade rejections (all five gates)
# ---------------------------------------------------------------------------


def test_alert_fires_on_portfolio_risk_rejection(monkeypatch):
    # Existing position risks $10 (entry 100, stop 90, qty 1); the new
    # trade would add $0.50; a $5 limit rejects it.
    position = _make_open_position(symbol="BTC", entry_price=100.0, stop_loss=90.0)
    service = _make_service(open_positions=[position])
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_RISK_LIMIT": 5.0},
    )

    alerts.trade_rejected.assert_awaited_once()
    trade_arg, reason_arg = alerts.trade_rejected.await_args.args
    assert trade_arg.symbol == "X"
    assert "risk" in reason_arg.lower()
    alerts.order_placed.assert_not_awaited()
    # Existing behavior intact:
    assert "Portfolio Risk Limit Exceeded" in all_text
    assert "The trade was not placed." in all_text


def test_alert_fires_on_exposure_rejection(monkeypatch):
    # Existing exposure $100 (entry 100 * qty 1); the new trade would
    # add $1.00; a $50 limit rejects it.
    position = _make_open_position(symbol="BTC", entry_price=100.0)
    service = _make_service(open_positions=[position])
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_EXPOSURE_LIMIT": 50.0},
    )

    alerts.trade_rejected.assert_awaited_once()
    alerts.order_placed.assert_not_awaited()
    assert "Portfolio Exposure Limit Exceeded" in all_text


def test_alert_fires_on_daily_loss_rejection(monkeypatch):
    today = datetime.now().strftime("%Y-%m-%d")
    closed_today = MagicMock()
    closed_today.realized_pnl = -150.0
    closed_today.exit_at = f"{today} 10:00:00"
    service = _make_service(closed_positions=[closed_today])
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_DAILY_LOSS_LIMIT": 100.0},
    )

    alerts.trade_rejected.assert_awaited_once()
    alerts.order_placed.assert_not_awaited()
    assert "Daily Loss Limit Reached" in all_text
    # The dedicated breaker alert also fires for this event.
    alerts.daily_loss_breaker.assert_awaited_once()


def test_alert_fires_on_max_open_positions_rejection(monkeypatch):
    service = _make_service(
        open_positions=[{"symbol": "BTC"}, {"symbol": "ETH"}]
    )
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_MAX_OPEN_POSITIONS": 2},
    )

    alerts.trade_rejected.assert_awaited_once()
    mop_arg = alerts.max_open_positions_reached.await_args.args[0]
    assert mop_arg.current_open_positions == 2
    assert mop_arg.limit == 2
    alerts.order_placed.assert_not_awaited()
    alerts.daily_loss_breaker.assert_not_awaited()
    assert "Maximum Open Positions Reached" in all_text


def test_alert_fires_on_symbol_concentration_rejection(monkeypatch):
    # One open position already on symbol X; a per-symbol cap of 1
    # rejects the second position on the same symbol.
    position = _make_open_position(symbol="X")
    service = _make_service(open_positions=[position])
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_MAX_POSITIONS_PER_SYMBOL": 1},
    )

    alerts.trade_rejected.assert_awaited_once()
    alerts.order_placed.assert_not_awaited()
    assert "Symbol Concentration Limit Exceeded" in all_text


def test_rejection_alert_receives_reason_text(monkeypatch):
    service = _make_service(
        open_positions=[{"symbol": "BTC"}, {"symbol": "ETH"}]
    )
    alerts = _make_alerts()

    _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_MAX_OPEN_POSITIONS": 2},
    )

    args, kwargs = alerts.trade_rejected.await_args
    reason = args[1] if len(args) > 1 else kwargs.get("reason", "")
    assert reason
    assert "position" in reason.lower()


def test_no_rejection_alert_when_trade_places(monkeypatch):
    service = _make_service()
    service.place_trade.return_value = _make_placed_order()
    alerts = _make_alerts()

    _run_analyze(monkeypatch, alerts=alerts, service=service)

    alerts.trade_rejected.assert_not_awaited()
    alerts.daily_loss_breaker.assert_not_awaited()
    alerts.max_open_positions_reached.assert_not_awaited()
    alerts.order_placed.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Daily loss breaker activation (dedicated event)
# ---------------------------------------------------------------------------


def test_daily_breaker_alert_receives_metrics(monkeypatch):
    today = datetime.now().strftime("%Y-%m-%d")
    closed_today = MagicMock()
    closed_today.realized_pnl = -150.0
    closed_today.exit_at = f"{today} 10:00:00"
    service = _make_service(closed_positions=[closed_today])
    alerts = _make_alerts()

    _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_DAILY_LOSS_LIMIT": 100.0},
    )

    metrics_arg = alerts.daily_loss_breaker.await_args.args[0]
    assert metrics_arg.breached is True
    assert metrics_arg.loss_limit == 100.0
    assert metrics_arg.realized_pnl_today == -150.0


def test_no_breaker_alert_when_not_breached(monkeypatch):
    today = datetime.now().strftime("%Y-%m-%d")
    closed_today = MagicMock()
    closed_today.realized_pnl = -20.0
    closed_today.exit_at = f"{today} 10:00:00"
    service = _make_service(closed_positions=[closed_today])
    alerts = _make_alerts()

    _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={"PORTFOLIO_DAILY_LOSS_LIMIT": 100.0},
    )

    alerts.daily_loss_breaker.assert_not_awaited()
    alerts.trade_rejected.assert_not_awaited()


# ---------------------------------------------------------------------------
# 7. Alerts disabled -> nothing fires, behavior unchanged
# ---------------------------------------------------------------------------


def test_disabled_alerts_never_fire(monkeypatch):
    service = _make_service(
        open_positions=[{"symbol": "BTC"}, {"symbol": "ETH"}]
    )
    alerts = _make_alerts()

    all_text, _ = _run_analyze(
        monkeypatch,
        alerts=alerts,
        service=service,
        settings_overrides={
            "PORTFOLIO_MAX_OPEN_POSITIONS": 2,
            "ALERTS_ENABLED": False,
        },
    )

    alerts.trade_rejected.assert_not_awaited()
    alerts.max_open_positions_reached.assert_not_awaited()
    alerts.order_placed.assert_not_awaited()
    # Existing behavior intact — the rejection message is still shown.
    assert "Maximum Open Positions Reached" in all_text


# ---------------------------------------------------------------------------
# Every alert failure is swallowed: trading continues
# ---------------------------------------------------------------------------


def test_alert_failure_does_not_break_analyze(monkeypatch):
    service = _make_service()
    service.place_trade.return_value = _make_placed_order()
    alerts = _make_alerts()
    alerts.order_placed = AsyncMock(side_effect=RuntimeError("boom"))

    all_text, _ = _run_analyze(monkeypatch, alerts=alerts, service=service)

    alerts.order_placed.assert_awaited_once()
    # The analysis flow completed and still reported the placed order.
    assert "Paper Order: PLACED" in all_text


# ---------------------------------------------------------------------------
# Lazy alert manager bootstrap
# ---------------------------------------------------------------------------


def test_get_alert_manager_returns_manager():
    manager = _handlers._get_alert_manager()
    assert isinstance(manager, AlertManager)


def test_get_alert_manager_is_cached():
    first = _handlers._get_alert_manager()
    second = _handlers._get_alert_manager()
    assert first is second
