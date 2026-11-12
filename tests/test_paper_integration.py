"""Tests for paper trading integration in Telegram handlers (Sprint 6A)."""

import pytest
from pathlib import Path

from telegram_bot.handlers import (
    PAPER_STATE_PATH,
    _get_paper_service,
)


@pytest.fixture
def mock_paper_state_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Mock PAPER_STATE_PATH to a temporary file."""
    path = tmp_path / "paper_state.json"
    monkeypatch.setattr(
        "telegram_bot.handlers.PAPER_STATE_PATH", path
    )
    return path


class TestPaperServiceLazyLoading:
    """Tests for the lazy-loading paper service mechanism."""

    def test_get_paper_service_creates_new_account_when_no_file_exists(
        self, mock_paper_state_path: Path
    ) -> None:
        import telegram_bot.handlers as handlers

        # Ensure no existing service or file
        handlers._paper_service = None
        assert not mock_paper_state_path.exists()

        service = handlers._get_paper_service()
        assert service is not None
        assert isinstance(service, handlers.PaperTradingService)

        # Verify a new account was created
        summary = service.summary()
        assert summary.equity == 1000.0  # default ACCOUNT_BALANCE

    def test_get_paper_service_loads_existing_account(
        self, mock_paper_state_path: Path
    ) -> None:
        import telegram_bot.handlers as handlers

        # First, create and save a service
        handlers._paper_service = None
        original_service = handlers.PaperTradingService.create_new_account(2500.0)
        mock_paper_state_path.parent.mkdir(parents=True, exist_ok=True)
        original_service.save_account(mock_paper_state_path)

        # Now load it via the lazy loader
        service = handlers._get_paper_service()
        assert service is not None
        summary = service.summary()
        assert summary.equity == 2500.0

    def test_get_paper_service_returns_same_instance_on_subsequent_calls(
        self, mock_paper_state_path: Path
    ) -> None:
        import telegram_bot.handlers as handlers

        handlers._paper_service = None
        mock_paper_state_path.parent.mkdir(parents=True, exist_ok=True)

        service1 = handlers._get_paper_service()
        service2 = handlers._get_paper_service()

        assert service1 is service2


class TestPaperTradingIntegrationLogic:
    """Tests for the paper trading integration logic (without Telegram)."""

    def test_place_approved_trade_via_service(self, mock_paper_state_path: Path) -> None:
        import telegram_bot.handlers as handlers

        handlers._paper_service = None
        mock_paper_state_path.parent.mkdir(parents=True, exist_ok=True)

        service = handlers._get_paper_service()

        from trader.trade import Trade
        trade = Trade(
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

        order = service.place_trade(trade)
        assert order is not None
        assert order.symbol == "BTC"
        assert order.status.name == "OPEN"

        # Save state
        service.save_account(mock_paper_state_path)
        assert mock_paper_state_path.exists()

    def test_place_unapproved_trade_returns_none(self, mock_paper_state_path: Path) -> None:
        import telegram_bot.handlers as handlers

        handlers._paper_service = None
        mock_paper_state_path.parent.mkdir(parents=True, exist_ok=True)

        service = handlers._get_paper_service()

        from trader.trade import Trade
        trade = Trade(
            symbol="BTC",
            price=100.0,
            trend="Bullish",
            signal="BUY",
            confidence=80,
            decision="WAIT",
            risk_verdict="BLOCKED",
            position_size=0.0,
            stop_loss=0.0,
            take_profit=0.0,
            side="LONG",
            approved=False,
        )

        order = service.place_trade(trade)
        assert order is None

    def test_paper_state_persists_after_placement(
        self, mock_paper_state_path: Path
    ) -> None:
        import telegram_bot.handlers as handlers

        handlers._paper_service = None
        mock_paper_state_path.parent.mkdir(parents=True, exist_ok=True)

        service = handlers._get_paper_service()

        from trader.trade import Trade
        trade = Trade(
            symbol="ETH",
            price=2000.0,
            trend="Bullish",
            signal="BUY",
            confidence=75,
            decision="BUY",
            risk_verdict="APPROVED",
            position_size=1.0,
            stop_loss=1960.0,
            take_profit=2080.0,
            side="LONG",
            approved=True,
        )

        service.place_trade(trade)
        service.save_account(mock_paper_state_path)

        # Load a fresh service from the saved state
        handlers._paper_service = None
        loaded_service = handlers.PaperTradingService.load_account(mock_paper_state_path)
        summary = loaded_service.summary()

        assert summary.closed_count == 0
        assert len(summary.open_positions) == 1
        assert summary.open_positions[0].symbol == "ETH"
        assert summary.equity == 1000.0

    def test_multiple_approved_trades_stack_positions(
        self, mock_paper_state_path: Path
    ) -> None:
        import telegram_bot.handlers as handlers

        handlers._paper_service = None
        mock_paper_state_path.parent.mkdir(parents=True, exist_ok=True)

        service = handlers._get_paper_service()

        from trader.trade import Trade

        btc_trade = Trade(
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
        eth_trade = Trade(
            symbol="ETH",
            price=2000.0,
            trend="Bullish",
            signal="BUY",
            confidence=75,
            decision="BUY",
            risk_verdict="APPROVED",
            position_size=1.0,
            stop_loss=1960.0,
            take_profit=2080.0,
            side="LONG",
            approved=True,
        )

        service.place_trade(btc_trade)
        service.place_trade(eth_trade)
        service.save_account(mock_paper_state_path)

        assert len(service.open_positions()) == 2

        # Reload and verify
        handlers._paper_service = None
        loaded = handlers.PaperTradingService.load_account(mock_paper_state_path)
        assert len(loaded.open_positions()) == 2

    def test_closing_position_updates_state(
        self, mock_paper_state_path: Path
    ) -> None:
        import telegram_bot.handlers as handlers

        handlers._paper_service = None
        mock_paper_state_path.parent.mkdir(parents=True, exist_ok=True)

        service = handlers._get_paper_service()

        from trader.trade import Trade
        from trader.paper.order import PaperOrderStatus

        trade = Trade(
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

        service.place_trade(trade)
        closed = service.close_position(
            symbol="BTC",
            exit_price=110.0,
            status=PaperOrderStatus.TAKEN_PROFIT,
        )

        assert closed is not None
        assert closed.realized_pnl == 100.0
        assert service.summary().equity == 1100.0

        service.save_account(mock_paper_state_path)

        # Reload and verify
        handlers._paper_service = None
        loaded = handlers.PaperTradingService.load_account(mock_paper_state_path)
        assert loaded.summary().equity == 1100.0
        assert loaded.summary().closed_count == 1


class TestExitStatusHelperRegression:
    """Sprint 9A regression tests.

    Real PaperOrder objects have no ``exit_status`` attribute; the
    terminal status lives in ``status`` after ``record_exit``. The
    display paths used to read ``exit_status`` directly and crashed
    (or silently aborted) on production data. These tests pin the
    fixed behavior using REAL PaperOrder objects, not mocks.
    """

    def _make_closed_real_order(self, status):
        from trader.paper.account import PaperAccount
        from trader.paper.broker import PaperBroker
        from trader.paper.order import PaperOrderStatus
        from trader.trade import Trade

        trade = Trade(
            symbol="BTC",
            price=100.0,
            trend="Bullish",
            signal="BUY",
            confidence=80,
            decision="BUY",
            risk_verdict="APPROVED",
            position_size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            side="LONG",
            approved=True,
        )
        account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
        broker = PaperBroker(account)
        order = broker.place_order(trade)
        return broker.close_order(order, 98.0, status)

    def test_real_order_has_no_exit_status_attribute(self):
        from trader.paper.order import PaperOrderStatus

        order = self._make_closed_real_order(PaperOrderStatus.STOPPED)
        assert not hasattr(order, "exit_status")
        assert order.status.value == "STOPPED"

    def test_paper_exit_status_reads_real_order(self):
        from trader.paper.order import PaperOrderStatus, paper_exit_status

        order = self._make_closed_real_order(PaperOrderStatus.STOPPED)
        assert paper_exit_status(order) == "STOPPED"

    def test_paper_exit_status_reads_real_tp_order(self):
        from trader.paper.order import PaperOrderStatus, paper_exit_status

        order = self._make_closed_real_order(PaperOrderStatus.TAKEN_PROFIT)
        assert paper_exit_status(order) == "TAKEN_PROFIT"

    def test_paper_exit_status_prefers_legacy_exit_status(self):
        from trader.paper.order import paper_exit_status
        from unittest.mock import MagicMock

        legacy = MagicMock()
        legacy.exit_status.value = "STOPPED"
        legacy.status.value = "TAKEN_PROFIT"
        assert paper_exit_status(legacy) == "STOPPED"

    def test_paper_exit_status_falls_back_to_status(self):
        from trader.paper.order import PaperOrderStatus, paper_exit_status

        order = self._make_closed_real_order(PaperOrderStatus.STOPPED)
        # Simulate the legacy attribute being absent entirely.
        assert paper_exit_status(order) == "STOPPED"

    def test_paper_exit_status_default_when_nothing_present(self):
        from trader.paper.order import paper_exit_status

        assert paper_exit_status(object(), default="CLOSED") == "CLOSED"
        assert paper_exit_status(None, default="CLOSED") == "CLOSED"

    def test_paper_exit_date_with_real_string_exit_at(self):
        from trader.paper.order import PaperOrderStatus, paper_exit_date

        order = self._make_closed_real_order(PaperOrderStatus.STOPPED)
        # Real exit_at is a string like '2026-09-08 14:36:48'.
        assert isinstance(order.exit_at, str)
        formatted = paper_exit_date(order)
        assert formatted == order.exit_at[:16]
        assert len(formatted) == 16

    def test_paper_exit_date_datetime_like_object(self):
        """If exit_at is ever a datetime, strftime is used unchanged."""
        from datetime import datetime
        from trader.paper.order import paper_exit_date
        from unittest.mock import MagicMock

        order = MagicMock()
        order.exit_at = datetime(2026, 9, 8, 14, 36)
        assert paper_exit_date(order) == "2026-09-08 14:36"
        assert paper_exit_date(order, format="%Y-%m-%d") == "2026-09-08"

    def test_paper_exit_date_missing_exits_at(self):
        from trader.paper.order import paper_exit_date
        from unittest.mock import MagicMock

        order = MagicMock()
        order.exit_at = None
        assert paper_exit_date(order) == "N/A"
        order2 = MagicMock()
        order2.exit_at = ""
        assert paper_exit_date(order2) == "N/A"

    def test_sltp_message_uses_helper_not_attribute(self):
        """The SL/TP closure message must render for a REAL order."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import asyncio

        import telegram_bot.handlers as h
        from trader.paper.order import PaperOrderStatus

        order = self._make_closed_real_order(PaperOrderStatus.STOPPED)

        service = MagicMock()
        service.get_position.return_value = order  # pre-existing position
        service.close_position.return_value = order
        service.place_trade.return_value = None  # scenario ends at the closure
        service._open_positions_fixture = []

        # Reuse the alerts-integration harness shape for a focused run.
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

        trade = MagicMock()
        trade.symbol = "BTC"
        trade.price = 50.0  # below stop_loss 98 -> SL closure
        trade.position_size = 1.0
        trade.stop_loss = 98.0
        trade.take_profit = 104.0
        trade.side = "LONG"
        trade.approved = True
        trade.risk_verdict = "APPROVED"
        trade.risk_reasons = []

        service._broker = MagicMock()
        service._broker.get_open_positions.return_value = ()
        service._broker.account.closed_positions = []

        from config import settings as cfg_settings

        original_alerts = cfg_settings.settings.ALERTS_ENABLED
        try:
            cfg_settings.settings.ALERTS_ENABLED = True
            with patch.object(h, "analyze_coin", return_value={"symbol": "BTC", "price": 50.0}), \
                 patch.object(h, "get_current_price", return_value=50.0), \
                 patch.object(h, "make_decision", return_value=MagicMock(approved=True)), \
                 patch.object(h, "validate_trade", return_value=MagicMock(approved=True)), \
                 patch.object(h, "build_trade", return_value=trade), \
                 patch.object(h, "log_trade", MagicMock()), \
                 patch.object(h, "_get_paper_service", return_value=service), \
                 patch.object(h, "_alert_manager", alerts), \
                 patch.object(h, "PAPER_STATE_PATH", "/tmp/ignore.json"):
                update = MagicMock()
                update.message.reply_text = AsyncMock()
                asyncio.run(h.analyze(update, MagicMock()))
        finally:
            cfg_settings.settings.ALERTS_ENABLED = original_alerts

        msg = update.message.reply_text.await_args[0][0]
        assert "Paper Order: CLOSED (SL/TP)" in msg
        assert "Exit Status: STOPPED" in msg  # was the crash point
        # The flow must not have aborted: the SL alert fired too.
        alerts.stop_loss_triggered.assert_awaited_once()

    def test_history_command_works_with_real_orders(self):
        """'/history used pos.exit_status.value and crashed on real
        orders; it must now render the full list.'"""
        from unittest.mock import AsyncMock, MagicMock, patch

        import asyncio

        import telegram_bot.handlers as h
        from trader.paper.account import PaperAccount
        from trader.paper.broker import PaperBroker
        from trader.paper.order import PaperOrderStatus
        from trader.trade import Trade

        trade = Trade(
            symbol="BTC",
            price=100.0,
            trend="Bullish",
            signal="BUY",
            confidence=80,
            decision="BUY",
            risk_verdict="APPROVED",
            position_size=1.0,
            stop_loss=98.0,
            take_profit=104.0,
            side="LONG",
            approved=True,
        )
        account = PaperAccount(initial_balance=1000.0, current_balance=1000.0)
        broker = PaperBroker(account)
        order = broker.place_order(trade)
        closed = broker.close_order(order, 104.0, PaperOrderStatus.TAKEN_PROFIT)

        service = MagicMock()
        service.closed_positions.return_value = [closed]

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        with patch.object(h, "_get_paper_service", return_value=service):
            asyncio.run(h.history_command(update, MagicMock()))

        msg = update.message.reply_text.await_args[0][0]
        assert "Closed Paper Trades" in msg
        assert "Closed: TAKEN_PROFIT" in msg  # was the crash point
        assert "Date: N/A" in msg or "Date: 2" in msg
