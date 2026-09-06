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
