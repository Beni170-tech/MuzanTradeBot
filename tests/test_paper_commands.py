"""
Tests for telegram_bot.handlers paper trading commands (Sprint 6B).
Tests the /paper, /positions, /history, and /close command handlers.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio

from telegram.ext import ContextTypes

from telegram_bot.handlers import (
    _format_pnl_label,
    close_command,
    history_command,
    paper_command,
    positions_command,
)
def _make_update():
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update
def _make_context():
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.args = []
    return context
def _make_service(
    open_positions=None,
    closed_positions=None,
    current_balance=10000.0,
    closed_count=0,
):
    service = MagicMock()
    service.summary.return_value = MagicMock(
        current_balance=current_balance,
        initial_balance=current_balance,
        equity=current_balance,
        total_realized_pnl=0.0,
        open_pnl=0.0,
        closed_count=closed_count,
    )
    service.open_positions.return_value = open_positions or []
    service.closed_positions.return_value = closed_positions or []
    service.close_position.return_value = None
    service.save_account = MagicMock()
    return service
def _make_order(symbol="BTC", side="LONG", entry_price=100.0, quantity=1.0,
                realized_pnl=0.0, exit_price=0.0, exit_status="CLOSED_MANUAL",
                exit_at=None, unrealized_pnl=0.0):
    order = MagicMock()
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side
    order.entry_price = entry_price
    order.quantity = quantity
    order.realized_pnl = realized_pnl
    order.exit_price = exit_price
    order.exit_status = MagicMock()
    order.exit_status.value = exit_status
    order.exit_at = exit_at
    order.unrealized_pnl_for.return_value = unrealized_pnl
    return order
class TestFormatPnlLabel:
    def test_win(self):
        assert _format_pnl_label(_make_order(realized_pnl=500.0)) == "WIN"
    def test_loss(self):
        assert _format_pnl_label(_make_order(realized_pnl=-300.0)) == "LOSS"
    def test_breakeven(self):
        assert _format_pnl_label(_make_order(realized_pnl=0.0)) == "BREAKEVEN"
    def test_small_positive(self):
        assert _format_pnl_label(_make_order(realized_pnl=0.01)) == "WIN"
    def test_small_negative(self):
        assert _format_pnl_label(_make_order(realized_pnl=-0.01)) == "LOSS"
class TestPaperCommand:
    @pytest.mark.asyncio
    async def test_empty_account(self):
        update = _make_update()
        context = _make_context()
        service = _make_service()
        with patch("telegram_bot.handlers._get_paper_service", return_value=service):
            await paper_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "📄 Paper Account Summary" in msg
        assert "Balance: $10000.00" in msg
        assert "Equity: $10000.00" in msg
        assert "Open Positions: 0" in msg
        assert "Closed Positions: 0" in msg
        assert "Realized PnL: $0.00" in msg
        assert "Unrealized PnL: $0.00" in msg
    @pytest.mark.asyncio
    async def test_with_open_position(self):
        update = _make_update()
        context = _make_context()
        pos = _make_order(symbol="BTC", side="LONG", entry_price=50000.0, quantity=0.5,
                          unrealized_pnl=150.0)
        service = _make_service(open_positions=[pos])
        with patch("telegram_bot.handlers._get_paper_service", return_value=service):
            await paper_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "📄 Paper Account Summary" in msg
        assert "Open Positions: 1" in msg
        assert "BTC LONG" in msg
        assert "Entry: $50000.00" in msg
        assert "Qty: 0.5" in msg
    @pytest.mark.asyncio
    async def test_with_closed_position(self):
        update = _make_update()
        context = _make_context()
        closed = _make_order(symbol="ETH", side="LONG", entry_price=2000.0,
                             quantity=1.0, realized_pnl=500.0, exit_price=2500.0,
                             exit_status="TAKEN_PROFIT")
        service = _make_service(closed_positions=[closed], closed_count=1,
                                current_balance=10500.0)
        with patch("telegram_bot.handlers._get_paper_service", return_value=service):
            await paper_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "📜 Recently Closed:" in msg
        assert "ETH - LONG" in msg
        assert "Exit: $2500.00" in msg
        assert "PnL: $500.00 (WIN)" in msg
    @pytest.mark.asyncio
    async def test_calls_service_once(self):
        update = _make_update()
        context = _make_context()
        with patch("telegram_bot.handlers._get_paper_service") as mock_get:
            mock_get.return_value = _make_service()
            await paper_command(update, context)
        mock_get.assert_called_once()
class TestPositionsCommand:
    @pytest.mark.asyncio
    async def test_no_positions(self):
        update = _make_update()
        context = _make_context()
        with patch("telegram_bot.handlers._get_paper_service") as mock_get:
            mock_get.return_value = _make_service()
            await positions_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "📭 No open paper positions." in msg
    @pytest.mark.asyncio
    async def test_single_position(self):
        update = _make_update()
        context = _make_context()
        pos = _make_order(symbol="BTC", side="LONG", entry_price=40000.0, quantity=0.25)
        with patch("telegram_bot.handlers._get_paper_service") as mock_get:
            mock_get.return_value = _make_service(open_positions=[pos])
            await positions_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "📌 Open Paper Positions:" in msg
        assert "BTC LONG" in msg
        assert "Entry: $40000.00" in msg
        assert "Qty: 0.25" in msg
    @pytest.mark.asyncio
    async def test_multiple_positions(self):
        update = _make_update()
        context = _make_context()
        btc = _make_order(symbol="BTC", side="LONG", entry_price=40000.0, quantity=0.25)
        eth = _make_order(symbol="ETH", side="SHORT", entry_price=2000.0, quantity=5.0)
        with patch("telegram_bot.handlers._get_paper_service") as mock_get:
            mock_get.return_value = _make_service(open_positions=[btc, eth])
            await positions_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "BTC LONG" in msg
        assert "ETH SHORT" in msg
class TestHistoryCommand:
    @pytest.mark.asyncio
    async def test_no_history(self):
        update = _make_update()
        context = _make_context()
        with patch("telegram_bot.handlers._get_paper_service") as mock_get:
            mock_get.return_value = _make_service()
            await history_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "📭 No closed paper trades yet." in msg
    @pytest.mark.asyncio
    async def test_single_trade(self):
        update = _make_update()
        context = _make_context()
        closed = _make_order(symbol="BTC", side="LONG", entry_price=30000.0,
                             quantity=0.5, realized_pnl=2500.0, exit_price=35000.0,
                             exit_status="TAKEN_PROFIT")
        service = _make_service(closed_positions=[closed], closed_count=1,
                                current_balance=102500.0)
        with patch("telegram_bot.handlers._get_paper_service", return_value=service):
            await history_command(update, context)
        msg = update.message.reply_text.call_args[0][0]
        assert "📜 Closed Paper Trades" in msg
        assert "BTC LONG" in msg
        assert "Entry: $30000.00" in msg
        assert "Exit: $35000.00" in msg
        assert "Qty: 0.5" in msg
        assert "Realized PnL: $2500.00 (WIN)" in msg
        assert "Closed: TAKEN_PROFIT" in msg
        assert "Date: N/A" in msg
    @pytest.mark.asyncio
    async def test_loss_label(self):
        update = _make_update()
        context = _make_context()
        closed = _make_order(
            symbol="ETH", side="SHORT", entry_price=2500.0,
            exit_status="STOPPED",
        )