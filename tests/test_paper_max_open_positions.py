"""Tests for Sprint 8B - Maximum Open Portfolio Positions limit.

The tests cover the pure helper functions, the manager wrapper, and the
handler-level behavior when PORTFOLIO_MAX_OPEN_POSITIONS is configured.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PROJECT_ROOT = "/c/Users/user/Desktop/MuzanTradeBot"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from trader.paper.max_open_positions import (
    MaxOpenPositionsManager,
    MaxOpenPositionsMetrics,
    MaxOpenPositionsResult,
    count_open_positions,
    evaluate,
    metrics,
)

import telegram_bot.handlers as _handlers


def test_count_open_positions_empty() -> None:
    assert count_open_positions(None) == 0
    assert count_open_positions([]) == 0


def test_count_open_positions_simple() -> None:
    positions = [
        {"symbol": "BTC", "side": "LONG"},
        {"symbol": "ETH", "side": "SHORT"},
    ]
    assert count_open_positions(positions) == 2


def test_count_open_positions_iterable_not_list() -> None:
    positions = ({"symbol": s} for s in ("BTC", "ETH", "SOL"))
    assert count_open_positions(positions) == 3


def test_evaluate_disabled() -> None:
    result = evaluate([{"symbol": "BTC"}], 0)
    assert result.allowed is True
    assert result.limit == 0
    assert result.reason == "portfolio max open positions limit disabled"


def test_evaluate_negative() -> None:
    result = evaluate([{"symbol": "BTC"}], -1)
    assert result.allowed is True
    assert result.limit == 0


def test_evaluate_allowed() -> None:
    result = evaluate([{"symbol": "BTC"}], 5)
    assert result.allowed is True
    assert result.current_open_positions == 1
    assert result.limit == 5
    assert result.remaining == 4
    assert result.reason == ""


def test_evaluate_exact_limit() -> None:
    positions = [{"symbol": s} for s in "ABCDE"]
    result = evaluate(positions, 5)
    assert result.allowed is False
    assert result.current_open_positions == 5
    assert result.limit == 5
    assert result.remaining == 0
    assert "maximum number of open positions" in result.reason


def test_evaluate_over_limit() -> None:
    positions = [{"symbol": s} for s in "ABCDEF"]
    result = evaluate(positions, 5)
    assert result.allowed is False
    assert result.current_open_positions == 6
    assert result.limit == 5
    assert result.remaining == 0


def test_metrics_disabled() -> None:
    m = metrics([{"symbol": "BTC"}], 0)
    assert m.enabled is False
    assert m.status == "disabled"
    assert m.usage_percent == 0.0
    assert m.remaining == -1


def test_metrics_ok() -> None:
    m = metrics([{"symbol": "BTC"}, {"symbol": "ETH"}], 5)
    assert m.enabled is True
    assert m.current_open_positions == 2
    assert m.limit == 5
    assert m.usage_percent == 40.0
    assert m.remaining == 3
    assert m.status == "ok"


def test_metrics_near_limit() -> None:
    m = metrics([{"symbol": s} for s in "ABCD"], 5)
    assert m.current_open_positions == 4
    assert m.limit == 5
    assert m.usage_percent == 80.0
    assert m.status == "near limit"


def test_metrics_at_limit() -> None:
    positions = [{"symbol": s} for s in "ABCDE"]
    m = metrics(positions, 5)
    assert m.current_open_positions == 5
    assert m.limit == 5
    assert m.usage_percent == 100.0
    assert m.remaining == 0
    assert m.status == "at limit"


def test_metrics_over_limit() -> None:
    positions = [{"symbol": s} for s in "ABCDEF"]
    m = metrics(positions, 5)
    assert m.current_open_positions == 6
    assert m.limit == 5
    assert m.usage_percent == 120.0
    assert m.remaining == -1
    assert m.status == "at limit"


def test_dataclass_frozen() -> None:
    result = evaluate([{"symbol": "BTC"}], 5)
    with pytest.raises(Exception):
        result.allowed = False  # type: ignore[misc]

    m = metrics([{"symbol": "BTC"}], 5)
    with pytest.raises(Exception):
        m.status = "changed"  # type: ignore[misc]


def _make_broker(positions: list[dict]) -> types.SimpleNamespace:
    broker = types.SimpleNamespace()
    broker.open_positions = positions
    return broker


def test_manager_disabled() -> None:
    broker = _make_broker([{"symbol": "BTC"}, {"symbol": "ETH"}])
    mgr = MaxOpenPositionsManager(broker, 0)
    assert mgr.evaluate().allowed is True
    assert mgr.metrics().enabled is False


def test_manager_allowed() -> None:
    broker = _make_broker([{"symbol": "BTC"}, {"symbol": "ETH"}])
    mgr = MaxOpenPositionsManager(broker, 5)
    result = mgr.evaluate()
    assert result.allowed is True
    assert result.current_open_positions == 2


def test_manager_blocked() -> None:
    broker = _make_broker([{"symbol": s} for s in "ABCDE"])
    mgr = MaxOpenPositionsManager(broker, 5)
    result = mgr.evaluate()
    assert result.allowed is False
    assert result.reason == "portfolio already has the maximum number of open positions"


def test_manager_account_attribute_name() -> None:
    account = types.SimpleNamespace()
    account.open_positions = [{"symbol": "BTC"}, {"symbol": "ETH"}]
    mgr = MaxOpenPositionsManager(account, 5)
    assert mgr.evaluate().current_open_positions == 2


def test_manager_missing_open_positions() -> None:
    account = types.SimpleNamespace()
    mgr = MaxOpenPositionsManager(account, 5)
    assert mgr.evaluate().current_open_positions == 0


def test_manager_none_open_positions() -> None:
    account = types.SimpleNamespace()
    account.open_positions = None  # type: ignore[assignment]
    mgr = MaxOpenPositionsManager(account, 5)
    assert mgr.evaluate().current_open_positions == 0


def _make_closed_positions(count: int) -> list:
    from trader.paper.order import PaperOrder, PaperOrderStatus, PaperOrderSide
    from trader.trade import Trade

    return [
        PaperOrder(
            order_id=f"fake-closed-{i}",
            symbol="BTC",
            side=PaperOrderSide.LONG,
            quantity=1.0,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=115.0,
            status=PaperOrderStatus.TAKEN_PROFIT,
            trade=Trade(
                symbol="BTC",
                price=100.0,
                trend="Bullish",
                signal="BUY",
                confidence=80,
                decision="BUY",
                risk_verdict="APPROVED",
                position_size=1.0,
                stop_loss=95.0,
                take_profit=115.0,
                side="LONG",
                approved=True,
                risk_reasons=[],
                timestamp="2026-01-01 00:00:00",
            ),
            created_at=None,
            filled_at=None,
            exit_at=None,
            exit_price=110.0,
            realized_pnl=10.0,
            unrealized_pnl=0.0,
            paper=True,
            notes=[],
        )
        for i in range(count)
    ]


def _make_open_positions(count: int) -> list:
    return [{"symbol": f"POS{i}", "side": "LONG"} for i in range(count)]


def _configure_and_run_analyze(
    monkeypatch,
    *,
    max_open_positions: int,
    open_positions: list,
    place_trade_return,
    fake_trade,
    analyze_coin_return,
    make_decision_return,
    validate_trade_return,
    log_trade_return,
):
    from config import settings as cfg_settings
    import telegram_bot.handlers as h

    original_paper = cfg_settings.settings.PAPER_TRADING
    original_limit = cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS
    original_daily = cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT
    original_risk = cfg_settings.settings.PORTFOLIO_RISK_LIMIT
    original_exposure = cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT
    original_conc_pos = cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL
    original_conc_exp = cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL
    try:
        cfg_settings.settings.PAPER_TRADING = True
        cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS = max_open_positions
        cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_RISK_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = 0
        cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL = 0.0

        broker = _make_broker(open_positions)
        service = MagicMock()
        service._broker = broker
        service.execution_backend.return_value = broker
        service.open_positions.return_value = open_positions
        # The mock service must behave like the real PaperTradingService:
        # get_position() returns None when there is no pre-existing open
        # position for the symbol. Without this, the auto-created MagicMock
        # position makes _before_analysis_paper_check raise TypeError
        # (`MagicMock <= 0`), which analyze() swallows, silently aborting
        # the paper-trading branch before place_trade is reached.
        service.get_position.return_value = None
        service.place_trade.return_value = place_trade_return

        monkeypatch.setattr(h, "analyze_coin", lambda _: analyze_coin_return)
        monkeypatch.setattr(h, "make_decision", lambda r: make_decision_return)
        monkeypatch.setattr(h, "validate_trade", lambda d, r, p: validate_trade_return)
        monkeypatch.setattr(h, "build_trade", lambda r, v, p: fake_trade)
        monkeypatch.setattr(h, "log_trade", log_trade_return)
        monkeypatch.setattr(h, "_get_paper_service", lambda: service)
        monkeypatch.setattr(h, "PAPER_STATE_PATH", "/tmp/ignore.json")

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        update.message.chat.id = 123
        update.args = ["X"]
        update.effective_user = MagicMock()
        update.effective_chat = MagicMock()
        update.effective_message = MagicMock()
        asyncio.run(h.analyze(update, MagicMock()))
        replies = update.message.reply_text.await_args_list
        if not replies:
            raise AssertionError("Expected analyze() to call reply_text at least once")
        last_reply = replies[-1][0][0]
        return last_reply, replies
    finally:
        cfg_settings.settings.PAPER_TRADING = original_paper
        cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS = original_limit
        cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT = original_daily
        cfg_settings.settings.PORTFOLIO_RISK_LIMIT = original_risk
        cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT = original_exposure
        cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = original_conc_pos
        cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL = original_conc_exp


def test_max_open_positions_gate_blocks_at_limit(monkeypatch) -> None:
    fake_trade = MagicMock()
    fake_trade.symbol = "X"
    fake_trade.price = 1.0
    fake_trade.position_size = 1.0
    fake_trade.stop_loss = 0.5
    fake_trade.take_profit = 2.0
    fake_trade.side = "LONG"
    fake_trade.approved = True
    fake_trade.risk_verdict = "APPROVED"
    fake_trade.risk_reasons = []

    reply, replies = _configure_and_run_analyze(
        monkeypatch,
        max_open_positions=2,
        open_positions=[{"symbol": "BTC"}, {"symbol": "ETH"}],
        place_trade_return=None,
        fake_trade=fake_trade,
        analyze_coin_return={"symbol": "X", "price": 1.0},
        make_decision_return=MagicMock(approved=True),
        validate_trade_return=MagicMock(approved=True),
        log_trade_return=MagicMock(),
    )

    all_text = "\n".join(r[0][0] for r in replies)

    assert "Maximum Open Positions Reached" in all_text
    assert "The trade was not placed" in all_text
    assert "Close existing positions to free up open-position capacity" in all_text


def test_max_open_positions_gate_allows_when_under_limit(monkeypatch) -> None:
    placed_order = MagicMock()
    placed_order.order_id = "PAPE-X-PLACE-001"
    placed_order.status = MagicMock()
    placed_order.status.value = "OPEN"
    placed_order.quantity = 1.0
    placed_order.entry_price = 1.0
    placed_order.stop_loss = 0.5
    placed_order.take_profit = 2.0
    placed_order.side = MagicMock()
    placed_order.side.value = "LONG"

    fake_trade = MagicMock()
    fake_trade.symbol = "X"
    fake_trade.price = 1.0
    fake_trade.position_size = 1.0
    fake_trade.stop_loss = 0.5
    fake_trade.take_profit = 2.0
    fake_trade.side = "LONG"
    fake_trade.approved = True
    fake_trade.risk_verdict = "APPROVED"
    fake_trade.risk_reasons = []

    reply, replies = _configure_and_run_analyze(
        monkeypatch,
        max_open_positions=5,
        open_positions=[{"symbol": "BTC"}],
        place_trade_return=placed_order,
        fake_trade=fake_trade,
        analyze_coin_return={"symbol": "X", "price": 1.0},
        make_decision_return=MagicMock(approved=True),
        validate_trade_return=MagicMock(approved=True),
        log_trade_return=MagicMock(),
    )

    all_text = "\n".join(r[0][0] for r in replies)

    if "Maximum Open Positions Reached" in all_text:
        assert False, "Expected paper order placement but got a max-open-positions rejection"

    assert "Paper Order: PLACED" in all_text
    assert "PAPE-X-PLACE-001" in all_text
    assert "Maximum Open Positions Reached" not in all_text




def _stats_runner(monkeypatch, max_open_positions: int, open_positions: list):
    from config import settings as cfg_settings

    original_limit = cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS
    original_risk = cfg_settings.settings.PORTFOLIO_RISK_LIMIT
    original_exposure = cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT
    original_daily = cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT
    original_conc_pos = cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL
    original_conc_exp = cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL
    try:
        cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS = max_open_positions
        cfg_settings.settings.PORTFOLIO_RISK_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = 0
        cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL = 0.0

        broker = _make_broker(open_positions)
        service = MagicMock()
        service.summary.return_value = MagicMock(
            current_balance=1050.0,
            initial_balance=1000.0,
            equity=1050.0,
            closed_count=2,
            total_realized_pnl=50.0,
            open_pnl=0.0,
        )
        service.closed_positions.return_value = _make_closed_positions(2)
        service.open_positions.return_value = open_positions
        service._broker = broker
        service.execution_backend.return_value = broker

        with patch("telegram_bot.handlers._get_paper_service", return_value=service):
            update = MagicMock()
            update.message.reply_text = AsyncMock()
            asyncio.run(_handlers.stats_command(update, MagicMock()))
            return update.message.reply_text.await_args[0][0]
    finally:
        cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS = original_limit
        cfg_settings.settings.PORTFOLIO_RISK_LIMIT = original_risk
        cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT = original_exposure
        cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT = original_daily
        cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = original_conc_pos
        cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL = original_conc_exp


def test_max_open_positions_stats_section_only_when_enabled(monkeypatch) -> None:
    called_text = _stats_runner(monkeypatch, 0, [])
    assert "Max Open Positions" not in called_text


def test_max_open_positions_stats_section_when_enabled(monkeypatch) -> None:
    called_text = _stats_runner(monkeypatch, 5, _make_open_positions(2))
    assert "Max Open Positions" in called_text
    assert "Current Open Positions: 2" in called_text
    assert "Limit: 5" in called_text
    assert "Capacity Used: 40.0%" in called_text
    assert "Remaining: 3" in called_text
    assert "Status: ok" in called_text


def test_max_open_positions_stats_section_near_limit(monkeypatch) -> None:
    called_text = _stats_runner(monkeypatch, 5, _make_open_positions(4))
    assert "Max Open Positions" in called_text
    assert "Current Open Positions: 4" in called_text
    assert "Limit: 5" in called_text
    assert "Capacity Used: 80.0%" in called_text
    assert "Remaining: 1" in called_text
    assert "Status: near limit" in called_text


def test_max_open_positions_stats_section_at_limit(monkeypatch) -> None:
    called_text = _stats_runner(monkeypatch, 5, _make_open_positions(5))
    assert "Max Open Positions" in called_text
    assert "Current Open Positions: 5" in called_text
    assert "Limit: 5" in called_text
    assert "Capacity Used: 100.0%" in called_text
    assert "Remaining: 0" in called_text
    assert "Status: at limit" in called_text


# ---------------------------------------------------------------------------
# Real execution backend integration (Task 3, Part 1)
# ---------------------------------------------------------------------------

def _real_trade(
    symbol: str = "BTC",
    price: float = 100.0,
    quantity: float = 1.0,
    side: str = "LONG",
):
    """Build a real approved Trade for placement through the real broker."""
    from trader.trade import Trade

    return Trade(
        symbol=symbol,
        price=price,
        trend="Bullish" if side == "LONG" else "Bearish",
        signal="BUY" if side == "LONG" else "SELL",
        confidence=80,
        decision="BUY" if side == "LONG" else "SELL",
        risk_verdict="APPROVED",
        position_size=quantity,
        stop_loss=price * 0.95,
        take_profit=price * 1.05,
        side=side,
        approved=True,
        risk_reasons=[],
        timestamp="2026-01-01 00:00:00",
    )


def _real_service_with(*trades):
    """Create a real PaperTradingService holding the given open positions."""
    from trader.paper.service import PaperTradingService

    service = PaperTradingService.create_new_account(1000.0)
    for trade in trades:
        assert service.place_trade(trade) is not None
    return service


def test_manager_counts_real_paper_broker_positions() -> None:
    service = _real_service_with(
        _real_trade("BTC", 100.0),
        _real_trade("ETH", 2000.0),
    )
    mgr = MaxOpenPositionsManager(service.execution_backend(), 5)

    result = mgr.evaluate()

    assert result.current_open_positions == 2
    assert result.allowed is True
    assert result.remaining == 3


def test_manager_blocks_real_paper_broker_at_limit() -> None:
    service = _real_service_with(
        _real_trade("BTC", 100.0),
        _real_trade("ETH", 2000.0),
    )
    mgr = MaxOpenPositionsManager(service.execution_backend(), 2)

    result = mgr.evaluate()

    assert result.allowed is False
    assert result.current_open_positions == 2
    assert result.reason == (
        "portfolio already has the maximum number of open positions"
    )


def test_manager_metrics_use_real_paper_broker() -> None:
    service = _real_service_with(
        _real_trade("BTC", 100.0),
        _real_trade("ETH", 2000.0),
    )
    mgr = MaxOpenPositionsManager(service.execution_backend(), 5)

    m = mgr.metrics()

    assert m.enabled is True
    assert m.current_open_positions == 2
    assert m.usage_percent == 40.0
    assert m.remaining == 3
    assert m.status == "ok"


def test_manager_counts_nothing_for_reloaded_empty_account() -> None:
    from trader.paper.service import PaperTradingService

    service = PaperTradingService.create_new_account(1000.0)
    mgr = MaxOpenPositionsManager(service.execution_backend(), 2)

    assert mgr.evaluate().current_open_positions == 0
    assert mgr.metrics().current_open_positions == 0


def test_manager_accepts_paper_account_directly() -> None:
    service = _real_service_with(_real_trade("BTC", 100.0))

    result = MaxOpenPositionsManager(service._account, 2).evaluate()

    assert result.current_open_positions == 1
    assert result.allowed is True


def _run_analyze_with_real_service(
    monkeypatch,
    tmp_path,
    *,
    service,
    new_trade,
    limit: int,
):
    """Run /analyze end-to-end against a real paper service and broker."""
    from config import settings as cfg_settings
    import telegram_bot.handlers as h

    names = (
        "PAPER_TRADING",
        "PORTFOLIO_MAX_OPEN_POSITIONS",
        "PORTFOLIO_DAILY_LOSS_LIMIT",
        "PORTFOLIO_RISK_LIMIT",
        "PORTFOLIO_EXPOSURE_LIMIT",
        "PORTFOLIO_MAX_POSITIONS_PER_SYMBOL",
        "PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL",
        "ALERTS_ENABLED",
    )
    originals = {name: getattr(cfg_settings.settings, name) for name in names}
    try:
        cfg_settings.settings.PAPER_TRADING = True
        cfg_settings.settings.PORTFOLIO_MAX_OPEN_POSITIONS = limit
        cfg_settings.settings.PORTFOLIO_DAILY_LOSS_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_RISK_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_EXPOSURE_LIMIT = 0.0
        cfg_settings.settings.PORTFOLIO_MAX_POSITIONS_PER_SYMBOL = 0
        cfg_settings.settings.PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL = 0.0
        cfg_settings.settings.ALERTS_ENABLED = False

        monkeypatch.setattr(h, "check_rate_limit", lambda *a, **k: True)
        monkeypatch.setattr(
            h,
            "analyze_coin",
            lambda _: {"symbol": new_trade.symbol, "price": new_trade.price},
        )
        monkeypatch.setattr(h, "make_decision", lambda r: MagicMock(approved=True))
        monkeypatch.setattr(
            h, "validate_trade", lambda d, r, p: MagicMock(approved=True)
        )
        monkeypatch.setattr(h, "build_trade", lambda r, v, p: new_trade)
        monkeypatch.setattr(h, "log_trade", MagicMock())
        monkeypatch.setattr(h, "_get_paper_service", lambda: service)
        monkeypatch.setattr(h, "PAPER_STATE_PATH", tmp_path / "paper_state.json")

        update = MagicMock()
        update.message.reply_text = AsyncMock()
        update.message.chat.id = 123
        asyncio.run(h.analyze(update, MagicMock()))
        return update.message.reply_text.await_args_list
    finally:
        for name, value in originals.items():
            setattr(cfg_settings.settings, name, value)


def test_analyze_blocks_new_trade_at_limit_with_real_service(
    monkeypatch, tmp_path
) -> None:
    service = _real_service_with(
        _real_trade("BTC", 100.0),
        _real_trade("ETH", 2000.0),
    )

    replies = _run_analyze_with_real_service(
        monkeypatch,
        tmp_path,
        service=service,
        new_trade=_real_trade("SOL", 50.0),
        limit=2,
    )
    text = "\n".join(reply[0][0] for reply in replies)

    assert "Maximum Open Positions Reached" in text
    assert "Current Open Positions: 2" in text
    assert "Max Open Positions: 2" in text
    assert "The trade was not placed." in text
    assert "Paper Order: PLACED" not in text
    assert len(service.open_positions()) == 2


def test_analyze_places_trade_under_limit_with_real_service(
    monkeypatch, tmp_path
) -> None:
    service = _real_service_with(
        _real_trade("BTC", 100.0),
        _real_trade("ETH", 2000.0),
    )

    replies = _run_analyze_with_real_service(
        monkeypatch,
        tmp_path,
        service=service,
        new_trade=_real_trade("SOL", 50.0),
        limit=3,
    )
    text = "\n".join(reply[0][0] for reply in replies)

    assert "Maximum Open Positions Reached" not in text
    assert "Paper Order: PLACED" in text
    assert len(service.open_positions()) == 3
    assert service.get_position("SOL") is not None
