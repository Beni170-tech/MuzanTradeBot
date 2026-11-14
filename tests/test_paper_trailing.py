"""
Tests for automatic trailing stop management for paper trading positions (Sprint 6E).

The _apply_trailing_stop helper runs during /analyze using the already-fetched
market price and adjusts an open paper position's stop loss once the position
is at least 1R into profit.
"""

from unittest.mock import MagicMock

import pytest

import telegram_bot.handlers as handlers
from telegram_bot.handlers import _apply_trailing_stop, _is_trail_active
from trader.paper.order import PaperOrder, PaperOrderSide
from trader.paper.service import PaperTradingService
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(
    symbol="BTC",
    side="LONG",
    entry_price=100.0,
    quantity=1.0,
    stop_loss=90.0,
    take_profit=110.0,
    status="OPEN",
    order_id="ORD-001",
    notes=None,
    initial_risk_distance=10.0,
    trade=None,
):
    """Build a MagicMock PaperOrder-like object for testing."""
    order = MagicMock()
    order.symbol = symbol
    order.side = MagicMock()
    order.side.value = side
    order.entry_price = entry_price
    order.quantity = quantity
    order.stop_loss = stop_loss
    order.take_profit = take_profit
    order.status = MagicMock()
    order.status.value = status
    order.order_id = order_id
    order.realized_pnl = 0.0
    order.exit_price = None
    order.exit_at = None
    order.notes = list(notes) if notes else []
    order.initial_risk_distance = initial_risk_distance
    order.trade = trade
    return order


def _make_service(open_position=None):
    """Build a MagicMock PaperTradingService with a configurable open position."""
    service = MagicMock()
    service.get_position.return_value = open_position
    service.save_account = MagicMock()
    return service


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestApplyTrailingStop:

    # --- No position / no stop loss -------------------------------------------

    def test_no_open_position_returns_none(self):
        service = _make_service(open_position=None)
        result = _apply_trailing_stop(service, "BTC", 50000.0)
        assert result is None
        service.save_account.assert_not_called()

    def test_no_stop_loss_returns_none(self):
        order = _make_order(stop_loss=0.0, take_profit=110.0)
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 50000.0)
        assert result is None
        service.save_account.assert_not_called()

    def test_negative_stop_loss_returns_none(self):
        order = _make_order(stop_loss=-5.0, take_profit=110.0)
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 50000.0)
        assert result is None
        service.save_account.assert_not_called()

    # --- LONG: below 1R ------------------------------------------------------

    def test_long_below_one_r_no_action(self):
        # entry=100, stop=90 → stop_distance=10 → 1R price=110
        order = _make_order(
            side="LONG", entry_price=100.0, stop_loss=90.0, take_profit=110.0
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 105.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 90.0  # unchanged

    def test_long_at_one_r_exact_no_action_yet(self):
        # At exactly 1R (110.0), still below threshold (strictly less than)
        order = _make_order(
            side="LONG", entry_price=100.0, stop_loss=90.0, take_profit=110.0
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 110.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 90.0

    # --- LONG: first 1R cross → break-even ---------------------------------

    def test_long_first_one_r_cross_to_breakeven(self):
        # entry=100, stop=90 → stop_distance=10 → 1R price=110
        # Price 115 > 110 → activate trailing, set stop to entry=100
        order = _make_order(
            side="LONG", entry_price=100.0, stop_loss=90.0, take_profit=110.0
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 115.0)
        assert result is not None
        changed, old_stop, new_stop = result
        assert changed is True
        assert old_stop == 90.0
        assert new_stop == 100.0
        assert order.stop_loss == 100.0
        assert _is_trail_active(order) is True
        service.save_account.assert_called_once()

    def test_long_first_one_r_cross_save_called(self):
        order = _make_order(
            side="LONG", entry_price=100.0, stop_loss=90.0, take_profit=110.0
        )
        service = _make_service(open_position=order)
        _apply_trailing_stop(service, "BTC", 120.0)
        service.save_account.assert_called_once()

    def test_long_already_at_breakeven_activation_still_saves(self):
        # Stop already at entry, trail not yet active. Activation must
        # still persist the marker and the original risk distance.
        order = _make_order(
            side="LONG", entry_price=100.0, stop_loss=100.0, take_profit=110.0
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 115.0)
        assert result is not None  # activation happens
        changed, old_stop, new_stop = result
        assert changed is True
        assert old_stop == 100.0
        assert new_stop == 100.0
        assert order.initial_risk_distance == 10.0
        service.save_account.assert_called_once()
        assert _is_trail_active(order) is True

    # --- LONG: already trail-active, price improves ------------------------

    def test_long_trail_active_price_moves_up_stop_follows(self):
        order = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=100.0,  # already at breakeven (trail active)
            take_profit=110.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        # The original risk distance is 10, so the trail stays 10 below
        # the market instead of hugging the current price.
        result = _apply_trailing_stop(service, "BTC", 115.0)
        assert result is not None
        changed, old_stop, new_stop = result
        assert changed is True
        assert old_stop == 100.0
        assert new_stop == 105.0
        assert order.stop_loss == 105.0
        service.save_account.assert_called_once()

    def test_long_trail_active_price_moves_up_more(self):
        order = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=105.0,  # already trailing, above breakeven
            take_profit=110.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        # The original risk distance is 10, not abs(100 - 105):
        # new_stop = max(105, 120 - 10) = 110
        result = _apply_trailing_stop(service, "BTC", 120.0)
        assert result is not None
        changed, old_stop, new_stop = result
        assert changed is True
        assert old_stop == 105.0
        assert new_stop == 110.0
        assert order.stop_loss == 110.0
        service.save_account.assert_called_once()

    def test_long_trail_active_price_unchanged_no_save(self):
        order = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=110.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        # new_stop = max(105, 110 - 10) = max(105, 100) = 105
        result = _apply_trailing_stop(service, "BTC", 110.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 105.0

    def test_long_trail_active_price_drops_no_backward_move(self):
        # Price drops but stop must NOT move backward
        order = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=110.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 100.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 105.0  # unchanged

    def test_long_trail_active_price_drops_below_stop_no_change(self):
        # Price below the stop — trailing does NOT close the position,
        # it just doesn't move the stop backward.
        order = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=110.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "BTC", 95.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 105.0

    # --- LONG: multiple trailing steps -------------------------------------

    def test_long_multiple_trailing_steps(self):
        order = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=110.0,
        )
        service = _make_service(open_position=order)
        # Step 1: price 115 → activate, stop → 100 (breakeven)
        r1 = _apply_trailing_stop(service, "BTC", 115.0)
        assert r1 is not None
        assert r1[2] == 100.0
        # The original risk distance (10) is fixed at activation.
        # Step 2: price 120 → stop 110
        r2 = _apply_trailing_stop(service, "BTC", 120.0)
        assert r2 is not None
        assert r2[2] == 110.0
        # Step 3: price 125 → stop 115
        r3 = _apply_trailing_stop(service, "BTC", 125.0)
        assert r3 is not None
        assert r3[2] == 115.0
        # Step 4: price 145 → stop 135
        r4 = _apply_trailing_stop(service, "BTC", 145.0)
        assert r4 is not None
        assert r4[2] == 135.0
        # Step 5: price 140 → no backward move
        r5 = _apply_trailing_stop(service, "BTC", 140.0)
        assert r5 is None
        assert order.stop_loss == 135.0
        # 4 saves (activation plus steps 2, 3 and 4)
        assert service.save_account.call_count == 4

    # --- LONG: take profit never modified ----------------------------------

    def test_long_take_profit_unchanged(self):
        original_tp = 110.0
        order = _make_order(
            side="LONG", entry_price=100.0, stop_loss=90.0, take_profit=original_tp
        )
        service = _make_service(open_position=order)
        _apply_trailing_stop(service, "BTC", 115.0)
        assert order.take_profit == original_tp

    # --- SHORT: mirror behavior ---------------------------------------------

    def test_short_below_one_r_no_action(self):
        # entry=100, stop=110 → stop_distance=10 → 1R price=90
        order = _make_order(
            side="SHORT", entry_price=100.0, stop_loss=110.0, take_profit=90.0
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "ETH", 95.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 110.0

    def test_short_at_one_r_exact_no_action_yet(self):
        # At exactly 1R (90.0), still above threshold
        order = _make_order(
            side="SHORT", entry_price=100.0, stop_loss=110.0, take_profit=90.0
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "ETH", 90.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 110.0

    def test_short_first_one_r_cross_to_breakeven(self):
        # entry=100, stop=110 → stop_distance=10 → 1R price=90
        # Price 85 < 90 → activate, stop → entry=100
        order = _make_order(
            side="SHORT", entry_price=100.0, stop_loss=110.0, take_profit=90.0
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "ETH", 85.0)
        assert result is not None
        changed, old_stop, new_stop = result
        assert changed is True
        assert old_stop == 110.0
        assert new_stop == 100.0
        assert order.stop_loss == 100.0
        assert _is_trail_active(order) is True
        service.save_account.assert_called_once()

    def test_short_trail_active_price_drops_stop_follows(self):
        order = _make_order(
            side="SHORT",
            entry_price=100.0,
            stop_loss=100.0,
            take_profit=90.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        # Original risk distance is 10: new_stop = min(100, 85 + 10) = 95
        result = _apply_trailing_stop(service, "ETH", 85.0)
        assert result is not None
        assert result[2] == 95.0
        assert order.stop_loss == 95.0
        service.save_account.assert_called_once()

    def test_short_trail_active_price_drops_more(self):
        order = _make_order(
            side="SHORT",
            entry_price=100.0,
            stop_loss=95.0,  # already trailing below breakeven
            take_profit=90.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        # Original risk distance is 10: new_stop = min(95, 80 + 10) = 90
        result = _apply_trailing_stop(service, "ETH", 80.0)
        assert result is not None
        assert result[2] == 90.0
        assert order.stop_loss == 90.0
        service.save_account.assert_called_once()

    def test_short_trail_active_uses_original_risk_distance(self):
        # At breakeven the trail must still keep the original 10-point
        # risk distance instead of hugging the current price.
        order = _make_order(
            side="SHORT",
            entry_price=100.0,
            stop_loss=100.0,  # breakeven, trail active
            take_profit=90.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "ETH", 85.0)
        assert result is not None
        assert result[2] == 95.0
        assert order.stop_loss == 95.0
        service.save_account.assert_called_once()

    def test_short_trail_active_price_rises_no_backward_move(self):
        order = _make_order(
            side="SHORT",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=90.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "ETH", 100.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 95.0

    def test_short_trail_active_price_rises_above_stop_no_change(self):
        order = _make_order(
            side="SHORT",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=90.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        result = _apply_trailing_stop(service, "ETH", 105.0)
        assert result is None
        service.save_account.assert_not_called()
        assert order.stop_loss == 95.0

    # --- SHORT: take profit never modified ----------------------------------

    def test_short_take_profit_unchanged(self):
        original_tp = 90.0
        order = _make_order(
            side="SHORT", entry_price=100.0, stop_loss=110.0, take_profit=original_tp
        )
        service = _make_service(open_position=order)
        _apply_trailing_stop(service, "ETH", 85.0)
        assert order.take_profit == original_tp

    # --- Closed positions are never fetched ---------------------------------

    def test_closed_position_not_fetched_by_get_position(self):
        # get_position only returns OPEN positions. The helper simply
        # returns None when get_position returns None, so closed positions
        # are naturally excluded without any extra logic.
        closed = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=110.0,
            status="CLOSED_MANUAL",
        )
        service = _make_service(open_position=closed)
        # If get_position incorrectly returns a closed order, the helper
        # still runs — but in production get_position filters by OPEN status.
        # This test documents that behavior by using the real contract:
        # get_position returns None for closed positions.
        service.get_position.return_value = None
        result = _apply_trailing_stop(service, "BTC", 115.0)
        assert result is None
        service.save_account.assert_not_called()

    # --- Save called only on change -----------------------------------------

    def test_save_not_called_when_no_change(self):
        order = _make_order(
            side="LONG",
            entry_price=100.0,
            stop_loss=100.0,  # already breakeven, trail active
            take_profit=110.0,
            notes=["trailing_active"],
        )
        service = _make_service(open_position=order)
        # With stop_distance=0, new_stop = max(100, 100 - 0) = 100 → no change
        result = _apply_trailing_stop(service, "BTC", 100.0)
        assert result is None
        service.save_account.assert_not_called()

    # --- Symbol passthrough -------------------------------------------------

    def test_symbol_passthrough(self):
        order = _make_order(side="LONG", entry_price=100.0, stop_loss=90.0)
        service = _make_service(open_position=order)
        _apply_trailing_stop(service, "btc", 115.0)
        service.get_position.assert_called_once_with("btc")


# ---------------------------------------------------------------------------
# Fixed original-risk distance
# ---------------------------------------------------------------------------

def test_activation_uses_original_risk_distance():
    order = _make_order(
        side="LONG",
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=130.0,
        initial_risk_distance=10.0,
    )
    service = _make_service(open_position=order)

    result = _apply_trailing_stop(service, "BTC", 115.0)

    assert result is not None
    assert result[2] == 100.0
    assert order.initial_risk_distance == 10.0


def test_subsequent_moves_use_fixed_risk_not_moved_stop():
    order = _make_order(
        side="LONG",
        entry_price=100.0,
        stop_loss=120.0,
        take_profit=200.0,
        notes=["trailing_active"],
        initial_risk_distance=10.0,
    )
    service = _make_service(open_position=order)

    # A recalculation from the current stop (120) would give 20 and
    # produce 130; the fixed risk distance gives 140.
    result = _apply_trailing_stop(service, "BTC", 150.0)

    assert result is not None
    assert result[1] == 120.0
    assert result[2] == 140.0
    assert order.stop_loss == 140.0


def test_initial_risk_distance_survives_restart(monkeypatch, tmp_path):
    state_path = tmp_path / "paper_state.json"
    monkeypatch.setattr(handlers, "PAPER_STATE_PATH", state_path)

    trade = Trade(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=10.0,
        stop_loss=90.0,
        take_profit=130.0,
        side="LONG",
        approved=True,
    )
    service = PaperTradingService.create_new_account(1000.0)
    order = service.place_trade(trade)
    assert order is not None
    assert order.initial_risk_distance == 10.0

    # Activate trailing at 1R and persist the state.
    activated = handlers._apply_trailing_stop_to_position(service, order, 115.0)
    assert activated is not None
    assert activated[2] == 100.0

    reloaded_service = PaperTradingService.load_account(state_path)
    reloaded = reloaded_service.get_position_by_id(order.order_id)
    assert reloaded is not None
    assert reloaded.initial_risk_distance == 10.0
    assert "trailing_active" in reloaded.notes

    # After the restart, a further rise must trail by the original 10.
    moved = handlers._apply_trailing_stop_to_position(
        reloaded_service, reloaded, 130.0
    )
    assert moved is not None
    assert moved[2] == 120.0
    assert reloaded.stop_loss == 120.0


def test_long_activation_never_lowers_tighter_stop():
    # A stop already tighter than entry (manual/legacy) must not be
    # moved back down to breakeven when trailing activates.
    order = _make_order(
        side="LONG",
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=130.0,
        initial_risk_distance=10.0,
    )
    service = _make_service(open_position=order)

    result = _apply_trailing_stop(service, "BTC", 115.0)

    assert result is not None
    assert result[1] == 105.0
    assert result[2] == 105.0
    assert order.stop_loss == 105.0
    assert _is_trail_active(order) is True
    service.save_account.assert_called_once()


def test_short_activation_never_raises_tighter_stop():
    # Mirror for SHORT: a stop already below entry must stay there.
    order = _make_order(
        side="SHORT",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=70.0,
        initial_risk_distance=10.0,
    )
    service = _make_service(open_position=order)

    result = _apply_trailing_stop(service, "ETH", 85.0)

    assert result is not None
    assert result[1] == 95.0
    assert result[2] == 95.0
    assert order.stop_loss == 95.0
    assert _is_trail_active(order) is True
    service.save_account.assert_called_once()


def test_invalid_initial_risk_distance_backfilled_from_trade():
    trade = Trade(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=1.0,
        stop_loss=90.0,
        take_profit=130.0,
        side="LONG",
        approved=True,
    )
    order = PaperOrder(
        order_id="LEGACY-2",
        symbol="BTC",
        side=PaperOrderSide.LONG,
        quantity=1.0,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=130.0,
        notes=["trailing_active"],
        initial_risk_distance=0.0,
        trade=trade,
    )
    service = _make_service(open_position=order)

    result = handlers._apply_trailing_stop_to_position(service, order, 120.0)

    assert result is not None
    assert result[2] == 110.0
    assert order.initial_risk_distance == 10.0


def test_huge_initial_risk_distance_is_not_persisted():
    # Overflow-sized malformed values must be ignored, not crash the
    # monitor, and the valid fallback distance must be stored instead.
    order = _make_order(
        side="LONG",
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=130.0,
        initial_risk_distance=10**400,
        trade=None,
    )
    service = _make_service(open_position=order)

    result = _apply_trailing_stop(service, "BTC", 115.0)

    assert result is not None
    assert result[2] == 100.0
    assert order.initial_risk_distance == 10.0


def test_legacy_active_position_recovers_original_risk_from_trade():
    trade = Trade(
        symbol="BTC",
        price=100.0,
        trend="Bullish",
        signal="BUY",
        confidence=80,
        decision="BUY",
        risk_verdict="APPROVED",
        position_size=1.0,
        stop_loss=90.0,
        take_profit=130.0,
        side="LONG",
        approved=True,
    )
    order = PaperOrder(
        order_id="LEGACY-1",
        symbol="BTC",
        side=PaperOrderSide.LONG,
        quantity=1.0,
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=130.0,
        notes=["trailing_active"],
        initial_risk_distance=None,
        trade=trade,
    )
    service = _make_service(open_position=order)

    result = handlers._apply_trailing_stop_to_position(service, order, 120.0)

    assert result is not None
    # Original trade stop 90 → distance 10, not abs(100 - 105) = 5.
    assert result[2] == 110.0
    assert order.initial_risk_distance == 10.0
