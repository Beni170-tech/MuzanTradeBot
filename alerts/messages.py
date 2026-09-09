"""
MuzanTradeBot - Alert message builders (Sprint 9A)

Pure functions that turn domain objects into human-readable Telegram
alert texts. One builder per alert event:

    - Paper order placed            -> order_placed_message
    - Stop-loss triggered           -> stop_loss_triggered_message
    - Take-profit triggered         -> take_profit_triggered_message
    - Trailing stop updates         -> trailing_stop_updated_message
    - Guardian/risk-limit rejection -> trade_rejected_message
    - Daily loss breaker activation -> daily_loss_breaker_message
    - Max open positions reached    -> max_open_positions_message

The builders are pure: they only read attributes from the objects they
are given and never touch the network, the filesystem, or the broker.
All formatting helpers are local so the builders stay independent from
the handler layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from trader.paper.order import paper_exit_status

if TYPE_CHECKING:
    from trader.paper.daily_risk import DailyRiskMetrics
    from trader.paper.max_open_positions import MaxOpenPositionsResult
    from trader.paper.order import PaperOrder
    from trader.trade import Trade


# ---------------------------------------------------------------------------
# Local formatting helpers
# ---------------------------------------------------------------------------


def _fmt_money(value: float) -> str:
    """Format a monetary amount without a sign (e.g. ``$104.00``).

    Matches the existing handler message style (``${value:.2f}``),
    without thousands separators.
    """
    return f"${float(value):.2f}"


def _fmt_signed_money(value: float) -> str:
    """Format a signed monetary amount (e.g. ``+$4.00`` / ``-$2.00``).

    Mirrors the handler layer's ``_format_pnl_value``.
    """
    value = float(value)
    if value > 0:
        return f"+${value:.2f}"
    if value < 0:
        return f"-${abs(value):.2f}"
    return "$0.00"


def _side_label(order: "PaperOrder") -> str:
    """Extract the side label from a PaperOrder-like object."""
    side = getattr(order, "side", None)
    value = getattr(side, "value", None)
    if isinstance(value, str) and value:
        return value.upper()
    return str(side).upper() if side is not None else "UNKNOWN"


def _exit_status_label(order: "PaperOrder") -> str:
    """Extract the exit-status label from a closed PaperOrder.

    Delegates to the single compatibility helper
    :func:`trader.paper.order.paper_exit_status` so the alerts layer
    and the handler layer never diverge on how a terminal status is
    read. Falls back to ``CLOSED`` for objects with no usable status.
    """
    return paper_exit_status(order, default="CLOSED")


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def order_placed_message(order: "PaperOrder") -> str:
    """Build the alert text for a newly placed paper order.

    Args:
        order: The placed PaperOrder.

    Returns:
        A Telegram-ready alert message describing the new position.
    """
    return (
        f"📄 Paper Order Placed\n\n"
        f"Symbol: {order.symbol}\n"
        f"Side: {_side_label(order)}\n"
        f"Quantity: {order.quantity}\n"
        f"Entry: {_fmt_money(order.entry_price)}\n"
        f"Stop Loss: {_fmt_money(order.stop_loss)}\n"
        f"Take Profit: {_fmt_money(order.take_profit)}\n"
        f"Order ID: {order.order_id}"
    )


def stop_loss_triggered_message(order: "PaperOrder") -> str:
    """Build the alert text for a stop-loss triggered closure.

    Args:
        order: The closed PaperOrder (status STOPPED).

    Returns:
        A Telegram-ready alert message with exit price and realized PnL.
    """
    return (
        f"🛑 Stop Loss Triggered\n\n"
        f"Symbol: {order.symbol}\n"
        f"Side: {_side_label(order)}\n"
        f"Entry: {_fmt_money(order.entry_price)}\n"
        f"Exit: {_fmt_money(order.exit_price)}\n"
        f"Realized PnL: {_fmt_signed_money(order.realized_pnl)}\n"
        f"Status: {_exit_status_label(order)}\n"
        f"Order ID: {order.order_id}"
    )


def take_profit_triggered_message(order: "PaperOrder") -> str:
    """Build the alert text for a take-profit triggered closure.

    Args:
        order: The closed PaperOrder (status TAKEN_PROFIT).

    Returns:
        A Telegram-ready alert message with exit price and realized PnL.
    """
    return (
        f"🎯 Take Profit Triggered\n\n"
        f"Symbol: {order.symbol}\n"
        f"Side: {_side_label(order)}\n"
        f"Entry: {_fmt_money(order.entry_price)}\n"
        f"Exit: {_fmt_money(order.exit_price)}\n"
        f"Realized PnL: {_fmt_signed_money(order.realized_pnl)}\n"
        f"Status: {_exit_status_label(order)}\n"
        f"Order ID: {order.order_id}"
    )


def trailing_stop_updated_message(
    symbol: str, old_stop: float, new_stop: float
) -> str:
    """Build the alert text for a trailing stop update.

    Args:
        symbol: The symbol whose stop was moved.
        old_stop: The stop loss before the update.
        new_stop: The stop loss after the update.

    Returns:
        A Telegram-ready alert message showing the stop movement.
    """
    return (
        f"📈 Trailing Stop Updated\n\n"
        f"Symbol: {symbol}\n"
        f"Stop Loss: {_fmt_money(old_stop)} → {_fmt_money(new_stop)}"
    )


def trade_rejected_message(trade: "Trade", reason: str) -> str:
    """Build the alert text for a Guardian/risk-limit trade rejection.

    Args:
        trade: The Trade that was not placed.
        reason: Human-readable rejection reason (e.g. "portfolio risk
            limit exceeded").

    Returns:
        A Telegram-ready alert message naming the reason and reminding
        the user that Guardian Mode blocked the trade.
    """
    return (
        f"🛡️ Trade Rejected\n\n"
        f"Guardian Mode blocked this trade.\n"
        f"Symbol: {trade.symbol}\n"
        f"Side: {trade.side}\n"
        f"Entry: {_fmt_money(trade.price)}\n"
        f"Reason: {reason}\n"
        f"The trade was not placed."
    )


def daily_loss_breaker_message(metrics: "DailyRiskMetrics") -> str:
    """Build the alert text for daily loss breaker activation.

    Args:
        metrics: The DailyRiskMetrics snapshot captured when the brake
            tripped.

    Returns:
        A Telegram-ready alert message with the day's PnL, the limit,
        and a note that trading resumes tomorrow.
    """
    return (
        f"⛔ Daily Loss Breaker Activated\n\n"
        f"Day: {metrics.day}\n"
        f"Today's Realized PnL: {_fmt_signed_money(metrics.realized_pnl_today)}\n"
        f"Daily Loss Limit: {_fmt_money(metrics.loss_limit)}\n"
        f"New paper trades are halted until tomorrow."
    )


def max_open_positions_message(result: "MaxOpenPositionsResult") -> str:
    """Build the alert text for the maximum open positions limit.

    Args:
        result: The MaxOpenPositionsResult captured when the limit was
            reached.

    Returns:
        A Telegram-ready alert message with current usage against the
        limit and a note that the trade was not placed.
    """
    return (
        f"🚧 Maximum Open Positions Reached\n\n"
        f"Open Positions: {result.current_open_positions} / {result.limit}\n"
        f"The trade was not placed.\n"
        f"Close existing positions to free up open-position capacity."
    )
