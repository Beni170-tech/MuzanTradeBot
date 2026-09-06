"""
MuzanTradeBot - Stop Loss / Take Profit Calculator

Computes the exit levels for a trade from the entry price and the
risk/reward percentages. In Guardian Mode, predefined exit levels are
a capital-preservation control: the stop loss caps the maximum loss
on any trade before it is entered. Levels are direction-aware: a long
stops below entry, a short stops above entry.
"""

import math

# Rounding keeps at least two decimals for regular-priced assets and
# adds precision automatically for sub-$1 assets so a stop level can
# never collapse to 0.0 (a position without a stop is forbidden).
_MIN_DECIMALS = 2
_MAX_DECIMALS = 8


def _level_decimals(entry_price: float) -> int:
    """Number of decimals to round exit levels to for a given entry.

    Assets worth $1 or more keep two decimals. Cheaper assets get more
    decimals so a level like 0.00098 is preserved rather than rounded
    down to a useless 0.0 stop loss.
    """
    if not isinstance(entry_price, (int, float)) or entry_price <= 0:
        return _MIN_DECIMALS
    if entry_price >= 1:
        return _MIN_DECIMALS
    decimals = int(abs(math.floor(math.log10(abs(entry_price))))) + 3
    return min(decimals, _MAX_DECIMALS)


def calculate_exit_levels(
    entry_price: float,
    stop_loss_percent: float,
    take_profit_percent: float,
    side: str = "LONG",
) -> dict:
    """Calculate stop-loss and take-profit price levels.

    Args:
        entry_price: The price at which the position is entered.
        stop_loss_percent: The stop-loss distance from entry, in
            percent (e.g. 2.0 means 2%).
        take_profit_percent: The take-profit distance from entry, in
            percent (e.g. 4.0 means 4%).
        side: "LONG" (default) or "SHORT". For a LONG the stop loss is
            below entry and take profit above; for a SHORT they are
            inverted.

    Returns:
        A dict with "stop_loss" and "take_profit" price levels.
        Returns {"stop_loss": 0, "take_profit": 0} if entry_price,
        stop_loss_percent, or take_profit_percent is not positive.
        Raises ValueError for any side other than LONG or SHORT, so a
        misconfigured direction fails loudly instead of producing
        misleading levels.

    Guardian Mode purpose:
        Exit levels are set before entry so losses are bounded and
        profits are secured automatically, removing emotion from the
        exit decision.
    """
    side = str(side).upper()
    if side not in ("LONG", "SHORT"):
        raise ValueError(f"side must be 'LONG' or 'SHORT', got {side!r}")

    if (
        entry_price <= 0
        or stop_loss_percent <= 0
        or take_profit_percent <= 0
    ):
        return {"stop_loss": 0, "take_profit": 0}

    decimals = _level_decimals(entry_price)

    if side == "LONG":
        stop_loss = entry_price * (1 - stop_loss_percent / 100)
        take_profit = entry_price * (1 + take_profit_percent / 100)
    else:  # SHORT: stop above entry, take profit below
        stop_loss = entry_price * (1 + stop_loss_percent / 100)
        take_profit = entry_price * (1 - take_profit_percent / 100)

    return {
        "stop_loss": round(stop_loss, decimals),
        "take_profit": round(take_profit, decimals),
    }
