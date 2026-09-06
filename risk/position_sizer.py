"""
MuzanTradeBot - Position Sizer

Calculates the quantity to trade from the available balance, the
percentage of capital at risk, the entry price, and the stop-loss
distance. In Guardian Mode, position sizing is a core
capital-preservation control: the quantity is chosen so that being
stopped out loses *exactly* the predefined risk budget and nothing
more, regardless of the entry price.
"""


def calculate_position_size(
    balance: float,
    risk_percent: float,
    entry_price: float,
    stop_loss_percent: float,
) -> float:
    """Calculate the position size (quantity) for a trade.

    Args:
        balance: The account balance available for trading.
        risk_percent: The percentage of the balance to risk on this
            trade (e.g. 2.0 means 2%).
        entry_price: The price at which the position is entered.
        stop_loss_percent: The stop-loss distance below (long) or
            above (short) the entry price, in percent (e.g. 2.0
            means 2%). The position is sized against this distance.

    Returns:
        The quantity to trade, rounded to 8 decimal places. Returns 0
        if balance, risk_percent, entry_price, or stop_loss_percent is
        not positive.

    Guardian Mode purpose:
        The quantity satisfies: quantity * stop_distance = risk_amount,
        where risk_amount = balance * (risk_percent / 100) and
        stop_distance = entry_price * (stop_loss_percent / 100). A
        stop-out therefore costs exactly `risk_amount`, so a single
        bad trade can never exceed the predefined risk budget.
    """
    if (
        balance <= 0
        or risk_percent <= 0
        or entry_price <= 0
        or stop_loss_percent <= 0
    ):
        return 0

    risk_amount = balance * (risk_percent / 100)
    stop_distance = entry_price * (stop_loss_percent / 100)
    position_size = risk_amount / stop_distance
    return round(position_size, 8)
