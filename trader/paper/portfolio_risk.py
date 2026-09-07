"""Portfolio-level risk management for paper trading positions.

Sprint 7A: Track aggregate risk across all open positions and prevent
new trades from being placed when they would exceed a configured
portfolio risk limit.

Risk is defined as the dollar amount lost if a position's stop loss
is hit:
    LONG:  risk = (entry_price - stop_loss) * quantity
    SHORT: risk = (stop_loss - entry_price) * quantity

The portfolio risk manager is a read-only analysis tool. It never
places, closes, or modifies orders. It answers three questions:
    1. How much total risk is currently open?
    2. How much risk would a new trade add?
    3. Would adding that trade exceed the configured limit?
"""

from __future__ import annotations

from dataclasses import dataclass

from trader.paper.broker import PaperBroker
from trader.paper.order import PaperOrder
from trader.trade import Trade


# ---------------------------------------------------------------------------
# Pure risk calculation helpers
# ---------------------------------------------------------------------------

def position_risk(order: PaperOrder) -> float:
    """Compute the dollar risk of a single paper position.

    Risk is the loss incurred if the stop loss is triggered, before
    any margin, leverage, or fees. Positions with no valid stop loss
    (stop_loss <= 0) or zero quantity contribute zero risk.

    Args:
        order: The paper order to evaluate.

    Returns:
        The dollar risk of the position. Always >= 0.
    """
    if order.quantity <= 0 or not order.stop_loss or order.stop_loss <= 0:
        return 0.0

    if order.side.name == "LONG":
        return max(0.0, (order.entry_price - order.stop_loss) * order.quantity)
    else:
        # SHORT (and any non-LONG side)
        return max(0.0, (order.stop_loss - order.entry_price) * order.quantity)


def total_portfolio_risk(broker: PaperBroker) -> float:
    """Compute the sum of risk across all open positions in a broker.

    Args:
        broker: The paper broker whose open positions to sum.

    Returns:
        The total portfolio risk as a non-negative float.
    """
    return sum(position_risk(order) for order in broker.get_open_positions())


def prospective_risk(broker: PaperBroker, trade: Trade) -> float:
    """Compute what the total portfolio risk would be after placing a trade.

    This does not place the trade. It computes the current total risk
    and adds the risk the new trade would contribute.

    Args:
        broker: The paper broker with current open positions.
        trade: The trade to evaluate for prospective placement.

    Returns:
        The hypothetical total portfolio risk after placing the trade.
    """
    current = total_portfolio_risk(broker)
    added = trade_risk(trade)
    return current + added


def trade_risk(trade: Trade) -> float:
    """Compute the risk contribution of a trade that has not been placed yet.

    Mirrors ``position_risk`` but operates on a ``Trade`` object rather
    than a ``PaperOrder``.

    Args:
        trade: The trade to evaluate.

    Returns:
        The dollar risk the trade would contribute if placed.
    """
    if trade.position_size <= 0 or not trade.stop_loss or trade.stop_loss <= 0:
        return 0.0

    if trade.side == "LONG":
        return max(0.0, (trade.price - trade.stop_loss) * trade.position_size)
    else:
        return max(0.0, (trade.stop_loss - trade.price) * trade.position_size)


# ---------------------------------------------------------------------------
# PortfolioRiskManager
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PortfolioRiskMetrics:
    """Snapshot of portfolio risk state.

    Attributes:
        current_risk: Total risk across all open positions.
        risk_limit: The configured maximum portfolio risk.
        usage_percent: Current risk as a percentage of the limit.
            0.0 when no limit is configured.
        remaining: Risk budget remaining before hitting the limit.
            Equals risk_limit when no limit is configured.
    """

    current_risk: float
    risk_limit: float
    usage_percent: float
    remaining: float


class PortfolioRiskManager:
    """Portfolio-level risk gate for paper trading.

    Wraps a PaperBroker and a risk limit, and provides methods to
    query current portfolio risk and check whether a new trade would
    exceed the configured limit.

    When the risk limit is 0 or negative, no enforcement is applied
    and all checks pass unconditionally. This preserves backward
    compatibility — existing installations without a configured limit
    are unaffected.

    Usage:
        manager = PortfolioRiskManager(broker, risk_limit=5000.0)
        if manager.can_open_trade(trade):
            broker.place_order(trade)
        else:
            # reject the trade
    """

    def __init__(self, broker: PaperBroker, risk_limit: float = 0.0) -> None:
        """Initialize the portfolio risk manager.

        Args:
            broker: The paper broker to monitor.
            risk_limit: Maximum total portfolio risk. When <= 0, no
                limit is enforced and all checks pass.
        """
        self._broker = broker
        self._risk_limit = risk_limit

    @property
    def risk_limit(self) -> float:
        """The configured maximum portfolio risk."""
        return self._risk_limit

    def total_portfolio_risk(self) -> float:
        """Return the sum of risk across all open positions."""
        return total_portfolio_risk(self._broker)

    def can_open_trade(self, trade: Trade) -> bool:
        """Check whether a trade can be opened without exceeding the risk limit.

        When no risk limit is configured (limit <= 0), this always
        returns True.

        Args:
            trade: The trade to evaluate.

        Returns:
            True when the trade can be opened, False when it would
            exceed the configured portfolio risk limit.
        """
        if self._risk_limit <= 0:
            return True
        return prospective_risk(self._broker, trade) <= self._risk_limit

    def metrics(self) -> PortfolioRiskMetrics:
        """Build a snapshot of current portfolio risk state.

        Returns:
            A PortfolioRiskMetrics with current risk, limit, usage
            percentage, and remaining budget.
        """
        current = self.total_portfolio_risk()
        limit = self._risk_limit

        if limit <= 0:
            return PortfolioRiskMetrics(
                current_risk=current,
                risk_limit=limit,
                usage_percent=0.0,
                remaining=0.0,
            )

        usage = (current / limit * 100.0) if limit != 0 else 0.0
        remaining = max(0.0, limit - current)
        return PortfolioRiskMetrics(
            current_risk=current,
            risk_limit=limit,
            usage_percent=usage,
            remaining=remaining,
        )
