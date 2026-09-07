"""Paper account summary dataclass.

Sprint 5E: Typed summary of paper trading account state for
display and reporting purposes.
"""

from dataclasses import dataclass

from trader.paper.order import PaperOrder


@dataclass(frozen=True, slots=True)
class PaperAccountSummary:
    """Snapshot of paper account state.

    Attributes:
        initial_balance: The starting balance of the account.
        current_balance: The current settled balance (after realized PnL).
        open_positions: List of currently open paper orders.
        closed_count: Number of closed positions.
        total_realized_pnl: Sum of realized PnL across all closed
            positions (positive = profit, negative = loss).
        open_pnl: Sum of unrealized PnL across all open positions
            (positive = gain, negative = loss).
        equity: Current account equity (balance + unrealized PnL).
    """

    initial_balance: float
    current_balance: float
    open_positions: list[PaperOrder]
    closed_count: int
    total_realized_pnl: float
    open_pnl: float
    equity: float
