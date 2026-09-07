"""
Paper Trading Performance Analytics (Sprint 6D).

Pure functions and frozen dataclasses for computing performance metrics
from paper trading account state. No I/O, no side effects, no Telegram
dependencies. All functions are deterministic and testable.
"""

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trader.paper.order import PaperOrder


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OverallPerformance:
    """Aggregate performance metrics across all closed paper trades."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    total_realized_pnl: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    expectancy: float


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    """Risk-oriented metrics derived from closed paper trades."""

    profit_factor: float
    win_loss_ratio: float | None
    payoff_ratio: float | None
    average_trade_pnl: float
    current_streak: int
    max_consecutive_losses: int


@dataclass(frozen=True, slots=True)
class SymbolPerformance:
    """Performance metrics for a single symbol."""

    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float


# ---------------------------------------------------------------------------
# Overall Performance
# ---------------------------------------------------------------------------


def compute_overall_performance(closed_positions: list["PaperOrder"]) -> OverallPerformance:
    """Compute aggregate performance metrics from closed paper orders.

    Args:
        closed_positions: List of PaperOrder objects representing closed
            paper positions.

    Returns:
        OverallPerformance with all metrics populated. When the list is
        empty, all counts are zero, win_rate is 0.0, and expectancy is 0.0.
    """
    if not closed_positions:
        return OverallPerformance(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            total_realized_pnl=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            expectancy=0.0,
        )

    pnl_values = [pos.realized_pnl for pos in closed_positions]
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]
    breakevens = [p for p in pnl_values if p == 0]

    total = len(closed_positions)
    win_count = len(wins)
    loss_count = len(losses)
    breakeven_count = len(breakevens)

    win_rate = (win_count / total) * 100.0 if total > 0 else 0.0
    total_pnl = sum(pnl_values)

    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(abs(p) for p in losses) if losses else 0.0
    largest_win = max(wins) if wins else 0.0
    largest_loss = min(losses) if losses else 0.0

    loss_rate = (loss_count / total) * 100.0 if total > 0 else 0.0
    expectancy = (win_rate / 100.0 * avg_win) - (loss_rate / 100.0 * avg_loss)

    return OverallPerformance(
        total_trades=total,
        winning_trades=win_count,
        losing_trades=loss_count,
        breakeven_trades=breakeven_count,
        win_rate=win_rate,
        total_realized_pnl=total_pnl,
        average_win=avg_win,
        average_loss=avg_loss,
        largest_win=largest_win,
        largest_loss=largest_loss,
        expectancy=expectancy,
    )


# ---------------------------------------------------------------------------
# Risk Metrics
# ---------------------------------------------------------------------------


def compute_risk_metrics(closed_positions: list["PaperOrder"]) -> RiskMetrics:
    """Compute risk-oriented metrics from closed paper orders.

    Args:
        closed_positions: List of PaperOrder objects representing closed
            paper positions.

    Returns:
        RiskMetrics with profit factor, win/loss ratio, payoff ratio,
        average trade PnL, current streak, and max consecutive losses.
    """
    if not closed_positions:
        return RiskMetrics(
            profit_factor=0.0,
            win_loss_ratio=None,
            payoff_ratio=None,
            average_trade_pnl=0.0,
            current_streak=0,
            max_consecutive_losses=0,
        )

    pnl_values = [pos.realized_pnl for pos in closed_positions]
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]

    sum_wins = sum(wins)
    sum_losses = sum(losses)
    abs_sum_losses = abs(sum_losses)

    if abs_sum_losses == 0.0:
        if sum_wins > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
    else:
        profit_factor = sum_wins / abs_sum_losses if sum_wins > 0.0 else 0.0

    win_count = len(wins)
    loss_count = len(losses)

    win_loss_ratio = (win_count / loss_count) if loss_count > 0 else None
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(abs(p) for p in losses) if losses else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0.0 else None
    average_trade_pnl = mean(pnl_values)

    current_streak = 0
    for pnl in reversed(pnl_values):
        if pnl > 0:
            current_streak += 1
        elif pnl < 0:
            current_streak -= 1
        else:
            break

    max_consecutive_losses = 0
    current_loss_streak = 0
    for pnl in pnl_values:
        if pnl < 0:
            current_loss_streak += 1
            if current_loss_streak > max_consecutive_losses:
                max_consecutive_losses = current_loss_streak
        else:
            current_loss_streak = 0

    return RiskMetrics(
        profit_factor=profit_factor,
        win_loss_ratio=win_loss_ratio,
        payoff_ratio=payoff_ratio,
        average_trade_pnl=average_trade_pnl,
        current_streak=current_streak,
        max_consecutive_losses=max_consecutive_losses,
    )


# ---------------------------------------------------------------------------
# Symbol Performance
# ---------------------------------------------------------------------------


def compute_symbol_performance(
    closed_positions: list["PaperOrder"], symbol: str
) -> SymbolPerformance:
    """Compute performance metrics for a single symbol.

    Args:
        closed_positions: List of all closed paper orders.
        symbol: The symbol to filter by (case-insensitive).

    Returns:
        SymbolPerformance for the requested symbol.
    """
    symbol_upper = symbol.upper()
    filtered = [p for p in closed_positions if p.symbol.upper() == symbol_upper]

    if not filtered:
        return SymbolPerformance(
            symbol=symbol_upper,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            profit_factor=0.0,
        )

    pnl_values = [p.realized_pnl for p in filtered]
    wins = [p for p in pnl_values if p > 0]
    losses = [p for p in pnl_values if p < 0]

    total = len(filtered)
    win_count = len(wins)
    loss_count = len(losses)

    win_rate = (win_count / total) * 100.0 if total > 0 else 0.0
    total_pnl = sum(pnl_values)

    sum_wins = sum(wins)
    abs_sum_losses = abs(sum(losses))

    if abs_sum_losses == 0.0:
        if sum_wins > 0.0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0
    else:
        profit_factor = sum_wins / abs_sum_losses if sum_wins > 0.0 else 0.0

    return SymbolPerformance(
        symbol=symbol_upper,
        total_trades=total,
        winning_trades=win_count,
        losing_trades=loss_count,
        breakeven_trades=len([p for p in pnl_values if p == 0]),
        win_rate=win_rate,
        total_pnl=total_pnl,
        profit_factor=profit_factor,
    )


def group_by_symbol(closed_positions: list["PaperOrder"]) -> list[SymbolPerformance]:
    """Group closed positions by symbol and compute per-symbol performance.

    Args:
        closed_positions: List of all closed paper orders.

    Returns:
        List of SymbolPerformance, sorted by total_pnl descending.
    """
    if not closed_positions:
        return []

    symbols = set(p.symbol.upper() for p in closed_positions)
    results = [
        compute_symbol_performance(closed_positions, sym)
        for sym in symbols
    ]
    results.sort(key=lambda sp: sp.total_pnl, reverse=True)
    return results


def get_recent_trades(
    closed_positions: list["PaperOrder"], limit: int = 10
) -> list["PaperOrder"]:
    """Return the most recent closed trades up to the given limit.

    Args:
        closed_positions: List of closed paper orders, expected to be
            sorted most-recent-first.
        limit: Maximum number of trades to return.

    Returns:
        A slice of the input list containing at most `limit` items.
    """
    if not closed_positions:
        return []
    return closed_positions[:limit]
