"""Pure performance analytics for historical backtest trades.

Only completed HistoricalTrade records contribute to realized PnL, win
rate, ending equity, and drawdown. This module performs no I/O and does
not change position sizing or trade execution.
"""

from dataclasses import dataclass
from datetime import datetime
import math

from backtest.trades import HistoricalTrade
from trader.paper.order import PaperOrderStatus

_CLOSED_STATUSES = {
    PaperOrderStatus.STOPPED,
    PaperOrderStatus.TAKEN_PROFIT,
    PaperOrderStatus.CLOSED_MANUAL,
    PaperOrderStatus.CLOSED_BY_MARKET,
}


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Aggregate metrics for a historical backtest trade set."""

    total_trades: int
    open_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    gross_profit: float
    gross_loss: float
    total_realized_pnl: float
    profit_factor: float | None
    ending_equity: float
    max_drawdown: float
    max_drawdown_percent: float


def _is_finite_positive(value) -> bool:
    """Return True only for a finite, strictly positive int/float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number > 0


def _validate_trade(trade) -> bool:
    """Validate one trade and return True when it is open."""
    if not isinstance(trade, HistoricalTrade):
        raise TypeError("trades must contain HistoricalTrade values")

    if not isinstance(trade.symbol, str) or not trade.symbol.strip():
        raise ValueError("trade.symbol must be a non-empty string")
    if not isinstance(trade.signal_timestamp, datetime):
        raise TypeError("signal_timestamp must be a datetime")
    if not isinstance(trade.entry_timestamp, datetime):
        raise TypeError("entry_timestamp must be a datetime")
    if not isinstance(trade.trade_timestamp, str) or not trade.trade_timestamp.strip():
        raise ValueError("trade_timestamp must be a non-empty string")

    side = trade.side.strip().upper() if isinstance(trade.side, str) else ""
    if side not in ("LONG", "SHORT"):
        raise ValueError("trade.side must be 'LONG' or 'SHORT'")

    for field_name, value in (
        ("entry_price", trade.entry_price),
        ("stop_loss", trade.stop_loss),
        ("take_profit", trade.take_profit),
        ("quantity", trade.quantity),
    ):
        if not _is_finite_positive(value):
            raise ValueError(
                f"trade.{field_name} must be a finite positive number"
            )

    exit_fields = (
        trade.exit_timestamp,
        trade.exit_price,
        trade.exit_status,
    )
    populated = [value is not None for value in exit_fields]
    if not any(populated):
        return True
    if not all(populated):
        raise ValueError(
            "trade exit fields must be all None or all populated"
        )

    if not isinstance(trade.exit_timestamp, datetime):
        raise TypeError("exit_timestamp must be a datetime")
    try:
        exit_before_entry = trade.exit_timestamp < trade.entry_timestamp
    except TypeError as exc:
        raise ValueError(
            "entry and exit timestamps must be mutually comparable"
        ) from exc
    if exit_before_entry:
        raise ValueError(
            "exit_timestamp must not be earlier than entry_timestamp"
        )
    if not _is_finite_positive(trade.exit_price):
        raise ValueError("exit_price must be a finite positive number")
    if not isinstance(trade.exit_status, PaperOrderStatus):
        raise TypeError("exit_status must be a PaperOrderStatus")
    if trade.exit_status not in _CLOSED_STATUSES:
        raise ValueError("exit_status must describe a closed trade")
    return False


def _realized_pnl(trade: HistoricalTrade) -> float:
    side = trade.side.strip().upper()
    entry = float(trade.entry_price)
    exit_price = float(trade.exit_price)
    quantity = float(trade.quantity)
    if side == "LONG":
        pnl = (exit_price - entry) * quantity
    else:
        pnl = (entry - exit_price) * quantity
    if not math.isfinite(pnl):
        raise ValueError("trade realized PnL is not finite")
    return pnl


def compute_backtest_metrics(
    trades: list[HistoricalTrade] | tuple[HistoricalTrade, ...],
    initial_balance: float,
) -> BacktestMetrics:
    """Compute realized performance metrics from historical trades.

    Open trades are those with all three exit fields set to None. They are
    counted in ``total_trades`` and ``open_trades`` but excluded from all
    realized calculations. Completed trades are ordered by exit timestamp;
    equal timestamps preserve their original input order.

    Args:
        trades: HistoricalTrade records, which are not modified.
        initial_balance: Starting equity; must be finite and positive.

    Returns:
        Immutable metrics. Win rate and max drawdown are percentages.
        Profit factor is None when there is no gross loss.

    Raises:
        TypeError: If trades are not a list/tuple of HistoricalTrade, or a
            timestamp/status has the wrong type.
        ValueError: If the initial balance is invalid, a trade field is
            invalid, exit fields are partially populated, or numeric
            calculations overflow.
    """
    if not isinstance(trades, (list, tuple)):
        raise TypeError("trades must be a list or tuple of HistoricalTrade")
    if not _is_finite_positive(initial_balance):
        raise ValueError("initial_balance must be a finite positive number")

    balance = float(initial_balance)
    completed = []
    open_trades = 0

    for input_index, trade in enumerate(trades):
        is_open = _validate_trade(trade)
        if is_open:
            open_trades += 1
            continue
        completed.append(
            (trade.exit_timestamp, input_index, _realized_pnl(trade))
        )

    try:
        completed.sort(key=lambda item: (item[0], item[1]))
    except TypeError as exc:
        raise ValueError(
            "exit timestamps must be mutually comparable"
        ) from exc

    pnl_values = [pnl for _, _, pnl in completed]
    wins = sum(1 for pnl in pnl_values if pnl > 0)
    losses = sum(1 for pnl in pnl_values if pnl < 0)
    breakeven = sum(1 for pnl in pnl_values if pnl == 0)

    try:
        gross_profit = math.fsum(pnl for pnl in pnl_values if pnl > 0)
        gross_loss = abs(math.fsum(pnl for pnl in pnl_values if pnl < 0))
        total_realized_pnl = math.fsum(pnl_values)
    except OverflowError as exc:
        raise ValueError("aggregate PnL overflowed") from exc
    if (
        not math.isfinite(gross_profit)
        or not math.isfinite(gross_loss)
        or not math.isfinite(total_realized_pnl)
    ):
        raise ValueError("aggregate PnL is not finite")

    equity = balance
    peak = equity
    max_drawdown = 0.0
    max_drawdown_percent = 0.0

    for _, _, pnl in completed:
        equity += pnl
        if not math.isfinite(equity):
            raise ValueError("equity calculation is not finite")

        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if not math.isfinite(drawdown):
            raise ValueError("drawdown calculation is not finite")
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        drawdown_percent = (drawdown / peak) * 100.0
        if not math.isfinite(drawdown_percent):
            raise ValueError("drawdown percentage is not finite")
        if drawdown_percent > max_drawdown_percent:
            max_drawdown_percent = drawdown_percent

    completed_count = len(completed)
    win_rate = (
        (wins / completed_count) * 100.0 if completed_count > 0 else 0.0
    )
    try:
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0.0 else None
        )
    except OverflowError as exc:
        raise ValueError("profit factor is not finite") from exc
    if profit_factor is not None and not math.isfinite(profit_factor):
        raise ValueError("profit factor is not finite")

    return BacktestMetrics(
        total_trades=len(trades),
        open_trades=open_trades,
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        total_realized_pnl=total_realized_pnl,
        profit_factor=profit_factor,
        ending_equity=equity,
        max_drawdown=max_drawdown,
        max_drawdown_percent=max_drawdown_percent,
    )
