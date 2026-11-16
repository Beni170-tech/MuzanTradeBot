"""Pure performance analytics for historical backtest trades.

Only completed HistoricalTrade records contribute to realized PnL, win
rate, ending equity, and drawdown. This module performs no I/O and does
not change position sizing or trade execution.

Gross PnL uses the recorded historical quote prices. Net PnL applies the
caller-supplied fee and adverse-slippage assumptions to both entry and
exit. Costs default to exactly zero, preserving legacy behavior.
"""

from dataclasses import dataclass
from datetime import datetime
import math

from backtest.costs import (
    TradeCostBreakdown,
    TradingCostModel,
    ZERO_COSTS,
    calculate_trade_costs,
)
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
    """Aggregate metrics for a historical backtest trade set.

    ``gross_profit`` and ``gross_loss`` describe quote-price outcomes.
    ``net_profit``, ``net_loss``, ``total_realized_pnl``, equity, and
    drawdown use net PnL after the explicit ``cost_model`` assumptions.
    """

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
    total_gross_pnl: float = 0.0
    net_profit: float = 0.0
    net_loss: float = 0.0
    net_profit_factor: float | None = None
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0
    total_costs: float = 0.0
    cost_model: TradingCostModel = ZERO_COSTS


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

    if trade.cost_breakdown is not None and not isinstance(
        trade.cost_breakdown, TradeCostBreakdown
    ):
        raise TypeError("trade.cost_breakdown must be a TradeCostBreakdown")

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


def _resolve_cost_model(
    completed_trades: list[HistoricalTrade],
    cost_model: TradingCostModel | None,
) -> TradingCostModel:
    """Resolve explicit costs, or infer one consistent model from trades."""
    if cost_model is not None:
        if not isinstance(cost_model, TradingCostModel):
            raise TypeError("cost_model must be a TradingCostModel or None")
        return cost_model

    models = set()
    missing = 0
    for trade in completed_trades:
        breakdown = trade.cost_breakdown
        if breakdown is None:
            missing += 1
            continue
        if not isinstance(breakdown.cost_model, TradingCostModel):
            raise TypeError(
                "trade cost breakdowns must contain a TradingCostModel"
            )
        models.add(breakdown.cost_model)

    if not models:
        return ZERO_COSTS
    if missing:
        raise ValueError(
            "completed trades must all include cost breakdowns or none"
        )
    if len(models) != 1:
        raise ValueError("completed trades must use a single cost model")
    return next(iter(models))


def _sum(values, overflow_message: str) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(overflow_message) from exc
    if not math.isfinite(total):
        raise ValueError(overflow_message)
    return total


def _profit_factor(profit: float, loss: float, name: str) -> float | None:
    try:
        factor = profit / loss if loss > 0.0 else None
    except OverflowError as exc:
        raise ValueError(f"{name} is not finite") from exc
    if factor is not None and not math.isfinite(factor):
        raise ValueError(f"{name} is not finite")
    return factor


def compute_backtest_metrics(
    trades: list[HistoricalTrade] | tuple[HistoricalTrade, ...],
    initial_balance: float,
    cost_model: TradingCostModel | None = None,
) -> BacktestMetrics:
    """Compute realized gross and net performance metrics.

    Open trades are those with all three exit fields set to None. They are
    counted in ``total_trades`` and ``open_trades`` but excluded from all
    realized calculations. Completed trades are ordered by exit timestamp;
    equal timestamps preserve their original input order.

    Costs are resolved in this order:

    1. An explicit ``cost_model`` argument is used.
    2. Otherwise, completed trades carrying cost breakdowns must all use
       one identical model, which is applied again to their quote prices.
    3. Otherwise, the exact zero-cost default is used.

    Wins, losses, breakeven, win rate, net profit factor, ending equity,
    and drawdown use net PnL. Gross profit and gross loss remain available
    for comparison with quote-price outcomes.

    Args:
        trades: HistoricalTrade records, which are not modified.
        initial_balance: Starting equity; must be finite and positive.
        cost_model: Optional explicit fee/slippage assumptions.

    Returns:
        Immutable metrics. Win rate and max drawdown are percentages.
        Profit factors are None when there is no corresponding loss.
        The exact assumptions are exposed in ``cost_model``.

    Raises:
        TypeError: If trades are not a list/tuple of HistoricalTrade, a
            timestamp/status has the wrong type, or a cost model is not a
            TradingCostModel.
        ValueError: If the initial balance is invalid, a trade field is
            invalid, exit fields are partially populated, cost models are
            inconsistent, or numeric calculations overflow.
    """
    if not isinstance(trades, (list, tuple)):
        raise TypeError("trades must be a list or tuple of HistoricalTrade")
    if not _is_finite_positive(initial_balance):
        raise ValueError("initial_balance must be a finite positive number")
    if cost_model is not None and not isinstance(cost_model, TradingCostModel):
        raise TypeError("cost_model must be a TradingCostModel or None")

    balance = float(initial_balance)
    completed_trades = []
    open_trades = 0

    for input_index, trade in enumerate(trades):
        is_open = _validate_trade(trade)
        if is_open:
            open_trades += 1
            continue
        completed_trades.append((input_index, trade))

    resolved_cost_model = _resolve_cost_model(
        [trade for _, trade in completed_trades],
        cost_model,
    )
    completed = []
    for input_index, trade in completed_trades:
        breakdown = calculate_trade_costs(
            side=trade.side,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            quantity=trade.quantity,
            cost_model=resolved_cost_model,
        )
        completed.append((trade.exit_timestamp, input_index, breakdown))

    try:
        completed.sort(key=lambda item: (item[0], item[1]))
    except TypeError as exc:
        raise ValueError(
            "exit timestamps must be mutually comparable"
        ) from exc

    gross_values = [breakdown.gross_pnl for _, _, breakdown in completed]
    net_values = [breakdown.net_pnl for _, _, breakdown in completed]
    wins = sum(1 for pnl in net_values if pnl > 0)
    losses = sum(1 for pnl in net_values if pnl < 0)
    breakeven = sum(1 for pnl in net_values if pnl == 0)

    gross_profit = _sum(
        (pnl for pnl in gross_values if pnl > 0),
        "aggregate PnL overflowed",
    )
    gross_loss = abs(
        _sum(
            (pnl for pnl in gross_values if pnl < 0),
            "aggregate PnL overflowed",
        )
    )
    net_profit = _sum(
        (pnl for pnl in net_values if pnl > 0),
        "aggregate PnL overflowed",
    )
    net_loss = abs(
        _sum(
            (pnl for pnl in net_values if pnl < 0),
            "aggregate PnL overflowed",
        )
    )
    total_gross_pnl = _sum(gross_values, "aggregate PnL overflowed")
    total_realized_pnl = _sum(net_values, "aggregate PnL overflowed")
    total_fees = _sum(
        (breakdown.total_fees for _, _, breakdown in completed),
        "aggregate costs overflowed",
    )
    total_slippage_cost = _sum(
        (breakdown.slippage_cost for _, _, breakdown in completed),
        "aggregate costs overflowed",
    )
    total_costs = _sum(
        (breakdown.total_costs for _, _, breakdown in completed),
        "aggregate costs overflowed",
    )

    equity = balance
    peak = equity
    max_drawdown = 0.0
    max_drawdown_percent = 0.0

    for _, _, breakdown in completed:
        equity += breakdown.net_pnl
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
    profit_factor = _profit_factor(
        gross_profit,
        gross_loss,
        "profit factor",
    )
    net_profit_factor = _profit_factor(
        net_profit,
        net_loss,
        "net profit factor",
    )

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
        total_gross_pnl=total_gross_pnl,
        net_profit=net_profit,
        net_loss=net_loss,
        net_profit_factor=net_profit_factor,
        total_fees=total_fees,
        total_slippage_cost=total_slippage_cost,
        total_costs=total_costs,
        cost_model=resolved_cost_model,
    )
