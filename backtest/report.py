"""Pure, deterministic reporting for completed historical backtests.

This module consumes existing HistoricalTrade records and delegates all
metric and cost resolution to :func:`backtest.analytics.compute_backtest_metrics`.
It performs no I/O, network access, paper-account changes, or execution.

Reports always expose the fee and slippage assumptions. Zero costs are
reported as an explicit configuration choice, never as realistic exchange
costs or actual market fills.
"""

from dataclasses import dataclass
from datetime import datetime

from backtest.analytics import BacktestMetrics, compute_backtest_metrics
from backtest.costs import (
    TradingCostModel,
    ZERO_COSTS,
    calculate_trade_costs,
)
from backtest.trades import HistoricalTrade
from trader.paper.order import PaperOrderStatus


@dataclass(frozen=True, slots=True)
class BacktestTradeSummary:
    """One deterministic summary row for an individual backtest trade.

    Completed trades include gross and net PnL plus modeled fills and
    costs. Open trades expose no unrealized PnL because HistoricalTrade
    has no mark price; their cost fields remain None.
    """

    number: int
    symbol: str
    side: str
    entry_timestamp: datetime
    exit_timestamp: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: float
    exit_status: PaperOrderStatus | None
    is_open: bool
    entry_fill_price: float | None
    exit_fill_price: float | None
    gross_pnl: float | None
    net_pnl: float | None
    total_fees: float | None
    slippage_cost: float | None
    total_costs: float | None


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """Immutable report summary, trade rows, and explicit limitations."""

    initial_equity: float
    ending_equity: float
    metrics: BacktestMetrics
    trades: tuple[BacktestTradeSummary, ...]
    warnings: tuple[str, ...]

    @property
    def cost_model(self) -> TradingCostModel:
        """Return the exact fee and slippage assumptions used."""
        return self.metrics.cost_model

    @property
    def total_trades(self) -> int:
        return self.metrics.total_trades

    @property
    def open_trades(self) -> int:
        return self.metrics.open_trades

    @property
    def wins(self) -> int:
        return self.metrics.wins

    @property
    def losses(self) -> int:
        return self.metrics.losses

    @property
    def win_rate(self) -> float:
        return self.metrics.win_rate

    @property
    def gross_pnl(self) -> float:
        return self.metrics.total_gross_pnl

    @property
    def net_pnl(self) -> float:
        return self.metrics.total_realized_pnl

    @property
    def total_costs(self) -> float:
        return self.metrics.total_costs

    @property
    def profit_factor(self) -> float | None:
        return self.metrics.profit_factor

    @property
    def max_drawdown(self) -> float:
        return self.metrics.max_drawdown


def _is_open(trade: HistoricalTrade) -> bool:
    return (
        trade.exit_timestamp is None
        and trade.exit_price is None
        and trade.exit_status is None
    )


def _build_trade_summaries(
    trades,
    cost_model: TradingCostModel,
) -> tuple[BacktestTradeSummary, ...]:
    summaries = []
    for number, trade in enumerate(trades, start=1):
        is_open = _is_open(trade)
        breakdown = None
        if not is_open:
            breakdown = calculate_trade_costs(
                side=trade.side,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                cost_model=cost_model,
            )

        summaries.append(
            BacktestTradeSummary(
                number=number,
                symbol=trade.symbol,
                side=trade.side.strip().upper(),
                entry_timestamp=trade.entry_timestamp,
                exit_timestamp=trade.exit_timestamp,
                entry_price=float(trade.entry_price),
                exit_price=(
                    None if trade.exit_price is None else float(trade.exit_price)
                ),
                quantity=float(trade.quantity),
                exit_status=trade.exit_status,
                is_open=is_open,
                entry_fill_price=(
                    None if breakdown is None else breakdown.entry_fill_price
                ),
                exit_fill_price=(
                    None if breakdown is None else breakdown.exit_fill_price
                ),
                gross_pnl=None if breakdown is None else breakdown.gross_pnl,
                net_pnl=None if breakdown is None else breakdown.net_pnl,
                total_fees=(
                    None if breakdown is None else breakdown.total_fees
                ),
                slippage_cost=(
                    None if breakdown is None else breakdown.slippage_cost
                ),
                total_costs=(
                    None if breakdown is None else breakdown.total_costs
                ),
            )
        )
    return tuple(summaries)


def _build_warnings(metrics: BacktestMetrics) -> tuple[str, ...]:
    warnings = [
        "Fees and slippage are modeled assumptions, not actual exchange "
        "costs or executed market fills."
    ]

    if metrics.cost_model == ZERO_COSTS:
        warnings.append(
            "Zero fees and zero slippage were configured; this does not "
            "represent realistic exchange costs."
        )

    completed = metrics.total_trades - metrics.open_trades
    if metrics.total_trades == 0:
        warnings.append(
            "No trades were generated; no performance conclusions can be "
            "drawn."
        )
    elif completed == 0:
        warnings.append(
            "All trades are open; realized performance cannot be evaluated."
        )

    if metrics.open_trades:
        noun = "trade is" if metrics.open_trades == 1 else "trades are"
        warnings.append(
            f"{metrics.open_trades} open {noun} not marked to market and "
            "excluded from realized PnL, ending equity, win rate, profit "
            "factor, and drawdown."
        )

    if completed > 0 and metrics.profit_factor is None:
        warnings.append(
            "Gross profit factor is unavailable because there were no gross "
            "losing completed trades."
        )
    if completed > 0 and metrics.net_profit_factor is None:
        warnings.append(
            "Net profit factor is unavailable because there were no net "
            "losing completed trades."
        )

    return tuple(warnings)


def build_backtest_report(
    trades: list[HistoricalTrade] | tuple[HistoricalTrade, ...],
    initial_balance: float,
    cost_model: TradingCostModel | None = None,
) -> BacktestReport:
    """Build a pure report from historical trades and explicit assumptions.

    Args:
        trades: HistoricalTrade records. The sequence is not modified.
        initial_balance: Starting equity; must be finite and positive.
        cost_model: Optional fee/slippage assumptions. When omitted, the
            analytics layer uses the zero-cost default or a single
            consistent model already attached to completed trades.

    Returns:
        An immutable BacktestReport with summary metrics, individual trade
        summaries, and limitations. Open trades are listed but excluded
        from realized PnL and other realized metrics.

    Raises:
        TypeError: If trades, the balance, or the cost model have invalid
            types.
        ValueError: If a trade, balance, or cost assumption is invalid, or
            calculations overflow.
    """
    metrics = compute_backtest_metrics(
        trades,
        initial_balance,
        cost_model,
    )
    summaries = _build_trade_summaries(trades, metrics.cost_model)
    return BacktestReport(
        initial_equity=float(initial_balance),
        ending_equity=metrics.ending_equity,
        metrics=metrics,
        trades=summaries,
        warnings=_build_warnings(metrics),
    )


def _money(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _number(value: float) -> str:
    return f"{value:.8g}"


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _timestamp(value: datetime | None) -> str:
    if value is None:
        return "not closed"
    return value.isoformat(sep=" ", timespec="seconds")


def _format_trade(trade: BacktestTradeSummary) -> list[str]:
    if trade.is_open:
        status = "OPEN"
    else:
        status = trade.exit_status.value if trade.exit_status else "CLOSED"

    lines = [
        f"Trade {trade.number}: {trade.symbol} {trade.side} - {status}",
        f"  Entry:    {_timestamp(trade.entry_timestamp)} @ "
        f"{_money(trade.entry_price)}",
    ]

    if trade.is_open:
        lines.extend(
            [
                "  Exit:     not closed",
                f"  Quantity: {_number(trade.quantity)}",
                "  Gross PnL: not realized",
                "  Net PnL:   not realized",
            ]
        )
        return lines

    lines.extend(
        [
            f"  Exit:     {_timestamp(trade.exit_timestamp)} @ "
            f"{_money(trade.exit_price)}",
            f"  Quantity: {_number(trade.quantity)}",
        ]
    )

    lines.extend(
        [
            f"  Fills:    {_money(trade.entry_fill_price)} -> "
            f"{_money(trade.exit_fill_price)}",
            f"  Gross PnL: {_money(trade.gross_pnl)}",
            f"  Fees:      {_money(trade.total_fees)}",
            f"  Slippage:  {_money(trade.slippage_cost)}",
            f"  Net PnL:   {_money(trade.net_pnl)}",
            f"  Costs:     {_money(trade.total_costs)}",
        ]
    )
    return lines


def format_backtest_report(
    report: BacktestReport,
    max_trades: int | None = None,
) -> str:
    """Format a report as deterministic plain text for terminals or chat.

    The output uses only ASCII characters and does not read the clock,
    network, filesystem, or any mutable runtime state.

    Args:
        report: Report returned by :func:`build_backtest_report`.
        max_trades: Optional maximum number of individual trades to print.
            All trade summaries remain available on ``report.trades``.

    Returns:
        A newline-terminated plain-text report.

    Raises:
        TypeError: If ``report`` is not a BacktestReport, or if
            ``max_trades`` is not an integer.
        ValueError: If ``max_trades`` is a negative integer.
    """
    if not isinstance(report, BacktestReport):
        raise TypeError("report must be a BacktestReport")
    if max_trades is not None:
        if isinstance(max_trades, bool) or not isinstance(max_trades, int):
            raise TypeError("max_trades must be a non-negative integer or None")
        if max_trades < 0:
            raise ValueError("max_trades must be a non-negative integer or None")

    metrics = report.metrics
    completed = metrics.total_trades - metrics.open_trades
    lines = [
        "BACKTEST REPORT",
        "===============",
        f"Initial equity: {_money(report.initial_equity)}",
        f"Ending equity (net): {_money(report.ending_equity)}",
        f"Net PnL:        {_money(report.net_pnl)}",
        f"Gross PnL:      {_money(report.gross_pnl)}",
        "",
        "Cost assumptions",
        "----------------",
        "Fees: "
        f"{report.cost_model.fee_rate * 100.0:.4f}% of notional per side "
        "(entry and exit)",
        "Slippage: "
        f"{report.cost_model.slippage_rate * 100.0:.4f}% adverse price move "
        "per side",
        "These are modeled assumptions, not exchange-reported rates or "
        "actual fills.",
        "",
        "Trades",
        "------",
        f"Total:      {metrics.total_trades}",
        f"Completed:  {completed}",
        f"Open:       {metrics.open_trades}",
        f"Wins:       {metrics.wins}",
        f"Losses:     {metrics.losses}",
        f"Breakeven:  {metrics.breakeven}",
        f"Win rate:   {metrics.win_rate:.2f}%",
        "",
        "Performance",
        "-----------",
        f"Gross PnL:        {_money(metrics.total_gross_pnl)}",
        f"Net PnL:          {_money(metrics.total_realized_pnl)}",
        f"Fees:             {_money(metrics.total_fees)}",
        f"Slippage cost:    {_money(metrics.total_slippage_cost)}",
        f"Total costs:      {_money(metrics.total_costs)}",
        f"Profit factor (gross): {_ratio(metrics.profit_factor)}",
        f"Profit factor (net):   {_ratio(metrics.net_profit_factor)}",
        f"Max drawdown (net): {_money(metrics.max_drawdown)} "
        f"({metrics.max_drawdown_percent:.2f}%)",
        "",
        "Warnings and limitations",
        "-----------------------",
    ]
    lines.extend(f"- {warning}" for warning in report.warnings)
    lines.extend(["", "Trades detail", "-------------"])

    shown = report.trades if max_trades is None else report.trades[:max_trades]
    if not shown:
        lines.append("No trade summaries to display.")
    else:
        for trade in shown:
            lines.extend(_format_trade(trade))
            lines.append("")

    omitted = len(report.trades) - len(shown)
    if omitted > 0:
        lines.append(f"... {omitted} more trade(s) not shown.")

    return "\n".join(lines).rstrip() + "\n"
