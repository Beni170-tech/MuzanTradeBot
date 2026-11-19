"""Rendering for the ``/report`` command (Milestone M-C).

Pure formatting over the ``/report`` read model, so the operational
report can be tested without Telegram, storage or configuration. The
handler supplies the view; this module never reads account state and
never places or modifies orders.

The report is deliberately a period summary: all time, today, the last
seven calendar days and per-symbol results. Entry-gate metrics remain on
``/stats``. Nothing here shows credentials.
"""

from __future__ import annotations

from trader.paper.read_model import PaperReportView, PerformanceWindow

_SECTION = "\u2501" * 19


def _format_pnl(value: float) -> str:
    """Format a PnL value with sign and currency symbol."""
    if value > 0:
        return f"+${value:.2f}"
    if value < 0:
        return f"-${abs(value):.2f}"
    return "$0.00"


def _format_percent(value: float) -> str:
    """Format a percentage with an explicit sign."""
    return f"{value:+.2f}%"


def _format_profit_factor(value: float | None) -> str:
    """Format a profit factor that may be infinite or undefined."""
    if value is None:
        return "N/A"
    if value == float("inf"):
        return "\u221e"
    if value == float("-inf"):
        return "-\u221e"
    return f"{value:.2f}"


def _format_window(window: PerformanceWindow) -> str:
    """Render the trade metrics shared by today and the last seven days."""
    overall = window.overall
    if overall.total_trades == 0:
        return "No trades closed."
    return (
        f"Closed Trades: {overall.total_trades}\n"
        f"Win Rate: {overall.win_rate:.1f}%\n"
        f"PnL: {_format_pnl(overall.total_realized_pnl)}"
    )


def format_report_message(view: PaperReportView) -> str:
    """Build the ``/report`` message from a paper report view.

    Args:
        view: The period and per-symbol performance read model.

    Returns:
        The complete report text. It contains no credentials.
    """
    summary = view.summary
    message = (
        "\U0001f4d1 Paper Trading Report\n\n"
        f"{_SECTION}\n"
        "\U0001f4cb Account\n"
        f"{_SECTION}\n"
        f"Balance: ${summary.current_balance:,.2f}\n"
        f"Initial: ${summary.initial_balance:,.2f}\n"
        f"Return: {_format_percent(view.total_return)}\n"
        f"Open Positions: {len(view.open_positions)}"
    )

    all_time = view.all_time
    if all_time is None:
        message += "\n\n\U0001f4ed No closed trades yet."
        return message

    overall = all_time.overall
    message += (
        f"\n\n{_SECTION}\n"
        "\U0001f5d3 All Time\n"
        f"{_SECTION}\n"
        f"Closed Trades: {overall.total_trades}\n"
        f"Win Rate: {overall.win_rate:.1f}%\n"
        f"Wins: {overall.winning_trades}  "
        f"Losses: {overall.losing_trades}  "
        f"Breakevens: {overall.breakeven_trades}\n"
        f"Total PnL: {_format_pnl(overall.total_realized_pnl)}\n"
        f"Expectancy: {_format_pnl(overall.expectancy)}/trade"
    )

    risk = all_time.risk
    if risk is not None:
        message += (
            f"\nProfit Factor: {_format_profit_factor(risk.profit_factor)}\n"
            f"Max Consec. Losses: {risk.max_consecutive_losses}"
        )

    message += (
        f"\n\n{_SECTION}\n"
        f"\U0001f305 Today ({view.today_label})\n"
        f"{_SECTION}\n"
        f"{_format_window(view.today)}"
        f"\n\n{_SECTION}\n"
        f"\U0001f4c5 Last 7 Days ({view.week_start_label} to "
        f"{view.today_label})\n"
        f"{_SECTION}\n"
        f"{_format_window(view.week)}"
    )

    if view.symbol_groups:
        message += (
            f"\n\n{_SECTION}\n"
            "\U0001fa99 By Symbol\n"
            f"{_SECTION}\n"
        )
        for symbol in view.symbol_groups:
            message += (
                f"{symbol.symbol}: {symbol.total_trades} trades, "
                f"WR {symbol.win_rate:.1f}%, "
                f"PnL {_format_pnl(symbol.total_pnl)}\n"
            )
        hidden = view.symbol_count - len(view.symbol_groups)
        if hidden > 0:
            message += f"(+{hidden} more symbol{'s' if hidden != 1 else ''})\n"

    message += "\nPaper trading only \u2014 no live orders are placed."
    return message
