"""Rendering for the /settings command (Milestone M-A).

Pure formatting so the operational settings report can be tested without
Telegram or configuration side effects. The caller (the handler) passes
live values; this module never imports settings, credentials or the
handler layer.

The report makes disabled safety limits explicit (``disabled``) instead
of printing a bare ``0`` that could be mistaken for a configured limit.
It never prints the bot token, exchange keys or any other secret.
"""

from __future__ import annotations

from typing import Mapping


def _format_limit(value: float | int) -> str:
    """Format one limit, making the disabled state explicit."""
    if value <= 0:
        return "disabled"
    if isinstance(value, int):
        return str(value)
    return f"{value:,.2f}"


def _format_money_limit(value: float) -> str:
    """Format a dollar limit, making the disabled state explicit."""
    if value <= 0:
        return "disabled"
    return f"${value:,.2f}"


def format_settings_message(
    *,
    execution_mode: str,
    paper_trading: bool,
    guardian_mode: bool,
    portfolio_risk_limit: float,
    portfolio_exposure_limit: float,
    daily_loss_limit: float,
    max_open_positions: int,
    max_positions_per_symbol: int,
    max_exposure_per_symbol: float,
    authorization_status: str,
    alerts_enabled: bool,
    alert_chat_configured: bool,
) -> str:
    """Build the /settings report.

    Args:
        execution_mode: The configured execution mode label.
        paper_trading: Whether paper trading is enabled at runtime.
        guardian_mode: Whether Guardian Mode is active.
        portfolio_risk_limit: Maximum total open risk, 0 = disabled.
        portfolio_exposure_limit: Maximum total exposure, 0 = disabled.
        daily_loss_limit: Maximum daily realized loss, 0 = disabled.
        max_open_positions: Portfolio open-position cap, 0 = disabled.
        max_positions_per_symbol: Per-symbol position cap, 0 = disabled.
        max_exposure_per_symbol: Per-symbol exposure cap, 0 = disabled.
        authorization_status: Secret-free access-control status line.
        alerts_enabled: Whether the alert master switch is on.
        alert_chat_configured: Whether an alert chat id is configured.

    Returns:
        The complete settings message. It contains no credentials.
    """
    paper_status = "Enabled" if paper_trading else "Disabled"
    guardian_status = "Active" if guardian_mode else "Inactive"
    alerts_status = "Enabled" if alerts_enabled else "Disabled"
    if not alerts_enabled:
        chat_status = "n/a (alerts disabled)"
    elif alert_chat_configured:
        chat_status = "Configured"
    else:
        chat_status = "Not configured (alerts will not be sent)"

    return (
        "⚙️ Bot Settings\n\n"
        "Execution:\n"
        f"  Mode: {execution_mode}\n"
        f"  Paper Trading: {paper_status}\n"
        f"  Guardian Mode: {guardian_status}\n\n"
        "Portfolio Limits:\n"
        "  Portfolio Risk Limit: "
        f"{_format_money_limit(portfolio_risk_limit)}\n"
        "  Portfolio Exposure Limit: "
        f"{_format_money_limit(portfolio_exposure_limit)}\n"
        "  Daily Loss Limit: "
        f"{_format_money_limit(daily_loss_limit)}\n"
        "  Max Open Positions: "
        f"{_format_limit(max_open_positions)}\n"
        "  Max Positions per Symbol: "
        f"{_format_limit(max_positions_per_symbol)}\n"
        "  Max Exposure per Symbol: "
        f"{_format_money_limit(max_exposure_per_symbol)}\n\n"
        "Access:\n"
        f"  Authorization: {authorization_status}\n\n"
        "Alerts:\n"
        f"  Alerts: {alerts_status}\n"
        f"  Alert Chat: {chat_status}"
    )
