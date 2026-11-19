"""Rendering for the ``/status`` command (Milestone M-C).

Pure formatting over values the handler gathers: runtime mode, whether
new entries are allowed, the background monitor, the paper account, the
trade journal and the rate limiter. The renderer never reads state
itself, never places orders and never prints credentials.

The point of the report is that a degraded component is visible: a
paused or unreadable pause control, a stopped monitor, or a failing
journal all appear explicitly rather than being silently absent.
"""

from __future__ import annotations

from journal.journal import JournalStatus
from trader.paper.pause import PauseState

_SECTION = "\u2501" * 19

_MONITOR_LABELS = {
    "running": "Running",
    "stopped": "Stopped (positions are no longer checked automatically)",
    "disabled": "Disabled by configuration",
    "unknown": "Unknown (status check failed)",
}


def _format_bytes(size: int) -> str:
    """Format a byte count compactly."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _entries_status(pause: PauseState) -> str:
    """Describe whether new paper entries are allowed."""
    if pause.degraded:
        return (
            "BLOCKED (pause control state is unreadable)\n"
            "   Fix or remove logs/paper_pause.json, then use /resume."
        )
    if pause.paused:
        status = "Paused \u2014 new entries are blocked"
        if pause.reason:
            status += f"\n   Reason: {pause.reason}"
        return status
    return "Allowed"


def _monitor_status(status: str, interval_seconds: float) -> str:
    """Describe the background monitor's state."""
    label = _MONITOR_LABELS.get(status, status.capitalize())
    if status == "running":
        return f"{label} (every {interval_seconds:g}s)"
    return label


def format_status_message(
    *,
    execution_mode: str,
    paper_trading: bool,
    guardian_mode: bool,
    authorization: str,
    pause: PauseState,
    monitor_status: str,
    monitor_interval: float,
    balance: float,
    open_positions: int,
    closed_trades: int,
    journal: JournalStatus,
    rate_limit_commands: int,
    rate_limit_window: float,
) -> str:
    """Build the operational status message.

    Args:
        execution_mode: The configured execution mode label.
        paper_trading: Whether paper trading is enabled at runtime.
        guardian_mode: Whether Guardian Mode is active.
        authorization: Secret-free access-control status line.
        pause: The global pause control state.
        monitor_status: One of running/stopped/disabled/unknown.
        monitor_interval: The monitor's configured interval in seconds.
        balance: Current paper account balance.
        open_positions: Number of open paper positions.
        closed_trades: Number of closed paper trades.
        journal: The trade journal's health snapshot.
        rate_limit_commands: Commands allowed per window (<= 0 disables).
        rate_limit_window: Window length in seconds.

    Returns:
        The complete status message. It contains no credentials.
    """
    paper_label = "Enabled" if paper_trading else "Disabled"
    guardian_label = "Active" if guardian_mode else "Inactive"

    if rate_limit_commands > 0 and rate_limit_window > 0:
        rate_limit_label = (
            f"{rate_limit_commands} commands / {rate_limit_window:g}s per chat"
        )
    else:
        rate_limit_label = "Disabled"

    message = (
        "\U0001fa7a Operational Status\n\n"
        f"{_SECTION}\n"
        "\u2699\ufe0f Runtime\n"
        f"{_SECTION}\n"
        f"Execution Mode: {execution_mode}\n"
        f"Paper Trading: {paper_label}\n"
        f"Guardian Mode: {guardian_label}\n"
        f"Authorization: {authorization}\n\n"
        f"{_SECTION}\n"
        "\u23f8 New Entries\n"
        f"{_SECTION}\n"
        f"{_entries_status(pause)}\n"
        f"Monitor: {_monitor_status(monitor_status, monitor_interval)}\n\n"
        f"{_SECTION}\n"
        "\U0001f4c4 Paper Account\n"
        f"{_SECTION}\n"
        f"Balance: ${balance:,.2f}\n"
        f"Open Positions: {open_positions}\n"
        f"Closed Trades: {closed_trades}\n\n"
        f"{_SECTION}\n"
        "\U0001f4d3 Trade Journal\n"
        f"{_SECTION}\n"
        f"Path: {journal.path}\n"
    )

    if journal.exists:
        message += (
            f"Size: {_format_bytes(journal.size_bytes)}\n"
            f"Rotated Backups: {journal.rotated_files}\n"
            f"Last Write: {journal.last_write_at}\n"
        )
    else:
        message += "Size: not written yet\n"

    if journal.last_error:
        message += f"Last Error: {journal.last_error}\n"

    message += (
        f"\n{_SECTION}\n"
        "\U0001f6a6 Rate Limit\n"
        f"{_SECTION}\n"
        f"{rate_limit_label}"
    )
    return message
