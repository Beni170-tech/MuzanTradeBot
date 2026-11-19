"""Paper sizing gates for one /analyze request.

Equity-based paper sizing needs two verified inputs before any risk
profile can be built:

    - A readable paper account equity. When it cannot be read, or is not
      a finite positive number, the trade must be blocked rather than
      sized from a configured balance that no longer matches the account.
    - A recent, valid market mark on every open position. Equity values
      open positions at recorded marks, so a missing or stale mark means
      the equity itself is unknown.

Both rules fail closed. This module owns those decisions; the Telegram
handler layer only supplies the live service factory, the settings and
its patchable equity reader, so it can keep its existing seams while the
business logic lives here.

The service access is injected as a factory because the handler layer
owns lazy loading and test overrides; nothing here caches a service.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

from strategy.trading_policy import TradingPolicy
from trader.paper.mark_freshness import (
    MarksFreshnessResult,
    check_marks_freshness,
    max_mark_age_seconds,
)

logger = logging.getLogger(__name__)


def read_paper_equity(service_factory: Callable[[], Any]) -> float:
    """Return the paper account equity used to size a new trade.

    Guardian Mode sizes from the capital that is actually at risk, so
    realized losses and unrealized results at recorded marks shrink or
    grow the risk budget instead of every trade sizing from the static
    configured balance.

    A failure to read the account fails closed with 0.0: the risk manager
    then rejects the trade (balance must be positive) rather than silently
    sizing from a configured balance that no longer matches the account.

    Args:
        service_factory: Returns the live paper trading service.

    Returns:
        The paper account equity, or 0.0 when it cannot be read.
    """
    try:
        return float(service_factory().summary().equity)
    except Exception:
        logger.warning(
            "Could not read paper account equity for position sizing; "
            "refusing to size this trade.",
            exc_info=True,
        )
        return 0.0


def resolve_trading_policy(
    strategy: Any,
    settings: Any,
    equity_reader: Callable[[], float],
) -> TradingPolicy | None:
    """Resolve the validated policy that drives this trade's sizing.

    Called as late as practical: after analysis, the fresh quote, and the
    market-mark freshness gate. In paper mode, sizing uses the paper
    account's current equity. When that equity cannot be read, or is not
    positive, no policy can be built and ``None`` is returned so the
    caller blocks the trade instead of falling back to a configured
    balance that no longer matches the account. When paper trading is
    disabled there is no paper account to read, so the configured balance
    is used as before.

    Args:
        strategy: The exact strategy instance that produced this
            decision's signal. It is carried unchanged by the policy.
        settings: The live settings object.
        equity_reader: Returns the paper account equity. Only called when
            paper trading is enabled, so a disabled paper mode never
            touches the account.

    Returns:
        The TradingPolicy for this analysis request, or None when paper
        equity is unusable.
    """
    if not settings.PAPER_TRADING:
        return TradingPolicy.from_settings(settings, strategy=strategy)

    equity = equity_reader()
    if not math.isfinite(equity) or equity <= 0:
        return None
    return TradingPolicy.from_settings(
        settings, balance=equity, strategy=strategy
    )


def paper_mark_block(
    *,
    paper_trading: bool,
    monitor_interval_seconds: Any,
    service_factory: Callable[[], Any],
) -> MarksFreshnessResult | None:
    """Return the freshness result that forbids sizing, or None.

    Equity-based sizing values open positions at recorded market marks,
    so every open position must have a valid, recent mark before a new
    trade may be sized. A missing/invalid/stale mark (or unreadable
    account state) blocks the trade; it never falls back to
    ``ACCOUNT_BALANCE``, an entry price, or an old mark.

    Args:
        paper_trading: Whether paper trading is enabled right now.
        monitor_interval_seconds: The configured background monitor
            interval, which sets the maximum accepted mark age.
        service_factory: Returns the live paper trading service.

    Returns:
        None when sizing may proceed (including when paper trading is
        disabled), otherwise the freshness result explaining the block.
    """
    if not paper_trading:
        return None

    max_age = max_mark_age_seconds(monitor_interval_seconds)
    try:
        positions = service_factory().open_positions()
    except Exception:
        logger.warning(
            "Could not read open paper positions for mark freshness; "
            "refusing to size this trade.",
            exc_info=True,
        )
        return MarksFreshnessResult(
            fresh=False,
            reason="paper account state could not be read",
            max_age_seconds=max_age,
        )

    result = check_marks_freshness(positions, max_age_seconds=max_age)
    return None if result.fresh else result
