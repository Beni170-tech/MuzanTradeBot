"""Portfolio-level open position count limit for paper trading.

Sprint 8B: Prevent new paper trades once the paper account already has
the configured number of open positions across all symbols.

Rules:
    - Limit is portfolio-wide, not per-symbol.
    - When ``current_open_positions >= limit``, new trades are rejected
      until existing positions are closed.
    - A limit ``<= 0`` disables the rule.

Public API:
    - ``max_open_positions.py`` exposes pure helpers for counting and
      limit checking, plus a manager used by ``telegram_bot/handlers.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class MaxOpenPositionsResult:
    """Frozen evaluation result for the maximum open positions rule."""

    allowed: bool
    current_open_positions: int
    limit: int
    remaining: int
    reason: str


@dataclass(frozen=True)
class MaxOpenPositionsMetrics:
    """Frozen metrics for the /stats section."""

    enabled: bool
    current_open_positions: int
    limit: int
    usage_percent: float
    remaining: int
    status: str


def count_open_positions(positions: Iterable[Mapping[str, Any]] | None) -> int:
    """Count open positions across the whole paper account.

    Counts each distinct open position once. Position input is expected to
    follow the existing broker/account open-positions shape already used by
    other paper-trader helpers.
    """
    if not positions:
        return 0
    return sum(1 for _ in positions)


def evaluate(
    positions: Iterable[Mapping[str, Any]] | None,
    limit: int,
) -> MaxOpenPositionsResult:
    """Evaluate whether a new paper trade would be allowed under the limit.

    Returns a frozen result with the current usage, remaining capacity, and
    a stable reason string suitable for Telegram messages.
    """
    current = count_open_positions(positions)
    if limit <= 0:
        return MaxOpenPositionsResult(
            allowed=True,
            current_open_positions=current,
            limit=0,
            remaining=-1,
            reason="portfolio max open positions limit disabled",
        )

    remaining = limit - current
    if current >= limit:
        return MaxOpenPositionsResult(
            allowed=False,
            current_open_positions=current,
            limit=limit,
            remaining=0,
            reason=(
                "portfolio already has the maximum number of open positions"
            ),
        )

    return MaxOpenPositionsResult(
        allowed=True,
        current_open_positions=current,
        limit=limit,
        remaining=remaining,
        reason="",
    )


def metrics(
    positions: Iterable[Mapping[str, Any]] | None,
    limit: int,
) -> MaxOpenPositionsMetrics:
    """Build frozen metrics for /stats when the limit is configured."""
    current = count_open_positions(positions)
    if limit <= 0:
        return MaxOpenPositionsMetrics(
            enabled=False,
            current_open_positions=current,
            limit=0,
            usage_percent=0.0,
            remaining=-1,
            status="disabled",
        )

    usage = (current / limit) if limit else 0.0
    remaining = limit - current
    if current >= limit:
        status = "at limit"
    elif current >= (limit * 0.8):
        status = "near limit"
    else:
        status = "ok"

    return MaxOpenPositionsMetrics(
        enabled=True,
        current_open_positions=current,
        limit=limit,
        usage_percent=round(usage * 100, 1),
        remaining=remaining,
        status=status,
    )


class MaxOpenPositionsManager:
    """Lightweight manager wrapping the account/broker used by the bot.

    Kept intentionally thin and side-effect free: the real work is done by
    :func:`evaluate` and :func:`metrics`. The manager exists so the handler
    layer can reuse the same shape as the other Sprint 7A-8A risk managers.

    The source is normally an execution backend (``PaperBroker``), which
    exposes ``get_open_positions()`` rather than an ``open_positions`` list.
    Reading the broker's contract method is what makes this rule actually
    enforce the portfolio-wide cap; the attribute form is kept only for
    backward compatibility with callers that pass a ``PaperAccount``-like
    object.
    """

    def __init__(self, account: Any, limit: int) -> None:
        self._account = account
        self._limit = limit

    def _positions(self) -> Any:
        """Return the current open positions from the wrapped source.

        Prefers the execution-backend contract (``get_open_positions()``)
        and falls back to an ``open_positions`` attribute for account-like
        objects. A backend failure is not swallowed: the handler layer
        treats a raising gate as "do not place", which fails closed for a
        capital-protection limit.
        """
        getter = getattr(self._account, "get_open_positions", None)
        if callable(getter):
            return getter() or []
        return getattr(self._account, "open_positions", None) or []

    def evaluate(self) -> MaxOpenPositionsResult:
        return evaluate(self._positions(), self._limit)

    def metrics(self) -> MaxOpenPositionsMetrics:
        return metrics(self._positions(), self._limit)
