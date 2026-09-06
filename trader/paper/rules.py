"""Paper trading validation rules.

This module defines the result type for paper trade validation and the
single, pure validation entry point. It performs no execution, no storage,
and no Telegram notifications.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trader.trade import Trade


@dataclass(frozen=True, slots=True)
class PaperRuleResult:
    """Outcome of validating a candidate paper trade.

    Attributes:
        ok: Whether the candidate passed all validation rules.
        reason: Human-readable explanation when ``ok`` is ``False``. If
            ``ok`` is ``True``, this is an empty string.
    """

    ok: bool
    reason: str


def validate_paper_trade(
    trade: Trade,
) -> PaperRuleResult:
    """Validate a candidate paper trade against paper-trading invariants.

    This function is pure: it performs no side effects, performs no
    persistence, and does not contact any market data feed or broker.
    It only checks structural and business-rule constraints that any
    paper trade must satisfy before it can be considered for execution.

    Args:
        trade: The :class:`Trade` candidate to validate.

    Returns:
        A :class:`PaperRuleResult` describing whether the candidate is
        valid and, if not, why.
    """
    if trade is None:
        return PaperRuleResult(
            ok=False,
            reason="trade must be provided",
        )

    symbol = trade.symbol
    side = trade.side
    quantity = trade.position_size
    price = trade.price

    if not symbol or not isinstance(symbol, str) or symbol.strip() == "":
        return PaperRuleResult(
            ok=False,
            reason="trade.symbol must be a non-empty string",
        )

    normalized_side = side.strip().upper() if isinstance(side, str) else ""
    if normalized_side not in ("LONG", "SHORT", "BUY", "SELL"):
        return PaperRuleResult(
            ok=False,
            reason="trade.side must be 'LONG', 'SHORT', 'BUY', or 'SELL'",
        )

    if not isinstance(quantity, (int, float)) or quantity <= 0:
        return PaperRuleResult(
            ok=False,
            reason="trade.position_size must be a positive number",
        )

    if not isinstance(price, (int, float)) or price <= 0:
        return PaperRuleResult(
            ok=False,
            reason="trade.price must be a positive number",
        )

    return PaperRuleResult(ok=True, reason="")
