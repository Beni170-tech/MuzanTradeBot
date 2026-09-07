"""Correlation / concentration risk management for paper trading.

Sprint 7C: Prevent excessive concentration by rejecting new paper
trades that would exceed configurable per-symbol rules:

    1. Position-count cap — no more than ``N`` open paper positions
       on the same symbol.
    2. Per-symbol exposure cap — total notional exposure
       (entry_price * quantity) on one symbol may not exceed ``L``.

Exposure arithmetic is deliberately *not* duplicated here: it reuses
``trader.paper.analytics.position_exposure``, the single source of
truth for notional exposure introduced in Sprint 7B.

The correlation risk manager is a read-only analysis tool. It never
places, closes, or modifies orders. It answers three questions per
candidate trade:
    1. How many open positions already exist on the trade's symbol?
    2. How much exposure is already open on that symbol?
    3. Would adding the trade exceed either configured limit?

Both rules are disabled by default (limit <= 0), preserving backward
compatibility — existing installations without limits are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trader.paper.analytics import position_exposure

if TYPE_CHECKING:
    from trader.trade import Trade


# ---------------------------------------------------------------------------
# Pure per-symbol calculation helpers
# ---------------------------------------------------------------------------

def open_position_count(broker, symbol: str) -> int:
    """Count open paper positions on a single symbol.

    Symbols are matched exactly, case-insensitively (uppercase), which
    is how the paper broker already stores them.

    Args:
        broker: A PaperBroker-like object providing
            ``get_open_positions()``.
        symbol: The symbol to count (case-insensitive).

    Returns:
        The number of currently open positions whose symbol matches.
    """
    symbol_upper = symbol.strip().upper()
    return sum(
        1 for position in broker.get_open_positions()
        if position.symbol == symbol_upper
    )


def symbol_exposure(broker, symbol: str) -> float:
    """Compute total notional exposure open on a single symbol.

    Exposure is the sum of ``position_exposure(order)`` across every
    open position whose symbol matches. Exposure math is delegated to
    the Sprint 7B helper so both features stay consistent.

    Args:
        broker: A PaperBroker-like object providing
            ``get_open_positions()``.
        symbol: The symbol to sum exposure for (case-insensitive).

    Returns:
        Total notional exposure (entry_price * quantity) currently
        open on the symbol. Always >= 0.
    """
    symbol_upper = symbol.strip().upper()
    return sum(
        position_exposure(position)
        for position in broker.get_open_positions()
        if position.symbol == symbol_upper
    )


def trade_exposure(trade: "Trade") -> float:
    """Compute the notional exposure a candidate trade would add.

    Mirrors ``position_exposure`` semantics for an unplaced trade:
    ``trade.price * trade.position_size``.

    Args:
        trade: The trade to evaluate.

    Returns:
        The dollar notional exposure the trade would contribute if
        placed. Always >= 0.
    """
    price = trade.price
    size = trade.position_size
    if not isinstance(price, (int, float)) or price <= 0:
        return 0.0
    if not isinstance(size, (int, float)) or size <= 0:
        return 0.0
    return float(price * size)


def prospective_symbol_exposure(broker, trade: "Trade") -> float:
    """Compute the symbol exposure if a candidate trade is added.

    Args:
        broker: A PaperBroker-like object providing
            ``get_open_positions()``.
        trade: The candidate Trade being considered for placement.

    Returns:
        Current exposure on the trade's symbol plus the exposure the
        trade itself would add.
    """
    return symbol_exposure(broker, trade.symbol) + trade_exposure(trade)


# ---------------------------------------------------------------------------
# Result and metrics dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CorrelationCheck:
    """Outcome of evaluating a candidate trade against the concentration rules.

    Attributes:
        allowed: True when the trade may be placed (both rules pass, or
            both rules are disabled).
        rule: The rule that rejected the trade — ``"positions"`` for
            the position-count cap, ``"exposure"`` for the per-symbol
            exposure cap. None when ``allowed`` is True.
    """

    allowed: bool
    rule: str | None = None


@dataclass(frozen=True, slots=True)
class CorrelationMetrics:
    """Per-symbol concentration snapshot for display.

    Attributes:
        symbol: The normalized (uppercase) symbol these metrics describe.
        open_positions: Open position count on the symbol.
        max_positions: The configured position cap. 0 when disabled.
        symbol_exposure: Total notional exposure open on the symbol.
        exposure_limit: The configured per-symbol exposure cap.
            0 when disabled.
        exposure_usage_percent: Symbol exposure as a percentage of the
            exposure limit. 0.0 when no exposure limit is configured.
        remaining_exposure: Exposure budget remaining on the symbol.
            0.0 when no exposure limit is configured.
    """

    symbol: str
    open_positions: int
    max_positions: int
    symbol_exposure: float
    exposure_limit: float
    exposure_usage_percent: float
    remaining_exposure: float


# ---------------------------------------------------------------------------
# CorrelationRiskManager
# ---------------------------------------------------------------------------

class CorrelationRiskManager:
    """Per-symbol concentration gate for paper trading.

    Wraps a PaperBroker plus the two configurable limits and provides
    methods to evaluate whether a new trade may be opened.

    When a limit is 0 or negative, that rule is not enforced. When
    both limits are disabled, all checks pass unconditionally. This
    preserves backward compatibility — existing installations without
    configured limits are unaffected.

    Usage:
        manager = CorrelationRiskManager(
            broker,
            max_positions_per_symbol=1,
            max_exposure_per_symbol=5000.0,
        )
        if manager.can_open_trade(trade):
            broker.place_order(trade)
        else:
            # reject the trade
    """

    def __init__(
        self,
        broker,
        max_positions_per_symbol: int = 0,
        max_exposure_per_symbol: float = 0.0,
    ) -> None:
        """Initialize the correlation risk manager.

        Args:
            broker: The paper broker to monitor.
            max_positions_per_symbol: Maximum open paper positions
                allowed on one symbol. When <= 0, the position-count
                rule is disabled.
            max_exposure_per_symbol: Maximum notional exposure allowed
                on one symbol. When <= 0, the exposure rule is
                disabled.
        """
        self._broker = broker
        self._max_positions_per_symbol = max_positions_per_symbol
        self._max_exposure_per_symbol = max_exposure_per_symbol

    @property
    def max_positions_per_symbol(self) -> int:
        """The configured per-symbol open-position cap."""
        return self._max_positions_per_symbol

    @property
    def max_exposure_per_symbol(self) -> float:
        """The configured per-symbol notional exposure cap."""
        return self._max_exposure_per_symbol

    def evaluate(self, trade: "Trade") -> CorrelationCheck:
        """Evaluate a candidate trade against both concentration rules.

        The position-count rule is evaluated first; when it rejects,
        the exposure rule is not consulted. When a rule's limit is
        disabled (<= 0), that rule always passes.

        Args:
            trade: The trade to evaluate.

        Returns:
            A CorrelationCheck describing whether the trade may be
            placed and, when rejected, which rule rejected it.
        """
        if trade is None:
            return CorrelationCheck(allowed=True)

        # Rule A: position-count cap (enforced only when configured).
        if self._max_positions_per_symbol > 0:
            count = open_position_count(self._broker, trade.symbol)
            if count >= self._max_positions_per_symbol:
                return CorrelationCheck(
                    allowed=False, rule="positions"
                )

        # Rule B: per-symbol exposure cap (enforced only when configured).
        if self._max_exposure_per_symbol > 0:
            if (
                prospective_symbol_exposure(self._broker, trade)
                > self._max_exposure_per_symbol
            ):
                return CorrelationCheck(allowed=False, rule="exposure")

        return CorrelationCheck(allowed=True)

    def can_open_trade(self, trade: "Trade") -> bool:
        """Check whether a trade can be opened without exceeding the limits.

        When both limits are disabled (<= 0), this always returns True.

        Args:
            trade: The trade to evaluate.

        Returns:
            True when the trade can be opened, False when it would
            exceed either configured per-symbol limit.
        """
        return self.evaluate(trade).allowed

    def metrics(self, symbol: str) -> CorrelationMetrics:
        """Build a per-symbol concentration snapshot.

        Args:
            symbol: The symbol to build metrics for (case-insensitive).

        Returns:
            A CorrelationMetrics snapshot with the current open count
            and exposure on the symbol, plus the configured limits and
            remaining budget.
        """
        normalized = symbol.strip().upper()
        count = open_position_count(self._broker, normalized)
        exposure = symbol_exposure(self._broker, normalized)
        limit = self._max_exposure_per_symbol

        if limit > 0:
            usage_percent = (exposure / limit) * 100.0
            remaining = max(0.0, limit - exposure)
        else:
            usage_percent = 0.0
            remaining = 0.0

        return CorrelationMetrics(
            symbol=normalized,
            open_positions=count,
            max_positions=self._max_positions_per_symbol,
            symbol_exposure=exposure,
            exposure_limit=limit,
            exposure_usage_percent=usage_percent,
            remaining_exposure=remaining,
        )

    def concentration_report(self) -> list[CorrelationMetrics]:
        """Build per-symbol metrics for every symbol with open positions.

        Returns:
            List of CorrelationMetrics, one per distinct symbol with at
            least one open position, sorted by symbol exposure
            descending. Empty when there are no open positions.
        """
        symbols = sorted(
            {position.symbol for position in self._broker.get_open_positions()}
        )
        results = [self.metrics(symbol) for symbol in symbols]
        results.sort(key=lambda m: m.symbol_exposure, reverse=True)
        return results
