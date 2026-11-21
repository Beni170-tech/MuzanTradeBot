"""Paper-trading entry gates: the pre-placement portfolio safety checks.

This module is the single domain-level home for the gates that decide
whether a *new* paper position may be opened:

    1. Portfolio risk        (Sprint 7A)
    2. Portfolio exposure    (Sprint 7B)
    3. Daily loss            (Sprint 8A)
    4. Maximum open positions (Sprint 8B)
    5. Symbol concentration  (Sprint 7C, count then exposure)

Two compositions share the same domain calculations:

    - ``evaluate_entry_gates`` answers "may this candidate trade be
      placed right now?". Gates run in their established order and
      evaluation stops at the first rejection, so the reported rule is
      always the highest-precedence failing gate.
    - ``collect_entry_gate_metrics`` answers "what does every enabled
      gate see right now?" for the portfolio dashboard. It never
      short-circuits, so a dashboard section for every enabled gate can
      be rendered from one call.

The gate sequence is preserved exactly as ``/analyze`` has always
applied it:

    - Portfolio risk rejects when the prospective total risk
      (current + trade risk) is strictly greater than the limit, i.e.
      it allows ``prospective <= limit``.
    - Portfolio exposure rejects when the current plus requested
      notional exposure is strictly greater than the limit.
    - The daily loss brake trips when the day's loss (settled PnL plus
      floating PnL at recorded marks) is ``<= -limit``. An open position
      with no usable mark cannot be valued, so the gate rejects rather
      than assuming the floating loss is zero.
    - Maximum open positions rejects when the current count is
      ``>= limit``.
    - Symbol concentration rejects when the symbol's open position
      count is ``>= cap`` and, separately, when the prospective symbol
      exposure is strictly greater than the exposure cap.

Every gate is disabled when its limit is ``<= 0``. A disabled gate is
skipped entirely, so it never reads account state.

Design constraints (Guardian Mode):
    - No chat-client imports, no settings imports, no formatting
      intended for a chat client.
    - No file or network I/O, no alert dispatch.
    - No account, broker, or order mutations: the wrapped managers are
      read-only analysis tools and this module only reads from them.
    - All configuration limits are supplied explicitly by the caller,
      so the result never depends on ambient global state.

All calculations are delegated to the existing domain managers and
helpers (``PortfolioRiskManager``, ``DailyRiskManager``,
``MaxOpenPositionsManager``, ``CorrelationRiskManager``,
``analytics.exposure_metrics`` and ``portfolio_risk.trade_risk``);
nothing is re-implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from trader.paper.analytics import ExposureMetrics, exposure_metrics
from trader.paper.correlation_risk import (
    CorrelationMetrics,
    CorrelationRiskManager,
    trade_exposure,
)
from trader.paper.daily_risk import DailyRiskManager, DailyRiskMetrics
from trader.paper.max_open_positions import (
    MaxOpenPositionsManager,
    MaxOpenPositionsMetrics,
)
from trader.paper.portfolio_risk import (
    PortfolioRiskManager,
    PortfolioRiskMetrics,
    trade_risk,
)

if TYPE_CHECKING:
    from trader.trade import Trade


# Stable rule identifiers reported in ``EntryGateOutcome.rule`` and
# ``GateSnapshot.rule``.
RULE_PORTFOLIO_RISK = "portfolio_risk"
RULE_PORTFOLIO_EXPOSURE = "portfolio_exposure"
RULE_DAILY_LOSS = "daily_loss"
RULE_MAX_OPEN_POSITIONS = "max_open_positions"
RULE_SYMBOL_POSITIONS = "symbol_positions"
RULE_SYMBOL_EXPOSURE = "symbol_exposure"
RULE_INVALID_TRADE = "invalid_trade"


@dataclass(frozen=True, slots=True)
class EntryGateOutcome:
    """Frozen result of evaluating the paper-trading entry gates.

    Attributes:
        allowed: True when every enabled gate passed and a new paper
            position may be considered for placement.
        rule: The stable identifier of the first rejecting gate. None
            when ``allowed`` is True.
        detail: A short human-readable explanation of the rejection,
            without any chat formatting. Empty when ``allowed`` is True.
        metrics: The frozen snapshot object produced by the rejecting
            gate's manager (``PortfolioRiskMetrics``,
            ``ExposureMetrics``, ``DailyRiskMetrics``,
            ``MaxOpenPositionsResult`` or ``CorrelationMetrics``).
            None when no gate rejected. Callers may use it to render
            their own messages without re-reading account state.
    """

    allowed: bool
    rule: str | None = None
    detail: str = ""
    metrics: object | None = None


@dataclass(frozen=True, slots=True)
class GateSnapshot:
    """Frozen result of evaluating a single gate.

    This is the shared per-gate unit that :data:`GATE_ORDER` composes.
    It is one gate's answer, not the combined entry decision.

    Attributes:
        rule: The gate's rule identifier. For the symbol concentration
            gate the primary identifier is ``RULE_SYMBOL_POSITIONS``;
            a rejection identifies the failing rule precisely.
        enabled: False when the gate's limit is ``<= 0`` and the gate
            was skipped without reading account state.
        allowed: True when the gate passes (always True when disabled).
        detail: A short human-readable explanation of a rejection,
            without any chat formatting. Empty when the gate passes.
        metrics: The gate's frozen snapshot when it was built. On a
            rejection this is always the snapshot the callers need; on
            a pass it is populated only when the caller asked for the
            gate's metrics.
    """

    rule: str
    enabled: bool
    allowed: bool
    detail: str = ""
    metrics: object | None = None


@dataclass(frozen=True, slots=True)
class _GateLimits:
    """The explicit limit configuration shared by every gate."""

    portfolio_risk_limit: float
    portfolio_exposure_limit: float
    daily_loss_limit: float
    max_open_positions: int
    max_positions_per_symbol: int
    max_exposure_per_symbol: float
    day: str | None = None


def _disabled(rule: str) -> GateSnapshot:
    """Build the snapshot of a gate skipped because its limit is <= 0."""
    return GateSnapshot(rule=rule, enabled=False, allowed=True)


def _allowed(rule: str, metrics: object | None = None) -> GateSnapshot:
    """Build a passing snapshot for one gate."""
    return GateSnapshot(rule=rule, enabled=True, allowed=True, metrics=metrics)


def _reject(
    rule: str, detail: str, metrics: object | None = None
) -> GateSnapshot:
    """Build a rejecting snapshot for one gate."""
    return GateSnapshot(
        rule=rule, enabled=True, allowed=False, detail=detail, metrics=metrics
    )


def _gate_portfolio_risk(
    backend, trade: "Trade", limits: _GateLimits, *, need_metrics: bool
) -> GateSnapshot:
    """Gate 1: prospective portfolio risk. Allows ``prospective <= limit``."""
    limit = limits.portfolio_risk_limit
    if limit <= 0:
        return _disabled(RULE_PORTFOLIO_RISK)

    manager = PortfolioRiskManager(backend, limit)
    allowed = manager.can_open_trade(trade)
    if allowed and not need_metrics:
        return _allowed(RULE_PORTFOLIO_RISK)

    risk_metrics = manager.metrics()
    if allowed:
        return _allowed(RULE_PORTFOLIO_RISK, risk_metrics)

    added_risk = trade_risk(trade)
    return _reject(
        RULE_PORTFOLIO_RISK,
        (
            f"current risk ${risk_metrics.current_risk:,.2f} plus "
            f"trade risk ${added_risk:,.2f} exceeds portfolio "
            f"risk limit ${limit:,.2f}"
        ),
        risk_metrics,
    )


def _gate_portfolio_exposure(
    backend, trade: "Trade", limits: _GateLimits, *, need_metrics: bool
) -> GateSnapshot:
    """Gate 2: prospective notional exposure. Rejects strictly above the limit.

    The current exposure and the requested notional are derived from a
    single broker read: ``exposure_metrics`` already reports the current
    exposure, and the prospective value is that figure plus
    ``price * position_size``, which is the exact arithmetic the
    pre-extraction check applied.
    """
    limit = limits.portfolio_exposure_limit
    if limit <= 0:
        return _disabled(RULE_PORTFOLIO_EXPOSURE)

    exposure = exposure_metrics(backend, limit)
    requested = trade.price * trade.position_size
    prospective = exposure.current_exposure + requested
    allowed = prospective <= limit
    if allowed and not need_metrics:
        return _allowed(RULE_PORTFOLIO_EXPOSURE)
    if allowed:
        return _allowed(RULE_PORTFOLIO_EXPOSURE, exposure)

    return _reject(
        RULE_PORTFOLIO_EXPOSURE,
        (
            f"current exposure ${exposure.current_exposure:,.2f} plus "
            f"requested exposure ${requested:,.2f} exceeds portfolio "
            f"exposure limit ${limit:,.2f}"
        ),
        exposure,
    )


def _gate_daily_loss(
    backend, trade: "Trade", limits: _GateLimits, *, need_metrics: bool
) -> GateSnapshot:
    """Gate 3: daily loss brake. Trips at ``day loss <= -limit``.

    The day's loss includes floating PnL at recorded marks. When an open
    position has no usable mark the day cannot be measured, and the gate
    rejects instead of treating the unknown as zero loss.
    """
    limit = limits.daily_loss_limit
    if limit <= 0:
        return _disabled(RULE_DAILY_LOSS)

    manager = DailyRiskManager(backend, limit)
    allowed = manager.evaluate(limits.day).allowed
    if allowed and not need_metrics:
        return _allowed(RULE_DAILY_LOSS)

    daily_metrics = manager.metrics(limits.day)
    if allowed:
        return _allowed(RULE_DAILY_LOSS, daily_metrics)

    if daily_metrics.day_unmeasurable:
        detail = (
            "the paper account could not report open positions; the day's "
            "floating loss cannot be measured"
        )
    elif daily_metrics.unpriced_symbols:
        detail = (
            "no usable market mark for open position(s) "
            + ", ".join(daily_metrics.unpriced_symbols)
            + "; the day's floating loss cannot be measured"
        )
    else:
        detail = (
            f"day loss ${daily_metrics.day_loss:,.2f} "
            f"(realized ${daily_metrics.realized_pnl_today:,.2f}, "
            f"floating ${daily_metrics.open_unrealized_pnl:,.2f}) "
            f"on {daily_metrics.day} reached the daily loss limit "
            f"${limit:,.2f}"
        )

    return _reject(RULE_DAILY_LOSS, detail, daily_metrics)


def _gate_max_open_positions(
    backend, trade: "Trade", limits: _GateLimits, *, need_metrics: bool
) -> GateSnapshot:
    """Gate 4: portfolio-wide open count. Rejects at ``current >= limit``."""
    limit = limits.max_open_positions
    if limit <= 0:
        return _disabled(RULE_MAX_OPEN_POSITIONS)

    manager = MaxOpenPositionsManager(backend, limit)
    result = manager.evaluate()
    if result.allowed and not need_metrics:
        return _allowed(RULE_MAX_OPEN_POSITIONS)
    if result.allowed:
        return _allowed(RULE_MAX_OPEN_POSITIONS, manager.metrics())

    return _reject(
        RULE_MAX_OPEN_POSITIONS,
        (
            f"{result.current_open_positions} open positions reached "
            f"the maximum of {result.limit}"
        ),
        result,
    )


def _gate_symbol_concentration(
    backend, trade: "Trade", limits: _GateLimits, *, need_metrics: bool
) -> GateSnapshot:
    """Gate 5: symbol concentration. Count cap first, then exposure cap."""
    count_cap = limits.max_positions_per_symbol
    exposure_cap = limits.max_exposure_per_symbol
    if count_cap <= 0 and exposure_cap <= 0:
        return _disabled(RULE_SYMBOL_POSITIONS)

    manager = CorrelationRiskManager(backend, count_cap, exposure_cap)
    check = manager.evaluate(trade)
    if check.allowed and not need_metrics:
        return _allowed(RULE_SYMBOL_POSITIONS)

    concentration = manager.metrics(trade.symbol)
    if check.allowed:
        return _allowed(RULE_SYMBOL_POSITIONS, concentration)

    if check.rule == "positions":
        return _reject(
            RULE_SYMBOL_POSITIONS,
            (
                f"{concentration.open_positions} open positions on "
                f"{concentration.symbol} reached the maximum of "
                f"{concentration.max_positions}"
            ),
            concentration,
        )

    return _reject(
        RULE_SYMBOL_EXPOSURE,
        (
            f"current exposure on {concentration.symbol} "
            f"${concentration.symbol_exposure:,.2f} plus requested "
            f"exposure ${trade_exposure(trade):,.2f} exceeds the "
            f"per-symbol exposure limit "
            f"${concentration.exposure_limit:,.2f}"
        ),
        concentration,
    )


#: The entry-gate precedence, highest first. Both compositions iterate
#: this single sequence, so the gate order has exactly one home.
GATE_ORDER: tuple[Callable[..., GateSnapshot], ...] = (
    _gate_portfolio_risk,
    _gate_portfolio_exposure,
    _gate_daily_loss,
    _gate_max_open_positions,
    _gate_symbol_concentration,
)


def evaluate_entry_gates(
    backend,
    trade: "Trade",
    *,
    portfolio_risk_limit: float,
    portfolio_exposure_limit: float,
    daily_loss_limit: float,
    max_open_positions: int,
    max_positions_per_symbol: int,
    max_exposure_per_symbol: float,
    day: str | None = None,
) -> EntryGateOutcome:
    """Evaluate every paper-trading entry gate for one candidate trade.

    Gates run in their established order and stop at the first
    rejection:

        portfolio risk -> portfolio exposure -> daily loss ->
        maximum open positions -> symbol concentration (count, then
        exposure)

    Every limit is explicit; a limit ``<= 0`` disables that gate. A
    disabled gate is skipped entirely, so it performs no account reads.

    Only the rejecting gate builds a metrics snapshot. A passing gate
    performs the reads its own decision needs and nothing more, so this
    path keeps the account-read profile of the original inline checks.

    Args:
        backend: A PaperBroker-like execution backend providing
            ``get_open_positions()`` and an ``account`` with
            ``closed_positions``.
        trade: The candidate trade to evaluate. It is never modified.
        portfolio_risk_limit: Maximum prospective portfolio risk.
        portfolio_exposure_limit: Maximum prospective notional exposure.
        daily_loss_limit: Maximum combined settled plus floating loss
            for the day.
        max_open_positions: Maximum portfolio-wide open positions.
        max_positions_per_symbol: Maximum open positions on one symbol.
        max_exposure_per_symbol: Maximum notional exposure on one symbol.
        day: ``YYYY-MM-DD`` day used by the daily loss brake. Defaults to
            the current local calendar day.

    Returns:
        An ``EntryGateOutcome`` describing the first rejection, or an
        allowing outcome when every enabled gate passes.
    """
    if trade is None:
        return EntryGateOutcome(
            allowed=False,
            rule=RULE_INVALID_TRADE,
            detail="no candidate trade was supplied for entry evaluation",
        )

    limits = _GateLimits(
        portfolio_risk_limit=portfolio_risk_limit,
        portfolio_exposure_limit=portfolio_exposure_limit,
        daily_loss_limit=daily_loss_limit,
        max_open_positions=max_open_positions,
        max_positions_per_symbol=max_positions_per_symbol,
        max_exposure_per_symbol=max_exposure_per_symbol,
        day=day,
    )

    for gate in GATE_ORDER:
        snapshot = gate(backend, trade, limits, need_metrics=False)
        if snapshot.enabled and not snapshot.allowed:
            return EntryGateOutcome(
                allowed=False,
                rule=snapshot.rule,
                detail=snapshot.detail,
                metrics=snapshot.metrics,
            )

    return EntryGateOutcome(allowed=True)


@dataclass(frozen=True, slots=True)
class EntryGateMetrics:
    """Frozen snapshots of every enabled gate, for the dashboard view.

    Each field is None when that gate's limit is ``<= 0``, mirroring the
    "disabled gate is skipped" rule, so the dashboard renders a section
    for exactly the enabled gates.

    Attributes:
        portfolio_risk: Current risk against the portfolio risk limit.
        portfolio_exposure: Current notional exposure against its limit.
        daily_loss: Today's settled plus floating PnL against the daily
            loss limit.
        max_open_positions: Open-position capacity usage.
        symbol_concentration: One snapshot per symbol with open
            positions, ordered by symbol exposure descending. This is a
            portfolio-wide report, not a per-candidate-trade view.
    """

    portfolio_risk: PortfolioRiskMetrics | None = None
    portfolio_exposure: ExposureMetrics | None = None
    daily_loss: DailyRiskMetrics | None = None
    max_open_positions: MaxOpenPositionsMetrics | None = None
    symbol_concentration: tuple[CorrelationMetrics, ...] | None = None


def collect_entry_gate_metrics(
    backend,
    *,
    portfolio_risk_limit: float,
    portfolio_exposure_limit: float,
    daily_loss_limit: float,
    max_open_positions: int,
    max_positions_per_symbol: int,
    max_exposure_per_symbol: float,
    day: str | None = None,
) -> EntryGateMetrics:
    """Collect the metrics of every enabled gate without short-circuiting.

    This is the dashboard companion to :func:`evaluate_entry_gates`. It
    evaluates no candidate trade: it reports what each gate currently
    sees, so a rendering layer can show every enabled gate even when
    another gate would reject a trade.

    Every figure comes from the same domain managers and helpers the
    entry gates use; nothing is recomputed here. Two gates expose a
    display projection rather than their rejection payload — the open
    position count uses ``MaxOpenPositionsMetrics`` (which adds usage
    and status) and symbol concentration uses
    ``concentration_report()`` (which covers every open symbol, not just
    one candidate trade's symbol).

    Args:
        backend: A PaperBroker-like execution backend providing
            ``get_open_positions()`` and an ``account`` with
            ``closed_positions``.
        portfolio_risk_limit: Maximum prospective portfolio risk.
        portfolio_exposure_limit: Maximum prospective notional exposure.
        daily_loss_limit: Maximum combined settled plus floating loss
            for the day.
        max_open_positions: Maximum portfolio-wide open positions.
        max_positions_per_symbol: Maximum open positions on one symbol.
        max_exposure_per_symbol: Maximum notional exposure on one symbol.
        day: ``YYYY-MM-DD`` day used by the daily loss snapshot.
            Defaults to the current local calendar day.

    Returns:
        An ``EntryGateMetrics`` with None for every disabled gate.
    """
    risk_metrics = (
        PortfolioRiskManager(backend, portfolio_risk_limit).metrics()
        if portfolio_risk_limit > 0
        else None
    )
    exposure = (
        exposure_metrics(backend, portfolio_exposure_limit)
        if portfolio_exposure_limit > 0
        else None
    )
    daily_metrics = (
        DailyRiskManager(backend, daily_loss_limit).metrics(day)
        if daily_loss_limit > 0
        else None
    )
    max_open_metrics = (
        MaxOpenPositionsManager(backend, max_open_positions).metrics()
        if max_open_positions > 0
        else None
    )

    concentration: tuple[CorrelationMetrics, ...] | None = None
    if max_positions_per_symbol > 0 or max_exposure_per_symbol > 0:
        concentration = tuple(
            CorrelationRiskManager(
                backend,
                max_positions_per_symbol,
                max_exposure_per_symbol,
            ).concentration_report()
        )

    return EntryGateMetrics(
        portfolio_risk=risk_metrics,
        portfolio_exposure=exposure,
        daily_loss=daily_metrics,
        max_open_positions=max_open_metrics,
        symbol_concentration=concentration,
    )
