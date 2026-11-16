"""Explicit trading-cost assumptions for historical backtests.

Fees and slippage are caller-supplied assumptions. This module never
fetches exchange fee schedules, estimates live liquidity, or claims that
modeled slippage matches actual fills. The default model is exactly zero
so existing backtests remain unchanged unless costs are opted into.

Rates are fractions, not percentages:

- ``fee_rate=0.001`` means 0.1% of notional on each entry and exit.
- ``slippage_rate=0.001`` means a 0.1% adverse price move on each fill.

Slippage is always adverse: LONG entries fill higher and exits lower;
SHORT entries fill lower and exits higher. Fees are charged on the
slipped fill notional for both sides.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TradingCostModel:
    """User-supplied fee and slippage assumptions for one backtest.

    Attributes:
        fee_rate: Fraction of notional charged on entry and again on
            exit. Must be finite and in the range ``[0, 1)``.
        slippage_rate: Adverse fraction applied to entry and exit fills.
            Must be finite and in the range ``[0, 1)``.

    The model is intentionally explicit: ``TradingCostModel()`` is the
    zero-cost default and does not imply any real exchange fee rate.
    """

    fee_rate: float = 0.0
    slippage_rate: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("fee_rate", "slippage_rate"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a number")
            try:
                rate = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise TypeError(f"{field_name} must be a number") from exc
            if not math.isfinite(rate) or rate < 0.0 or rate >= 1.0:
                raise ValueError(
                    f"{field_name} must be finite and in the range [0, 1)"
                )
            object.__setattr__(self, field_name, rate)


ZERO_COSTS = TradingCostModel()


@dataclass(frozen=True, slots=True)
class TradeCostBreakdown:
    """Gross and net PnL for one completed trade with explicit costs."""

    cost_model: TradingCostModel
    side: str
    gross_pnl: float
    gross_pnl_after_slippage: float
    entry_price: float
    exit_price: float
    entry_fill_price: float
    exit_fill_price: float
    entry_fee: float
    exit_fee: float
    total_fees: float
    slippage_cost: float
    total_costs: float
    net_pnl: float


def _normalize_side(side) -> str:
    if not isinstance(side, str):
        raise ValueError("side must be 'LONG' or 'SHORT'")
    normalized = side.strip().upper()
    if normalized not in ("LONG", "SHORT"):
        raise ValueError("side must be 'LONG' or 'SHORT'")
    return normalized


def _is_finite_positive(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number > 0


def _finite(value, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


def calculate_trade_costs(
    side: str,
    entry_price: float,
    exit_price: float,
    quantity: float,
    cost_model: TradingCostModel | None = None,
) -> TradeCostBreakdown:
    """Calculate gross PnL and net PnL after modeled fees and slippage.

    Args:
        side: ``"LONG"`` or ``"SHORT"``.
        entry_price: Unadjusted historical entry price.
        exit_price: Unadjusted historical exit price.
        quantity: Position quantity in base units.
        cost_model: Explicit assumptions. None means exact zero costs.

    Returns:
        An immutable breakdown that preserves the original quote prices,
        the adverse slipped fill prices, fees on both sides, and gross
        and net PnL.

    Raises:
        TypeError: If ``cost_model`` has the wrong type.
        ValueError: If the side, prices, quantity, or any intermediate
            result is invalid or non-finite.
    """
    if cost_model is None:
        cost_model = ZERO_COSTS
    if not isinstance(cost_model, TradingCostModel):
        raise TypeError("cost_model must be a TradingCostModel")

    side = _normalize_side(side)
    for name, value in (
        ("entry_price", entry_price),
        ("exit_price", exit_price),
        ("quantity", quantity),
    ):
        if not _is_finite_positive(value):
            raise ValueError(f"{name} must be a finite positive number")

    entry = float(entry_price)
    exit_value = float(exit_price)
    qty = float(quantity)
    slippage = cost_model.slippage_rate
    fee_rate = cost_model.fee_rate

    try:
        if side == "LONG":
            entry_fill = entry * (1.0 + slippage)
            exit_fill = exit_value * (1.0 - slippage)
            gross_pnl = (exit_value - entry) * qty
            gross_after_slippage = (exit_fill - entry_fill) * qty
        else:
            entry_fill = entry * (1.0 - slippage)
            exit_fill = exit_value * (1.0 + slippage)
            gross_pnl = (entry - exit_value) * qty
            gross_after_slippage = (entry_fill - exit_fill) * qty

        if fee_rate == 0.0:
            # Avoid overflowing a huge notional when fees are disabled.
            entry_fee = 0.0
            exit_fee = 0.0
            total_fees = 0.0
        else:
            entry_fee = entry_fill * qty * fee_rate
            exit_fee = exit_fill * qty * fee_rate
            total_fees = entry_fee + exit_fee
        slippage_cost = gross_pnl - gross_after_slippage
        net_pnl = gross_after_slippage - total_fees
        total_costs = slippage_cost + total_fees
    except OverflowError as exc:
        raise ValueError("trade cost calculation overflowed") from exc

    for name, value in (
        ("entry_fill_price", entry_fill),
        ("exit_fill_price", exit_fill),
        ("gross_pnl", gross_pnl),
        ("gross_pnl_after_slippage", gross_after_slippage),
        ("entry_fee", entry_fee),
        ("exit_fee", exit_fee),
        ("total_fees", total_fees),
        ("slippage_cost", slippage_cost),
        ("net_pnl", net_pnl),
        ("total_costs", total_costs),
    ):
        _finite(value, name)

    if entry_fill <= 0.0 or exit_fill <= 0.0:
        raise ValueError("slipped fill prices must be positive")

    return TradeCostBreakdown(
        cost_model=cost_model,
        side=side,
        gross_pnl=gross_pnl,
        gross_pnl_after_slippage=gross_after_slippage,
        entry_price=entry,
        exit_price=exit_value,
        entry_fill_price=entry_fill,
        exit_fill_price=exit_fill,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        total_fees=total_fees,
        slippage_cost=slippage_cost,
        total_costs=total_costs,
        net_pnl=net_pnl,
    )
