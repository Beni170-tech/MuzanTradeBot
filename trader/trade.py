"""
MuzanTradeBot - Trade Model

A single consolidated object describing one analyzed trade: the
market analysis, the final decision, the risk verdict, the position
size, and the exit levels. In Guardian Mode, the Trade is the single
contract passed from analysis through risk to execution, so every
stage sees identical data.
"""

from dataclasses import dataclass, field
from datetime import datetime

from risk.position_sizer import calculate_position_size
from risk.risk_manager import RiskVerdict
from risk.risk_profile import RiskProfile
from risk.stop_loss import calculate_exit_levels

SIDE_BY_DECISION = {"BUY": "LONG", "SELL": "SHORT"}


@dataclass
class Trade:
    """One analyzed trade, fully described for risk and execution."""

    symbol: str
    price: float | str
    trend: str
    signal: str
    confidence: int | float
    decision: str  # Final decision after risk validation: BUY/SELL/WAIT.
    risk_verdict: str  # "APPROVED" or "BLOCKED".
    position_size: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    side: str = "LONG"
    approved: bool = False
    risk_reasons: list[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


def build_trade(
    analysis: dict,
    verdict: RiskVerdict,
    profile: RiskProfile | None = None,
) -> Trade:
    """Assemble a fully populated Trade from the pipeline output.

    Args:
        analysis: The dict from market.analyzer.analyze_coin.
        verdict: The RiskVerdict from risk.risk_manager.validate_trade.
            Only approved verdicts are sized; blocked ones keep zeros
            for position_size, stop_loss, and take_profit.
        profile: The RiskProfile used for sizing and exit levels.
            Defaults to a standard RiskProfile() when not provided.

    Returns:
        A Trade with all fields populated. The decision stored is the
        *final* one from the risk verdict, and the verdict's reasons
        are carried along so the caller can explain any rejection.
        If the analysis price is not numeric (e.g. "Unknown") or the
        verdict is not approved, position_size, stop_loss, and
        take_profit are 0, preserving the current placeholder behavior
        for unsupported coins.
    """
    if profile is None:
        profile = RiskProfile()

    approved = verdict.approved
    decision = verdict.decision if approved else "WAIT"
    risk_verdict = "APPROVED" if approved else "BLOCKED"
    side = SIDE_BY_DECISION.get(decision, "LONG")

    position_size = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    if approved:
        price = analysis["price"] if isinstance(analysis["price"], (int, float)) else 0
        if price > 0:
            position_size = calculate_position_size(
                balance=profile.balance,
                risk_percent=profile.risk_percent,
                entry_price=price,
                stop_loss_percent=profile.stop_loss_percent,
            )
            exit_levels = calculate_exit_levels(
                entry_price=price,
                stop_loss_percent=profile.stop_loss_percent,
                take_profit_percent=profile.take_profit_percent,
                side=side,
            )
            stop_loss = exit_levels["stop_loss"]
            take_profit = exit_levels["take_profit"]

    return Trade(
        symbol=analysis["symbol"],
        price=analysis["price"],
        trend=analysis["trend"],
        signal=analysis["signal"],
        confidence=analysis["confidence"],
        decision=decision,
        risk_verdict=risk_verdict,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit,
        side=side,
        approved=approved,
        risk_reasons=list(verdict.reasons),
    )
