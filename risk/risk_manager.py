"""
MuzanTradeBot - Risk Manager

Risk validation for trade decisions. In Guardian Mode, no trade is
executed until it passes risk checks; this module is the gateway that
confirms a decision is backed by sound market data and a sane risk
profile before it moves forward.

The decision engine owns *signal quality* (its confidence threshold).
This module owns *data integrity and risk budgeting*: it rejects
decisions backed by unusable prices, malformed confidence values,
conflicting trends, or a misconfigured risk profile, and it explains
every rejection so no silent veto ever happens.
"""

from dataclasses import dataclass, field

from risk.risk_profile import RiskProfile

VALID_TRENDS = ("Bullish", "Bearish")
DIRECTION_TREND = {"BUY": "Bullish", "SELL": "Bearish"}


@dataclass
class RiskVerdict:
    """Outcome of risk validation for one proposed decision."""

    decision: str  # Final decision: "BUY", "SELL", or "WAIT".
    approved: bool  # Whether the trade passed all risk checks.
    reasons: list[str] = field(default_factory=list)  # Human-readable explanations.


def validate_trade(
    decision: str,
    analysis: dict,
    profile: RiskProfile | None = None,
) -> RiskVerdict:
    """Validate a trade decision against Guardian Mode risk checks.

    Args:
        decision: The proposed decision, e.g. "BUY", "SELL", or "WAIT"
            from brain.decision_engine.make_decision.
        analysis: The market analysis dict (from market.analyzer.
            analyze_coin) with symbol, price, trend, signal, and
            confidence.
        profile: The RiskProfile to validate against. Defaults to a
            standard RiskProfile() when not provided.

    Returns:
        A RiskVerdict. Only a BUY or SELL backed by a positive numeric
        price, a confidence value in [0, 100], a direction-consistent
        trend, and a sane risk profile is approved. Anything malformed,
        missing, or contradictory resolves to a WAIT verdict with
        reasons explaining why.
    """
    if profile is None:
        profile = RiskProfile()

    if not isinstance(analysis, dict):
        return RiskVerdict(
            decision="WAIT",
            approved=False,
            reasons=["No market analysis was provided."],
        )

    # Only BUY/SELL decisions need risk validation; WAIT is already safe.
    if decision not in ("BUY", "SELL"):
        return RiskVerdict(
            decision="WAIT",
            approved=False,
            reasons=["No actionable trade signal (decision engine returned WAIT)."],
        )

    reasons: list[str] = []

    # --- Market data integrity checks -------------------------------
    confidence = analysis.get("confidence")
    if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 100):
        reasons.append(
            f"Confidence is invalid ({confidence!r}); expected a number between 0 and 100."
        )

    price = analysis.get("price")
    if not isinstance(price, (int, float)) or price <= 0:
        reasons.append("No valid market price available; cannot size or protect a position.")

    trend = analysis.get("trend")
    if trend not in VALID_TRENDS:
        reasons.append("Market conditions are unknown; cannot confirm the signal.")
    elif trend != DIRECTION_TREND.get(decision):
        reasons.append(f"{decision} signal conflicts with the current {trend} trend.")

    # --- Risk profile checks ----------------------------------------
    if profile.balance <= 0:
        reasons.append("Account balance must be positive.")
    if profile.risk_percent <= 0:
        reasons.append("Risk percent must be positive.")
    if profile.stop_loss_percent <= 0:
        reasons.append("Stop-loss percent must be positive; every position needs a stop.")
    if profile.take_profit_percent <= 0:
        reasons.append("Take-profit percent must be positive.")

    if reasons:
        return RiskVerdict(decision="WAIT", approved=False, reasons=reasons)

    return RiskVerdict(
        decision=decision,
        approved=True,
        reasons=["Risk checks passed."],
    )
