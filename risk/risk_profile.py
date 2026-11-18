"""
MuzanTradeBot - Risk Profile

A single immutable bundle of the risk parameters that drive position
sizing and exit levels. In Guardian Mode the profile is the source of
truth for how much capital a trade may expose, so every layer of the
pipeline reads identical numbers instead of scattered hard-coded
defaults.

Every field is validated on construction: non-numeric, non-finite, and
non-positive values are rejected before they can reach sizing or exit
arithmetic, and a stop-loss distance of 100% or more is rejected because
it would put the stop at or below zero. The same rules are exported so
callers that wrap a profile (such as a trading policy) can re-check the
contract without duplicating the ranges.
"""

from dataclasses import dataclass
import math


def _validate_risk_number(
    name: str,
    value,
    *,
    upper_exclusive: float | None = None,
) -> None:
    """Validate one risk number in place.

    Args:
        name: Field name used in error messages.
        value: Candidate value. Booleans are rejected even though they
            subclass ``int``.
        upper_exclusive: Optional exclusive upper bound.

    Raises:
        TypeError: If the value is not a real number.
        ValueError: If the value is non-finite, not positive, or not
            strictly below ``upper_exclusive``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError):
        # An integer too large for a float is not a usable risk value.
        raise ValueError(f"{name} must be finite") from None
    if not finite:
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if upper_exclusive is not None and value >= upper_exclusive:
        raise ValueError(
            f"{name} must be less than {upper_exclusive:g}"
        )


def validate_risk_configuration(
    *,
    balance,
    risk_percent,
    stop_loss_percent,
    take_profit_percent,
) -> None:
    """Validate one complete set of risk numbers.

    This is the single source of truth for risk-domain invariants. It is
    used by :class:`RiskProfile` and re-used by wrappers such as
    :class:`strategy.trading_policy.TradingPolicy` so no second, subtly
    different rule set can appear.

    Args:
        balance: Account balance used for sizing.
        risk_percent: Percentage of the balance risked per trade. Must be
            strictly below 100; a trade can never risk the whole account
            (or more than it) on one decision.
        stop_loss_percent: Stop distance from entry, in percent. Must be
            strictly below 100.
        take_profit_percent: Target distance from entry, in percent.

    Raises:
        TypeError: If a value is not a real number, or is a bool.
        ValueError: If a value is non-finite, not positive, or a risk
            percentage or stop-loss distance is 100% or more.
    """
    _validate_risk_number("balance", balance)
    _validate_risk_number(
        "risk_percent",
        risk_percent,
        upper_exclusive=100.0,
    )
    _validate_risk_number(
        "stop_loss_percent",
        stop_loss_percent,
        upper_exclusive=100.0,
    )
    _validate_risk_number("take_profit_percent", take_profit_percent)


@dataclass(frozen=True)
class RiskProfile:
    """Risk parameters for the account and for each trade decision.

    Raises:
        TypeError: If a field is not a real number (bool is rejected).
        ValueError: If a field is non-finite or non-positive, or if
            ``risk_percent`` or ``stop_loss_percent`` is 100 or more.
    """

    balance: float = 1000.0
    risk_percent: float = 2.0
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 4.0

    def __post_init__(self) -> None:
        validate_risk_configuration(
            balance=self.balance,
            risk_percent=self.risk_percent,
            stop_loss_percent=self.stop_loss_percent,
            take_profit_percent=self.take_profit_percent,
        )

    @classmethod
    def from_settings(
        cls, settings, balance: float | None = None
    ) -> "RiskProfile":
        """Build a profile from the central Settings object.

        Args:
            settings: The central Settings object.
            balance: Optional balance override used for position sizing.
                Callers pass the live account balance/equity here so the
                risk budget follows the account instead of a static
                configuration value. When None, the configured
                ``ACCOUNT_BALANCE`` is used.

        Raises:
            TypeError: If a resolved value is not a real number.
            ValueError: If a resolved value is non-finite, non-positive,
                or the risk percentage or stop-loss distance is 100% or
                more.
        """
        return cls(
            balance=settings.ACCOUNT_BALANCE if balance is None else balance,
            risk_percent=settings.RISK_PERCENT,
            stop_loss_percent=settings.STOP_LOSS_PERCENT,
            take_profit_percent=settings.TAKE_PROFIT_PERCENT,
        )

    @property
    def risk_amount(self) -> float:
        """Absolute capital at risk per trade: balance * risk_percent."""
        return self.balance * (self.risk_percent / 100)
