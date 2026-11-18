"""Immutable, validated trading policy for one live decision.

A decision needs exactly two pieces of configuration: the strategy that
produces the signal and the risk profile that sizes it. This module binds
them into one object so a single decision can never mix a strategy from
one source with risk numbers from another, and so the analyzer, decision
engine, risk manager, and trade builder all receive the same instances.

This is deliberately not an optimizer: it does not search, rank, or
select parameters. It only carries the configuration the caller chose,
defaulting to the existing strategy and settings-derived risk values.

Numeric validity is enforced by the value objects themselves:

* ``StrategyParameters`` validates its own fields on construction.
* ``RiskProfile`` validates its own fields on construction, and its
  shared validator is re-applied here so a policy can never carry a
  profile that skipped that validation.
"""

from dataclasses import dataclass

from risk.risk_profile import RiskProfile, validate_risk_configuration
from strategy.parameters import (
    DEFAULT_STRATEGY_PARAMETERS,
    StrategyParameters,
)


@dataclass(frozen=True, slots=True)
class TradingPolicy:
    """One validated strategy + risk configuration pair.

    Attributes:
        strategy: Immutable strategy parameters used to produce the
            signal and the final decision.
        risk_profile: Immutable risk profile used to size the trade and
            set its stop-loss and take-profit levels.

    Raises:
        TypeError: If either field has the wrong type.
        ValueError: If the carried risk numbers are non-finite,
            non-positive, or a risk percentage or stop-loss distance of
            100% or more.
    """

    strategy: StrategyParameters
    risk_profile: RiskProfile

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, StrategyParameters):
            raise TypeError("strategy must be a StrategyParameters")
        if not isinstance(self.risk_profile, RiskProfile):
            raise TypeError("risk_profile must be a RiskProfile")

        # Re-apply the single risk rule set. RiskProfile already validated
        # these values at construction; this closes the gap for instances
        # that skipped construction (subclasses or restored objects) so a
        # policy is always safe to hand to sizing and exit calculations.
        validate_risk_configuration(
            balance=self.risk_profile.balance,
            risk_percent=self.risk_profile.risk_percent,
            stop_loss_percent=self.risk_profile.stop_loss_percent,
            take_profit_percent=self.risk_profile.take_profit_percent,
        )

    @classmethod
    def from_settings(
        cls,
        settings,
        *,
        balance: float | None = None,
        strategy: StrategyParameters | None = None,
    ) -> "TradingPolicy":
        """Resolve one policy from the central Settings object.

        This is the only settings-based construction path. Existing
        defaults are preserved exactly: omitting ``strategy`` uses
        :data:`DEFAULT_STRATEGY_PARAMETERS`, and omitting ``balance``
        uses the configured ``ACCOUNT_BALANCE`` for sizing.

        Args:
            settings: The central Settings object.
            balance: Optional live account balance/equity to size from.
                ``None`` uses ``settings.ACCOUNT_BALANCE``.
            strategy: Optional explicit strategy parameters. ``None``
                uses the existing default strategy.

        Returns:
            A validated, immutable TradingPolicy.

        Raises:
            TypeError: If ``strategy`` is not a StrategyParameters or
                None, or if a resolved risk value is not a number.
            ValueError: If a resolved risk value is non-finite,
                non-positive, or a risk percentage or stop-loss distance
                of 100% or more.
        """
        if strategy is None:
            parameters = DEFAULT_STRATEGY_PARAMETERS
        elif isinstance(strategy, StrategyParameters):
            parameters = strategy
        else:
            raise TypeError(
                "strategy must be a StrategyParameters or None"
            )

        return cls(
            strategy=parameters,
            risk_profile=RiskProfile.from_settings(
                settings,
                balance=balance,
            ),
        )


__all__ = ["TradingPolicy"]
