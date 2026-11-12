"""
MuzanTradeBot - Risk Profile

A single immutable bundle of the risk parameters that drive position
sizing and exit levels. In Guardian Mode the profile is the source of
truth for how much capital a trade may expose, so every layer of the
pipeline reads identical numbers instead of scattered hard-coded
defaults.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    """Risk parameters for the account and for each trade decision."""

    balance: float = 1000.0
    risk_percent: float = 2.0
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 4.0

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
