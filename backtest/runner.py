"""Pure orchestration for one historical backtest run.

The runner connects the existing chronological replay, trade simulation,
cost assumptions, and reporting layers. It performs no I/O, network
access, Telegram calls, paper-account changes, or live execution.

The risk profile is the single source of truth for starting equity and
position sizing. Insufficient history is not an error: it produces an
empty report with the existing explicit warnings and marks the run as
having insufficient history.

Example:
    from backtest.costs import TradingCostModel
    from backtest.report import format_backtest_report
    from backtest.runner import BacktestRunConfig, run_backtest
    from risk.risk_profile import RiskProfile

    config = BacktestRunConfig(
        symbol="BTC",
        risk_profile=RiskProfile(
            balance=1000.0,
            risk_percent=2.0,
            stop_loss_percent=2.0,
            take_profit_percent=4.0,
        ),
        cost_model=TradingCostModel(fee_rate=0.001, slippage_rate=0.001),
    )
    result = run_backtest(bars, config)
    print(format_backtest_report(result.report, max_trades=20))
"""

from dataclasses import dataclass
from datetime import datetime
import math

from backtest.costs import TradingCostModel, ZERO_COSTS
from backtest.report import BacktestReport, build_backtest_report
from backtest.trades import HistoricalTrade, simulate_trades
from market.indicators import MIN_CANDLES
from risk.risk_profile import RiskProfile


@dataclass(frozen=True, slots=True)
class BacktestRunConfig:
    """Immutable inputs for one historical backtest.

    Attributes:
        symbol: Symbol associated with the supplied bar series. Values
            are stripped and uppercased.
        risk_profile: Account balance and risk parameters. Its positive
            ``balance`` is also the report's starting equity, so sizing
            and reported results cannot use different balances.
        cost_model: Explicit fee and slippage assumptions. The default is
            the existing exact zero-cost model.
    """

    symbol: str
    risk_profile: RiskProfile
    cost_model: TradingCostModel = ZERO_COSTS

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.risk_profile, RiskProfile):
            raise TypeError("risk_profile must be a RiskProfile")
        if not isinstance(self.cost_model, TradingCostModel):
            raise TypeError("cost_model must be a TradingCostModel")

        for field_name in (
            "balance",
            "risk_percent",
            "stop_loss_percent",
            "take_profit_percent",
        ):
            value = getattr(self.risk_profile, field_name)
            if not _is_finite_positive(value):
                raise ValueError(
                    f"risk_profile.{field_name} must be a finite positive "
                    "number"
                )

        object.__setattr__(self, "symbol", self.symbol.strip().upper())

    @property
    def initial_balance(self) -> float:
        """Return the starting equity used for sizing and reporting."""
        return float(self.risk_profile.balance)


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    """Immutable output and metadata from one historical backtest run.

    Attributes:
        report: Summary metrics, warnings, and individual trade rows.
        trades: Original HistoricalTrade records in chronological order.
            Open trades have no exit information and are excluded from
            realized metrics by the report.
        config: Exact normalized inputs used for the run.
        bar_count: Number of historical bars supplied.
        first_bar_timestamp: First bar timestamp, or None when no bars
            were supplied.
        last_bar_timestamp: Last bar timestamp, or None when no bars
            were supplied.
        minimum_history_bars: Bars required before the first signal can
            be evaluated. One additional bar is required for that signal
            to enter on the next close.
    """

    report: BacktestReport
    trades: tuple[HistoricalTrade, ...]
    config: BacktestRunConfig
    bar_count: int
    first_bar_timestamp: datetime | None
    last_bar_timestamp: datetime | None
    minimum_history_bars: int

    @property
    def history_sufficient(self) -> bool:
        """Return whether enough bars exist to evaluate any signal."""
        return self.bar_count >= self.minimum_history_bars

    @property
    def completed_trades(self) -> int:
        """Return the number of trades with realized exit information."""
        return self.report.total_trades - self.report.open_trades

    @property
    def open_trades(self) -> int:
        """Return the number of trades still open at the end of the data."""
        return self.report.open_trades


def _is_finite_positive(value) -> bool:
    """Return True only for finite, strictly positive int/float values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and number > 0.0


def run_backtest(
    bars: list | tuple,
    config: BacktestRunConfig,
) -> BacktestRunResult:
    """Run one pure historical backtest over supplied chronological bars.

    Signal replay and exit behavior are delegated to
    :func:`backtest.trades.simulate_trades`, using the supplied risk
    profile and cost model. The resulting trades are passed to
    :func:`backtest.report.build_backtest_report` with the same cost
    assumptions and the risk profile's balance.

    Args:
        bars: Ordered HistoricalBar values, oldest first. At least
            ``MIN_CANDLES`` bars are needed for a signal and one more for
            that signal to enter. Shorter inputs return an empty-trade
            report instead of raising.
        config: Validated run inputs.

    Returns:
        An immutable result containing the report, original trades, exact
        normalized configuration, bar span, and history metadata.

    Raises:
        TypeError: If ``bars`` or ``config`` have invalid types, or a bar
            value is not a HistoricalBar.
        ValueError: If a bar price, volume, or timestamp ordering is
            invalid.
    """
    if not isinstance(config, BacktestRunConfig):
        raise TypeError("config must be a BacktestRunConfig")
    if not isinstance(bars, (list, tuple)):
        raise TypeError("bars must be a list or tuple of HistoricalBar values")

    # Snapshot the sequence so later caller mutation cannot change the run
    # metadata after validation has completed.
    bar_snapshot = tuple(bars)

    trades = simulate_trades(
        config.symbol,
        bar_snapshot,
        config.risk_profile,
        cost_model=config.cost_model,
    )
    report = build_backtest_report(
        trades,
        config.initial_balance,
        config.cost_model,
    )

    return BacktestRunResult(
        report=report,
        trades=tuple(trades),
        config=config,
        bar_count=len(bar_snapshot),
        first_bar_timestamp=(
            bar_snapshot[0].timestamp if bar_snapshot else None
        ),
        last_bar_timestamp=(
            bar_snapshot[-1].timestamp if bar_snapshot else None
        ),
        minimum_history_bars=MIN_CANDLES,
    )
