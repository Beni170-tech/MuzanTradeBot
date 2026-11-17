"""Pure strategy validation across historical market conditions.

This module splits supplied chronological bars into contiguous,
non-overlapping full windows, classifies each window deterministically,
and runs :func:`backtest.runner.run_backtest` independently per window.

Each window starts with the same BacktestRunConfig and risk profile, so
window results are independent backtests. Condition summaries deliberately
do not present a combined ending balance or continuous equity curve.

No network access, Telegram calls, paper-account changes, live execution,
or filesystem I/O occurs here.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import math

from backtest.replay import HistoricalBar
from backtest.runner import BacktestRunConfig, BacktestRunResult, run_backtest
from market.indicators import MIN_CANDLES


MIN_WINDOW_BARS = MIN_CANDLES + 2
DEFAULT_WINDOW_BARS = MIN_CANDLES * 3
DEFAULT_TREND_THRESHOLD_PERCENT = 2.0
DEFAULT_MIN_DIRECTIONAL_EFFICIENCY = 0.25


class MarketCondition(str, Enum):
    """Deterministic classification of one validation window."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


@dataclass(frozen=True, slots=True)
class StrategyValidationConfig:
    """Controls window size and deterministic market classification.

    Attributes:
        window_bars: Number of bars in each non-overlapping window. Must
            allow at least one completed trade: ``MIN_CANDLES + 2``.
        trend_threshold_percent: Minimum absolute close-to-close window
            return required for a directional classification.
        min_directional_efficiency: Minimum ratio of net movement to total
            close-to-close path length required for a directional
            classification. Must be in ``[0, 1]``.
    """

    window_bars: int = DEFAULT_WINDOW_BARS
    trend_threshold_percent: float = DEFAULT_TREND_THRESHOLD_PERCENT
    min_directional_efficiency: float = (
        DEFAULT_MIN_DIRECTIONAL_EFFICIENCY
    )

    def __post_init__(self) -> None:
        if isinstance(self.window_bars, bool) or not isinstance(
            self.window_bars, int
        ):
            raise TypeError("window_bars must be an integer")
        if self.window_bars < MIN_WINDOW_BARS:
            raise ValueError(
                f"window_bars must be at least {MIN_WINDOW_BARS}"
            )

        threshold, efficiency = _validate_classifier_parameters(
            self.trend_threshold_percent,
            self.min_directional_efficiency,
        )
        object.__setattr__(self, "trend_threshold_percent", threshold)
        object.__setattr__(self, "min_directional_efficiency", efficiency)


@dataclass(frozen=True, slots=True)
class MarketWindowResult:
    """One classified window and its independent backtest result."""

    index: int
    condition: MarketCondition
    first_timestamp: datetime
    last_timestamp: datetime
    bar_count: int
    return_percent: float
    directional_efficiency: float
    backtest: BacktestRunResult


@dataclass(frozen=True, slots=True)
class MarketConditionSummary:
    """Metrics aggregated across independent windows of one condition.

    ``worst_max_drawdown`` and ``worst_max_drawdown_percent`` describe the
    worst individual window. They are not a continuous portfolio drawdown.
    """

    condition: MarketCondition
    window_count: int
    bar_count: int
    total_trades: int
    completed_trades: int
    open_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    total_costs: float
    gross_profit_factor: float | None
    net_profit_factor: float | None
    worst_max_drawdown: float
    worst_max_drawdown_percent: float
    best_window_net_pnl: float
    worst_window_net_pnl: float


@dataclass(frozen=True, slots=True)
class StrategyValidationResult:
    """Immutable validation windows, condition aggregates, and warnings."""

    config: BacktestRunConfig
    validation_config: StrategyValidationConfig
    bar_count: int
    unused_bars: int
    windows: tuple[MarketWindowResult, ...]
    conditions: tuple[MarketConditionSummary, ...]
    warnings: tuple[str, ...]

    @property
    def window_count(self) -> int:
        """Return the number of full windows evaluated."""
        return len(self.windows)


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_bars(bars) -> tuple[HistoricalBar, ...]:
    """Validate bars without sorting, repairing, or dropping any."""
    if not isinstance(bars, (list, tuple)):
        raise TypeError("bars must be a list or tuple of HistoricalBar values")

    previous_timestamp = None
    for index, bar in enumerate(bars):
        if not isinstance(bar, HistoricalBar):
            raise TypeError(
                f"bars[{index}] must be a HistoricalBar, got "
                f"{type(bar).__name__}"
            )
        if not isinstance(bar.timestamp, datetime):
            raise TypeError(
                f"bars[{index}].timestamp must be a datetime, got "
                f"{type(bar.timestamp).__name__}"
            )
        if not _is_finite_number(bar.close) or bar.close <= 0:
            raise ValueError(
                f"bars[{index}].close must be a positive finite number"
            )
        if not _is_finite_number(bar.volume) or bar.volume < 0:
            raise ValueError(
                f"bars[{index}].volume must be a finite non-negative number"
            )

        if previous_timestamp is not None:
            try:
                in_order = bar.timestamp > previous_timestamp
            except TypeError as exc:
                raise ValueError(
                    "bar timestamps must be mutually comparable"
                ) from exc
            if not in_order:
                raise ValueError(
                    "bar timestamps must be strictly increasing in input order"
                )
        previous_timestamp = bar.timestamp

    return tuple(bars)


def _validate_classifier_parameters(
    trend_threshold_percent,
    min_directional_efficiency,
) -> tuple[float, float]:
    if isinstance(trend_threshold_percent, bool) or not isinstance(
        trend_threshold_percent, (int, float)
    ):
        raise TypeError("trend_threshold_percent must be a number")
    try:
        threshold = float(trend_threshold_percent)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("trend_threshold_percent must be a number") from exc
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(
            "trend_threshold_percent must be finite and positive"
        )

    if isinstance(min_directional_efficiency, bool) or not isinstance(
        min_directional_efficiency, (int, float)
    ):
        raise TypeError("min_directional_efficiency must be a number")
    try:
        efficiency = float(min_directional_efficiency)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "min_directional_efficiency must be a number"
        ) from exc
    if not math.isfinite(efficiency) or not 0.0 <= efficiency <= 1.0:
        raise ValueError(
            "min_directional_efficiency must be finite and in the range [0, 1]"
        )

    return threshold, efficiency


def _window_measurements(
    bars: tuple[HistoricalBar, ...],
) -> tuple[float, float]:
    """Return window return percentage and directional efficiency."""
    first = float(bars[0].close)
    last = float(bars[-1].close)
    try:
        return_percent = ((last / first) - 1.0) * 100.0
        path_length = math.fsum(
            abs(float(current.close) - float(previous.close))
            for previous, current in zip(bars, bars[1:])
        )
        efficiency = (
            0.0
            if path_length == 0.0
            else abs(last - first) / path_length
        )
    except OverflowError as exc:
        raise ValueError("window price measurement overflowed") from exc
    except ZeroDivisionError as exc:
        raise ValueError(
            "window price measurement divided by zero"
        ) from exc

    if not math.isfinite(return_percent) or not math.isfinite(path_length):
        raise ValueError("window price measurement is not finite")
    if not math.isfinite(efficiency):
        raise ValueError("window directional efficiency is not finite")
    efficiency = max(0.0, min(1.0, efficiency))
    return return_percent, efficiency


def _classify_validated_window(
    bars: tuple[HistoricalBar, ...],
    validation_config: StrategyValidationConfig,
) -> tuple[MarketCondition, float, float]:
    return_percent, efficiency = _window_measurements(bars)
    threshold = validation_config.trend_threshold_percent
    minimum_efficiency = validation_config.min_directional_efficiency

    if (
        return_percent >= threshold
        and efficiency >= minimum_efficiency
    ):
        condition = MarketCondition.BULLISH
    elif (
        return_percent <= -threshold
        and efficiency >= minimum_efficiency
    ):
        condition = MarketCondition.BEARISH
    else:
        condition = MarketCondition.SIDEWAYS
    return condition, return_percent, efficiency


def classify_market_condition(
    bars,
    *,
    trend_threshold_percent: float = DEFAULT_TREND_THRESHOLD_PERCENT,
    min_directional_efficiency: float = (
        DEFAULT_MIN_DIRECTIONAL_EFFICIENCY
    ),
) -> MarketCondition:
    """Classify one supplied bar sequence as bullish, bearish, or sideways.

    The classification uses close-to-close return and directional
    efficiency (net movement divided by total absolute movement). A
    window must contain at least two bars.

    Raises:
        TypeError: If bars or classifier parameters have invalid types.
        ValueError: If bars or classifier values are invalid, or fewer
            than two bars are supplied.
    """
    threshold, efficiency = _validate_classifier_parameters(
        trend_threshold_percent,
        min_directional_efficiency,
    )
    validated = _validate_bars(bars)
    if len(validated) < 2:
        raise ValueError("at least two bars are required for classification")

    config = StrategyValidationConfig(
        window_bars=DEFAULT_WINDOW_BARS,
        trend_threshold_percent=threshold,
        min_directional_efficiency=efficiency,
    )
    condition, _, _ = _classify_validated_window(validated, config)
    return condition


def _sum_values(values, name: str) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"aggregated {name} overflowed") from exc
    if not math.isfinite(total):
        raise ValueError(f"aggregated {name} is not finite")
    return total


def _aggregate_profit_factor(
    profit: float,
    loss: float,
    name: str,
) -> float | None:
    if loss == 0.0:
        return None
    factor = profit / loss
    if not math.isfinite(factor):
        raise ValueError(f"aggregated {name} is not finite")
    return factor


def _summarize_condition(
    condition: MarketCondition,
    windows: tuple[MarketWindowResult, ...],
) -> MarketConditionSummary:
    reports = tuple(window.backtest.report for window in windows)
    metrics = tuple(report.metrics for report in reports)
    total_trades = sum(item.total_trades for item in metrics)
    open_trades = sum(item.open_trades for item in metrics)
    completed_trades = total_trades - open_trades
    wins = sum(item.wins for item in metrics)
    losses = sum(item.losses for item in metrics)
    breakeven = sum(item.breakeven for item in metrics)
    win_rate = (
        (wins / completed_trades) * 100.0 if completed_trades else 0.0
    )

    gross_profit = _sum_values(
        (item.gross_profit for item in metrics),
        "gross profit",
    )
    gross_loss = _sum_values(
        (item.gross_loss for item in metrics),
        "gross loss",
    )
    net_profit = _sum_values(
        (item.net_profit for item in metrics),
        "net profit",
    )
    net_loss = _sum_values(
        (item.net_loss for item in metrics),
        "net loss",
    )
    net_pnls = _sum_values(
        (report.net_pnl for report in reports),
        "net PnL",
    )
    gross_pnls = _sum_values(
        (report.gross_pnl for report in reports),
        "gross PnL",
    )
    total_costs = _sum_values(
        (report.total_costs for report in reports),
        "costs",
    )

    return MarketConditionSummary(
        condition=condition,
        window_count=len(windows),
        bar_count=sum(window.bar_count for window in windows),
        total_trades=total_trades,
        completed_trades=completed_trades,
        open_trades=open_trades,
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=win_rate,
        gross_pnl=gross_pnls,
        net_pnl=net_pnls,
        total_costs=total_costs,
        gross_profit_factor=_aggregate_profit_factor(
            gross_profit,
            gross_loss,
            "gross profit factor",
        ),
        net_profit_factor=_aggregate_profit_factor(
            net_profit,
            net_loss,
            "net profit factor",
        ),
        worst_max_drawdown=max(
            report.max_drawdown for report in reports
        ),
        worst_max_drawdown_percent=max(
            item.max_drawdown_percent for item in metrics
        ),
        best_window_net_pnl=max(report.net_pnl for report in reports),
        worst_window_net_pnl=min(report.net_pnl for report in reports),
    )


def validate_strategy(
    bars,
    config: BacktestRunConfig,
    validation_config: StrategyValidationConfig | None = None,
) -> StrategyValidationResult:
    """Validate one strategy across deterministic market-condition windows.

    Only complete, contiguous, non-overlapping windows are evaluated.
    Trailing bars that cannot form a full window are not evaluated and are
    reported through ``unused_bars``. Every window receives the exact same
    BacktestRunConfig, including its risk profile and cost model.

    Condition summaries sum independent window results. They intentionally
    expose no combined ending equity or continuous drawdown because each
    window starts from the profile balance.

    Args:
        bars: Chronological HistoricalBar values, oldest first.
        config: Exact backtest configuration reused for every window.
        validation_config: Window and classification settings. Defaults to
            a 180-bar window, 2% trend threshold, and 0.25 minimum
            directional efficiency.

    Returns:
        Immutable window results, condition summaries, and warnings. Empty
        or insufficient history returns an empty result rather than raising.

    Raises:
        TypeError: If bars, config, or validation config have invalid types.
        ValueError: If bars are invalid or contain duplicate/unordered
            timestamps, config values are invalid, or calculations overflow.
    """
    if not isinstance(config, BacktestRunConfig):
        raise TypeError("config must be a BacktestRunConfig")
    if validation_config is None:
        validation_config = StrategyValidationConfig()
    elif not isinstance(validation_config, StrategyValidationConfig):
        raise TypeError(
            "validation_config must be a StrategyValidationConfig or None"
        )

    validated = _validate_bars(bars)
    window_count = len(validated) // validation_config.window_bars
    unused_bars = len(validated) - (
        window_count * validation_config.window_bars
    )

    windows = []
    for index in range(window_count):
        start = index * validation_config.window_bars
        end = start + validation_config.window_bars
        window_bars = validated[start:end]
        condition, return_percent, efficiency = _classify_validated_window(
            window_bars,
            validation_config,
        )
        windows.append(
            MarketWindowResult(
                index=index,
                condition=condition,
                first_timestamp=window_bars[0].timestamp,
                last_timestamp=window_bars[-1].timestamp,
                bar_count=len(window_bars),
                return_percent=return_percent,
                directional_efficiency=efficiency,
                backtest=run_backtest(window_bars, config),
            )
        )

    condition_summaries = []
    for condition in (
        MarketCondition.BULLISH,
        MarketCondition.BEARISH,
        MarketCondition.SIDEWAYS,
    ):
        matching = tuple(
            window for window in windows if window.condition is condition
        )
        if matching:
            condition_summaries.append(
                _summarize_condition(condition, matching)
            )

    warnings = []
    if not validated:
        warnings.append(
            "No historical bars were supplied; no validation windows were "
            "evaluated."
        )
    elif not windows:
        warnings.append(
            f"Insufficient history: {len(validated)} bars supplied; at "
            f"least {validation_config.window_bars} are required for one "
            "validation window."
        )
    elif unused_bars:
        warnings.append(
            f"{unused_bars} trailing bars were not evaluated because "
            "validation uses full windows only."
        )

    return StrategyValidationResult(
        config=config,
        validation_config=validation_config,
        bar_count=len(validated),
        unused_bars=unused_bars,
        windows=tuple(windows),
        conditions=tuple(condition_summaries),
        warnings=tuple(warnings),
    )
