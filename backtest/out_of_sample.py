"""Deterministic chronological out-of-sample validation.

This module splits supplied chronological bars into one development, one
validation, and one final test period. Each period is evaluated as a
single, independent backtest starting from the profile balance, so period
results never share positions, equity, or indicator state.

The evaluated window is anchored to the end of the supplied history, so
the test period always contains the latest supplied bars. Any extra
history is left unused before the development period.

It is deliberately not an optimizer: no parameter is searched, ranked, or
selected, and no best configuration is reported. Walk-forward analysis is
out of scope, because in walk-forward a later fold's training window can
contain an earlier fold's test data.

No network access, Telegram calls, paper-account changes, live execution,
or filesystem I/O occurs here. Historical bars must already be supplied by
the caller.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from backtest.replay import HistoricalBar
from backtest.runner import (
    BacktestRunConfig,
    BacktestRunResult,
    run_backtest,
)
from backtest.validation import (
    DEFAULT_WINDOW_BARS,
    MarketCondition,
    StrategyValidationConfig,
    classify_market_window,
    validate_historical_bars,
)


INDEPENDENT_PERIODS_WARNING = (
    "Development, validation, and test periods are independent "
    "backtests; their results do not form one continuous portfolio "
    "equity curve."
)
OUT_OF_SAMPLE_PROOF_WARNING = (
    "Genuine out-of-sample status cannot be proven by software: freeze the "
    "strategy configuration before reviewing test-period results. If the "
    "test period or its results informed any parameter choice, the result "
    "is no longer out of sample."
)


class PeriodRole(str, Enum):
    """Chronological role of one independently evaluated period."""

    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


PERIOD_ROLES = (
    PeriodRole.DEVELOPMENT,
    PeriodRole.VALIDATION,
    PeriodRole.TEST,
)


@dataclass(frozen=True, slots=True)
class ChronologicalSplitConfig:
    """Explicit period sizes in bars and an optional embargo.

    Attributes:
        development_bars: Bar count of the development period.
        validation_bars: Bar count of the validation period.
        test_bars: Bar count of the final test period.
        embargo_bars: Bars skipped between development and validation and
            between validation and test. Embargo bars belong to no period.

    Raises:
        TypeError: If a count is not an integer.
        ValueError: If a period count is not positive or the embargo is
            negative.
    """

    development_bars: int
    validation_bars: int
    test_bars: int
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        for name in ("development_bars", "validation_bars", "test_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        embargo = self.embargo_bars
        if isinstance(embargo, bool) or not isinstance(embargo, int):
            raise TypeError("embargo_bars must be an integer")
        if embargo < 0:
            raise ValueError("embargo_bars must be non-negative")

    @property
    def period_bar_count(self) -> int:
        """Return the total number of bars assigned to periods."""
        return (
            self.development_bars
            + self.validation_bars
            + self.test_bars
        )

    @property
    def embargo_bar_count(self) -> int:
        """Return the total number of embargo bars between periods."""
        return 2 * self.embargo_bars

    @property
    def required_bars(self) -> int:
        """Return the minimum history length the split can consume."""
        return self.period_bar_count + self.embargo_bar_count


@dataclass(frozen=True, slots=True)
class PeriodBoundary:
    """One half-open ``[start_index, end_index)`` period slice."""

    role: PeriodRole
    start_index: int
    end_index: int
    start_timestamp: datetime
    end_timestamp: datetime
    bar_count: int


@dataclass(frozen=True, slots=True)
class EmbargoSpan:
    """Bars between two adjacent periods that belong to no period."""

    start_index: int
    end_index: int
    preceding_role: PeriodRole
    following_role: PeriodRole

    @property
    def bar_count(self) -> int:
        """Return the number of embargo bars in this span."""
        return self.end_index - self.start_index

    @property
    def bar_indices(self) -> tuple[int, ...]:
        """Return every embargo bar index in this span."""
        return tuple(range(self.start_index, self.end_index))


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Deterministic chronological boundaries and embargo spans."""

    split_config: ChronologicalSplitConfig
    bar_count: int
    boundaries: tuple[PeriodBoundary, ...]
    embargo_spans: tuple[EmbargoSpan, ...]
    unused_leading_bars: int

    def boundary(self, role: PeriodRole) -> PeriodBoundary:
        """Return the boundary for ``role``.

        Raises:
            TypeError: If ``role`` is not a PeriodRole.
            ValueError: If no boundary exists for ``role``.
        """
        if not isinstance(role, PeriodRole):
            raise TypeError("role must be a PeriodRole")
        for boundary in self.boundaries:
            if boundary.role is role:
                return boundary
        raise ValueError(f"unknown period role: {role!r}")

    @property
    def embargo_bar_count(self) -> int:
        """Return the total number of embargo bars."""
        return sum(span.bar_count for span in self.embargo_spans)

    @property
    def embargo_bar_indices(self) -> tuple[int, ...]:
        """Return every embargo bar index, in chronological order."""
        indices = []
        for span in self.embargo_spans:
            indices.extend(span.bar_indices)
        return tuple(indices)

    @property
    def evaluated_bar_count(self) -> int:
        """Return the total number of bars assigned to period slices."""
        return sum(boundary.bar_count for boundary in self.boundaries)


@dataclass(frozen=True, slots=True)
class PeriodResult:
    """One period's deterministic classification and independent backtest.

    The embedded report is the existing backtest report, so open trades
    remain excluded from realized metrics and period results never share
    equity with another period.
    """

    boundary: PeriodBoundary
    condition: MarketCondition
    return_percent: float
    directional_efficiency: float
    backtest: BacktestRunResult

    @property
    def role(self) -> PeriodRole:
        """Return this period's chronological role."""
        return self.boundary.role

    @property
    def report(self):
        """Return the existing backtest report for this period."""
        return self.backtest.report

    @property
    def total_trades(self) -> int:
        return self.report.total_trades

    @property
    def completed_trades(self) -> int:
        return self.backtest.completed_trades

    @property
    def open_trades(self) -> int:
        return self.backtest.open_trades

    @property
    def net_pnl(self) -> float:
        return self.report.net_pnl

    @property
    def total_costs(self) -> float:
        return self.report.total_costs

    @property
    def win_rate(self) -> float:
        return self.report.win_rate

    @property
    def max_drawdown(self) -> float:
        return self.report.max_drawdown


@dataclass(frozen=True, slots=True)
class OutOfSampleResult:
    """Immutable split, per-period results, and explicit limitations.

    This type exposes no selection, ranking, or best-configuration field.
    Adding one would turn the validation layer into an optimizer.
    """

    config: BacktestRunConfig
    validation_config: StrategyValidationConfig
    split: ChronologicalSplit
    periods: tuple[PeriodResult, ...]
    warnings: tuple[str, ...]

    def period(self, role: PeriodRole) -> PeriodResult:
        """Return the period result for ``role``.

        Raises:
            TypeError: If ``role`` is not a PeriodRole.
            ValueError: If no result exists for ``role``.
        """
        if not isinstance(role, PeriodRole):
            raise TypeError("role must be a PeriodRole")
        for period in self.periods:
            if period.role is role:
                return period
        raise ValueError(f"unknown period role: {role!r}")

    @property
    def split_config(self) -> ChronologicalSplitConfig:
        """Return the exact split configuration used."""
        return self.split.split_config

    @property
    def boundaries(self) -> tuple[PeriodBoundary, ...]:
        """Return the period boundaries in chronological order."""
        return self.split.boundaries

    @property
    def embargo_spans(self) -> tuple[EmbargoSpan, ...]:
        """Return the embargo spans between adjacent periods."""
        return self.split.embargo_spans

    @property
    def embargo_bar_count(self) -> int:
        """Return the total number of embargo bars."""
        return self.split.embargo_bar_count

    @property
    def embargo_bar_indices(self) -> tuple[int, ...]:
        """Return every embargo bar index that no period evaluates."""
        return self.split.embargo_bar_indices

    @property
    def bar_count(self) -> int:
        """Return the number of supplied bars."""
        return self.split.bar_count

    @property
    def unused_leading_bars(self) -> int:
        """Return bars before the development period not evaluated."""
        return self.split.unused_leading_bars

    @property
    def development_period(self) -> PeriodResult:
        """Return the development period result."""
        return self.period(PeriodRole.DEVELOPMENT)

    @property
    def validation_period(self) -> PeriodResult:
        """Return the validation period result."""
        return self.period(PeriodRole.VALIDATION)

    @property
    def test_period(self) -> PeriodResult:
        """Return the final out-of-sample test period result."""
        return self.period(PeriodRole.TEST)


def _build_split(
    validated: tuple[HistoricalBar, ...],
    split_config: ChronologicalSplitConfig,
) -> ChronologicalSplit:
    """Build end-anchored boundaries for validated bars, no reordering."""
    required = split_config.required_bars
    supplied = len(validated)
    if supplied < required:
        raise ValueError(
            f"insufficient history: {supplied} bars supplied; {required} "
            f"are required for {split_config.development_bars} "
            "development, "
            f"{split_config.validation_bars} validation, "
            f"{split_config.test_bars} test, and "
            f"{split_config.embargo_bar_count} embargo bars"
        )

    # Anchor to the end of the supplied history so the test period always
    # ends at the final supplied bar. Older bars stay unevaluated.
    leading_unused = supplied - required

    counts = (
        split_config.development_bars,
        split_config.validation_bars,
        split_config.test_bars,
    )
    boundaries = []
    embargo_spans = []
    offset = leading_unused
    previous_role = None
    for role, count in zip(PERIOD_ROLES, counts):
        if previous_role is not None and split_config.embargo_bars:
            embargo_spans.append(
                EmbargoSpan(
                    start_index=offset,
                    end_index=offset + split_config.embargo_bars,
                    preceding_role=previous_role,
                    following_role=role,
                )
            )
            offset += split_config.embargo_bars

        start_index = offset
        end_index = start_index + count
        boundaries.append(
            PeriodBoundary(
                role=role,
                start_index=start_index,
                end_index=end_index,
                start_timestamp=validated[start_index].timestamp,
                end_timestamp=validated[end_index - 1].timestamp,
                bar_count=count,
            )
        )
        previous_role = role
        offset = end_index

    return ChronologicalSplit(
        split_config=split_config,
        bar_count=supplied,
        boundaries=tuple(boundaries),
        embargo_spans=tuple(embargo_spans),
        unused_leading_bars=leading_unused,
    )


def split_chronological_bars(
    bars,
    split_config: ChronologicalSplitConfig,
) -> ChronologicalSplit:
    """Split supplied bars into development, validation, and test periods.

    Periods are contiguous in the supplied order, never sorted or
    repaired, and anchored to the end of the supplied history: the test
    period ends at the final supplied bar. Bars older than the configured
    window stay unevaluated and are reported through
    ``unused_leading_bars``.

    Args:
        bars: Chronological HistoricalBar values with timezone-aware
            timestamps, oldest first.
        split_config: Explicit period bar counts and embargo size.

    Returns:
        Immutable boundaries, embargo spans, and unused leading count.

    Raises:
        TypeError: If ``bars`` or ``split_config`` have invalid types.
        ValueError: If bars are invalid, timezone-naive, duplicated,
            unordered, or history is insufficient for the configured
            split and embargo.
    """
    if not isinstance(split_config, ChronologicalSplitConfig):
        raise TypeError(
            "split_config must be a ChronologicalSplitConfig"
        )
    validated = validate_historical_bars(
        bars,
        require_timezone_aware=True,
    )
    return _build_split(validated, split_config)


def _period_warnings(periods: tuple[PeriodResult, ...]) -> list[str]:
    warnings = []
    for period in periods:
        report = period.report
        if report.total_trades == 0:
            warnings.append(
                f"{period.role.value.capitalize()} period generated no "
                "trades; no performance conclusions can be drawn from it."
            )
        elif report.open_trades == report.total_trades:
            warnings.append(
                f"All trades in the {period.role.value} period are open; "
                "realized performance cannot be evaluated for it."
            )
    return warnings


def _build_warnings(
    split: ChronologicalSplit,
    periods: tuple[PeriodResult, ...],
    validation_config: StrategyValidationConfig,
) -> tuple[str, ...]:
    warnings = [INDEPENDENT_PERIODS_WARNING, OUT_OF_SAMPLE_PROOF_WARNING]

    if split.embargo_spans:
        count = split.embargo_bar_count
        noun = "bar" if count == 1 else "bars"
        warnings.append(
            f"{count} embargo {noun} between periods were excluded from "
            "every evaluation."
        )
    if split.unused_leading_bars:
        warnings.append(
            f"{split.unused_leading_bars} leading bars before the "
            "development period were not evaluated."
        )
    if validation_config.window_bars != DEFAULT_WINDOW_BARS:
        warnings.append(
            "validation_config.window_bars is not used by out-of-sample "
            "validation: each period is one independent backtest rather "
            "than nested Task 9 windows."
        )

    warnings.extend(_period_warnings(periods))
    return tuple(warnings)


def run_out_of_sample_validation(
    bars,
    config: BacktestRunConfig,
    split_config: ChronologicalSplitConfig,
    validation_config: StrategyValidationConfig | None = None,
) -> OutOfSampleResult:
    """Evaluate one fixed configuration across three chronological periods.

    Each period receives only its own bars exactly once and is evaluated
    as one independent backtest with the same ``config``. No open
    position, equity, or indicator state carries across a boundary, and
    no parameter is varied, ranked, or selected. The evaluated window is
    anchored to the end of the supplied history, so the test period ends
    at the final supplied bar and older surplus bars stay unused.

    ``validation_config`` supplies the deterministic market-condition
    thresholds only. Its ``window_bars`` field is not used, because each
    period is a single independent backtest rather than a set of nested
    Task 9 windows.

    Args:
        bars: Chronological HistoricalBar values with timezone-aware
            timestamps, oldest first.
        config: Exact backtest configuration reused for every period.
        split_config: Explicit period bar counts and embargo size.
        validation_config: Optional classification thresholds. ``None``
            uses the Task 9 defaults.

    Returns:
        An immutable result containing the split, one result per period,
        and explicit limitations.

    Raises:
        TypeError: If bars, config, split config, or validation config
            have invalid types.
        ValueError: If bars are invalid or timezone-naive, history is
            insufficient for the configured split and embargo, or any
            period cannot satisfy the configured strategy warmup.
    """
    if not isinstance(config, BacktestRunConfig):
        raise TypeError("config must be a BacktestRunConfig")
    if not isinstance(split_config, ChronologicalSplitConfig):
        raise TypeError(
            "split_config must be a ChronologicalSplitConfig"
        )
    if validation_config is None:
        validation_config = StrategyValidationConfig()
    elif not isinstance(validation_config, StrategyValidationConfig):
        raise TypeError(
            "validation_config must be a StrategyValidationConfig or None"
        )

    validated = validate_historical_bars(
        bars,
        require_timezone_aware=True,
    )
    split = _build_split(validated, split_config)

    minimum_period_bars = config.strategy.min_candles + 2
    for boundary in split.boundaries:
        if boundary.bar_count < minimum_period_bars:
            raise ValueError(
                f"{boundary.role.value} period has "
                f"{boundary.bar_count} bars; at least "
                f"{minimum_period_bars} are required for the configured "
                f"min_candles={config.strategy.min_candles}"
            )

    periods = []
    for boundary in split.boundaries:
        period_bars = validated[
            boundary.start_index:boundary.end_index
        ]
        condition, return_percent, efficiency = (
            classify_market_window(
                period_bars,
                trend_threshold_percent=(
                    validation_config.trend_threshold_percent
                ),
                min_directional_efficiency=(
                    validation_config.min_directional_efficiency
                ),
            )
        )
        periods.append(
            PeriodResult(
                boundary=boundary,
                condition=condition,
                return_percent=return_percent,
                directional_efficiency=efficiency,
                backtest=run_backtest(period_bars, config),
            )
        )

    period_results = tuple(periods)
    return OutOfSampleResult(
        config=config,
        validation_config=validation_config,
        split=split,
        periods=period_results,
        warnings=_build_warnings(
            split,
            period_results,
            validation_config,
        ),
    )


__all__ = [
    "ChronologicalSplit",
    "ChronologicalSplitConfig",
    "EmbargoSpan",
    "OutOfSampleResult",
    "PERIOD_ROLES",
    "PeriodBoundary",
    "PeriodResult",
    "PeriodRole",
    "run_out_of_sample_validation",
    "split_chronological_bars",
]
