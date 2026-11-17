"""Descriptive strategy parameter sensitivity over historical bars.

This module runs one-at-a-time parameter variants through the existing
Task 9 validation pipeline. It is deliberately not an optimizer: it does
not rank variants, select "best" parameters, persist results, or change
paper/live settings.

The same historical bars, risk profile, cost model, validation
configuration, and window size are reused for every variant. Each Task 9
window is an independent backtest starting from the profile balance, so
summed window PnL is not a continuous portfolio equity curve.

All comparisons are in-sample. One-at-a-time sweeps do not measure
parameter interactions, and the results must not be treated as claims of
statistical significance.
"""

from dataclasses import dataclass, fields, replace
import math

from backtest import validation as validation_module
from backtest.runner import BacktestRunConfig
from backtest.validation import (
    MarketCondition,
    MarketConditionSummary,
    StrategyValidationConfig,
    StrategyValidationResult,
)
from strategy.parameters import StrategyParameters


_CONFIGURABLE_FIELDS = frozenset(
    field.name for field in fields(StrategyParameters)
)


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not finite:
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class ParameterSweep:
    """One strategy field and the distinct values to evaluate.

    Values equal to the baseline are rejected as redundant. A sweep never
    creates a Cartesian product with another sweep.
    """

    field: str
    values: tuple[int | float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("field must be a non-empty string")
        normalized_field = self.field.strip()
        if normalized_field not in _CONFIGURABLE_FIELDS:
            raise ValueError(
                f"unknown or non-configurable strategy field: {self.field!r}"
            )
        object.__setattr__(self, "field", normalized_field)

        if isinstance(self.values, (str, bytes)) or not isinstance(
            self.values, (list, tuple)
        ):
            raise TypeError("values must be a list or tuple of numbers")
        values = tuple(self.values)
        if not values:
            raise ValueError("values must contain at least one value")

        seen = set()
        for value in values:
            _finite_number(value, "sweep value")
            if value in seen:
                raise ValueError(
                    f"duplicate value for {normalized_field}: {value!r}"
                )
            seen.add(value)
        object.__setattr__(self, "values", values)


def _sum_values(values, name: str) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"aggregated {name} overflowed") from exc
    if not math.isfinite(total):
        raise ValueError(f"aggregated {name} is not finite")
    return total


@dataclass(frozen=True, slots=True)
class SensitivityVariantResult:
    """One baseline or one-at-a-time variant and its Task 9 validation.

    All metrics are descriptive. Window results remain independent
    backtests and are not a continuous portfolio equity curve.
    """

    label: str
    parameter_name: str | None
    parameter_value: int | float | None
    parameters: StrategyParameters
    validation: StrategyValidationResult

    def _reports(self):
        return tuple(window.backtest.report for window in self.validation.windows)

    @property
    def total_trades(self) -> int:
        return sum(report.total_trades for report in self._reports())

    @property
    def completed_trades(self) -> int:
        return self.total_trades - self.open_trades

    @property
    def open_trades(self) -> int:
        return sum(report.open_trades for report in self._reports())

    @property
    def wins(self) -> int:
        return sum(report.wins for report in self._reports())

    @property
    def losses(self) -> int:
        return sum(report.losses for report in self._reports())

    @property
    def breakeven(self) -> int:
        return sum(report.metrics.breakeven for report in self._reports())

    @property
    def win_rate(self) -> float:
        completed = self.completed_trades
        return (self.wins / completed) * 100.0 if completed else 0.0

    @property
    def gross_pnl(self) -> float:
        return _sum_values(
            (report.gross_pnl for report in self._reports()),
            "gross PnL",
        )

    @property
    def net_pnl(self) -> float:
        return _sum_values(
            (report.net_pnl for report in self._reports()),
            "net PnL",
        )

    @property
    def total_costs(self) -> float:
        return _sum_values(
            (report.total_costs for report in self._reports()),
            "costs",
        )

    @property
    def net_profit_factor(self) -> float | None:
        profit = _sum_values(
            (report.metrics.net_profit for report in self._reports()),
            "net profit",
        )
        loss = _sum_values(
            (report.metrics.net_loss for report in self._reports()),
            "net loss",
        )
        if loss == 0.0:
            return None
        factor = profit / loss
        if not math.isfinite(factor):
            raise ValueError("aggregated net profit factor is not finite")
        return factor

    @property
    def worst_max_drawdown(self) -> float:
        reports = self._reports()
        return max((report.max_drawdown for report in reports), default=0.0)

    @property
    def worst_max_drawdown_percent(self) -> float:
        reports = self._reports()
        return max(
            (report.metrics.max_drawdown_percent for report in reports),
            default=0.0,
        )

    @property
    def conditions(self) -> tuple[MarketConditionSummary, ...]:
        return self.validation.conditions


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Baseline, every one-at-a-time variant, and comparison constraints."""

    base_config: BacktestRunConfig
    validation_config: StrategyValidationConfig
    bar_count: int
    unused_bars: int
    window_conditions: tuple[MarketCondition, ...]
    baseline: SensitivityVariantResult
    variants: tuple[SensitivityVariantResult, ...]
    warnings: tuple[str, ...]

    @property
    def variant_count(self) -> int:
        return len(self.variants)


def _window_signature(result: StrategyValidationResult):
    return tuple(
        (
            window.index,
            window.first_timestamp,
            window.last_timestamp,
            window.bar_count,
            window.condition,
        )
        for window in result.windows
    )


def _validate_sweeps(sweeps, base_parameters: StrategyParameters, max_variants: int):
    if isinstance(sweeps, (str, bytes)) or not isinstance(
        sweeps, (list, tuple)
    ):
        raise TypeError("sweeps must be a list or tuple of ParameterSweep values")
    if not sweeps:
        raise ValueError("sweeps must contain at least one ParameterSweep")

    normalized = []
    seen_fields = set()
    for sweep in sweeps:
        if not isinstance(sweep, ParameterSweep):
            raise TypeError("sweeps must contain ParameterSweep values")
        if sweep.field in seen_fields:
            raise ValueError(
                f"duplicate sweep field is not supported: {sweep.field!r}"
            )
        seen_fields.add(sweep.field)
        normalized.append(sweep)

    total_variants = sum(len(sweep.values) for sweep in normalized)
    if total_variants > max_variants:
        raise ValueError(
            f"sweep generates {total_variants} variants, exceeding "
            f"max_variants={max_variants}"
        )

    variants = []
    for sweep in normalized:
        baseline_value = getattr(base_parameters, sweep.field)
        for value in sweep.values:
            if value == baseline_value:
                raise ValueError(
                    f"value {value!r} for {sweep.field!r} equals the "
                    "baseline and is redundant"
                )
            try:
                parameters = replace(
                    base_parameters,
                    **{sweep.field: value},
                )
            except TypeError as exc:
                raise TypeError(
                    f"invalid value for {sweep.field!r}: {exc}"
                ) from exc
            except ValueError as exc:
                raise ValueError(
                    f"invalid value for {sweep.field!r}: {exc}"
                ) from exc
            variants.append((sweep.field, value, parameters))

    return tuple(variants)


def run_parameter_sensitivity(
    bars,
    base_config: BacktestRunConfig,
    sweeps,
    validation_config: StrategyValidationConfig | None = None,
    *,
    max_variants: int = 100,
) -> SensitivityResult:
    """Run baseline and one-at-a-time sensitivity variants.

    Args:
        bars: Chronological HistoricalBar values, oldest first. The same
            immutable bar sequence is used for every run.
        base_config: Baseline backtest configuration. Its strategy, risk
            profile, and cost model are held fixed except for the single
            swept strategy field in each variant.
        sweeps: Non-empty sequence of ParameterSweep values. Duplicate
            fields, duplicate values, and baseline-equal values are
            rejected. No Cartesian product is created.
        validation_config: Task 9 window and classification settings.
            ``None`` uses StrategyValidationConfig defaults.
        max_variants: Maximum total number of variants, excluding the
            baseline. Must be a positive integer.

    Returns:
        An immutable SensitivityResult containing the baseline, every
        variant in sweep order, and descriptive warnings. No variant is
        selected or ranked.

    Raises:
        TypeError: If bars, config, sweeps, validation config, or
            max_variants have invalid types.
        ValueError: If sweeps or parameters are invalid, the variant cap
            is exceeded, the window size cannot fit every configured
            warmup, or variants produce inconsistent windows.
    """
    if isinstance(max_variants, bool) or not isinstance(max_variants, int):
        raise TypeError("max_variants must be an integer")
    if max_variants <= 0:
        raise ValueError("max_variants must be positive")

    if not isinstance(base_config, BacktestRunConfig):
        raise TypeError("base_config must be a BacktestRunConfig")
    if not isinstance(bars, (list, tuple)):
        raise TypeError("bars must be a list or tuple of HistoricalBar values")

    if validation_config is None:
        validation_config = StrategyValidationConfig()
    elif not isinstance(validation_config, StrategyValidationConfig):
        raise TypeError(
            "validation_config must be a StrategyValidationConfig or None"
        )

    variants = _validate_sweeps(
        sweeps,
        base_config.strategy,
        max_variants,
    )

    minimum_candles = max(
        [base_config.strategy.min_candles]
        + [parameters.min_candles for _, _, parameters in variants]
    )
    required_window = minimum_candles + 2
    if validation_config.window_bars < required_window:
        raise ValueError(
            f"window_bars must be at least {required_window} for the "
            "configured min_candles across the sweep"
        )

    baseline_validation = validation_module.validate_strategy(
        bars,
        base_config,
        validation_config,
    )
    baseline = SensitivityVariantResult(
        label="baseline",
        parameter_name=None,
        parameter_value=None,
        parameters=base_config.strategy,
        validation=baseline_validation,
    )

    baseline_signature = _window_signature(baseline_validation)
    variant_results = []
    for field, value, parameters in variants:
        variant_config = replace(base_config, strategy=parameters)
        validation = validation_module.validate_strategy(
            bars,
            variant_config,
            validation_config,
        )
        if (
            validation.bar_count != baseline_validation.bar_count
            or validation.unused_bars != baseline_validation.unused_bars
            or _window_signature(validation) != baseline_signature
        ):
            raise ValueError(
                "sensitivity variant produced inconsistent window "
                "boundaries or market-condition classifications"
            )
        variant_results.append(
            SensitivityVariantResult(
                label=f"{field}={value!r}",
                parameter_name=field,
                parameter_value=value,
                parameters=parameters,
                validation=validation,
            )
        )

    warnings = (
        "Sensitivity results are descriptive and in-sample; they are not "
        "out-of-sample validation or statistical evidence.",
        "One-at-a-time sweeps do not measure parameter interactions.",
        "Task 9 windows are independently backtested and do not form one "
        "continuous portfolio equity curve.",
        "Sensitivity results never automatically change paper or live "
        "trading settings.",
    )

    return SensitivityResult(
        base_config=base_config,
        validation_config=validation_config,
        bar_count=baseline_validation.bar_count,
        unused_bars=baseline_validation.unused_bars,
        window_conditions=tuple(
            window.condition for window in baseline_validation.windows
        ),
        baseline=baseline,
        variants=tuple(variant_results),
        warnings=warnings,
    )


__all__ = [
    "ParameterSweep",
    "SensitivityResult",
    "SensitivityVariantResult",
    "run_parameter_sensitivity",
]
