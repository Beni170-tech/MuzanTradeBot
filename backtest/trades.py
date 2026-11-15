"""Connect historical signals to validated, non-overlapping trades.

This module is a deterministic orchestration layer. It reuses the live
decision, risk, sizing, trade-building, and exit rules without touching
paper accounts, live execution, the network, or the filesystem.
"""

from dataclasses import dataclass, replace
from datetime import datetime

from backtest.execution import BacktestPosition, simulate_exit
from backtest.replay import HistoricalBar, replay_signals
from risk.risk_manager import validate_trade
from risk.risk_profile import RiskProfile
from trader.paper.order import PaperOrderStatus
from trader.trade import build_trade

TRADE_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True, slots=True)
class HistoricalTrade:
    """One signal-to-entry-to-exit historical trade record."""

    symbol: str
    signal_timestamp: datetime
    entry_timestamp: datetime
    entry_price: float
    side: str
    quantity: float
    stop_loss: float
    take_profit: float
    trade_timestamp: str
    exit_timestamp: datetime | None = None
    exit_price: float | None = None
    exit_status: PaperOrderStatus | None = None


def _analysis_from_record(symbol: str, record, entry_price: float) -> dict:
    """Build the analysis contract expected by the risk layer."""
    return {
        "symbol": symbol,
        "price": entry_price,
        "trend": record.trend,
        "signal": record.signal,
        "confidence": record.confidence,
    }


def simulate_trades(
    symbol: str,
    bars: list[HistoricalBar] | tuple[HistoricalBar, ...],
    profile: RiskProfile,
) -> list[HistoricalTrade]:
    """Replay signals and simulate non-overlapping historical trades.

    A tradable signal at bar N enters at bar N+1's close. WAIT and blocked
    signals create no trade. Only one position may be open at a time, and
    the next signal is considered only after the exit bar reported by the
    exit simulator. An unclosed trade remains open through the end of the
    data and prevents later entries.

    Args:
        symbol: Symbol associated with the whole bar series.
        bars: Chronological HistoricalBar values, oldest first.
        profile: Risk profile used for validation, sizing, and exits.

    Returns:
        HistoricalTrade records in chronological order. The final record
        has no exit information when the position never closed.

    Raises:
        ValueError: If symbol is empty or invalid.
        TypeError: If profile is not a RiskProfile.
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(profile, RiskProfile):
        raise TypeError("profile must be a RiskProfile")

    normalized_symbol = symbol.strip().upper()

    # Replay must run exactly once per simulation; all decisions below use
    # these records rather than recomputing indicators.
    records = replay_signals(bars)
    if not records:
        return []

    first_signal_index = len(bars) - len(records)
    records_by_index = {
        first_signal_index + offset: record
        for offset, record in enumerate(records)
    }

    trades: list[HistoricalTrade] = []
    signal_index = first_signal_index

    while signal_index < len(bars):
        record = records_by_index.get(signal_index)
        if record is None:
            signal_index += 1
            continue

        if record.decision == "WAIT":
            signal_index += 1
            continue

        entry_index = signal_index + 1
        if entry_index >= len(bars):
            break

        entry_bar = bars[entry_index]
        analysis = _analysis_from_record(
            normalized_symbol,
            record,
            entry_bar.close,
        )
        verdict = validate_trade(record.decision, analysis, profile)
        if not verdict.approved:
            signal_index += 1
            continue

        trade = build_trade(analysis, verdict, profile)
        trade = replace(
            trade,
            timestamp=entry_bar.timestamp.strftime(TRADE_TIMESTAMP_FORMAT),
        )

        position = BacktestPosition(
            side=trade.side,
            entry_price=float(trade.price),
            stop_loss=trade.stop_loss,
            take_profit=trade.take_profit,
        )
        simulation = simulate_exit(position, bars[entry_index + 1:])
        exit_event = simulation.exit_event

        trades.append(
            HistoricalTrade(
                symbol=normalized_symbol,
                signal_timestamp=record.timestamp,
                entry_timestamp=entry_bar.timestamp,
                entry_price=float(trade.price),
                side=trade.side,
                quantity=trade.position_size,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                trade_timestamp=trade.timestamp,
                exit_timestamp=(
                    exit_event.timestamp if exit_event is not None else None
                ),
                exit_price=(
                    exit_event.price if exit_event is not None else None
                ),
                exit_status=(
                    exit_event.status if exit_event is not None else None
                ),
            )
        )

        if exit_event is None:
            # The position remains open through the end of the data.
            break

        # simulate_exit processes bars[entry_index + 1:], so its count
        # identifies the exact exit bar without re-deriving the rules.
        exit_index = entry_index + simulation.bars_processed
        signal_index = exit_index + 1

    return trades
