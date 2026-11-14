"""Paper trading persistence layer.

Sprint 5F: JSON-based storage for PaperAccount and PaperOrder.

This module provides functions to save and load paper trading state
to/from JSON files. It uses only the Python standard library and
produces fully functional PaperAccount and PaperOrder objects on load.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from trader.paper.account import PaperAccount
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.trade import Trade


def _reject_non_finite_json_constant(constant: str) -> None:
    """Reject JSON ``NaN``, ``Infinity`` and ``-Infinity`` on load.

    Python's ``json`` module accepts these non-standard literals by
    default, which would let a corrupted file inject non-finite numbers
    into account balances and PnL calculations.

    Raises:
        ValueError: Always, naming the rejected constant.
    """
    raise ValueError(
        "Invalid paper trading data: non-finite JSON number "
        f"{constant!r} is not allowed."
    )


def _validate_saved_number(
    value: Any,
    field: str,
    *,
    context: str,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    """Validate one numeric field read from saved paper trading data.

    Args:
        value: The raw JSON value to validate.
        field: Field name used in the error message.
        context: Description of the containing record.
        minimum: Optional lower bound for the value.
        strictly_greater: When True, ``minimum`` is exclusive (``>``)
            instead of inclusive (``>=``).

    Returns:
        The value converted to a finite float.

    Raises:
        ValueError: If the value is a boolean, not a number, not finite,
            or outside the requested range.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Invalid {context}: field {field!r} must be a number, "
            f"got {value!r}."
        )

    try:
        number = float(value)
    except OverflowError:
        raise ValueError(
            f"Invalid {context}: field {field!r} must be finite, "
            f"got {value!r}."
        ) from None

    if not math.isfinite(number):
        raise ValueError(
            f"Invalid {context}: field {field!r} must be finite, "
            f"got {value!r}."
        )

    if minimum is not None:
        if strictly_greater and number <= minimum:
            raise ValueError(
                f"Invalid {context}: field {field!r} must be greater than "
                f"{minimum:g}, got {value!r}."
            )
        if not strictly_greater and number < minimum:
            raise ValueError(
                f"Invalid {context}: field {field!r} must be at least "
                f"{minimum:g}, got {value!r}."
            )

    return number


def _required_field(data: dict[str, Any], field: str, *, context: str) -> Any:
    """Return a required field or raise a clear ValueError."""
    try:
        return data[field]
    except KeyError:
        raise ValueError(
            f"Invalid {context}: missing required field {field!r}."
        ) from None


def _optional_positive_price(
    data: dict[str, Any], field: str, *, context: str
) -> float | None:
    """Validate an optional price that must be finite and positive."""
    value = data.get(field)
    if value is None:
        return None
    return _validate_saved_number(
        value,
        field,
        context=context,
        minimum=0.0,
        strictly_greater=True,
    )


def _order_to_dict(order: PaperOrder) -> dict[str, Any]:
    """Convert a PaperOrder to a JSON-serializable dict."""
    return {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side.value,
        "quantity": order.quantity,
        "entry_price": order.entry_price,
        "stop_loss": order.stop_loss,
        "take_profit": order.take_profit,
        "status": order.status.value,
        "trade": (
            {
                "symbol": order.trade.symbol,
                "price": order.trade.price,
                "trend": order.trade.trend,
                "signal": order.trade.signal,
                "confidence": order.trade.confidence,
                "decision": order.trade.decision,
                "risk_verdict": order.trade.risk_verdict,
                "position_size": order.trade.position_size,
                "stop_loss": order.trade.stop_loss,
                "take_profit": order.trade.take_profit,
                "side": order.trade.side,
                "approved": order.trade.approved,
                "risk_reasons": order.trade.risk_reasons,
                "timestamp": order.trade.timestamp,
            }
            if order.trade is not None
            else None
        ),
        "created_at": order.created_at,
        "filled_at": order.filled_at,
        "exit_at": order.exit_at,
        "exit_price": order.exit_price,
        "realized_pnl": order.realized_pnl,
        "unrealized_pnl": order.unrealized_pnl,
        "mark_price": order.mark_price,
        "mark_at": order.mark_at,
        "initial_risk_distance": order.initial_risk_distance,
        "paper": order.paper,
        "notes": order.notes,
    }


def _dict_to_order(data: dict[str, Any]) -> PaperOrder:
    """Reconstruct a validated PaperOrder from a JSON dict.

    Raises:
        ValueError: If the record is not an object, a required field is
            missing, or any saved numeric field is not a finite number
            with a valid range.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid paper order data: expected a JSON object.")

    raw_order_id = data.get("order_id")
    context = (
        f"paper order {raw_order_id!r}"
        if isinstance(raw_order_id, str) and raw_order_id
        else "paper order"
    )

    order_id = _required_field(data, "order_id", context=context)
    symbol = _required_field(data, "symbol", context=context)

    raw_side = _required_field(data, "side", context=context)
    try:
        side = PaperOrderSide(raw_side)
    except ValueError:
        raise ValueError(
            f"Invalid {context}: unknown side {raw_side!r}."
        ) from None

    raw_status = _required_field(data, "status", context=context)
    try:
        status = PaperOrderStatus(raw_status)
    except ValueError:
        raise ValueError(
            f"Invalid {context}: unknown status {raw_status!r}."
        ) from None

    quantity = _validate_saved_number(
        _required_field(data, "quantity", context=context),
        "quantity",
        context=context,
        minimum=0.0,
        strictly_greater=True,
    )
    entry_price = _validate_saved_number(
        _required_field(data, "entry_price", context=context),
        "entry_price",
        context=context,
        minimum=0.0,
        strictly_greater=True,
    )
    stop_loss = _validate_saved_number(
        _required_field(data, "stop_loss", context=context),
        "stop_loss",
        context=context,
        minimum=0.0,
    )
    take_profit = _validate_saved_number(
        _required_field(data, "take_profit", context=context),
        "take_profit",
        context=context,
        minimum=0.0,
    )
    realized_pnl = _validate_saved_number(
        data.get("realized_pnl", 0.0), "realized_pnl", context=context
    )
    unrealized_pnl = _validate_saved_number(
        data.get("unrealized_pnl", 0.0), "unrealized_pnl", context=context
    )
    mark_price = _optional_positive_price(data, "mark_price", context=context)
    exit_price = _optional_positive_price(data, "exit_price", context=context)
    initial_risk_distance = _optional_positive_price(
        data, "initial_risk_distance", context=context
    )

    trade_data = data.get("trade")
    trade = None
    if trade_data is not None:
        if not isinstance(trade_data, dict):
            raise ValueError(
                f"Invalid {context}: field 'trade' must be a JSON object "
                "or null."
            )
        trade_context = f"trade for {context}"
        trade = Trade(
            symbol=_required_field(trade_data, "symbol", context=trade_context),
            price=_required_field(trade_data, "price", context=trade_context),
            trend=_required_field(trade_data, "trend", context=trade_context),
            signal=_required_field(trade_data, "signal", context=trade_context),
            confidence=_required_field(
                trade_data, "confidence", context=trade_context
            ),
            decision=_required_field(trade_data, "decision", context=trade_context),
            risk_verdict=_required_field(
                trade_data, "risk_verdict", context=trade_context
            ),
            position_size=_required_field(
                trade_data, "position_size", context=trade_context
            ),
            stop_loss=_required_field(
                trade_data, "stop_loss", context=trade_context
            ),
            take_profit=_required_field(
                trade_data, "take_profit", context=trade_context
            ),
            side=_required_field(trade_data, "side", context=trade_context),
            approved=_required_field(
                trade_data, "approved", context=trade_context
            ),
            risk_reasons=list(trade_data.get("risk_reasons", [])),
            timestamp=trade_data.get("timestamp", ""),
        )

    return PaperOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        status=status,
        trade=trade,
        created_at=data.get("created_at", ""),
        filled_at=data.get("filled_at"),
        exit_at=data.get("exit_at"),
        exit_price=exit_price,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        mark_price=mark_price,
        mark_at=data.get("mark_at"),
        initial_risk_distance=initial_risk_distance,
        paper=data.get("paper", True),
        notes=list(data.get("notes", [])),
    )


def _account_to_dict(account: PaperAccount) -> dict[str, Any]:
    """Convert a PaperAccount to a JSON-serializable dict."""
    return {
        "initial_balance": account.initial_balance,
        "current_balance": account.current_balance,
        "open_positions": [_order_to_dict(o) for o in account.open_positions],
        "closed_positions": [_order_to_dict(o) for o in account.closed_positions],
    }


def _dict_to_account(data: dict[str, Any]) -> PaperAccount:
    """Reconstruct a validated PaperAccount from a JSON dict.

    Raises:
        ValueError: If the record is not an object, a required field is
            missing, a positions field is not an array, or any saved
            numeric value is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError("Invalid paper account data: expected a JSON object.")

    context = "paper account"
    initial_balance = _validate_saved_number(
        _required_field(data, "initial_balance", context=context),
        "initial_balance",
        context=context,
    )
    current_balance = _validate_saved_number(
        _required_field(data, "current_balance", context=context),
        "current_balance",
        context=context,
    )

    open_raw = data.get("open_positions", [])
    closed_raw = data.get("closed_positions", [])
    for field, value in (
        ("open_positions", open_raw),
        ("closed_positions", closed_raw),
    ):
        if not isinstance(value, list):
            raise ValueError(
                f"Invalid {context}: field {field!r} must be a JSON array."
            )

    open_positions = [_dict_to_order(o) for o in open_raw]
    closed_positions = [_dict_to_order(o) for o in closed_raw]
    return PaperAccount(
        initial_balance=initial_balance,
        current_balance=current_balance,
        open_positions=open_positions,
        closed_positions=closed_positions,
    )


def save_account(account: PaperAccount, path: str | Path) -> None:
    """Save a PaperAccount to a JSON file.

    Args:
        account: The PaperAccount to save.
        path: File path to write the JSON to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _account_to_dict(account)
    # Serialize fully before writing so a non-finite value cannot leave a
    # truncated account file behind.
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(payload, encoding="utf-8")


def load_account(path: str | Path) -> PaperAccount:
    """Load a PaperAccount from a JSON file.

    Args:
        path: File path to read the JSON from.

    Returns:
        A fully functional PaperAccount reconstructed from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or missing required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Account file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, parse_constant=_reject_non_finite_json_constant)

    if not isinstance(data, dict):
        raise ValueError("Invalid account data: expected a JSON object")

    return _dict_to_account(data)


def save_orders(orders: list[PaperOrder], path: str | Path) -> None:
    """Save a list of PaperOrder objects to a JSON file.

    Args:
        orders: List of PaperOrder objects to save.
        path: File path to write the JSON to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [_order_to_dict(o) for o in orders]
    # Serialize fully before writing so a non-finite value cannot leave a
    # truncated orders file behind.
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(payload, encoding="utf-8")


def load_orders(path: str | Path) -> list[PaperOrder]:
    """Load a list of PaperOrder objects from a JSON file.

    Args:
        path: File path to read the JSON from.

    Returns:
        List of fully functional PaperOrder objects reconstructed from
        the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or missing required
            fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Orders file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, parse_constant=_reject_non_finite_json_constant)

    if not isinstance(data, list):
        raise ValueError("Invalid orders data: expected a JSON array")

    return [_dict_to_order(item) for item in data]
