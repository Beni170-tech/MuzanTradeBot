"""Paper trading persistence layer.

Sprint 5F: JSON-based storage for PaperAccount and PaperOrder.

This module provides functions to save and load paper trading state
to/from JSON files. It uses only the Python standard library and
produces fully functional PaperAccount and PaperOrder objects on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from trader.paper.account import PaperAccount
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.trade import Trade


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
        "paper": order.paper,
        "notes": order.notes,
    }


def _dict_to_order(data: dict[str, Any]) -> PaperOrder:
    """Reconstruct a PaperOrder from a JSON dict."""
    trade_data = data.get("trade")
    trade = None
    if trade_data is not None:
        trade = Trade(
            symbol=trade_data["symbol"],
            price=trade_data["price"],
            trend=trade_data["trend"],
            signal=trade_data["signal"],
            confidence=trade_data["confidence"],
            decision=trade_data["decision"],
            risk_verdict=trade_data["risk_verdict"],
            position_size=trade_data["position_size"],
            stop_loss=trade_data["stop_loss"],
            take_profit=trade_data["take_profit"],
            side=trade_data["side"],
            approved=trade_data["approved"],
            risk_reasons=list(trade_data.get("risk_reasons", [])),
            timestamp=trade_data.get("timestamp", ""),
        )

    return PaperOrder(
        order_id=data["order_id"],
        symbol=data["symbol"],
        side=PaperOrderSide(data["side"]),
        quantity=data["quantity"],
        entry_price=data["entry_price"],
        stop_loss=data["stop_loss"],
        take_profit=data["take_profit"],
        status=PaperOrderStatus(data["status"]),
        trade=trade,
        created_at=data.get("created_at", ""),
        filled_at=data.get("filled_at"),
        exit_at=data.get("exit_at"),
        exit_price=data.get("exit_price"),
        realized_pnl=data.get("realized_pnl", 0.0),
        unrealized_pnl=data.get("unrealized_pnl", 0.0),
        mark_price=data.get("mark_price"),
        mark_at=data.get("mark_at"),
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
    """Reconstruct a PaperAccount from a JSON dict."""
    open_positions = [_dict_to_order(o) for o in data.get("open_positions", [])]
    closed_positions = [_dict_to_order(o) for o in data.get("closed_positions", [])]
    return PaperAccount(
        initial_balance=data["initial_balance"],
        current_balance=data["current_balance"],
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
        data = json.load(f)

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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Invalid orders data: expected a JSON array")

    return [_dict_to_order(item) for item in data]
