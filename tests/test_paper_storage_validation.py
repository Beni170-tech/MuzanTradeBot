"""Regression tests for validation of saved paper trading data (D2)."""

import json
import math
from pathlib import Path

import pytest

from trader.paper.account import PaperAccount
from trader.paper.order import PaperOrder, PaperOrderSide, PaperOrderStatus
from trader.paper.storage import load_account, load_orders, save_account, save_orders


NON_FINITE_VALUES = [
    pytest.param(float("nan"), "NaN", id="nan"),
    pytest.param(float("inf"), "Infinity", id="infinity"),
    pytest.param(float("-inf"), "-Infinity", id="negative-infinity"),
]

NUMERIC_ORDER_FIELDS = [
    "quantity",
    "entry_price",
    "stop_loss",
    "take_profit",
    "mark_price",
    "exit_price",
    "realized_pnl",
    "unrealized_pnl",
]


def _order_dict(**overrides: object) -> dict:
    """Return a valid saved order record, with optional overrides."""
    data = {
        "order_id": "ORD-1",
        "symbol": "BTC",
        "side": "LONG",
        "quantity": 10.0,
        "entry_price": 100.0,
        "stop_loss": 98.0,
        "take_profit": 104.0,
        "status": "OPEN",
        "trade": None,
        "created_at": "2025-01-01 12:00:00",
        "filled_at": "2025-01-01 12:05:00",
        "exit_at": None,
        "exit_price": None,
        "realized_pnl": 0.0,
        "unrealized_pnl": 50.0,
        "mark_price": None,
        "mark_at": None,
        "paper": True,
        "notes": ["test note"],
    }
    data.update(overrides)
    return data


def _account_dict(**overrides: object) -> dict:
    """Return a valid saved account record, with optional overrides."""
    data = {
        "initial_balance": 1000.0,
        "current_balance": 1000.0,
        "open_positions": [],
        "closed_positions": [],
    }
    data.update(overrides)
    return data


def _write(path: Path, payload: object) -> Path:
    """Write payload as JSON, letting Python emit NaN/Infinity literals."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_order(
    *,
    order_id: str = "ORD-1",
    quantity: float = 10.0,
    entry_price: float = 100.0,
    stop_loss: float = 98.0,
    take_profit: float = 104.0,
    realized_pnl: float = 0.0,
    notes: list[str] | None = None,
) -> PaperOrder:
    return PaperOrder(
        order_id=order_id,
        symbol="BTC",
        side=PaperOrderSide.LONG,
        quantity=quantity,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        status=PaperOrderStatus.OPEN,
        realized_pnl=realized_pnl,
        notes=notes if notes is not None else ["test note"],
    )


# ---------------------------------------------------------------------------
# JSON NaN / Infinity / -Infinity rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, token", NON_FINITE_VALUES)
@pytest.mark.parametrize("field", ["initial_balance", "current_balance"])
def test_load_account_rejects_non_finite_json_balance(
    tmp_path: Path, field: str, value: float, token: str
) -> None:
    path = tmp_path / "account.json"
    path.write_text(
        json.dumps(_account_dict(**{field: value})), encoding="utf-8"
    )
    assert token in path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_account(path)


@pytest.mark.parametrize("value, token", NON_FINITE_VALUES)
def test_load_account_rejects_non_finite_position_values(
    tmp_path: Path, value: float, token: str
) -> None:
    path = _write(
        tmp_path / "account.json",
        _account_dict(open_positions=[_order_dict(entry_price=value)]),
    )
    assert token in path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_account(path)


@pytest.mark.parametrize("value, token", NON_FINITE_VALUES)
@pytest.mark.parametrize(
    "field", ["quantity", "entry_price", "realized_pnl"]
)
def test_load_orders_rejects_non_finite_json_values(
    tmp_path: Path, field: str, value: float, token: str
) -> None:
    path = _write(
        tmp_path / "orders.json", [_order_dict(**{field: value})]
    )
    assert token in path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="finite"):
        load_orders(path)


# ---------------------------------------------------------------------------
# Account balance validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [True, False, "1000.0", [1000.0]])
@pytest.mark.parametrize("field", ["initial_balance", "current_balance"])
def test_load_account_rejects_non_numeric_balances(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _write(tmp_path / "account.json", _account_dict(**{field: value}))

    with pytest.raises(ValueError, match=f"field '{field}' must be a number"):
        load_account(path)


def test_load_account_accepts_negative_finite_balances(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "account.json",
        _account_dict(initial_balance=1000.0, current_balance=-250.5),
    )

    account = load_account(path)

    assert account.initial_balance == 1000.0
    assert account.current_balance == -250.5
    assert account.equity() == -250.5


# ---------------------------------------------------------------------------
# Order numeric validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", NUMERIC_ORDER_FIELDS)
@pytest.mark.parametrize("value", [True, False, "10.0"])
def test_load_orders_rejects_boolean_and_string_numbers(
    tmp_path: Path, field: str, value: object
) -> None:
    path = _write(tmp_path / "orders.json", [_order_dict(**{field: value})])

    with pytest.raises(ValueError, match=f"field '{field}' must be a number"):
        load_orders(path)


@pytest.mark.parametrize("field", NUMERIC_ORDER_FIELDS)
@pytest.mark.parametrize("value, token", NON_FINITE_VALUES)
def test_load_orders_rejects_non_finite_numbers(
    tmp_path: Path, field: str, value: float, token: str
) -> None:
    path = _write(tmp_path / "orders.json", [_order_dict(**{field: value})])

    with pytest.raises(ValueError, match="finite"):
        load_orders(path)


@pytest.mark.parametrize("field", ["quantity", "entry_price", "mark_price", "exit_price"])
@pytest.mark.parametrize("value", [0.0, -1.0])
def test_load_orders_rejects_non_positive_required_prices(
    tmp_path: Path, field: str, value: float
) -> None:
    path = _write(tmp_path / "orders.json", [_order_dict(**{field: value})])

    with pytest.raises(ValueError, match=f"field '{field}' must be greater than 0"):
        load_orders(path)


@pytest.mark.parametrize("field", ["stop_loss", "take_profit"])
def test_load_orders_rejects_negative_levels(tmp_path: Path, field: str) -> None:
    path = _write(tmp_path / "orders.json", [_order_dict(**{field: -0.01})])

    with pytest.raises(ValueError, match=f"field '{field}' must be at least 0"):
        load_orders(path)


def test_zero_stop_loss_and_take_profit_are_accepted(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "orders.json",
        [_order_dict(stop_loss=0.0, take_profit=0.0)],
    )

    (order,) = load_orders(path)

    assert order.stop_loss == 0.0
    assert order.take_profit == 0.0


def test_negative_realized_pnl_is_accepted(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "orders.json",
        [
            _order_dict(
                status="STOPPED",
                exit_price=95.0,
                realized_pnl=-50.0,
                unrealized_pnl=-25.0,
            )
        ],
    )

    (order,) = load_orders(path)

    assert order.realized_pnl == -50.0
    assert order.unrealized_pnl == -25.0


def test_one_invalid_position_rejects_the_whole_account(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "account.json",
        _account_dict(
            open_positions=[
                _order_dict(order_id="GOOD-1"),
                _order_dict(order_id="BAD-1", entry_price=0.0),
                _order_dict(order_id="GOOD-2", symbol="ETH"),
            ]
        ),
    )

    with pytest.raises(
        ValueError, match="paper order 'BAD-1'.*field 'entry_price'"
    ):
        load_account(path)


# ---------------------------------------------------------------------------
# Missing and malformed structure
# ---------------------------------------------------------------------------

def test_missing_account_balance_raises_value_error(tmp_path: Path) -> None:
    data = _account_dict()
    del data["current_balance"]
    path = _write(tmp_path / "account.json", data)

    with pytest.raises(ValueError, match="missing required field 'current_balance'"):
        load_account(path)


@pytest.mark.parametrize("field", ["quantity", "entry_price", "side", "status"])
def test_missing_order_field_raises_value_error(
    tmp_path: Path, field: str
) -> None:
    data = _order_dict()
    del data[field]
    path = _write(tmp_path / "orders.json", [data])

    with pytest.raises(
        ValueError, match=f"paper order 'ORD-1'.*missing required field '{field}'"
    ):
        load_orders(path)


@pytest.mark.parametrize("container", ["positions", "order"])
def test_wrong_structure_raises_value_error(
    tmp_path: Path, container: str
) -> None:
    if container == "positions":
        payload = _account_dict(open_positions={"not": "a list"})
        path = _write(tmp_path / "account.json", payload)
        expected = "field 'open_positions' must be a JSON array"
        loader = load_account
    else:
        path = _write(tmp_path / "orders.json", ["not an object"])
        expected = "expected a JSON object"
        loader = load_orders

    with pytest.raises(ValueError, match=expected):
        loader(path)


def test_unknown_order_side_raises_clear_value_error(tmp_path: Path) -> None:
    path = _write(tmp_path / "orders.json", [_order_dict(side="SIDEWAYS")])

    with pytest.raises(ValueError, match="paper order 'ORD-1'.*unknown side"):
        load_orders(path)


# ---------------------------------------------------------------------------
# Unrealized P&L hardening
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "10.0", 0.0, -5.0])
@pytest.mark.parametrize("field", ["entry_price", "quantity"])
def test_invalid_entry_or_quantity_cannot_contaminate_pnl(
    field: str, value: object
) -> None:
    kwargs = {field: value}
    order = _make_order(**kwargs)
    order.mark_price = 100.0

    assert order.unrealized_pnl_for(110.0) == 0.0

    account = PaperAccount(
        initial_balance=1000.0,
        current_balance=1000.0,
        open_positions=[order],
    )
    assert account.equity() == 1000.0


@pytest.mark.parametrize("value", [10**400, -(10**400)])
@pytest.mark.parametrize("field", ["entry_price", "quantity"])
def test_extremely_large_integers_cannot_contaminate_pnl(
    field: str, value: int
) -> None:
    order = _make_order(**{field: value})
    order.mark_price = 100.0

    assert order.unrealized_pnl_for(110.0) == 0.0

    account = PaperAccount(
        initial_balance=1000.0,
        current_balance=1000.0,
        open_positions=[order],
    )
    assert account.equity() == 1000.0


def test_valid_unrealized_pnl_is_unchanged() -> None:
    long_order = _make_order(quantity=10.0, entry_price=100.0)
    short_order = PaperOrder(
        order_id="ORD-2",
        symbol="BTC",
        side=PaperOrderSide.SHORT,
        quantity=10.0,
        entry_price=100.0,
        stop_loss=102.0,
        take_profit=96.0,
        status=PaperOrderStatus.OPEN,
    )

    assert long_order.unrealized_pnl_for(110.0) == pytest.approx(100.0)
    assert short_order.unrealized_pnl_for(110.0) == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# Save-time rejection and round trips
# ---------------------------------------------------------------------------

def test_save_account_rejects_non_finite_values_without_truncating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "account.json"
    path.write_text("original contents", encoding="utf-8")
    account = PaperAccount(
        initial_balance=1000.0, current_balance=float("nan")
    )

    with pytest.raises(ValueError):
        save_account(account, path)

    assert path.read_text(encoding="utf-8") == "original contents"


def test_save_orders_rejects_non_finite_values_without_truncating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.json"
    path.write_text("original contents", encoding="utf-8")
    order = _make_order(entry_price=float("inf"))

    with pytest.raises(ValueError):
        save_orders([order], path)

    assert path.read_text(encoding="utf-8") == "original contents"


def test_valid_account_and_orders_round_trip(tmp_path: Path) -> None:
    order = _make_order()
    order.mark_to_market(105.0)
    account = PaperAccount(
        initial_balance=1000.0,
        current_balance=950.0,
        open_positions=[order],
    )
    account_path = tmp_path / "account.json"
    orders_path = tmp_path / "orders.json"

    save_account(account, account_path)
    save_orders([order], orders_path)
    loaded_account = load_account(account_path)
    (loaded_order,) = load_orders(orders_path)

    assert loaded_account.initial_balance == 1000.0
    assert loaded_account.current_balance == 950.0
    assert loaded_account.open_count == 1
    assert loaded_account.open_positions[0].order_id == "ORD-1"
    assert loaded_account.open_positions[0].mark_price == pytest.approx(105.0)
    assert loaded_order.quantity == 10.0
    assert loaded_order.entry_price == 100.0
    assert loaded_order.stop_loss == 98.0
    assert loaded_order.take_profit == 104.0
    assert loaded_order.notes == ["test note"]


def test_legacy_order_without_optional_fields_loads(tmp_path: Path) -> None:
    data = _order_dict()
    for field in [
        "trade",
        "created_at",
        "filled_at",
        "exit_at",
        "exit_price",
        "realized_pnl",
        "unrealized_pnl",
        "mark_price",
        "mark_at",
        "paper",
        "notes",
    ]:
        data.pop(field)
    path = _write(tmp_path / "orders.json", [data])

    (order,) = load_orders(path)

    assert order.mark_price is None
    assert order.exit_price is None
    assert order.realized_pnl == 0.0
    assert order.unrealized_pnl == 0.0
    assert order.notes == []


def test_legacy_account_without_position_lists_loads(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "account.json",
        {"initial_balance": 1000.0, "current_balance": 900.0},
    )

    account = load_account(path)

    assert account.current_balance == 900.0
    assert math.isfinite(account.equity())
    assert account.open_count == 0
    assert account.closed_count == 0
