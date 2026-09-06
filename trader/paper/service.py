"""Paper trading service — high-level interface.

Sprint 5G: PaperTradingService is the single entry point for paper
trading operations. It internally owns a PaperAccount, a PaperBroker,
and uses the storage and executor modules to provide a simple API that
hides broker/account details from callers.

The service is a thin orchestration layer — it does not duplicate
any logic from the underlying paper modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from trader.paper.account import PaperAccount
from trader.paper.broker import PaperBroker
from trader.paper.executor import (
    close_paper_position,
    get_open_position,
    get_paper_summary,
    place_paper_trade,
)
from trader.paper.storage import load_account as storage_load_account
from trader.paper.storage import save_account as storage_save_account
from trader.paper.summary import PaperAccountSummary

if TYPE_CHECKING:
    from collections.abc import Sequence

    from trader.paper.order import PaperOrder
    from trader.trade import Trade


class PaperTradingService:
    """High-level paper trading service.

    This class is the single interface for paper trading operations.
    It internally owns a PaperAccount and a PaperBroker, and uses the
    executor and storage modules for placement, closing, and persistence.

    The service hides broker/account details from callers. All position
    and account state is accessed through this service.

    Usage:
        service = PaperTradingService.create_new_account(1000.0)
        order = service.place_trade(approved_trade)
        summary = service.summary()
        service.save_account("paper_state.json")
    """

    def __init__(self, account: PaperAccount, broker: PaperBroker) -> None:
        """Initialize the service with an existing account and broker.

        This constructor is intended for internal use and for the
        factory methods. Callers should use
        :meth:`create_new_account` or :meth:`load_account` instead.

        Args:
            account: The PaperAccount that holds state.
            broker: The PaperBroker wired to that account.
        """
        self._account = account
        self._broker = broker

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def create_new_account(cls, initial_balance: float = 1000.0) -> PaperTradingService:
        """Create a new paper trading service with a fresh account.

        Args:
            initial_balance: The starting balance for the new account.

        Returns:
            A new PaperTradingService with an empty account and a fresh
            PaperBroker.
        """
        account = PaperAccount(
            initial_balance=initial_balance,
            current_balance=initial_balance,
        )
        broker = PaperBroker(account)
        return cls(account, broker)

    @classmethod
    def load_account(cls, path: str | Path) -> PaperTradingService:
        """Load a paper trading service from a saved JSON account file.

        The loaded account is wired to a new PaperBroker so that
        placement and closing work normally after loading.

        Args:
            path: Path to the JSON file saved by :meth:`save_account`.

        Returns:
            A new PaperTradingService with the loaded account and a
            fresh PaperBroker.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid data.
        """
        account = storage_load_account(path)
        broker = PaperBroker(account)
        return cls(account, broker)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_account(self, path: str | Path) -> None:
        """Save the current account state to a JSON file.

        Args:
            path: File path to write the JSON to. Parent directories are
                created automatically if they do not exist.
        """
        storage_save_account(self._account, path)

    # ------------------------------------------------------------------
    # Trading operations
    # ------------------------------------------------------------------

    def place_trade(self, trade: Trade) -> PaperOrder | None:
        """Place an approved Trade as a paper order.

        The trade must be approved (``trade.approved is True``) and must
        pass paper trading validation. Unapproved or invalid trades are
        rejected without placing an order.

        Args:
            trade: The Trade to place.

        Returns:
            The placed PaperOrder on success, or None when the trade is
            not approved or fails validation.
        """
        return place_paper_trade(self._broker, trade)

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        status: PaperOrderStatus,
        exit_at: str | None = None,
    ) -> PaperOrder | None:
        """Close an open paper position for a symbol.

        Args:
            symbol: The symbol to close (case-insensitive).
            exit_price: The price at which the position is closed.
            status: The terminal status for the closed order. Must be
                provided explicitly.
            exit_at: Optional timestamp for the exit.

        Returns:
            The closed PaperOrder on success, or None when no open
            position exists for the symbol.
        """
        return close_paper_position(
            self._broker, symbol, exit_price, status, exit_at
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def summary(self) -> PaperAccountSummary:
        """Build a typed summary of the current paper account state.

        Returns:
            A PaperAccountSummary with open positions, closed count,
            and realized/unrealized PnL totals.
        """
        return get_paper_summary(self._broker)

    def open_positions(self) -> list[PaperOrder]:
        """Return a copy of the currently open paper positions.

        Returns:
            A list of PaperOrder objects currently open. The returned
            list is a copy; modifications to it do not affect the
            service's internal state.
        """
        return list(self._broker.get_open_positions())

    def closed_positions(self) -> list[PaperOrder]:
        """Return a copy of the closed paper positions.

        Returns:
            A list of PaperOrder objects that have been closed. The
            returned list is a copy; modifications to it do not affect
            the service's internal state.
        """
        return list(self._account.closed_positions)

    def get_position(self, symbol: str) -> PaperOrder | None:
        """Return the open position for a symbol, or None if none exists.

        Args:
            symbol: The symbol to look up (case-insensitive).

        Returns:
            The open PaperOrder for the symbol, or None when no open
            position exists.
        """
        return get_open_position(self._broker, symbol)
