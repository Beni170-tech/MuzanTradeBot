"""Execution contract package for MuzanTradeBot.

Sprint 10.1: backend-agnostic contract for order placement and order
closure so future paper and live backends can be swapped behind the
same interface.
"""

from trader.execution.contract import ExecutionBackend

__all__ = ["ExecutionBackend"]
