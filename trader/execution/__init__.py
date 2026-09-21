"""Execution contract package for MuzanTradeBot.

Sprint 10.1/10.2: backend-agnostic contract for order placement, order
closure, and portfolio-state queries, with a small accessor layer so
service objects can expose their underlying execution backend without
changing existing placement behavior.
"""

from trader.execution.contract import ExecutionBackend

__all__ = ["ExecutionBackend"]
