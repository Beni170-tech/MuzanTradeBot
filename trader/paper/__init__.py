"""
MuzanTradeBot - Paper Trading Foundation (Sprint 5A)

Paper trading data models and account state.
This package is the foundation for the paper trading engine.
Execution, persistence, feed, and Telegram integration
are intentionally out of scope for Sprint 5A.
"""

from trader.paper.order import PaperOrderStatus, PaperOrder
from trader.paper.account import PaperAccount

# Sprint 8B: expose max open positions limit helpers at package level.
__all__ = [
    "PaperOrderStatus",
    "PaperOrder",
    "PaperAccount",
    "MaxOpenPositionsResult",
    "MaxOpenPositionsMetrics",
    "MaxOpenPositionsManager",
]


def __getattr__(name):
    if name in {"MaxOpenPositionsResult", "MaxOpenPositionsMetrics", "MaxOpenPositionsManager"}:
        import sys as _sys
        import pathlib as _pathlib
        project_root = _pathlib.Path(__file__).resolve().parent.parent
        if str(project_root) not in _sys.path:
            _sys.path.insert(0, str(project_root))
        from trader.paper.max_open_positions import (
            MaxOpenPositionsResult as _Result,
            MaxOpenPositionsMetrics as _Metrics,
            MaxOpenPositionsManager as _Manager,
        )
        if name == "MaxOpenPositionsResult":
            return _Result
        if name == "MaxOpenPositionsMetrics":
            return _Metrics
        return _Manager
    raise AttributeError(name)

