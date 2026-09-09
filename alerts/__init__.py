"""
MuzanTradeBot - Alerts package (Sprint 9A)

Notification and alert system. Exposes the public alert API:

    - AlertManager: one semantic method per alert event, fail-safe.
    - TelegramNotifier / NullNotifier: message delivery.
    - Message builders in alerts.messages: pure text builders.

Alert events (Sprint 9A scope):
    - Paper order placed
    - Stop-loss triggered
    - Take-profit triggered
    - Trailing stop updates
    - Guardian/risk-limit trade rejections
    - Daily loss breaker activation
    - Maximum open positions reached
"""

from alerts.manager import AlertManager, build_alert_manager
from alerts.notifier import NullNotifier, TelegramNotifier

__all__ = [
    "AlertManager",
    "build_alert_manager",
    "NullNotifier",
    "TelegramNotifier",
]
