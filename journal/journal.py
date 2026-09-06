"""
MuzanTradeBot - Trade Journal

Records every validated trade decision to a plain-text log file so
the bot's activity can be reviewed after the fact. Guardian Mode
relies on this journal for accountability and auditing.
"""

from datetime import datetime


def log_trade(symbol: str, decision: str, risk_verdict: str) -> None:
    """Append one trade entry to the trade journal file.

    Args:
        symbol: The cryptocurrency symbol (e.g. "BTC").
        decision: The decision from the decision engine.
        risk_verdict: The verdict from the risk manager.

    Returns:
        None. Writes a line to logs/trade_journal.txt in the format:
        YYYY-MM-DD HH:MM:SS | SYMBOL | Decision: X | Risk: Y
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {symbol} | Decision: {decision} | Risk: {risk_verdict}\n"

    with open("logs/trade_journal.txt", "a", encoding="utf-8") as journal:
        journal.write(line)