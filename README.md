# MuzanTradeBot

A professional Telegram crypto trading assistant built to protect capital first.

## Purpose

MuzanTradeBot is designed to provide automated trading assistance via Telegram with a primary focus on capital preservation and risk management. The bot will execute trades only after passing rigorous safety checks.

## Current Version

0.1.0 - Foundation Ready

## Project Structure

- `brain/` - Core AI and decision-making logic
- `market/` - Market data and analysis
- `trader/` - Trade execution logic
- `alerts/` - Notification and alert system
- `risk/` - Risk management and safety checks
- `journal/` - Trade journaling and logging
- `dashboard/` - Monitoring and status dashboard
- `config/` - Configuration management
- `logs/` - Application logs (git-ignored)
- `tests/` - Test suite
- `docs/` - Documentation

## Background Paper Position Monitor

When paper trading is enabled, the bot runs a background monitor that
checks every open paper position on the interval configured by
`PAPER_MONITOR_INTERVAL_SECONDS` (default 45 seconds). It updates trailing
stops and closes positions when stop-loss or take-profit levels are
reached.

Each cycle obtains a fresh USD market price from CoinGecko's simple-price
endpoint for the supported symbols (BTC, ETH, SOL), reusing the existing
market-data integration. The synchronous HTTP lookup runs in a worker
thread, so a slow request cannot block Telegram polling. If a price
cannot be retrieved or is not a valid positive number, that position is
skipped for the cycle and retried on the next one; the entry price is
never used as a substitute.

On shutdown, the monitor task is cancelled and awaited from the Telegram
application's `post_shutdown` hook while the event loop is still open.

All monitoring actions remain simulated paper-trading operations. No live
orders are placed.
