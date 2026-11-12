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

## Paper Position Marks, Equity and the Open-Position Cap

Open paper positions record a market mark (`mark_price`), the last
validated market price seen for that position:

- The background monitor records the fresh price it already fetched for
  the exact position it is checking, and persists the paper account.
- `/analyze` records the analysis price for an existing position in the
  same symbol. No extra market request is made.
- Unrealized PnL and paper equity are valued at the recorded marks, for
  LONG and SHORT positions. A position without a mark contributes 0.0;
  the entry price is never substituted, so a missing mark can never look
  like a flat position. Marks are saved in the paper account JSON.
- Telegram command handlers never fetch prices, so `/paper`,
  `/positions` and `/stats` label marked values as "last recorded
  marks" and report missing marks as "unavailable (no market mark
  recorded yet)" instead of a misleading `$0.00`.

`PORTFOLIO_MAX_OPEN_POSITIONS` counts open paper positions through the
active execution backend (`PaperBroker.get_open_positions()`), so the
configured portfolio-wide cap blocks new paper trades once the limit is
reached. The count and cap are also reported by `/stats`.

Position sizing follows the paper account: each new paper trade risks
`RISK_PERCENT` of the account's current equity (settled balance plus
unrealized PnL at recorded marks), so realized losses and floating
results shrink or grow the risk budget. If the paper account cannot be
read, sizing fails closed and the trade is rejected rather than sized
from a stale configured balance. `ACCOUNT_BALANCE` remains the starting
balance for a new paper account and the sizing fallback only when paper
trading is disabled.

### /analyze Pricing

`/analyze` keeps using historical hourly candles for trend, signal,
confidence and the indicator snapshot, but it prices the trade from a
fresh quote fetched with the existing simple-price helper. The current
quote is used for the trade price, position sizing, stop-loss and
take-profit levels, and the stop-loss/take-profit check against an
existing position in that symbol. The last historical close is never
used as an execution price: if the quote is missing, non-numeric,
non-finite, zero or negative, the trade is not placed and the reply
explains that the current market quote is unavailable. Only one quote
request is made per `/analyze`.

### Multiple positions on one symbol

The bot may hold several open paper positions on the same symbol. The
background monitor treats each position independently: it records that
position's own market mark and runs stop-loss, take-profit and
trailing-stop checks against that exact position, closing it by order
ID. Closing or trailing one position never touches another position
with the same symbol.

`/close <SYMBOL> <EXIT_PRICE>` keeps working when exactly one position
exists for that symbol. When several positions share the symbol, the
bot closes nothing and replies with each position's order ID; re-run
`/close <SYMBOL> <EXIT_PRICE> <ORDER_ID>` to close exactly that
position. An unknown or mismatched ID closes nothing, and a manual
close never closes more than one position.

### Stale market marks

Every recorded mark also stores a UTC timestamp (`mark_at`). Before a
new `/analyze` trade is sized, every open paper position must have a
valid mark price and a timestamp no older than **two monitor polling
intervals** (`2 × PAPER_MONITOR_INTERVAL_SECONDS`; 90 seconds with the
default 45-second interval). One missed monitor cycle plus scheduling
jitter is tolerated. When the background monitor is disabled
(`PAPER_MONITOR_INTERVAL_SECONDS <= 0`), the same 90-second default
applies, because nothing refreshes marks in the background.

If any open position has a missing mark, a missing timestamp, or a mark
older than the threshold, `/analyze` refuses to size the trade. It
never falls back to `ACCOUNT_BALANCE`, an entry price, or an old mark,
and the Telegram reply says the trade was blocked because market-price
data is stale or unavailable. Accounts saved before timestamps existed
load normally, but their unverified marks cannot be used for sizing
until the monitor records a fresh price. Failed market reads never
refresh a mark's timestamp.
