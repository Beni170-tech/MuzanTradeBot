# MuzanTradeBot

A professional Telegram crypto trading assistant built to protect capital first.

## Purpose

MuzanTradeBot is designed to provide automated trading assistance via Telegram with a primary focus on capital preservation and risk management. The bot will execute trades only after passing rigorous safety checks.

## Paper Trading Only

Every execution path in this repository is simulated paper trading. No
live order is ever placed, and there is no live execution backend:

- `EXECUTION_MODE` accepts only `PAPER`. An unrecognized value is an
  unsafe configuration and the bot refuses to start.
- `PAPER_TRADING` remains the runtime switch. A `PAPER_TRADING=true`
  configuration must also use `EXECUTION_MODE=PAPER`.
- Telegram replies label the results as paper trades, and `/report`
  ends with a paper-only notice.

## Current Version

0.1.0 - Foundation Ready

## Installation

Requirements: Python 3.12 (the version this project is developed and
tested with), `python-telegram-bot>=22.0` and `python-dotenv>=1.0.0`.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env             # then edit .env
python main.py
```

`main.py` refuses to start when configuration validation finds an error
(a negative limit, a non-finite number, a malformed value for any risk
setting, a zero `RISK_PERCENT`), when `EXECUTION_MODE` is not valid for
the current `PAPER_TRADING` intent, or when `TELEGRAM_BOT_TOKEN` is
missing. Warnings for disabled safety limits are printed without
blocking startup. Values are never printed, only setting names and the
reason.

## Configuration

Settings are read from `.env` in the project root (see `.env.example`
for the annotated reference). The file is git-ignored; never commit real
credentials. `/settings` and `/status` show configuration and access
state but never a token, API key or authorized id.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | - (required) | Bot token from @BotFather. |
| `TELEGRAM_ALLOWED_CHAT_IDS` | empty | Authorized user and/or chat ids. Empty locks the bot. |
| `EXECUTION_MODE` | `PAPER` | Execution backend. Only `PAPER` is valid. |
| `PAPER_TRADING` | `true` | Runtime paper-trading switch. |
| `GUARDIAN_MODE` | `true` | Guardian Mode label and safety posture. |
| `ACCOUNT_BALANCE` | `1000` | Starting balance for a new paper account. |
| `RISK_PERCENT` | `2.0` | Percent of current paper equity risked per trade. |
| `STOP_LOSS_PERCENT` | `2.0` | Stop-loss distance from the entry price. |
| `TAKE_PROFIT_PERCENT` | `4.0` | Take-profit distance from the entry price. |
| `PORTFOLIO_RISK_LIMIT` | 6% of balance | Maximum total open risk. |
| `PORTFOLIO_EXPOSURE_LIMIT` | 300% of balance | Maximum total notional exposure. |
| `PORTFOLIO_DAILY_LOSS_LIMIT` | 3% of balance | Maximum combined realized and floating daily loss. |
| `PORTFOLIO_MAX_OPEN_POSITIONS` | `3` | Maximum open positions. |
| `PORTFOLIO_MAX_POSITIONS_PER_SYMBOL` | `1` | Maximum open positions per symbol. |
| `PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL` | 200% of balance | Maximum notional exposure per symbol. |
| `PAPER_MONITOR_INTERVAL_SECONDS` | `45` | Background monitor cycle; `<= 0` disables it. |
| `TELEGRAM_COMMAND_RATE_LIMIT_PER_WINDOW` | `12` | Commands per chat per window; `<= 0` disables. |
| `TELEGRAM_COMMAND_RATE_WINDOW_SECONDS` | `10` | Rate-limit window length; `<= 0` disables. |
| `ALERTS_ENABLED` | `false` | Push-notification master switch. |
| `TELEGRAM_ALERT_CHAT_ID` | empty | Chat alerts are sent to. |

Portfolio limits (`PORTFOLIO_RISK_LIMIT`,
`PORTFOLIO_EXPOSURE_LIMIT`, `PORTFOLIO_DAILY_LOSS_LIMIT`,
`PORTFOLIO_MAX_OPEN_POSITIONS`, `PORTFOLIO_MAX_POSITIONS_PER_SYMBOL`,
`PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL`) are enabled by default with
conservative values derived from `ACCOUNT_BALANCE`; set one to `0` to
disable that gate explicitly. See
[Guardian Limits](#guardian-limits-and-entry-gates).

`ACCOUNT_BALANCE`, `RISK_PERCENT`, `STOP_LOSS_PERCENT` and
`TAKE_PROFIT_PERCENT` are parsed strictly, because every limit and every
position size is derived from them. Unset or blank keeps the default
shown above; a malformed or non-finite value (`10O0`, `2%`, `nan`) is a
configuration error that names the variable and blocks startup, instead
of silently reverting to a default that would rescale the risk budget.
Exchange API keys (`BINANCE_*`, `BYBIT_*`, `KRAKEN_*`) may be set for
future use; no live backend consumes them.

## Telegram Authorization

Access is denied by default and enforced in one place:

1. Set `TELEGRAM_ALLOWED_CHAT_IDS` to a comma-separated list of
   Telegram user ids and/or chat ids, for example
   `TELEGRAM_ALLOWED_CHAT_IDS=123456789,-1001234567890`. Use
   @userinfobot to find your user id; a group is listed by its negative
   chat id.
2. A private chat can be authorized by user id, a group by chat id: an
   update is allowed when **either** its effective user id or its
   effective chat id is listed.
3. Leave the variable empty and the bot stays **locked**: no command,
   callback or account read runs.
4. Invalid entries are ignored rather than treated as a wildcard. If no
   entry parses to a usable id, the bot remains locked.

A single `TypeHandler` authorization gate is installed in handler group
`-1`, before any command handler, so new commands are covered
automatically. Unauthorized updates receive
`⛔ Unauthorized. This bot is private.` and are stopped from
propagating; the denial identifies no account, limit or configuration
detail. The allowlist is re-read on every update, so a corrected `.env`
plus restart is enough to change access. `/settings` and `/status`
report only `locked` or a count of authorized ids, never the ids.

## Commands

Every command except `/stats` shares the per-chat rate limit above.

| Command | Description |
| --- | --- |
| `/start` | Show Guardian Mode and paper-trading status. |
| `/help` | List the available commands. |
| `/ping` | Liveness check. |
| `/version` | Bot version. |
| `/analyze <COIN>` | Analyze a coin (BTC, ETH, SOL), price it from a fresh quote, run the Guardian checks and place a paper trade only if every check passes. |
| `/paper` | Paper account summary: balance, equity, open and closed counts. |
| `/positions` | Open paper positions with entry price, quantity and the last recorded market mark (reported as unavailable when none exists). |
| `/history` | Closed paper trades. |
| `/close <SYMBOL> <EXIT_PRICE> [ORDER_ID]` | Close a paper position at an explicit exit price. `ORDER_ID` is required when several positions share the symbol. |
| `/stats` | Paper performance dashboard, including the entry gates and current limits. |
| `/settings` | Execution mode, Guardian limits, authorization state and alert configuration. |
| `/report` | Period report: account, all time, today, last 7 days and per-symbol results. Entry-gate metrics stay on `/stats`. |
| `/pause [reason]` | Block new paper entries; open positions stay monitored. |
| `/resume` | Allow new paper entries again. |
| `/status` | Operational status: runtime, new entries, monitor, paper account, journal and rate limit. |

`/report` and `/status` render from `dashboard/report.py` and
`dashboard/status.py`; the handlers only gather values, so the reports
never read account state or place orders themselves.

## Guardian Limits and Entry Gates

`/analyze` runs these checks in this order and stops at the first
rejection:

1. Historical analysis, signal and confidence from hourly candles.
2. A fresh market quote for the trade price (never the last candle
   close).
3. Risk validation of the decision and position sizing from current
   paper equity.
4. Stop-loss/take-profit checks on any **existing** position in the
   symbol, then trailing-stop updates.
5. The global pause control (see below).
6. Portfolio entry gates: portfolio risk -> portfolio exposure -> daily
   loss -> maximum open positions -> symbol concentration (position
   count, then exposure).
7. Paper order placement.

| Limit | Meaning when above 0 |
| --- | --- |
| `PORTFOLIO_RISK_LIMIT` | Maximum total dollar risk across open positions. |
| `PORTFOLIO_EXPOSURE_LIMIT` | Maximum total notional exposure. |
| `PORTFOLIO_DAILY_LOSS_LIMIT` | Maximum combined realized and floating loss per calendar day; new entries halt until the next day. |
| `PORTFOLIO_MAX_OPEN_POSITIONS` | Maximum open positions across all symbols. |
| `PORTFOLIO_MAX_POSITIONS_PER_SYMBOL` | Maximum open positions on one symbol. |
| `PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL` | Maximum notional exposure on one symbol. |

Limits are configured by environment variable and follow one rule:

- **Unset or blank** uses the conservative default shown below. Defaults
  scale with `ACCOUNT_BALANCE` (default 1000), so a smaller account gets
  proportionally smaller dollar limits.
- **An explicit `0`** disables that gate entirely, so it performs no
  account reads.
- **A negative, non-finite or non-numeric value** is a configuration
  error that names the variable and blocks startup. A mistyped limit
  never silently falls back to a default or to disabled.

| Limit | Default |
| --- | --- |
| `PORTFOLIO_RISK_LIMIT` | 6% of `ACCOUNT_BALANCE` |
| `PORTFOLIO_EXPOSURE_LIMIT` | 300% of `ACCOUNT_BALANCE` |
| `PORTFOLIO_DAILY_LOSS_LIMIT` | 3% of `ACCOUNT_BALANCE` |
| `PORTFOLIO_MAX_OPEN_POSITIONS` | 3 |
| `PORTFOLIO_MAX_POSITIONS_PER_SYMBOL` | 1 |
| `PORTFOLIO_MAX_EXPOSURE_PER_SYMBOL` | 200% of `ACCOUNT_BALANCE` |

The exposure defaults leave room for the position counts they accompany:
with the default 2% risk and 2% stop, one trade risks about 2% of equity
and uses about its full value as notional.

Stop-loss and take-profit levels are mandatory on every paper trade, and
a disabled or stale market mark blocks new sizing rather than falling
back to an entry price or the configured balance.

The daily-loss brake tests the whole day, not just settled trades: it
counts realized PnL for the calendar day **plus** floating PnL on open
positions at their recorded marks. A position without a usable mark is
never treated as zero loss — the brake fails closed and new entries are
halted until every open position can be priced, naming the affected
symbols. This keeps the daily budget meaningful before positions are
closed, and it never lets an unreadable account look profitable.

### Migrating an existing `.env`

An explicit value in `.env` always wins. Configurations that already set
a limit to `0` keep that gate disabled after upgrading; only variables
that are missing or blank change behaviour and pick up the conservative
default. Run `/settings` after restarting: it reports each limit and how
many of the six backstops are enabled.

Startup also prints a summary warning when all six limits are disabled,
so an all-off Guardian posture is visible rather than silent.

## Pausing and Resuming

`/pause [reason]` and `/resume` write a small control file,
`logs/paper_pause.json`, so the state survives a restart:

```json
{"paused": true, "reason": "maintenance", "updated_at": "2026-11-19 10:00:00"}
```

- **Pause blocks new entries only.** Existing positions keep being
  monitored; stop-loss, take-profit, trailing stops and `/close` all
  keep working.
- Writes are atomic. If the state cannot be written, the command replies
  that it could not save the pause state and paper trading is unchanged
  instead of claiming success.
- A missing file means no pause was requested. Corrupt JSON, an
  unexpected shape or a non-boolean `paused` field is treated as
  **paused** (fail closed), and `/status` shows `BLOCKED (pause control
  state is unreadable)`; fix or remove the file, then use `/resume`.
- The reason is collapsed to one line and truncated at 200 characters.

## Trade Journal and State Files

All state lives under `logs/`, which is git-ignored:

| File | Contents |
| --- | --- |
| `logs/trade_journal.jsonl` | Structured trade journal, one JSON object per line. |
| `logs/trade_journal.jsonl.1` .. `.5` | Rotated backups; `.1` is the newest. |
| `logs/paper_state.json` | Paper account, positions and marks. |
| `logs/paper_pause.json` | Pause control (see above). |
| `logs/trade_journal.txt` | Legacy plain-text journal, left untouched. |

Each journal record is `{"timestamp", "symbol", "decision",
"risk_verdict"}` for every validated trade decision, including rejected
ones. The journal rotates before appending once the active file would
exceed 1 MB, keeping five backups. A write failure is logged and
reported through `/status` (`Last Error`), never raised: a broken
journal cannot block an `/analyze` reply or look like a placed trade.
`/status` shows the path, size, backup count and last write time.

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

`requirements-dev.txt` adds `pytest` and `pytest-asyncio`, which
provides the async test fixtures. The suite runs without Telegram,
network access or a real `.env`: tests redirect state files to temporary
directories and monkeypatch the existing handler seams.

## Project Structure

- `brain/` - Core AI and decision-making logic
- `market/` - Market data and analysis
- `trader/` - Trade execution logic (`trader/paper/` holds the paper
  account, gates, monitor, pause control and read models)
- `alerts/` - Notification and alert system
- `risk/` - Risk management and safety checks
- `journal/` - Structured trade journal
- `dashboard/` - `/report` and `/status` rendering
- `telegram_bot/` - Handlers, authorization gate, rate limiting and
  bot wiring
- `config/` - Configuration and startup validation
- `logs/` - Application state and logs (git-ignored)
- `tests/` - Test suite

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

The exit price must be a finite number greater than 0. Zero,
negatives, NaN, infinity, booleans and non-numeric input are rejected
before anything is looked up or changed: the position stays open and
the account balance, realized P&L, saved state and alerts are left
untouched. The same rule is enforced by the broker, order and executor
close functions, so a direct caller cannot bypass it.

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
