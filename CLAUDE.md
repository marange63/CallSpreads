# CLAUDE.md

Guidance for working in this repo. Keep it current when conventions change.

## What this is
A personal tool for finding and monitoring **call debit spreads** on US equity/index options. Two surfaces:
- **Call Spread Finder** (`/`) — scans an option chain for spreads matching filters, with a results table and a theoretical-value chart.
- **My Positions monitor** (`/positions`) — live risk/P&L for saved positions.

## Architecture
- **Everything is in one file: `spx_call_spread_finder.py`** (~3500 lines). Pure stdlib `http.server`/`socketserver`; the two pages are hand-written HTML/CSS/JS held as big Python strings:
  - `HTML_PAGE` → the Finder (`/`).
  - `POSITIONS_PAGE` → the monitor (`/positions`).
- `main.py` is an **unused** PyCharm stub — ignore it.
- Backend entry points: `fetch_and_find_spreads(...)` (Finder scan) and `fetch_position_quotes(...)` (monitor). Shared Black-Scholes toolkit (`bs_call_price`, `bs_call_delta`, `bs_gamma`, `bs_vega`, `bs_call_theta`, `implied_vol`) is defined once near the top and reused by both.

## Running & testing
- Launch: `Launch CallSpreads.bat`, or `python spx_call_spread_finder.py`. It **auto-detects a free port** starting at 8765 and prints a banner with the URL.
- The startup banner uses box-drawing characters; `main()` reconfigures stdout/stderr to UTF-8 so redirecting output to a file works. If you invoke Python yourself in a cp1252 shell, set `PYTHONIOENCODING=utf-8`.
- Quotes come from **Yahoo via `yfinance`** (auto-installed if missing). Option quotes are **~15 min delayed**; the underlying `regularMarketPrice` is fresher.
- **Yahoo rate-limits repeated option-chain scans** — heavy back-to-back Finder scans during testing will start hanging/returning empty. Space them out; prefer verifying page structure (serving `/` doesn't hit Yahoo) over repeated live scans.
- When testing, parse the port from the banner. Launched server processes hold ports (8765+); kill stray ones when done so they don't accumulate.

## Data files (all gitignored — runtime state)
- `positions.json` — saved monitor positions.
- `templates.json` — saved Finder parameter sets.
- `beta_cache.json` — per-day beta cache for the monitor's beta column.

## Conventions & gotchas
- **`HTML_PAGE` is a raw string**; `{RISK_FREE_RATE_PCT}` is substituted with `.replace()` at serve time — it is *not* an f-string, so literal `{}` in the page's JS is safe.
- **Units:** premium/dollar UI fields are **dollars = option points × 100** (the 100x multiplier). Width is in **points**. Mixing them bites: e.g. a $9,000 *min premium* against a 100-pt spread (whose max value is only $10,000) returns nothing.
- **Per-contract vs. position-level:** many metrics exist in both forms. Screen a spread's *character* on the **per-contract** value (independent of size): `Min Net Delta` filters per-contract Δ/Contract; `Min Short-Leg Price` filters the short call's quoted price. `contracts` is auto-sized = `floor(Max Premium ÷ cost-per-spread)`, so position-level values scale with budget — don't filter structure on them.

## Finder form (organized by function)
- **Scan** (Ticker, Expirations) · **Filters** (screen which spreads return) · **Output & scenario** (shape result *columns*, don't screen) · **Model inputs** (Rate, Commission — collapsed, feed the math).
- Two easily-confused knobs, different units: **Recovery move %** = an *underlying* move (drives the "P&L X% $" column); **P&L target (% of cost)** = a *return on premium* (drives the path-aware "P(+X%)" Monte-Carlo probability column). Neither screens spreads.

## Monitor (`/positions`)
- Compact detail table (merged columns: Position, Quote, Entry/Liq, Cost/Value) + Portfolio Summary.
- **Adj P&L** = haircut on gains + round-trip commission. **Daily Theo P&L** = BS reprice for the underlying's 1-day move. **Own-vol ±Nσ** and **β·index ±Nσ** columns use one shared reprice engine; **beta** is a 2yr daily regression vs. the index (default `^GSPC`), cached daily, with the index's σ from its VIX-family implied vol. The **Scenario P&L** summary box combines these.
- Global controls: refresh interval, Adj P&L haircut %, profit target %, beta index, **Std devs** (σ multiplier driving all ±σ columns).

## Git & workflow
- Solo repo, commits go **direct to `main`** (remote `github.com/marange63/CallSpreads`).
- `.claude/` is gitignored — it holds local settings and two personal slash commands: **`/pos`** (scope a request to the My Positions page) and **`/finder`** (scope to the Finder page).
