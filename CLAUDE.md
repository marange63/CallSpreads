# CLAUDE.md

Guidance for working in this repo. Keep it current when conventions change.

## What this is
A personal tool for finding and monitoring **call debit spreads** on US equity/index options. Three surfaces:
- **Call Spread Finder** (`/`) — scans an option chain for spreads matching filters, with a results table, a composite **Score** column, and a theoretical-value chart.
- **My Positions monitor** (`/positions`) — live risk/P&L for saved positions.
- **Scatter** (`/scatter`) — plots the Finder's current results as two side-by-side, independently-configurable scatter panels; opened from the Finder and kept in sync with it (see Scatter below).

## Architecture
- **Two Python files.** `spx_call_spread_finder.py` (~4800 lines) is the app: pure stdlib `http.server`/`socketserver`; the three pages are hand-written HTML/CSS/JS held as big Python strings:
  - `HTML_PAGE` → the Finder (`/`).
  - `POSITIONS_PAGE` → the monitor (`/positions`).
  - `SCATTER_PAGE` → the Scatter view (`/scatter`).
- **`price_sources.py` is the market-data seam**: `PriceSource` ABC + `YahooSource` + registry (`SOURCE_CLASSES`/`SOURCES`, `get_source`/`set_active_source`/`register_sources`). It owns the yfinance import/auto-install; **the main file must contain no direct `yf.*` calls**. Contract: all methods take Yahoo-style symbols (`^SPX`, `^VIX` — subclasses translate via `map_symbol`) and return yfinance-shaped data (Yahoo column/key names). A new vendor = subclass implementing `get_expirations` + `get_option_chain` (Tier 1) + one `SOURCE_CLASSES` entry; Tier-2 reference data (`get_daily_closes`, `get_daily_closes_batch`, `get_previous_close`) defaults to delegating to Yahoo. Source failures surface as errors — deliberately no silent Yahoo fallback. The active source is server-global: dropdown on both pages, `GET/POST /api/source`, persisted in `sources.json`.
- `main.py` is an **unused** PyCharm stub — ignore it.
- Backend entry points: `fetch_and_find_spreads(...)` (Finder scan) and `fetch_position_quotes(...)` (monitor). Shared Black-Scholes toolkit (`bs_call_price`, `bs_call_delta`, `bs_gamma`, `bs_vega`, `bs_call_theta`, `implied_vol`) is defined once near the top and reused by both.

## Running & testing
- Launch: `Launch CallSpreads.bat`, or `python spx_call_spread_finder.py`. It **auto-detects a free port** starting at 8765 and prints a banner with the URL.
- The startup banner uses box-drawing characters; `main()` reconfigures stdout/stderr to UTF-8 so redirecting output to a file works. If you invoke Python yourself in a cp1252 shell, set `PYTHONIOENCODING=utf-8`.
- Quotes come from **Yahoo via `yfinance`** (auto-installed if missing). Option quotes are **~15 min delayed**; the underlying `regularMarketPrice` is fresher.
- **Yahoo rate-limits repeated option-chain scans** — heavy back-to-back Finder scans during testing will start hanging/returning empty. Space them out; prefer verifying page structure (serving `/` doesn't hit Yahoo) over repeated live scans.
- **Test mode** is the intended way to test with the market closed: it caches each underlying's option chain so repeated Finder scans and the monitor's 30s auto-refresh reuse frozen data instead of re-fetching. Toggle it live via the **LIVE/TEST badge** on both pages; it sets `?test=1` on the API calls. See the Test/Production mode gotcha below.
- **Offline verification (preferred over live scans):** `python -c "import ast; ast.parse(open('spx_call_spread_finder.py',encoding='utf-8').read())"` for Python; for the page JS, extract the `<script>` blocks (substituting placeholders — `{RISK_FREE_RATE_PCT}` for the Finder; `__PNL_CHART__`/`__SPREAD_TOOLTIP__` for Scatter) and run `node --check`; and use fake-ticker unit tests for backend logic. This sidesteps Yahoo throttling entirely.
- When testing, parse the port from the banner. Launched server processes hold ports (8765+); kill stray ones when done so they don't accumulate.

## Data files (all gitignored — runtime state)
- `positions.json` — saved monitor positions.
- `templates.json` — saved Finder parameter sets.
- `beta_cache.json` — per-day beta cache for the monitor's beta column.
- `chain_cache.pkl` — disk mirror of the Test-mode option-chain cache (pickled chain snapshots + expiration lists), loaded at startup, deleted by "Clear cache".
- `alerts.json` — Adj P&L alert config (secret random ntfy topic + thresholds) and the once-per-day sent latches; auto-created on first run.
- `sources.json` — active price source + per-source config (future vendor API keys live here). Missing file = Yahoo.

## Conventions & gotchas
- **`HTML_PAGE` is a raw string**; `{RISK_FREE_RATE_PCT}` is substituted with `.replace()` at serve time — it is *not* an f-string, so literal `{}` in the page's JS is safe.
- **`SCATTER_PAGE` is also a raw string** with the same trap: `__PNL_CHART__` and `__SPREAD_TOOLTIP__` are `.replace()`-substituted at serve time with the `PNL_CHART_JS` / `SPREAD_TOOLTIP_JS` constants (self-contained copies of the Finder's BS/chart/tooltip helpers).
- **Test/Production mode (chain caching):** shared helpers `get_option_chain(symbol, exp, test_mode)` / `get_expirations(symbol, test_mode)` / `clear_test_cache()` back a module-level cache **keyed by the active source's name** (`(source, symbol, exp)`), so frozen Yahoo data is never served while another source is active. In test mode each chain and expiration list is fetched **once** and reused; in prod every call re-fetches (from the active `PriceSource`). Wired through both `fetch_and_find_spreads(..., test_mode=)` and `fetch_position_quotes(..., test_mode=)` (incl. the monitor's `get_atm_iv_30d` / `get_index_sigma_1d`). Routes: `?test=1` on `/api/spreads` and `/api/positions/quotes`; `/api/clear_cache` empties it. **The cache is shared across the Finder and monitor**, so "Clear cache" on either page clears both. Beta is separately disk-cached daily; the index σ comes from a small `^VIX`-family quote, not an option chain.
- **The chain cache is disk-persistent** (`chain_cache.pkl`, format v2 = source-prefixed keys; v1 files migrate to `yahoo` keys on load): every `PriceSource` returns chains pre-frozen as plain picklable snapshots (`.calls`/`.puts`/`.underlying`/`fetched_at` — Yahoo's normalizer is `_snap_chain` in `price_sources.py`) and test-mode fetches are write-through saved — but **only chains with ≥1 live two-sided quote** (`_chain_has_live_quotes`), so weekend/after-hours fetches with zeroed bid/ask can't poison the cache (they still cache in memory for the session). `main()` reloads the file at startup; "Clear cache" deletes memory **and** the file. Capture workflow: scan in Test mode while quotes are live (e.g. Friday session) and weekend restarts keep working off that frozen data. **Yahoo zeroes option bid/ask outside market hours (worst on weekends)** — live scans then return no spreads by design (the OTM filter requires `bid>0 & ask>0`).
- **Runtime UI state lives in `localStorage`** (namespaced, per-browser): `finderScoreWeights` / `finderScoreOiTarget` / `finderScoreOpen` (Score panel), `finderTestMode`, `posTestMode`, `finderScatterData` (see Scatter), `betaIndexSymbol`, `sigmaMult`, plus saved templates. Reuse these keys rather than inventing parallel ones.
- **Units:** premium/dollar UI fields are **dollars = option points × 100** (the 100x multiplier). Width is in **points**. Mixing them bites: e.g. a $9,000 *min premium* against a 100-pt spread (whose max value is only $10,000) returns nothing.
- **Per-contract vs. position-level:** many metrics exist in both forms. Screen a spread's *character* on the **per-contract** value (independent of size): `Min Net Delta` filters per-contract Δ/Contract; `Min Short-Leg Price` filters the short call's quoted price. `contracts` is auto-sized = `floor(Max Premium ÷ cost-per-spread)`, so position-level values scale with budget — don't filter structure on them.

## Finder form (organized by function)
- **Scan** (Ticker, Expirations) · **Filters** (screen which spreads return) · **Output & scenario** (shape result *columns*, don't screen) · **Model inputs** (Rate, Commission — collapsed, feed the math).
- Two easily-confused knobs, different units: **Recovery move %** = an *underlying* move (drives the "Return @ +X%" column); **P&L target (% of cost)** = a *return on premium* (drives the path-aware "P(+X%)" Monte-Carlo probability column and its siblings below). Neither screens spreads.
- **Scenario columns are exact BS reprices, not Taylor expansions**: Leverage, Return @ +X% and Return 1σ 1d all come from repricing both legs at the shocked underlying (leg IVs and T held fixed), so they respect the spread's value cap at width. Δ/Prem is the linear part and **Γ/Prem is the full convexity residual** (`Γ/Prem = Return1σ − Δ/Prem`, so the additive identity still holds exactly). The old Taylor dollar fields (`pnl1pct`/`pnl1sigma`/`pnl2sigma`) are gone.
- **MC expectancy columns** (same zero-drift paths as P(+X%), all commission/haircut-aware, `spread_mc_stats`): **P(prof exp)** = terminal Adjusted P&L > 0 at expiration; **Med d→tgt** = median first-touch day among target-hitting paths; **EV %** = expected Adjusted P&L (% of cost) under exit-at-target (hitting paths bank the target, the rest ride to expiry — a conservative no-edge floor, usually negative). Hover popup adds P(−50%) loss tail and **Velocity** (target % ÷ median days). All ride into Scatter automatically.
- **Staleness flags**: each leg carries `lastTrade*` epochs; a leg untraded for > `STALE_TRADE_AGE_SECS` (1 day) vs. the scan flags the row with an amber ⚠ beside Premium (per-leg ages in the popup). The monitor flags individual legs in its Quote column the same way (vs. the snapshot's `quoteTime`).
- **Score column** (first column, default sort): a 0–100 composite for risk-adjusted capital efficiency, computed **client-side over the returned set** (percentile-rank blend of `returnAtMove`, `probTarget`, `gammaPrem`, `rewardRisk`, `thetaPrem`, times a `min(1, worstLegOI/target)` liquidity gate). Weights are tuned live via the collapsible **Score weights** strip above the table (sliders + Reset/Aggressive/Conservative presets), persisted in `localStorage`; re-weighting re-ranks instantly with **no new Yahoo scan**. It ranks *within the current scan* — re-read after any filter change. Full formula + worked example live in the plan file `~/.claude/plans/curently-we-are-getting-idempotent-rocket.md`.

## Monitor (`/positions`)
- Compact detail table (merged columns: Position, Quote, Entry/Liq, Cost/Value) + Portfolio Summary.
- **P&L and Adj P&L come in best/worst pairs** stacked in their cells: **best** = both legs exited at bid/ask mid (`pnl`/`adjPnl` fields), **worst** = pure liquidation, long sold at bid + short bought at ask (`pnlWorst`/`adjPnlWorst`). The **ntfy alerts, P(+X%) calibration, and P&L history all anchor to Best (mid)**; worst is the liquidation floor. Caveat: a leg with no live two-sided quote has mid = `max(bid, ask, last)`, so best degrades to the stale last (the ⚠ staleness flag marks those rows). **Adj P&L** = haircut on gains + round-trip commission. **Daily Theo P&L** = BS reprice for the underlying's 1-day move. **Own-vol ±Nσ** and **β·index ±Nσ** columns use one shared reprice engine; **beta** is a 2yr daily regression vs. the index (default `^GSPC`), cached daily, with the index's σ from its VIX-family implied vol. The **Scenario P&L** summary box combines these, plus a **β-wtd Δ** row = Σ per-position `betaDollarDeltaPer1Pct` (theoretical $ P&L for a +1% index move) — the portfolio's aggregate calibrated leverage in index terms.
- Global controls: refresh interval, Adj P&L haircut %, profit target %, beta index, **Std devs** (σ multiplier driving all ±σ columns).
- **Adj P&L push alerts** (`check_pnl_alerts`): every live (non-test) `/api/positions/quotes` refresh sends an ntfy.sh phone push the first time a position's Adj P&L % rises above each threshold in `alerts.json` (default +2.5%, +5%) — latched once per (position, threshold) per calendar day, so re-crossings don't re-fire until the date rolls. The ntfy topic is a gitignored random secret printed in the startup banner; alerts only fire while something is polling the quotes endpoint (i.e. a monitor tab is open).

## Scatter (`/scatter`)
- Two side-by-side, independently-configurable panels; each point is one spread, with the same row-hover popup as the Finder. Axes/color come from a shared `COLS` list (mirrors the results-table columns, incl. `score`).
- **Data handoff:** the Finder stashes its current results (with live Score) into `localStorage['finderScatterData']` and opens `/scatter` in a new tab. The Scatter page reads that key on load.
- **Live sync:** the Finder re-stashes on every search **and** on Score-weight changes (debounced), each write carrying a changing `ts`. An open Scatter tab listens for the `storage` event and re-reads/re-renders automatically, preserving the user's axis/color picks. So an open Scatter tab tracks the Finder; clicking Scatter again still opens a fresh tab.

## Git & workflow
- Solo repo, commits go **direct to `main`** (remote `github.com/marange63/CallSpreads`).
- `.claude/` is gitignored — it holds local settings and two personal slash commands: **`/pos`** (scope a request to the My Positions page) and **`/finder`** (scope to the Finder page).
