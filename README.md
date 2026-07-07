# CallSpreads

Bull call spread finder and position monitor using live Yahoo Finance option data.

## Features

- **Spread Finder** — scans an underlying's option chain for bull call spread
  candidates, ranked by leverage, with Greeks, breakeven, and 1σ/2σ P&L columns.
- **My Positions monitor** — tracks saved call spreads with live quotes, P&L
  (raw and haircut-adjusted), per-position Greeks, and a portfolio summary
  (net Δ+Γ 1σ P&L, net theta, net vega).

## Requirements

- Python 3.9+
- Dependencies in `requirements.txt` (`yfinance` is also auto-installed on first run if missing)

```bash
pip install -r requirements.txt
```

## Running

```bash
python spx_call_spread_finder.py
```

Then open <http://localhost:8765> in your browser.

## Data files

The app stores state in JSON files next to the script. These are **not** tracked
in git (they contain personal position data) and are created automatically:

- `positions.json` — your saved positions
- `templates.json` — saved spread templates

## Extra scripts

- `qqq_rolling_63d.py` — plots QQQ's rolling 63 trading-day cumulative return
  and saves `qqq_rolling_63d.png`.
