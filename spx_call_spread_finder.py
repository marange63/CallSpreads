#!/usr/bin/env python3
"""
Call Spread Finder
==================
Finds bull call spread candidates using live Yahoo Finance option data.

Usage:
    python spx_call_spread_finder.py

Then open http://localhost:8765 in your browser.
"""

import http.server
import json
import math
import os
import sys
import subprocess
import threading
import uuid
import webbrowser
import socketserver
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Auto-install yfinance if missing
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

PORT = 8765

# ---------------------------------------------------------------------------
# Fetch current risk-free rate (3-month Treasury bill yield from Yahoo Finance)
# ---------------------------------------------------------------------------

def fetch_risk_free_rate():
    """Fetch the current 3-month T-bill rate from Yahoo Finance (^IRX).
    Returns the rate as a percentage (e.g. 4.5 for 4.5%). Falls back to 4.5 on failure."""
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            rate = float(hist['Close'].iloc[-1])
            print(f"  Risk-free rate (3-mo T-bill): {rate:.2f}%")
            return round(rate, 2)
    except Exception as e:
        print(f"  Warning: Could not fetch T-bill rate ({e}), using 4.5% default")
    return 4.5

RISK_FREE_RATE_PCT = fetch_risk_free_rate()

# ---------------------------------------------------------------------------
# Black-Scholes helpers (European options — perfect for SPX)
# ---------------------------------------------------------------------------

def norm_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_delta(S, K, T, r, sigma):
    """Black-Scholes call delta for European option."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


def bs_gamma(S, K, T, r, sigma):
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price for European option."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (sensitivity of price to volatility)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * norm_pdf(d1) * math.sqrt(T)


def bs_call_theta(S, K, T, r, sigma):
    """Black-Scholes theta per year for a European call. Divide by 365 for per-day."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return (-S * norm_pdf(d1) * sigma / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm_cdf(d2))


def implied_vol(price, S, K, T, r, max_iter=50, tol=1e-6):
    """
    Back out implied volatility from an option's market price using
    Newton-Raphson with vega.  Falls back to bisection if Newton diverges.
    Returns None if no solution is found.
    """
    if T <= 0 or price <= 0:
        return None

    intrinsic = max(S - K * math.exp(-r * T), 0.0)
    if price < intrinsic:
        return None  # below intrinsic — no valid IV

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2.0 * math.pi / T) * (price / S)
    sigma = max(0.01, min(sigma, 5.0))

    # Newton-Raphson
    for _ in range(max_iter):
        p = bs_call_price(S, K, T, r, sigma)
        v = bs_vega(S, K, T, r, sigma)
        if v < 1e-12:
            break  # vega too small, switch to bisection
        diff = p - price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / v
        if sigma <= 0.001 or sigma > 10.0:
            break  # diverged

    # Bisection fallback
    lo, hi = 0.001, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        p = bs_call_price(S, K, T, r, mid)
        if abs(p - price) < tol:
            return mid
        if p > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Data fetching & spread finding
# ---------------------------------------------------------------------------

def fetch_and_find_spreads(min_premium, max_premium, min_leverage, max_width=100, max_otm=5.0, risk_free_rate=None, expiration_filter="all", min_net_delta=0.33, min_reward_risk=0.5, commission=35.80, min_dte=30, max_leg_premium=20000, min_leg_premium=0, symbol="^SPX", move_pct=1.0):
    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE_PCT / 100.0
    """
    Fetch option chains and find bull call spreads matching criteria.

    Returns dict with 'spot', 'spreads', and metadata.
    """
    ticker = yf.Ticker(symbol)

    # Get current price
    info = ticker.history(period="1d")
    if info.empty:
        raise ValueError(f"Could not fetch {symbol} price. Market may be closed or Yahoo Finance unavailable.")
    spot = float(info["Close"].iloc[-1])

    # Get all expiration dates
    expirations = ticker.options
    if not expirations:
        raise ValueError("No option expiration dates available.")

    # Filter to specific expiration(s) if requested
    if expiration_filter and expiration_filter != "all":
        requested_dates = [d.strip() for d in expiration_filter.split(",") if d.strip()]
        matched = []
        for req in requested_dates:
            # Exact match first
            exact = [e for e in expirations if e == req]
            if exact:
                matched.extend(exact)
            else:
                # Try to find the nearest available expiration within 3 days
                target = datetime.strptime(req, "%Y-%m-%d")
                for e in expirations:
                    ed = datetime.strptime(e, "%Y-%m-%d")
                    if abs((ed - target).days) <= 3 and e not in matched:
                        matched.append(e)
        if not matched:
            raise ValueError(f"No expirations found near {expiration_filter}. Available: {', '.join(expirations[:10])}")
        expirations = matched

    now = datetime.now()
    spreads = []

    # Previous close (for day move)
    prev_close = None
    try:
        prev_close = float(ticker.fast_info.previous_close)
    except Exception:
        try:
            hist = ticker.history(period="5d", interval="1d")
            if len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
        except Exception:
            pass

    # ATM IV from the absolutely nearest expiration (independent of min_dte filter)
    atm_iv = None
    near_atm_exp = None
    try:
        all_exps = ticker.options
        if all_exps:
            near_atm_exp = all_exps[0]
            nc = ticker.option_chain(near_atm_exp).calls
            nc = nc[nc["impliedVolatility"] > 0]
            if not nc.empty:
                idx = (nc["strike"] - spot).abs().idxmin()
                atm_iv = float(nc.loc[idx, "impliedVolatility"])
    except Exception:
        pass

    # Day move metrics
    day_move = None
    day_move_pct = None
    day_move_sigma = None
    if prev_close is not None and prev_close > 0:
        day_move = spot - prev_close
        day_move_pct = (day_move / prev_close) * 100
        if atm_iv is not None and atm_iv > 0:
            expected_daily = spot * atm_iv * math.sqrt(1 / 252)
            if expected_daily > 0:
                day_move_sigma = day_move / expected_daily

    print(f"  {symbol} spot: {spot}")
    print(f"  Expirations to scan: {len(expirations)}")

    for exp_date_str in expirations:
        try:
            exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
            dte = (exp_date - now).days
            if dte < max(1, min_dte):
                print(f"  {exp_date_str}: skipped ({dte}d < {min_dte}d min DTE)")
                continue
            T = dte / 365.0

            chain = ticker.option_chain(exp_date_str)
            calls = chain.calls

            if calls.empty:
                print(f"  {exp_date_str}: skipped (empty chain)")
                continue

            # ATM IV for this expiration: pick the call strike closest to spot with a valid IV.
            exp_atm_iv = None
            valid_iv_calls = calls[calls["impliedVolatility"] > 0]
            if not valid_iv_calls.empty:
                idx = (valid_iv_calls["strike"] - spot).abs().idxmin()
                exp_atm_iv = float(valid_iv_calls.loc[idx, "impliedVolatility"])

            print(f"  {exp_date_str} ({dte}d): {len(calls)} total calls, ", end="")

            # Filter to OTM calls (strike > spot) with valid bid/ask
            otm_calls = calls[
                (calls["strike"] > spot) &
                (calls["bid"] > 0) &
                (calls["ask"] > 0) &
                (calls["impliedVolatility"] > 0)
            ].copy()

            print(f"{len(otm_calls)} OTM w/ valid bid/ask/IV")

            if len(otm_calls) < 2:
                continue

            # Show a sample of the data
            if len(otm_calls) > 0:
                sample = otm_calls.iloc[0]
                print(f"    Sample: strike={sample['strike']}, bid={sample['bid']}, ask={sample['ask']}, IV={sample['impliedVolatility']}")

            otm_calls = otm_calls.sort_values("strike").reset_index(drop=True)

            # Build candidate spreads
            strikes = otm_calls["strike"].values
            skipped_premium_zero = 0
            skipped_premium_high = 0
            skipped_leg_premium = 0
            skipped_leg_premium_min = 0
            skipped_leverage = 0
            skipped_delta = 0
            skipped_rr = 0
            found = 0

            for i in range(len(otm_calls)):
                for j in range(i + 1, len(otm_calls)):
                    row_buy = otm_calls.iloc[i]
                    row_sell = otm_calls.iloc[j]

                    K1 = float(row_buy["strike"])  # lower strike (buy)
                    K2 = float(row_sell["strike"])  # higher strike (sell)

                    # Skip if spread is wider than max width
                    if (K2 - K1) > max_width:
                        continue

                    # Skip if buy strike is too far OTM
                    pct_otm = ((K1 / spot) - 1) * 100
                    if pct_otm > max_otm:
                        continue

                    # Cost: pay ask for long, receive bid for short
                    buy_price = float(row_buy["ask"])
                    sell_price = float(row_sell["bid"])
                    net_premium = buy_price - sell_price

                    if net_premium <= 0:
                        skipped_premium_zero += 1
                        continue  # no cost or credit — skip

                    # Auto-calculate number of contracts:
                    # min/max_premium are in quoted points; multiply by 100 for dollars
                    # contracts = how many fit in the premium budget range
                    per_contract_dollars = net_premium * 100
                    max_contracts = int(max_premium * 100 / per_contract_dollars)
                    min_contracts = max(1, math.ceil(min_premium * 100 / per_contract_dollars))

                    if max_contracts < 1:
                        skipped_premium_high += 1
                        continue  # even 1 contract exceeds max budget
                    if min_contracts > max_contracts:
                        continue  # can't reach min budget without exceeding max

                    # Use the max contracts that fit in the budget
                    contracts = max_contracts

                    # Check per-leg premium constraint (in quoted points × contracts)
                    buy_leg_dollars = buy_price * 100 * contracts
                    sell_leg_dollars = sell_price * 100 * contracts
                    if max_leg_premium > 0:
                        if buy_leg_dollars > max_leg_premium or sell_leg_dollars > max_leg_premium:
                            skipped_leg_premium += 1
                            continue
                    if min_leg_premium > 0:
                        if buy_leg_dollars < min_leg_premium or sell_leg_dollars < min_leg_premium:
                            skipped_leg_premium_min += 1
                            continue

                    total_premium = net_premium * contracts/ide

                    # Compute deltas using implied vol
                    iv_buy = float(row_buy["impliedVolatility"])
                    iv_sell = float(row_sell["impliedVolatility"])

                    delta_buy = bs_call_delta(spot, K1, T, risk_free_rate, iv_buy)
                    delta_sell = bs_call_delta(spot, K2, T, risk_free_rate, iv_sell)
                    net_delta = delta_buy - delta_sell

                    gamma_buy = bs_gamma(spot, K1, T, risk_free_rate, iv_buy)
                    gamma_sell = bs_gamma(spot, K2, T, risk_free_rate, iv_sell)
                    net_gamma = gamma_buy - gamma_sell

                    # 2nd-order P&L for a 1% move (per contract): delta*ΔS + ½*gamma*ΔS²
                    move_frac = move_pct / 100.0
                    ds_1 = spot * move_frac
                    pnl_1pct_per = net_delta * ds_1 + 0.5 * net_gamma * ds_1 * ds_1

                    # P&L for a ±1σ / ±2σ *one-day* underlying move using the expiration's ATM IV.
                    # Daily σ = spot * IV * sqrt(1/252) (trading-day convention).
                    if exp_atm_iv and T > 0:
                        one_sigma_dS = spot * exp_atm_iv * math.sqrt(1 / 252)
                        pnl_1sigma_per = net_delta * one_sigma_dS + 0.5 * net_gamma * one_sigma_dS * one_sigma_dS
                        two_sigma_dS = 2 * one_sigma_dS
                        pnl_2sigma_per = net_delta * two_sigma_dS + 0.5 * net_gamma * two_sigma_dS * two_sigma_dS
                    else:
                        pnl_1sigma_per = 0.0
                        pnl_2sigma_per = 0.0

                    # Leverage: normalized to a 1% move so it stays comparable regardless of move_pct.
                    leverage = (pnl_1pct_per / net_premium) / move_frac if net_premium > 0 and move_frac > 0 else 0

                    if leverage < min_leverage:
                        skipped_leverage += 1
                        continue

                    if net_delta * contracts < min_net_delta:
                        skipped_delta += 1
                        continue

                    # Max profit & other metrics (per contract in quoted points)
                    # Commission: convert $/spread round-trip to points (÷100)
                    comm_pts = commission / 100.0
                    spread_width = K2 - K1
                    max_profit = spread_width - net_premium - comm_pts
                    breakeven = K1 + net_premium + comm_pts
                    pct_otm_buy = ((K1 / spot) - 1) * 100
                    pct_otm_sell = ((K2 / spot) - 1) * 100
                    effective_cost = net_premium + comm_pts
                    reward_risk = max_profit / effective_cost if effective_cost > 0 else 0

                    if reward_risk < min_reward_risk:
                        skipped_rr += 1
                        continue

                    found += 1

                    # Mid prices for reference
                    mid_buy = (float(row_buy["bid"]) + buy_price) / 2
                    mid_sell = (sell_price + float(row_sell["ask"])) / 2
                    mid_premium = mid_buy - mid_sell

                    # Scale by contracts for total position
                    pnl_1pct_total = pnl_1pct_per * contracts
                    pnl_1sigma_total = pnl_1sigma_per * contracts
                    pnl_2sigma_total = pnl_2sigma_per * contracts
                    breakeven_move_pct = (breakeven - spot) / spot * 100
                    iv_avg = (iv_buy + iv_sell) / 2
                    # Use ATM IV as the sigma yardstick so BE distance is comparable across strikes
                    # and consistent with the P&L 1σ/2σ columns. Fall back to leg-average IV if ATM IV
                    # for the expiration isn't available.
                    sigma_iv = exp_atm_iv if exp_atm_iv else iv_avg
                    be_move_sigma = (
                        (breakeven - spot) / (spot * sigma_iv * math.sqrt(T))
                        if sigma_iv > 0 and T > 0 else 0
                    )
                    rr_per_sigma = reward_risk / be_move_sigma if be_move_sigma > 0 else 0

                    spreads.append({
                        "expiration": exp_date_str,
                        "dte": dte,
                        "contracts": contracts,
                        "buyStrike": K1,
                        "sellStrike": K2,
                        "buyAsk": buy_price,
                        "sellBid": sell_price,
                        "netPremium": round(net_premium, 2),
                        "totalPremium": round(total_premium, 2),
                        "midPremium": round(mid_premium, 2),
                        "spreadWidth": round(spread_width, 2),
                        "maxProfit": round(max_profit * contracts, 2),
                        "breakeven": round(breakeven, 2),
                        "leverage": round(leverage, 2),
                        "pnl1pct": round(pnl_1pct_total, 4),
                        "pnl1sigma": round(pnl_1sigma_total, 4),
                        "pnl2sigma": round(pnl_2sigma_total, 4),
                        "expAtmIv": round(exp_atm_iv * 100, 1) if exp_atm_iv else None,
                        "breakevenMovePct": round(breakeven_move_pct, 2),
                        "breakevenMoveSigma": round(be_move_sigma, 2),
                        "netDelta": round(net_delta * contracts, 4),
                        "netDeltaPer": round(net_delta, 4),
                        "netGamma": round(net_gamma * contracts, 6),
                        "deltaBuy": round(delta_buy, 4),
                        "deltaSell": round(delta_sell, 4),
                        "ivBuy": round(iv_buy * 100, 1),
                        "ivSell": round(iv_sell * 100, 1),
                        "pctOtmBuy": round(pct_otm_buy, 2),
                        "pctOtmSell": round(pct_otm_sell, 2),
                        "rewardRisk": round(reward_risk, 2),
                        "rrPerSigma": round(rr_per_sigma, 2),
                        "volume_buy": int(row_buy.get("volume", 0) or 0),
                        "volume_sell": int(row_sell.get("volume", 0) or 0),
                        "oi_buy": int(row_buy.get("openInterest", 0) or 0),
                        "oi_sell": int(row_sell.get("openInterest", 0) or 0),
                        "commissionPerSpread": round(commission, 2),
                        "totalCommission": round(commission * contracts, 2),
                    })

            print(f"    => {found} matched | skipped: {skipped_premium_zero} zero/neg prem, {skipped_premium_high} over max prem, {skipped_leg_premium} over max leg prem, {skipped_leg_premium_min} under min leg prem, {skipped_leverage} under min leverage, {skipped_delta} under min delta, {skipped_rr} under min R/R")

        except Exception as e:
            print(f"  Skipping {exp_date_str}: {e}")
            continue

    print(f"\n  TOTAL SPREADS FOUND: {len(spreads)}")
    # Sort by leverage descending
    spreads.sort(key=lambda x: x["leverage"], reverse=True)

    return {
        "symbol": symbol,
        "spot": round(spot, 2),
        "movePct": move_pct,
        "prevClose": round(prev_close, 2) if prev_close is not None else None,
        "dayMove": round(day_move, 2) if day_move is not None else None,
        "dayMovePct": round(day_move_pct, 2) if day_move_pct is not None else None,
        "atmIv": round(atm_iv * 100, 1) if atm_iv is not None else None,
        "dayMoveSigma": round(day_move_sigma, 2) if day_move_sigma is not None else None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expirations_scanned": len(expirations),
        "total_spreads": len(spreads),
        "spreads": spreads
    }


# ---------------------------------------------------------------------------
# Position persistence + quote fetching
# ---------------------------------------------------------------------------

POSITIONS_FILE = Path(__file__).parent / "positions.json"
TEMPLATES_FILE = Path(__file__).parent / "templates.json"


def _load_json_list(path):
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _save_json_list(path, items):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def load_positions():
    return _load_json_list(POSITIONS_FILE)


def save_positions(positions):
    _save_json_list(POSITIONS_FILE, positions)


def load_templates():
    return _load_json_list(TEMPLATES_FILE)


def save_templates(templates):
    _save_json_list(TEMPLATES_FILE, templates)


def _leg_snapshot(row):
    """Extract bid/ask/last/volume/IV from a single calls-DataFrame row."""
    bid = float(row["bid"])
    ask = float(row["ask"])
    last = float(row["lastPrice"]) if "lastPrice" in row else 0.0
    vol = int(row["volume"]) if row.get("volume") and not math.isnan(float(row["volume"])) else 0
    iv_raw = float(row["impliedVolatility"]) if "impliedVolatility" in row else 0.0
    iv = iv_raw if iv_raw > 0 else None
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask, last)
    return {"bid": round(bid, 2), "ask": round(ask, 2), "last": round(last, 2),
            "mid": round(mid, 2), "volume": vol,
            "iv": round(iv * 100, 1) if iv is not None else None,
            "_iv_raw": iv}


# ---- P&L history (in-memory, per-server-session) ----
_PNL_HISTORY = {}   # position_id -> list of [timestamp_ms, pnl]
_HISTORY_MAX = 240  # ~2 hours at 30s cadence


def _record_pnl(position_id, pnl):
    if pnl is None:
        return
    ts_ms = int(datetime.now().timestamp() * 1000)
    hist = _PNL_HISTORY.setdefault(position_id, [])
    # Avoid dupes if the same fetch fires twice in <1s
    if hist and (ts_ms - hist[-1][0]) < 500:
        return
    hist.append([ts_ms, round(pnl, 2)])
    if len(hist) > _HISTORY_MAX:
        del hist[:len(hist) - _HISTORY_MAX]


def fetch_position_quotes(positions, haircut_pct=0.80):
    """Fetch live leg data for every saved position, batching chain calls.

    haircut_pct: multiplier applied to (raw P&L − entry commission − exit commission)
    to produce Adjusted P&L. Exit commission is assumed equal to entry commission.
    """
    results = []
    chain_cache = {}   # (symbol, expiration) -> calls DataFrame
    spot_cache = {}    # symbol -> float

    for p in positions:
        symbol = p["symbol"]
        exp = p["expiration"]
        contracts = int(p["contracts"])
        long_entry = float(p.get("longEntryPrice", 0))
        short_entry = float(p.get("shortEntryPrice", 0))
        entry_commission = float(p.get("entryCommission", 0))

        result = dict(p)  # copy stored fields
        result["error"] = None
        result["long"] = None
        result["short"] = None
        result["spot"] = None
        result["spreadMid"] = None
        result["spreadLiquidation"] = None
        result["currentValue"] = None
        result["liquidationValue"] = None
        # Entry cost is the pure premium outlay (commission-free) so raw P&L reflects the market move
        # only. Adjusted P&L nets out entry + mirrored exit commission and then applies haircut_pct.
        result["entryCost"] = round((long_entry - short_entry) * 100 * contracts, 2)
        result["entryCommission"] = round(entry_commission, 2)
        result["exitCommission"] = round(entry_commission, 2)
        result["totalCommission"] = round(entry_commission * 2, 2)
        result["haircutPct"] = round(haircut_pct * 100, 2)
        result["pnl"] = None
        result["pnlPct"] = None
        result["adjPnl"] = None
        result["adjPnlPct"] = None
        result["netDelta"] = None
        result["netThetaPerDay"] = None
        result["netVega"] = None
        result["oneSigmaMove"] = None
        result["oneSigmaPnl"] = None
        result["dte"] = None
        result["history"] = _PNL_HISTORY.get(p.get("id"), [])

        try:
            key = (symbol, exp)
            if key not in chain_cache:
                tk = yf.Ticker(symbol)
                chain_cache[key] = tk.option_chain(exp).calls
                if symbol not in spot_cache:
                    spot_cache[symbol] = float(tk.history(period="1d")["Close"].iloc[-1])
            calls = chain_cache[key]
            spot = spot_cache.get(symbol)
            result["spot"] = round(spot, 2) if spot is not None else None

            long_match = calls[calls["strike"] == float(p["longStrike"])]
            short_match = calls[calls["strike"] == float(p["shortStrike"])]

            if long_match.empty or short_match.empty:
                result["error"] = "Strike not found in chain"
            else:
                long_leg = _leg_snapshot(long_match.iloc[0])
                short_leg = _leg_snapshot(short_match.iloc[0])
                result["long"] = long_leg
                result["short"] = short_leg

                spread_mid = long_leg["mid"] - short_leg["mid"]
                spread_liq = long_leg["bid"] - short_leg["ask"]
                # currentValue is the pure market liquidation (sell long at bid, buy short at ask),
                # commission-free. Commissions and haircut are applied only to adjPnl.
                current_value = round(spread_liq * 100 * contracts, 2)
                liquidation_value = current_value

                result["spreadMid"] = round(spread_mid, 2)
                result["spreadLiquidation"] = round(spread_liq, 2)
                result["currentValue"] = current_value
                result["liquidationValue"] = liquidation_value
                if result["entryCost"]:
                    result["pnl"] = round(current_value - result["entryCost"], 2)
                    result["pnlPct"] = round(result["pnl"] / abs(result["entryCost"]) * 100, 2)
                    total_comm = entry_commission * 2  # exit assumed equal to entry
                    # Haircut applied only when raw P&L is positive (models exit slippage on gains);
                    # losses pass through untouched. Commissions are then netted in full.
                    if result["pnl"] > 0:
                        after_haircut = result["pnl"] * haircut_pct
                    else:
                        after_haircut = result["pnl"]
                    result["adjPnl"] = round(after_haircut - total_comm, 2)
                    result["adjPnlPct"] = round(result["adjPnl"] / abs(result["entryCost"]) * 100, 2)

                # Greeks — use both leg IVs; skip if either IV missing
                exp_dt = datetime.strptime(exp, "%Y-%m-%d")
                dte = max(0, (exp_dt - datetime.now()).days)
                T = dte / 365.0
                result["dte"] = dte
                r = RISK_FREE_RATE_PCT / 100.0
                iv_l = long_leg.get("_iv_raw")
                iv_s = short_leg.get("_iv_raw")
                if iv_l and iv_s and T > 0:
                    K1 = float(p["longStrike"])
                    K2 = float(p["shortStrike"])
                    delta_l = bs_call_delta(spot, K1, T, r, iv_l)
                    delta_s = bs_call_delta(spot, K2, T, r, iv_s)
                    gamma_l = bs_gamma(spot, K1, T, r, iv_l)
                    gamma_s = bs_gamma(spot, K2, T, r, iv_s)
                    theta_l = bs_call_theta(spot, K1, T, r, iv_l)
                    theta_s = bs_call_theta(spot, K2, T, r, iv_s)
                    vega_l = bs_vega(spot, K1, T, r, iv_l)
                    vega_s = bs_vega(spot, K2, T, r, iv_s)
                    net_delta_per = delta_l - delta_s
                    net_gamma_per = gamma_l - gamma_s
                    # Position-level: long minus short, scaled by contracts * 100 multiplier
                    result["netDelta"] = round(net_delta_per * 100 * contracts, 2)
                    # Theta per calendar day, in $ (annual / 365)
                    result["netThetaPerDay"] = round((theta_l - theta_s) / 365.0 * 100 * contracts, 2)
                    # Vega in $ per 1 vol-point (1%) move
                    result["netVega"] = round((vega_l - vega_s) * 100 * contracts * 0.01, 2)

                    # Dollar P&L for a +1σ one-day underlying move, using delta AND gamma
                    # (2nd-order Taylor: Δ·ΔS + ½·Γ·ΔS²) for maximal accuracy.
                    # Daily σ move = spot · ATM_IV · √(1/252) (trading-day convention).
                    atm_iv = None
                    valid_iv_calls = calls[calls["impliedVolatility"] > 0]
                    if not valid_iv_calls.empty:
                        idx = (valid_iv_calls["strike"] - spot).abs().idxmin()
                        atm_iv = float(valid_iv_calls.loc[idx, "impliedVolatility"])
                    if not atm_iv or atm_iv <= 0:
                        atm_iv = (iv_l + iv_s) / 2.0
                    one_sigma_dS = spot * atm_iv * math.sqrt(1 / 252)
                    one_sigma_pnl_per = net_delta_per * one_sigma_dS + 0.5 * net_gamma_per * one_sigma_dS ** 2
                    result["oneSigmaMove"] = round(one_sigma_dS, 2)
                    result["oneSigmaPnl"] = round(one_sigma_pnl_per * 100 * contracts, 2)

                # Strip internal keys before returning
                long_leg.pop("_iv_raw", None)
                short_leg.pop("_iv_raw", None)

                # Record P&L history (in-memory)
                if p.get("id") and result["pnl"] is not None:
                    _record_pnl(p["id"], result["pnl"])
                    result["history"] = _PNL_HISTORY.get(p["id"], [])
        except Exception as e:
            result["error"] = str(e)

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# HTML Frontend (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Call Spread Finder</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242837;
    --border: #2e3348;
    --text: #e4e6f0;
    --text-dim: #8b8fa3;
    --accent: #4f8ff7;
    --accent-hover: #6ba1ff;
    --green: #34d399;
    --green-dim: #065f46;
    --red: #f87171;
    --yellow: #fbbf24;
    --font: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
  }

  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
  }

  .header h1 {
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .header h1 .tag {
    font-size: 11px;
    background: var(--accent);
    color: #fff;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
    letter-spacing: 0.5px;
  }

  .spot-display {
    font-family: var(--mono);
    font-size: 20px;
    color: var(--green);
    font-weight: 600;
  }

  .positions-link {
    margin-left: auto;
    margin-right: 24px;
    padding: 6px 14px;
    border: 1px solid var(--accent);
    border-radius: 6px;
    color: var(--accent);
    text-decoration: none;
    font-size: 13px;
    font-weight: 500;
    transition: background 0.2s, color 0.2s;
  }
  .positions-link:hover { background: var(--accent); color: #fff; }

  .spot-display .label {
    font-size: 12px;
    color: var(--text-dim);
    font-family: var(--font);
    font-weight: 400;
    margin-right: 6px;
  }

  .controls {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: flex-end;
    gap: 20px;
    flex-wrap: wrap;
  }

  .input-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .input-group label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .input-group input, .input-group select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    font-family: var(--mono);
    font-size: 15px;
    padding: 10px 14px;
    border-radius: 6px;
    width: 180px;
    outline: none;
    transition: border-color 0.2s;
  }

  .input-group input:focus, .input-group select:focus {
    border-color: var(--accent);
  }

  .input-group select[multiple] option {
    padding: 4px 8px;
  }

  .input-group select[multiple] option:checked {
    background: var(--accent);
    color: #fff;
  }

  .input-group .hint {
    font-size: 11px;
    color: var(--text-dim);
  }

  button.primary {
    background: var(--accent);
    color: #fff;
    border: none;
    font-size: 14px;
    font-weight: 600;
    padding: 11px 28px;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s;
    white-space: nowrap;
  }

  button.primary:hover { background: var(--accent-hover); }
  button.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .status-bar {
    padding: 12px 32px;
    font-size: 13px;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
  }

  .status-bar .meta {
    display: flex;
    gap: 20px;
  }

  .status-bar .meta span {
    display: flex;
    align-items: center;
    gap: 5px;
  }

  .badge {
    background: var(--surface2);
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 12px;
    color: var(--accent);
  }

  .table-wrapper {
    padding: 0 32px 32px;
    overflow-x: auto;
    overflow-y: auto;
    max-height: calc(100vh - 280px);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }

  thead th {
    background: var(--bg);
    padding: 10px 12px;
    text-align: right;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    border-bottom: 2px solid var(--border);
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    position: sticky;
    top: 0;
    z-index: 3;
  }

  thead th:hover { color: var(--accent); }
  thead th.sorted-asc::after { content: ' ▲'; color: var(--accent); }
  thead th.sorted-desc::after { content: ' ▼'; color: var(--accent); }

  thead th:first-child { text-align: left; }

  tbody tr {
    border-bottom: 1px solid var(--border);
    transition: background 0.15s;
  }

  tbody tr { position: relative; }
  tbody tr:hover { background: var(--surface2); }

  .row-tooltip {
    display: none;
    position: fixed;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    z-index: 20;
    white-space: nowrap;
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.7;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    pointer-events: none;
  }

  .row-tooltip .tt-header {
    font-family: var(--font);
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-bottom: 6px;
  }

  .row-tooltip .tt-buy { color: var(--green); }
  .row-tooltip .tt-sell { color: var(--red); }
  .row-tooltip .tt-net { color: var(--accent); font-weight: 600; }
  .row-tooltip .tt-dim { color: var(--text-dim); }
  .row-tooltip .tt-sep {
    border-top: 1px solid var(--border);
    margin: 6px 0;
  }

  /* tooltip shown/positioned via JS */

  tbody td {
    padding: 9px 12px;
    text-align: right;
    font-family: var(--mono);
    font-size: 13px;
    white-space: nowrap;
  }

  tbody td:first-child {
    text-align: left;
    font-family: var(--font);
  }

  .highlight { color: var(--green); font-weight: 600; }
  .dim { color: var(--text-dim); }

  .loading-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(15, 17, 23, 0.85);
    z-index: 100;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 16px;
  }

  .loading-overlay.active { display: flex; }

  .spinner {
    width: 40px; height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin { to { transform: rotate(360deg); } }

  .loading-text {
    color: var(--text-dim);
    font-size: 14px;
    text-align: center;
    max-width: 350px;
    line-height: 1.5;
  }

  .empty-state {
    text-align: center;
    padding: 80px 32px;
    color: var(--text-dim);
  }

  .empty-state h2 {
    font-size: 18px;
    margin-bottom: 8px;
    color: var(--text);
    font-weight: 500;
  }

  .error-msg {
    background: rgba(248, 113, 113, 0.1);
    border: 1px solid var(--red);
    color: var(--red);
    padding: 12px 20px;
    margin: 16px 32px;
    border-radius: 6px;
    font-size: 13px;
    display: none;
  }

  .tooltip-container { position: relative; }
  .tooltip-container .tooltip-text {
    display: none;
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    color: var(--text);
    white-space: nowrap;
    z-index: 10;
    font-family: var(--font);
  }
  .tooltip-container:hover .tooltip-text { display: block; }

  .filters {
    padding: 12px 32px;
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    align-items: center;
  }

  .filters .filter-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.2s;
  }

  .filters .filter-chip:hover,
  .filters .filter-chip.active {
    border-color: var(--accent);
    color: var(--accent);
  }

  @media (max-width: 768px) {
    .controls { flex-direction: column; align-items: stretch; }
    .input-group input { width: 100%; }
    .table-wrapper { padding: 0 12px 12px; }
    .header, .controls, .status-bar, .filters { padding-left: 16px; padding-right: 16px; }
  }
</style>
</head>
<body>

<div class="header">
  <h1>Call Spread Finder <span class="tag">LIVE</span></h1>
  <a href="/positions" target="_blank" class="positions-link">My Positions &rarr;</a>
  <div class="spot-display">
    <span class="label" id="spotLabel">SPX Last:</span>
    <span id="spotPrice">--</span>
  </div>
</div>

<div class="controls">
  <div class="input-group tooltip-container">
    <label>Ticker</label>
    <input type="text" id="ticker" value="^SPX" style="width:120px;text-transform:uppercase;">
    <span class="hint">Yahoo Finance symbol</span>
    <span class="tooltip-text">Enter any optionable ticker — e.g. ^SPX, AAPL, TSLA, QQQ, SPY. Use ^ prefix for indices.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Premium ($)</label>
    <input type="number" id="minPremium" value="9000" min="0" step="100">
    <span class="hint">Min net dollars laid out</span>
    <span class="tooltip-text">Minimum net cash outlay (long call ask − short call bid) × 100 multiplier</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Max Premium ($)</label>
    <input type="number" id="maxPremium" value="11000" min="100" step="100">
    <span class="hint">Max net dollars laid out</span>
    <span class="tooltip-text">Maximum net cash outlay (long call ask − short call bid) × 100 multiplier</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Leverage (x)</label>
    <input type="number" id="minLeverage" value="2" min="0.1" step="0.5">
    <span class="hint">Profit / premium for 1% move</span>
    <span class="tooltip-text">Minimum ratio of dollar profit from a 1% up move to premium paid</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Max Width (pts)</label>
    <input type="number" id="maxWidth" value="100" min="5" step="5">
    <span class="hint">Max strike spread in points</span>
    <span class="tooltip-text">Maximum distance between strikes — e.g. 50 pts = $5,000 max risk (×100 multiplier)</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Max % OTM</label>
    <input type="number" id="maxOtm" value="5" min="0.1" step="0.5">
    <span class="hint">Buy strike max % above spot</span>
    <span class="tooltip-text">Maximum % the lower (buy) strike is above the current underlying price</span>
  </div>
  <div class="input-group tooltip-container">
    <label>P&amp;L Move %</label>
    <input type="number" id="movePct" value="1" min="0.1" step="0.5">
    <span class="hint">Move used for P&amp;L $ column</span>
    <span class="tooltip-text">The underlying % move used to compute the "P&amp;L X% $" column (Δ·dS + ½·Γ·dS²). Leverage stays normalized to per-1% for comparability.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Expirations</label>
    <div id="expirationCheckboxes" style="display:flex;flex-direction:column;gap:4px;padding:6px 0;"></div>
    <span class="hint">Check one or more expirations</span>
    <span class="tooltip-text">3rd Friday of each month — check the expirations you want to scan, or "All" to scan everything</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Risk-Free Rate (%)</label>
    <input type="number" id="riskFreeRate" value="{RISK_FREE_RATE_PCT}" min="0" max="20" step="0.1">
    <span class="hint">For Black-Scholes delta calc</span>
    <span class="tooltip-text">Used in delta calculation — approximate current Treasury yield</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Net Delta</label>
    <input type="number" id="minNetDelta" value="0.33" min="0" max="1" step="0.01">
    <span class="hint">Per-contract net delta floor</span>
    <span class="tooltip-text">Minimum net delta (long delta − short delta) per contract — filters out low-directional spreads</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Reward/Risk</label>
    <input type="number" id="minRewardRisk" value="0.5" min="0" step="0.1">
    <span class="hint">Min max-profit / premium ratio</span>
    <span class="tooltip-text">Minimum ratio of max profit to premium paid — e.g. 1.0 means max profit ≥ premium</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Commission ($/spread)</label>
    <input type="number" id="commission" value="35.80" min="0" step="0.25">
    <span class="hint">Round-trip per spread</span>
    <span class="tooltip-text">Total commission per spread for opening + closing (all legs, both ways). Default $35.80 = $8.95/leg &times; 2 legs &times; 2 sides. Deducted from max profit, breakeven, and reward/risk.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Max Leg Premium ($)</label>
    <input type="number" id="maxLegPremium" value="20000" min="0" step="500">
    <span class="hint">Max $ per individual leg</span>
    <span class="tooltip-text">Maximum absolute dollar value (price × 100 × contracts) for either the long or short leg individually</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Leg Premium ($)</label>
    <input type="number" id="minLegPremium" value="0" min="0" step="500">
    <span class="hint">Min $ per individual leg</span>
    <span class="tooltip-text">Minimum absolute dollar value (price × 100 × contracts) required for both the long and short legs. Use 0 to disable.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min DTE</label>
    <input type="number" id="minDte" value="30" min="0" step="1">
    <span class="hint">Min days to expiration</span>
    <span class="tooltip-text">Skip expirations with fewer than this many days remaining</span>
  </div>
  <div class="input-group">
    <label>Sort By</label>
    <select id="sortBy" onchange="applySortDropdown()">
      <option value="leverage">Leverage (high first)</option>
      <option value="totalPremium">Premium (low first)</option>
      <option value="rewardRisk">Reward/Risk (high first)</option>
      <option value="dte">DTE (near first)</option>
      <option value="pctOtmBuy">% OTM (near first)</option>
    </select>
  </div>
  <button class="primary" id="searchBtn" onclick="doSearch()">Find Spreads</button>
</div>

<div class="controls" style="border-top:none;padding-top:0;">
  <div class="input-group" style="min-width:280px;">
    <label>Templates</label>
    <select id="templateSelect" onchange="onTemplatePick()">
      <option value="">(none — current values)</option>
    </select>
    <span class="hint">Saved parameter sets — pick to load</span>
  </div>
  <button type="button" class="primary" id="tplLoadBtn" style="background:transparent;color:var(--accent);border:1px solid var(--accent);" onclick="loadSelectedTemplate()">Load</button>
  <button type="button" class="primary" id="tplUpdateBtn" style="background:transparent;color:var(--accent);border:1px solid var(--accent);" onclick="updateSelectedTemplate()">Update Selected</button>
  <button type="button" class="primary" id="tplSaveAsBtn" style="background:transparent;color:var(--accent);border:1px solid var(--accent);" onclick="saveTemplateAs()">Save As New</button>
  <button type="button" class="primary" id="tplDeleteBtn" style="background:transparent;color:#ef4444;border:1px solid #ef4444;" onclick="deleteSelectedTemplate()">Delete</button>
</div>

<div class="error-msg" id="errorMsg"></div>

<div class="filters" id="filtersBar" style="display:none;">
  <span style="font-size:12px;color:var(--text-dim);margin-right:4px;">Filter DTE:</span>
  <span class="filter-chip active" data-dte="all" onclick="filterDte(this)">All</span>
</div>

<div class="status-bar" id="statusBar" style="display:none;">
  <span id="statusText"></span>
  <div class="meta">
    <span><span id="statusSymbol">--</span> @ <span class="badge" id="statusSpot">--</span></span>
    <span>Day: <span class="badge" id="statusDayMove">--</span></span>
    <span>Expirations: <span class="badge" id="expCount">0</span></span>
    <span>Matches: <span class="badge" id="matchCount">0</span></span>
    <span>As of: <span class="badge" id="timestamp">--</span></span>
  </div>
</div>

<div class="table-wrapper" id="tableWrapper">
  <div class="empty-state" id="emptyState">
    <h2>Set your criteria and hit "Find Spreads"</h2>
    <p>Searches all available expirations for OTM bull call spreads matching your criteria.</p>
  </div>
  <table id="resultsTable" style="display:none;">
    <thead>
      <tr>
        <th data-col="expiration" onclick="sortTable('expiration')">Expiration</th>
        <th data-col="dte" onclick="sortTable('dte')">DTE</th>
        <th data-col="contracts" onclick="sortTable('contracts')">Contracts</th>
        <th data-col="buyStrike" onclick="sortTable('buyStrike')">Buy Strike</th>
        <th data-col="sellStrike" onclick="sortTable('sellStrike')">Sell Strike</th>
        <th data-col="pctOtmBuy" onclick="sortTable('pctOtmBuy')">% OTM</th>
        <th data-col="spreadWidth" onclick="sortTable('spreadWidth')">Width (pts)</th>
        <th data-col="netPremium" onclick="sortTable('netPremium')">Premium $</th>
        <th data-col="midPremium" onclick="sortTable('midPremium')">Mid Prem $</th>
        <th data-col="maxProfit" onclick="sortTable('maxProfit')">Max Profit $</th>
        <th data-col="rewardRisk" onclick="sortTable('rewardRisk')">Reward/Risk</th>
        <th data-col="rrPerSigma" onclick="sortTable('rrPerSigma')">R/R per &sigma;</th>
        <th data-col="leverage" onclick="sortTable('leverage')">Leverage</th>
        <th data-col="pnl1pct" onclick="sortTable('pnl1pct')" id="pnlMoveHeader">P&amp;L 1% $</th>
        <th data-col="pnl1sigma" onclick="sortTable('pnl1sigma')">P&amp;L 1&sigma; 1d $</th>
        <th data-col="pnl2sigma" onclick="sortTable('pnl2sigma')">P&amp;L 2&sigma; 1d $</th>
        <th data-col="breakevenMovePct" onclick="sortTable('breakevenMovePct')">BE Move %</th>
        <th data-col="breakevenMoveSigma" onclick="sortTable('breakevenMoveSigma')">BE Move &sigma;</th>
        <th data-col="netDelta" onclick="sortTable('netDelta')">Net Delta</th>
        <th data-col="netDeltaPer" onclick="sortTable('netDeltaPer')">&Delta;/Contract</th>
        <th data-col="netGamma" onclick="sortTable('netGamma')">Net Gamma</th>
        <th data-col="breakeven" onclick="sortTable('breakeven')">Breakeven</th>
        <th data-col="ivBuy" onclick="sortTable('ivBuy')">IV Buy</th>
        <th data-col="ivSell" onclick="sortTable('ivSell')">IV Sell</th>
        <th data-col="oi_buy" onclick="sortTable('oi_buy')">OI Buy</th>
        <th data-col="oi_sell" onclick="sortTable('oi_sell')">OI Sell</th>
      </tr>
    </thead>
    <tbody id="resultsBody"></tbody>
  </table>
</div>

<div class="loading-overlay" id="loadingOverlay">
  <div class="spinner"></div>
  <div class="loading-text" id="loadingText">
    Fetching option chains from Yahoo Finance...<br>
    <span style="font-size:12px;color:var(--text-dim);">This may take 30-60 seconds depending on the number of expirations.</span>
  </div>
</div>

<script>
let allSpreads = [];
let currentSort = { col: 'leverage', asc: false };
let activeDteFilter = 'all';
let currentSpot = null;
let currentSymbol = '^SPX';

// Populate expiration checkboxes with 3rd-Friday monthly expirations up to 6 months out
function populateExpirations() {
  const container = document.getElementById('expirationCheckboxes');
  container.innerHTML = '';
  const today = new Date();
  const cutoff = new Date(today);
  cutoff.setMonth(cutoff.getMonth() + 6);

  // "All" checkbox
  const allLabel = document.createElement('label');
  allLabel.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px;';
  allLabel.innerHTML = '<input type="checkbox" id="exp_all" value="all" checked style="accent-color:var(--accent);width:16px;height:16px;"> <span>All expirations</span>';
  container.appendChild(allLabel);

  // Date checkboxes
  let d = new Date(today.getFullYear(), today.getMonth(), 1);
  while (d <= cutoff) {
    const first = new Date(d.getFullYear(), d.getMonth(), 1);
    const dayOfWeek = first.getDay();
    const firstFriday = 1 + ((5 - dayOfWeek + 7) % 7);
    const thirdFriday = new Date(d.getFullYear(), d.getMonth(), firstFriday + 14);
    if (thirdFriday > today) {
      const yyyy = thirdFriday.getFullYear();
      const mm = String(thirdFriday.getMonth() + 1).padStart(2, '0');
      const dd = String(thirdFriday.getDate()).padStart(2, '0');
      const val = `${yyyy}-${mm}-${dd}`;
      const label = thirdFriday.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      const dte = Math.round((thirdFriday - today) / 86400000);
      const lbl = document.createElement('label');
      lbl.style.cssText = 'display:flex;align-items:center;gap:6px;cursor:pointer;font-size:13px;';
      lbl.innerHTML = `<input type="checkbox" class="exp-date-cb" value="${val}" style="accent-color:var(--accent);width:16px;height:16px;"> <span>${label} (${dte}d)</span>`;
      container.appendChild(lbl);
    }
    d.setMonth(d.getMonth() + 1);
  }

  // Logic: "All" unchecks individual dates; individual dates uncheck "All"
  const allCb = document.getElementById('exp_all');
  allCb.addEventListener('change', () => {
    if (allCb.checked) {
      container.querySelectorAll('.exp-date-cb').forEach(cb => cb.checked = false);
    }
  });
  container.querySelectorAll('.exp-date-cb').forEach(cb => {
    cb.addEventListener('change', () => {
      const anyDateChecked = [...container.querySelectorAll('.exp-date-cb')].some(c => c.checked);
      if (anyDateChecked) {
        allCb.checked = false;
      } else {
        allCb.checked = true;
      }
    });
  });
}
populateExpirations();

// ---------------- Templates (saved parameter sets) ----------------

// Input IDs to capture in a template (order matters for restoring)
const TEMPLATE_INPUT_IDS = [
  'ticker','minPremium','maxPremium','minLeverage','maxWidth','maxOtm','movePct',
  'riskFreeRate','minNetDelta','minRewardRisk','commission','maxLegPremium',
  'minLegPremium','minDte','sortBy'
];

let savedTemplates = [];  // full list from /api/templates

function captureCurrentParams() {
  const params = {};
  for (const id of TEMPLATE_INPUT_IDS) {
    const el = document.getElementById(id);
    if (el) params[id] = el.value;
  }
  // Expiration filter state
  const allCb = document.getElementById('exp_all');
  if (allCb && !allCb.checked) {
    const checked = [...document.querySelectorAll('.exp-date-cb:checked')].map(cb => cb.value);
    params.expiration = checked.length ? checked.join(',') : 'all';
  } else {
    params.expiration = 'all';
  }
  return params;
}

function applyParams(params) {
  if (!params) return;
  for (const id of TEMPLATE_INPUT_IDS) {
    if (id in params) {
      const el = document.getElementById(id);
      if (el) el.value = params[id];
    }
  }
  // Expiration checkbox state
  if ('expiration' in params) {
    const allCb = document.getElementById('exp_all');
    const dateCbs = [...document.querySelectorAll('.exp-date-cb')];
    if (params.expiration === 'all' || !params.expiration) {
      if (allCb) allCb.checked = true;
      dateCbs.forEach(cb => cb.checked = false);
    } else {
      if (allCb) allCb.checked = false;
      const wanted = new Set(params.expiration.split(',').map(s => s.trim()));
      dateCbs.forEach(cb => cb.checked = wanted.has(cb.value));
    }
  }
}

async function fetchTemplates() {
  try {
    const r = await fetch('/api/templates');
    savedTemplates = await r.json();
    renderTemplateSelect();
  } catch (e) { console.error('template list failed:', e); }
}

function renderTemplateSelect() {
  const sel = document.getElementById('templateSelect');
  const prev = sel.value;
  sel.innerHTML = '<option value="">(none — current values)</option>';
  for (const t of savedTemplates) {
    const opt = document.createElement('option');
    opt.value = t.id;
    opt.textContent = t.name;
    sel.appendChild(opt);
  }
  if (savedTemplates.some(t => t.id === prev)) sel.value = prev;
}

function onTemplatePick() { /* placeholder for future auto-load behavior */ }

async function loadSelectedTemplate() {
  const sel = document.getElementById('templateSelect');
  const t = savedTemplates.find(x => x.id === sel.value);
  if (!t) { showError('No template selected.'); return; }
  applyParams(t.params);
  hideError();
}

async function saveTemplateAs() {
  const name = prompt('Name this template:');
  if (!name || !name.trim()) return;
  const payload = { name: name.trim(), params: captureCurrentParams() };
  try {
    const r = await fetch('/api/templates', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (j.error) { showError(j.error); return; }
    await fetchTemplates();
    document.getElementById('templateSelect').value = j.id;
    hideError();
  } catch (e) { showError('Save failed: ' + e.message); }
}

async function updateSelectedTemplate() {
  const sel = document.getElementById('templateSelect');
  const t = savedTemplates.find(x => x.id === sel.value);
  if (!t) { showError('Select a template to update, or use "Save As New".'); return; }
  if (!confirm(`Overwrite template "${t.name}" with the current parameter values?`)) return;
  try {
    const r = await fetch('/api/templates/' + t.id, {
      method: 'PUT', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ name: t.name, params: captureCurrentParams() })
    });
    const j = await r.json();
    if (j.error) { showError(j.error); return; }
    await fetchTemplates();
    document.getElementById('templateSelect').value = t.id;
    hideError();
  } catch (e) { showError('Update failed: ' + e.message); }
}

async function deleteSelectedTemplate() {
  const sel = document.getElementById('templateSelect');
  const t = savedTemplates.find(x => x.id === sel.value);
  if (!t) { showError('Select a template to delete.'); return; }
  if (!confirm(`Delete template "${t.name}"?`)) return;
  try {
    const r = await fetch('/api/templates/' + t.id, {method: 'DELETE'});
    const j = await r.json();
    if (j.error) { showError(j.error); return; }
    await fetchTemplates();
    hideError();
  } catch (e) { showError('Delete failed: ' + e.message); }
}

fetchTemplates();

async function doSearch() {
  const symbol = document.getElementById('ticker').value.trim().toUpperCase();
  if (!symbol) { showError('Please enter a ticker symbol.'); return; }
  const minPremiumDollars = parseFloat(document.getElementById('minPremium').value);
  const maxPremiumDollars = parseFloat(document.getElementById('maxPremium').value);
  const minLeverage = parseFloat(document.getElementById('minLeverage').value);
  const maxWidth = parseFloat(document.getElementById('maxWidth').value);
  const maxOtm = parseFloat(document.getElementById('maxOtm').value);
  const riskFreeRate = parseFloat(document.getElementById('riskFreeRate').value) / 100;
  const minNetDelta = parseFloat(document.getElementById('minNetDelta').value);
  const minRewardRisk = parseFloat(document.getElementById('minRewardRisk').value);
  const commission = parseFloat(document.getElementById('commission').value);
  const maxLegPremium = parseFloat(document.getElementById('maxLegPremium').value);
  const minLegPremium = parseFloat(document.getElementById('minLegPremium').value) || 0;
  const minDte = parseInt(document.getElementById('minDte').value) || 0;
  const movePct = parseFloat(document.getElementById('movePct').value) || 1.0;
  const allCb = document.getElementById('exp_all');
  let expiration = 'all';
  if (!allCb.checked) {
    const checked = [...document.querySelectorAll('.exp-date-cb:checked')].map(cb => cb.value);
    expiration = checked.length > 0 ? checked.join(',') : 'all';
  }

  // Convert actual dollars to quoted points (options multiplier = 100)
  const minPremium = minPremiumDollars / 100;
  const maxPremium = maxPremiumDollars / 100;

  if (isNaN(minPremiumDollars) || minPremiumDollars < 0) {
    showError('Please enter a valid min premium (0 or greater).');
    return;
  }
  if (isNaN(maxPremiumDollars) || maxPremiumDollars <= 0) {
    showError('Please enter a valid max premium greater than 0.');
    return;
  }
  if (minPremiumDollars > maxPremiumDollars) {
    showError('Min premium cannot exceed max premium.');
    return;
  }
  if (isNaN(minLeverage) || minLeverage <= 0) {
    showError('Please enter a valid min leverage greater than 0.');
    return;
  }
  if (isNaN(maxWidth) || maxWidth <= 0) {
    showError('Please enter a valid max width greater than 0.');
    return;
  }
  if (isNaN(maxOtm) || maxOtm <= 0) {
    showError('Please enter a valid max % OTM greater than 0.');
    return;
  }

  hideError();
  document.getElementById('searchBtn').disabled = true;
  document.getElementById('loadingOverlay').classList.add('active');

  try {
    const params = new URLSearchParams({
      symbol: symbol,
      min_premium: minPremium,
      max_premium: maxPremium,
      min_leverage: minLeverage,
      max_width: maxWidth,
      max_otm: maxOtm,
      risk_free_rate: riskFreeRate,
      min_net_delta: minNetDelta,
      min_reward_risk: minRewardRisk,
      commission: commission,
      max_leg_premium: maxLegPremium,
      min_leg_premium: minLegPremium,
      min_dte: minDte,
      move_pct: movePct,
      expiration: expiration
    });

    const resp = await fetch(`/api/spreads?${params}`);
    const data = await resp.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    currentSpot = data.spot;
    currentSymbol = data.symbol || symbol;
    document.getElementById('spotLabel').textContent = currentSymbol + ' Last:';
    document.getElementById('spotPrice').textContent = data.spot.toLocaleString('en-US', {minimumFractionDigits: 2});
    document.getElementById('statusSymbol').textContent = currentSymbol;
    document.getElementById('statusSpot').textContent = '$' + data.spot.toLocaleString('en-US', {minimumFractionDigits: 2});
    if (data.movePct !== undefined && data.movePct !== null) {
      const mp = Number(data.movePct);
      const label = (Number.isInteger(mp) ? mp.toFixed(0) : mp.toString()) + '%';
      document.getElementById('pnlMoveHeader').innerHTML = 'P&amp;L ' + label + ' $';
    }
    {
      const parts = [];
      if (data.dayMove !== null && data.dayMove !== undefined) {
        const dm = data.dayMove;
        const dmStr = (dm >= 0 ? '+$' : '-$') + Math.abs(dm).toFixed(2);
        let pctStr = '';
        if (data.dayMovePct !== null && data.dayMovePct !== undefined) {
          pctStr = ' (' + (data.dayMovePct >= 0 ? '+' : '') + data.dayMovePct.toFixed(2) + '%)';
        }
        parts.push(dmStr + pctStr);
        if (data.dayMoveSigma !== null && data.dayMoveSigma !== undefined) {
          parts.push((data.dayMoveSigma >= 0 ? '+' : '') + data.dayMoveSigma.toFixed(2) + 'σ');
        }
      }
      document.getElementById('statusDayMove').textContent = parts.length ? parts.join(' | ') : '--';
    }
    document.getElementById('expCount').textContent = data.expirations_scanned;
    document.getElementById('matchCount').textContent = data.total_spreads;
    document.getElementById('timestamp').textContent = data.timestamp;
    document.getElementById('statusBar').style.display = 'flex';

    allSpreads = data.spreads;
    buildDteFilters();
    renderTable();

    if (allSpreads.length === 0) {
      document.getElementById('emptyState').innerHTML = '<h2>No spreads found</h2><p>Try increasing the max premium or decreasing the min leverage.</p>';
      document.getElementById('emptyState').style.display = 'block';
      document.getElementById('resultsTable').style.display = 'none';
    } else {
      document.getElementById('emptyState').style.display = 'none';
      document.getElementById('resultsTable').style.display = 'table';
    }
  } catch (err) {
    showError('Failed to connect to server: ' + err.message);
  } finally {
    document.getElementById('searchBtn').disabled = false;
    document.getElementById('loadingOverlay').classList.remove('active');
  }
}

function buildDteFilters() {
  const bar = document.getElementById('filtersBar');
  const dtes = [...new Set(allSpreads.map(s => s.dte))].sort((a, b) => a - b);
  if (dtes.length === 0) { bar.style.display = 'none'; return; }

  // Group into buckets
  const buckets = [
    { label: 'All', value: 'all' },
    { label: '0-7d', value: '0-7', min: 0, max: 7 },
    { label: '8-30d', value: '8-30', min: 8, max: 30 },
    { label: '31-60d', value: '31-60', min: 31, max: 60 },
    { label: '61-120d', value: '61-120', min: 61, max: 120 },
    { label: '120d+', value: '120+', min: 121, max: 99999 },
  ];

  bar.innerHTML = '<span style="font-size:12px;color:var(--text-dim);margin-right:4px;">Filter DTE:</span>';
  for (const b of buckets) {
    let count = 0;
    if (b.value === 'all') {
      count = allSpreads.length;
    } else {
      count = allSpreads.filter(s => s.dte >= b.min && s.dte <= b.max).length;
    }
    if (count === 0 && b.value !== 'all') continue;
    const chip = document.createElement('span');
    chip.className = 'filter-chip' + (b.value === activeDteFilter ? ' active' : '');
    chip.dataset.dte = b.value;
    chip.textContent = `${b.label} (${count})`;
    chip.onclick = () => filterDte(chip);
    bar.appendChild(chip);
  }
  bar.style.display = 'flex';
}

function filterDte(el) {
  activeDteFilter = el.dataset.dte;
  document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  renderTable();
}

function getFilteredSpreads() {
  if (activeDteFilter === 'all') return [...allSpreads];
  const [minStr, maxStr] = activeDteFilter.includes('+')
    ? [activeDteFilter.replace('+', ''), '99999']
    : activeDteFilter.split('-');
  const min = parseInt(minStr), max = parseInt(maxStr);
  return allSpreads.filter(s => s.dte >= min && s.dte <= max);
}

function sortTable(col) {
  if (currentSort.col === col) {
    currentSort.asc = !currentSort.asc;
  } else {
    currentSort.col = col;
    // Default sort direction per column
    currentSort.asc = ['totalPremium', 'netPremium', 'dte', 'pctOtmBuy', 'buyStrike', 'expiration'].includes(col);
  }
  // Sync dropdown if it matches
  const dd = document.getElementById('sortBy');
  const match = [...dd.options].find(o => o.value === col);
  if (match) dd.value = col;
  renderTable();
}

function applySortDropdown() {
  const col = document.getElementById('sortBy').value;
  currentSort.col = col;
  // "high first" = descending, "low/near first" = ascending
  currentSort.asc = ['totalPremium', 'dte', 'pctOtmBuy'].includes(col);
  renderTable();
}

// Black-Scholes helpers for theoretical value curve
function jsNormCdf(x) {
  const a1=0.254829592, a2=-0.284496736, a3=1.421413741, a4=-1.453152027, a5=1.061405429, p=0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.sqrt(2);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t * Math.exp(-x*x);
  return 0.5 * (1.0 + sign * y);
}

function jsBsCallPrice(S, K, T, r, sigma) {
  if (T <= 0) return Math.max(S - K, 0);
  if (sigma <= 0) return Math.max(S - K * Math.exp(-r * T), 0);
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  return S * jsNormCdf(d1) - K * Math.exp(-r * T) * jsNormCdf(d2);
}

function buildPnlChart(s) {
  const m = 100;
  const c = s.contracts;
  const K1 = s.buyStrike;
  const K2 = s.sellStrike;
  const prem = s.netPremium * m * c;                   // option premium total $
  const totalComm = s.totalCommission;                  // total commission $
  const totalCost = prem + totalComm;                   // total $ outlay incl commission
  const maxProf = s.maxProfit * m;                      // total $ max profit (already net of commission)
  const width = s.spreadWidth;

  // P&L at expiration:
  //   below K1: -totalCost (max loss = premium + commission)
  //   at K1: -totalCost
  //   at K2: maxProf
  //   above K2: maxProf
  //   breakeven already includes commission

  const maxLoss = -totalCost;
  const maxGain = maxProf;
  const be = s.breakeven;

  // Chart dimensions
  const W = 320, H = 300;
  const pad = {l: 55, r: 15, t: 15, b: 30};
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  // X range: include spot, K1, and K2 with padding
  const spot = currentSpot || K1;
  const xPad = width * 0.3;
  const xMin = Math.min(K1, spot) - xPad;
  const xMax = K2 + xPad;
  const xScale = (v) => pad.l + (v - xMin) / (xMax - xMin) * cw;

  // Y range: maxLoss to maxGain with padding
  const yPadding = Math.max(Math.abs(maxLoss), Math.abs(maxGain)) * 0.15;
  const yMin = maxLoss - yPadding;
  const yMax = maxGain + yPadding;
  const yScale = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * ch;

  // Key points for the payoff line
  const points = [
    {x: xMin, y: maxLoss},
    {x: K1,   y: maxLoss},
    {x: K2,   y: maxGain},
    {x: xMax, y: maxGain},
  ];

  const line = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`
  ).join(' ');

  // Zero line Y
  const zeroY = yScale(0).toFixed(1);

  // Fill: green above zero, red below zero
  // Build the fill polygon clipped to below-zero (loss region)
  const lossFill = `M${xScale(xMin).toFixed(1)},${zeroY} ` +
    points.filter(p => p.x <= be + 0.1).map(p =>
      `L${xScale(p.x).toFixed(1)},${yScale(Math.min(p.y, 0)).toFixed(1)}`
    ).join(' ') +
    ` L${xScale(be).toFixed(1)},${zeroY} Z`;

  // Profit fill: from breakeven up
  const profFill = `M${xScale(be).toFixed(1)},${zeroY} ` +
    `L${xScale(K2).toFixed(1)},${yScale(maxGain).toFixed(1)} ` +
    `L${xScale(xMax).toFixed(1)},${yScale(maxGain).toFixed(1)} ` +
    `L${xScale(xMax).toFixed(1)},${zeroY} Z`;

  // Theoretical value curve (current P&L as a function of underlying price)
  const r = parseFloat(document.getElementById('riskFreeRate').value) / 100;
  const T = s.dte / 365;
  const ivB = s.ivBuy / 100;   // stored as percentage
  const ivS = s.ivSell / 100;
  const nSteps = 60;
  const theoPoints = [];
  for (let i = 0; i <= nSteps; i++) {
    const sx = xMin + (xMax - xMin) * i / nSteps;
    const callBuy = jsBsCallPrice(sx, K1, T, r, ivB);
    const callSell = jsBsCallPrice(sx, K2, T, r, ivS);
    const spreadVal = (callBuy - callSell) * m * c;  // current spread value in $
    const pnl = spreadVal - totalCost;                  // P&L = current value - cost (incl commission)
    theoPoints.push({x: sx, y: pnl});
  }
  const theoLine = theoPoints.map((p, i) =>
    `${i === 0 ? 'M' : 'L'}${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`
  ).join(' ');

  // Current position value dot (at spot)
  const spotCallBuy = jsBsCallPrice(spot, K1, T, r, ivB);
  const spotCallSell = jsBsCallPrice(spot, K2, T, r, ivS);
  const spotPnl = (spotCallBuy - spotCallSell) * m * c - totalCost;

  // Format dollar labels
  const fmtK = (v) => v.toLocaleString('en-US', {maximumFractionDigits: 0});
  const fmtD = (v) => (v >= 0 ? '+' : '') + '$' + Math.abs(v).toLocaleString('en-US', {maximumFractionDigits: 0});

  // Y-axis ticks
  const yTicks = [maxLoss, 0, maxGain];

  return `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin-top:6px;">
    <!-- Grid -->
    <line x1="${pad.l}" y1="${zeroY}" x2="${W - pad.r}" y2="${zeroY}" stroke="#2e3348" stroke-width="1" stroke-dasharray="4,3"/>

    <!-- Loss fill -->
    <path d="${lossFill}" fill="rgba(248,113,113,0.15)"/>

    <!-- Profit fill -->
    <path d="${profFill}" fill="rgba(52,211,153,0.15)"/>

    <!-- Payoff at expiration -->
    <path d="${line}" fill="none" stroke="#e4e6f0" stroke-width="1.5" stroke-opacity="0.5"/>

    <!-- Theoretical value curve (now) -->
    <path d="${theoLine}" fill="none" stroke="#c084fc" stroke-width="2"/>

    <!-- Current value dot at spot -->
    <circle cx="${xScale(spot).toFixed(1)}" cy="${yScale(spotPnl).toFixed(1)}" r="3.5" fill="#c084fc"/>

    <!-- Breakeven dot -->
    <circle cx="${xScale(be).toFixed(1)}" cy="${zeroY}" r="3" fill="#fbbf24"/>
    <text x="${xScale(be).toFixed(1)}" y="${parseFloat(zeroY) - 7}" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="sans-serif">BE ${fmtK(be)}</text>

    <!-- X-axis labels -->
    <text x="${xScale(K1).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#8b8fa3" font-size="9" font-family="sans-serif">${fmtK(K1)}</text>
    <text x="${xScale(K2).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#8b8fa3" font-size="9" font-family="sans-serif">${fmtK(K2)}</text>

    <!-- Y-axis labels -->
    ${yTicks.map(v => `<text x="${pad.l - 5}" y="${(parseFloat(yScale(v)) + 3).toFixed(1)}" text-anchor="end" fill="${v > 0 ? '#34d399' : v < 0 ? '#f87171' : '#8b8fa3'}" font-size="9" font-family="sans-serif">${fmtD(v)}</text>`).join('')}

    <!-- Strike lines -->
    <line x1="${xScale(K1).toFixed(1)}" y1="${pad.t}" x2="${xScale(K1).toFixed(1)}" y2="${H - pad.b}" stroke="#2e3348" stroke-width="1" stroke-dasharray="2,2"/>
    <line x1="${xScale(K2).toFixed(1)}" y1="${pad.t}" x2="${xScale(K2).toFixed(1)}" y2="${H - pad.b}" stroke="#2e3348" stroke-width="1" stroke-dasharray="2,2"/>

    <!-- Current spot -->
    <line x1="${xScale(spot).toFixed(1)}" y1="${pad.t}" x2="${xScale(spot).toFixed(1)}" y2="${H - pad.b}" stroke="#60a5fa" stroke-width="1.5" stroke-dasharray="4,2"/>
    <text x="${xScale(spot).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#60a5fa" font-size="9" font-weight="600" font-family="sans-serif">${currentSymbol} ${fmtK(spot)}</text>

    <!-- Legend -->
    <line x1="${W - 130}" y1="8" x2="${W - 115}" y2="8" stroke="#e4e6f0" stroke-width="1.5" stroke-opacity="0.5"/>
    <text x="${W - 112}" y="11" fill="#8b8fa3" font-size="8" font-family="sans-serif">At expiry</text>
    <line x1="${W - 65}" y1="8" x2="${W - 50}" y2="8" stroke="#c084fc" stroke-width="2"/>
    <text x="${W - 47}" y="11" fill="#c084fc" font-size="8" font-family="sans-serif">Now</text>
  </svg>`;
}

function renderTable() {
  const spreads = getFilteredSpreads();
  const col = currentSort.col;
  const mult = currentSort.asc ? 1 : -1;

  spreads.sort((a, b) => {
    if (col === 'expiration') return mult * a[col].localeCompare(b[col]);
    return mult * ((a[col] || 0) - (b[col] || 0));
  });

  // Update column header styling
  document.querySelectorAll('thead th').forEach(th => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.col === col) {
      th.classList.add(currentSort.asc ? 'sorted-asc' : 'sorted-desc');
    }
  });

  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';

  // Show up to 500 rows
  const display = spreads.slice(0, 500);

  for (const s of display) {
    const tr = document.createElement('tr');
    const m = 100; // options contract multiplier
    const c = s.contracts;
    const buyEach = (s.buyAsk * m).toLocaleString('en-US', {maximumFractionDigits: 0});
    const sellEach = (s.sellBid * m).toLocaleString('en-US', {maximumFractionDigits: 0});
    const netEach = (s.netPremium * m).toLocaleString('en-US', {maximumFractionDigits: 0});
    const totalDollars = (s.totalPremium * m).toLocaleString('en-US', {maximumFractionDigits: 0});
    tr.innerHTML = `
      <td>${s.expiration}
        <div class="row-tooltip">
          <div class="tt-header">Spread Detail — ${s.expiration} (${s.dte}d) — ${c} contract${c > 1 ? 's' : ''}</div>
          <span class="tt-buy">BUY</span>  ${s.buyStrike.toFixed(0)} call &nbsp;×${c} &nbsp;@ $${s.buyAsk.toFixed(2)} ask <span class="tt-dim">&nbsp; Vol: ${s.volume_buy.toLocaleString()} &nbsp; OI: ${s.oi_buy.toLocaleString()}</span><br>
          <span class="tt-sell">SELL</span> ${s.sellStrike.toFixed(0)} call ×${c} &nbsp;@ $${s.sellBid.toFixed(2)} bid <span class="tt-dim">&nbsp; Vol: ${s.volume_sell.toLocaleString()} &nbsp; OI: ${s.oi_sell.toLocaleString()}</span>
          <div class="tt-sep"></div>
          <span class="tt-buy">Pay:</span> &nbsp;$${buyEach} × ${c} = $${(s.buyAsk * m * c).toLocaleString('en-US', {maximumFractionDigits: 0})}<br>
          <span class="tt-sell">Recv:</span> $${sellEach} × ${c} = $${(s.sellBid * m * c).toLocaleString('en-US', {maximumFractionDigits: 0})}<br>
          <span class="tt-net">Net:&nbsp; $${netEach} × ${c} = $${totalDollars}</span><br>
          <span class="tt-dim">Comm: $${s.commissionPerSpread.toFixed(2)} × ${c} = $${s.totalCommission.toFixed(2)} RT</span>
          <div class="tt-sep"></div>
          ${buildPnlChart(s)}
        </div>
      </td>
      <td>${s.dte}</td>
      <td>${c}</td>
      <td>${s.buyStrike.toFixed(0)}</td>
      <td>${s.sellStrike.toFixed(0)}</td>
      <td class="dim">${s.pctOtmBuy.toFixed(1)}%</td>
      <td>${s.spreadWidth.toFixed(0)}</td>
      <td class="highlight">$${totalDollars}</td>
      <td class="dim">$${(s.midPremium * m * c).toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>$${(s.maxProfit * m).toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>${s.rewardRisk.toFixed(1)}x</td>
      <td>${s.rrPerSigma.toFixed(2)}</td>
      <td class="highlight">${s.leverage.toFixed(1)}x</td>
      <td>$${(s.pnl1pct * m).toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>$${(s.pnl1sigma * m).toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>$${(s.pnl2sigma * m).toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>${s.breakevenMovePct >= 0 ? '+' : ''}${s.breakevenMovePct.toFixed(2)}%</td>
      <td>${s.breakevenMoveSigma >= 0 ? '+' : ''}${s.breakevenMoveSigma.toFixed(2)}σ</td>
      <td>${s.netDelta.toFixed(4)}</td>
      <td>${s.netDeltaPer.toFixed(4)}</td>
      <td>${s.netGamma.toFixed(6)}</td>
      <td>${s.breakeven.toFixed(0)}</td>
      <td class="dim">${s.ivBuy.toFixed(1)}%</td>
      <td class="dim">${s.ivSell.toFixed(1)}%</td>
      <td class="dim">${s.oi_buy.toLocaleString()}</td>
      <td class="dim">${s.oi_sell.toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  }

  document.getElementById('matchCount').textContent =
    spreads.length + (spreads.length !== allSpreads.length ? ` / ${allSpreads.length}` : '');

  if (display.length < spreads.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="21" style="text-align:center;color:var(--text-dim);padding:16px;">
      Showing ${display.length} of ${spreads.length} results. Tighten your criteria to see fewer, more targeted spreads.
    </td>`;
    tbody.appendChild(tr);
  }
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg;
  el.style.display = 'block';
}

function hideError() {
  document.getElementById('errorMsg').style.display = 'none';
}

// Keyboard shortcut
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !document.getElementById('searchBtn').disabled) {
    doSearch();
  }
});

// Tooltip positioning — show near the mouse, flipping if it would overflow
document.addEventListener('mouseover', (e) => {
  const tr = e.target.closest('tbody tr');
  if (!tr) return;
  const tip = tr.querySelector('.row-tooltip');
  if (!tip) return;

  tip.style.display = 'block';

  const rect = tr.getBoundingClientRect();
  const tipRect = tip.getBoundingClientRect();
  const pad = 8;

  // Horizontal: center on the row, but clamp to viewport
  let left = rect.left + rect.width / 2 - tipRect.width / 2;
  left = Math.max(pad, Math.min(left, window.innerWidth - tipRect.width - pad));

  // Vertical: prefer above the row, flip below if clipped
  let top = rect.top - tipRect.height - pad;
  if (top < pad) {
    top = rect.bottom + pad;
  }

  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
});

document.addEventListener('mouseout', (e) => {
  const tr = e.target.closest('tbody tr');
  if (!tr) return;
  const related = e.relatedTarget;
  if (related && tr.contains(related)) return;
  const tip = tr.querySelector('.row-tooltip');
  if (tip) tip.style.display = 'none';
});
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Positions watch page (embedded)
# ---------------------------------------------------------------------------

POSITIONS_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>My Positions</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #232734;
    --border: #2a2f3d;
    --text: #e6e9ef;
    --text-dim: #8b92a5;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --mono: 'JetBrains Mono', 'Fira Code', Consolas, monospace;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; min-height: 100vh; }

  .header { background: var(--surface); border-bottom: 1px solid var(--border); padding: 18px 32px;
            display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header .tag { background: var(--accent); color: #fff; font-size: 11px; padding: 2px 8px;
                 border-radius: 4px; font-weight: 500; letter-spacing: 0.5px; }
  .header .right { margin-left: auto; display: flex; align-items: center; gap: 16px; }
  .header .right .timestamp { font-family: var(--mono); color: var(--text-dim); font-size: 12px; }
  .header .right .timestamp span { color: var(--text); }
  .header a.back { color: var(--accent); text-decoration: none; font-size: 13px; }
  .header a.back:hover { text-decoration: underline; }

  .panel { background: var(--surface); border-bottom: 1px solid var(--border); padding: 20px 32px; }
  .panel h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px;
              color: var(--text-dim); margin-bottom: 14px; }

  .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
               gap: 14px; align-items: end; }
  .form-grid.with-label { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .field input { background: var(--bg); border: 1px solid var(--border); color: var(--text);
                 font-family: var(--mono); font-size: 14px; padding: 9px 12px; border-radius: 6px; outline: none; }
  .field input:focus { border-color: var(--accent); }
  .actions { display: flex; gap: 8px; margin-top: 14px; }
  button.primary { background: var(--accent); color: #fff; border: none; font-size: 14px;
                   font-weight: 600; padding: 10px 22px; border-radius: 6px; cursor: pointer; }
  button.primary:hover { background: var(--accent-hover); }
  button.secondary { background: transparent; color: var(--text); border: 1px solid var(--border);
                     font-size: 14px; padding: 10px 18px; border-radius: 6px; cursor: pointer; }
  button.secondary:hover { border-color: var(--accent); color: var(--accent); }
  button.danger { background: transparent; color: var(--red); border: 1px solid var(--red);
                  font-size: 12px; padding: 5px 10px; border-radius: 4px; cursor: pointer; }
  button.danger:hover { background: var(--red); color: #fff; }
  button.ghost { background: transparent; color: var(--text-dim); border: 1px solid var(--border);
                 font-size: 12px; padding: 5px 10px; border-radius: 4px; cursor: pointer; }
  button.ghost:hover { color: var(--accent); border-color: var(--accent); }

  .refresh-bar { display: flex; align-items: center; gap: 14px; padding: 14px 32px;
                 background: var(--surface2); border-bottom: 1px solid var(--border); font-size: 13px; }
  .refresh-bar .field { flex-direction: row; align-items: center; gap: 8px; }
  .refresh-bar .field label { text-transform: none; letter-spacing: 0; font-size: 13px; color: var(--text-dim); }
  .refresh-bar input[type=number] { width: 70px; padding: 6px 8px; }

  .table-wrap { padding: 20px 32px; overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px;
          overflow: hidden; font-size: 13px; }
  th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
  th:first-child, td:first-child { text-align: left; }
  th { background: var(--surface2); color: var(--text-dim); font-weight: 500;
       text-transform: uppercase; letter-spacing: 0.5px; font-size: 11px; }
  tbody tr:hover { background: var(--surface2); }
  .mono { font-family: var(--mono); }
  .dim { color: var(--text-dim); }
  .pnl-pos { color: var(--green); font-weight: 600; }
  .pnl-neg { color: var(--red); font-weight: 600; }
  .err-row { color: var(--yellow); }

  .empty { padding: 60px 32px; text-align: center; color: var(--text-dim); }
  .err-banner { background: rgba(239, 68, 68, 0.1); color: var(--red); padding: 10px 32px;
                border-bottom: 1px solid var(--red); font-size: 13px; display: none; }
  .err-banner.show { display: block; }

  .leg-block { display: flex; flex-direction: column; gap: 2px; font-family: var(--mono); }
  .leg-block .strike { font-weight: 600; color: var(--text); }
  .leg-block .quote { font-size: 11px; color: var(--text-dim); }

  .pnl-block { display: flex; flex-direction: column; gap: 2px; font-family: var(--mono); }
  .pnl-block .pct { font-size: 11px; }
  .pnl-block .tt-dim { font-size: 10px; color: var(--text-dim); font-family: var(--mono); }

  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .stat { background: var(--surface2); border-radius: 6px; padding: 12px 14px;
          display: flex; flex-direction: column; gap: 4px; }
  .stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-value { font-size: 18px; font-weight: 600; }

  .spark { display: inline-block; vertical-align: middle; }
  .greek-block { display: flex; flex-direction: column; gap: 2px; font-family: var(--mono); font-size: 12px; }
  .greek-block .lbl { color: var(--text-dim); font-size: 10px; }
</style>
</head>
<body>

<div class="header">
  <h1>My Positions <span class="tag">WATCH</span></h1>
  <div class="right">
    <span class="timestamp">Updated: <span id="lastUpdate">--</span></span>
    <span class="timestamp">Next refresh in: <span id="countdown">--</span>s</span>
    <a class="back" href="/">&larr; Spread Finder</a>
  </div>
</div>

<div class="err-banner" id="errBanner"></div>

<div class="panel">
  <h2 id="formTitle">Add New Position</h2>
  <form id="positionForm">
    <input type="hidden" id="positionId" />
    <div class="form-grid">
      <div class="field"><label>Symbol</label><input id="symbol" required placeholder="QQQ" style="text-transform:uppercase;"></div>
      <div class="field"><label>Expiration (YYYY-MM-DD)</label><input id="expiration" required placeholder="2026-12-18"></div>
      <div class="field"><label>Long Strike</label><input id="longStrike" type="number" step="0.5" required></div>
      <div class="field"><label>Short Strike</label><input id="shortStrike" type="number" step="0.5" required></div>
      <div class="field"><label>Contracts</label><input id="contracts" type="number" min="1" step="1" value="1" required></div>
      <div class="field"><label>Long Entry $</label><input id="longEntryPrice" type="number" step="0.01" required></div>
      <div class="field"><label>Short Entry $</label><input id="shortEntryPrice" type="number" step="0.01" required></div>
      <div class="field"><label>Entry Commission $</label><input id="entryCommission" type="number" step="0.01" value="8.95"></div>
      <div class="field"><label>Label (optional)</label><input id="label" placeholder="QQQ Dec 750/795 bull"></div>
    </div>
    <div class="actions">
      <button type="submit" class="primary" id="saveBtn">Save Position</button>
      <button type="button" class="secondary" id="cancelEditBtn" style="display:none;">Cancel Edit</button>
    </div>
  </form>
</div>

<div class="refresh-bar">
  <div class="field">
    <label for="refreshSec">Refresh every</label>
    <input id="refreshSec" type="number" min="5" step="5" value="30">
    <span class="dim">seconds</span>
  </div>
  <div class="field">
    <label for="haircutPct">Adj P&amp;L haircut</label>
    <input id="haircutPct" type="number" min="0" max="100" step="1" value="80">
    <span class="dim">%</span>
  </div>
  <button class="ghost" id="refreshNowBtn">Refresh now</button>
  <span class="dim" id="autoStatus" style="margin-left:auto;">auto-refresh: <span style="color:var(--green);">on</span></span>
</div>

<div class="panel" id="summaryPanel" style="display:none;">
  <h2>Portfolio Summary</h2>
  <div class="summary-grid">
    <div class="stat"><div class="stat-label">Positions</div><div class="stat-value mono" id="sumCount">0</div></div>
    <div class="stat"><div class="stat-label">Entry Cost</div><div class="stat-value mono" id="sumEntry">--</div></div>
    <div class="stat"><div class="stat-label">Current Value</div><div class="stat-value mono" id="sumCurrent">--</div></div>
    <div class="stat"><div class="stat-label">Total P&amp;L</div><div class="stat-value mono" id="sumPnl">--</div></div>
    <div class="stat"><div class="stat-label" id="sumAdjPnlLabel">Adj P&amp;L (80%)</div><div class="stat-value mono" id="sumAdjPnl">--</div></div>
    <div class="stat"><div class="stat-label">Total Return</div><div class="stat-value mono" id="sumRet">--</div></div>
    <div class="stat"><div class="stat-label">Net &Delta;+&Gamma; P&amp;L (1&sigma; move)</div><div class="stat-value mono" id="sumDelta">--</div></div>
    <div class="stat"><div class="stat-label">Net &Theta; ($/day)</div><div class="stat-value mono" id="sumTheta">--</div></div>
    <div class="stat"><div class="stat-label">Net Vega ($/1% IV)</div><div class="stat-value mono" id="sumVega">--</div></div>
  </div>
</div>

<div class="table-wrap">
  <div class="empty" id="emptyState">No positions saved. Add one above to start watching.</div>
  <table id="posTable" style="display:none;">
    <thead>
      <tr>
        <th>Label / Symbol</th>
        <th>Expiration (DTE)</th>
        <th>Spot</th>
        <th>Long Leg (bid / ask / last / vol / IV)</th>
        <th>Short Leg (bid / ask / last / vol / IV)</th>
        <th>Contracts</th>
        <th>Spread Mid</th>
        <th>Liquidation</th>
        <th>Entry Cost</th>
        <th>Current Value</th>
        <th>P&amp;L</th>
        <th id="colAdjPnlLabel">Adj P&amp;L (80%)</th>
        <th>Greeks (&Delta;+&Gamma; 1&sigma; P&amp;L / &Theta;$/d / Vega)</th>
        <th>Chart</th>
        <th></th>
      </tr>
    </thead>
    <tbody id="posBody"></tbody>
  </table>
</div>

<script>
let refreshTimer = null;
let countdownTimer = null;
let nextRefreshAt = 0;
let positions = [];     // raw from /api/positions
let lastQuotes = [];    // joined w/ quotes

function $(id) { return document.getElementById(id); }
function dollarFmt(v, dp=2) { return (v >= 0 ? '$' : '-$') + Math.abs(v).toFixed(dp); }
function sign(v) { return (v >= 0 ? '+' : ''); }

function showErr(msg) { const b = $('errBanner'); b.textContent = msg; b.classList.add('show'); }
function clearErr() { $('errBanner').classList.remove('show'); }

function dteFromExp(expStr) {
  const exp = new Date(expStr + 'T16:00:00');
  return Math.round((exp - new Date()) / 86400000);
}

function sparkline(history, w=100, h=28) {
  if (!history || history.length < 2) return '<span class="dim">–</span>';
  const ys = history.map(p => p[1]);
  const min = Math.min(...ys), max = Math.max(...ys);
  const range = max - min || 1;
  const pts = ys.map((y, i) => {
    const x = (i / (ys.length - 1)) * (w - 2) + 1;
    const py = h - 2 - ((y - min) / range) * (h - 4);
    return `${x.toFixed(1)},${py.toFixed(1)}`;
  }).join(' ');
  const color = ys[ys.length - 1] >= ys[0] ? 'var(--green)' : 'var(--red)';
  const zeroLine = min <= 0 && max >= 0
    ? `<line x1="1" x2="${w-1}" y1="${(h - 2 - ((0 - min) / range) * (h - 4)).toFixed(1)}" y2="${(h - 2 - ((0 - min) / range) * (h - 4)).toFixed(1)}" stroke="var(--text-dim)" stroke-width="0.5" stroke-dasharray="2,2"/>`
    : '';
  return `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    ${zeroLine}
    <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.3"/>
  </svg>`;
}

function fmtSignedDollar(v, dp=0) {
  return (v >= 0 ? '+$' : '-$') + Math.abs(v).toFixed(dp);
}

function updateSummary(rows) {
  const withData = rows.filter(r => r.pnl !== null && r.pnl !== undefined && r.entryCost !== null);
  const panel = $('summaryPanel');
  if (withData.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';

  const totalEntry = withData.reduce((s, r) => s + r.entryCost, 0);
  const totalCurrent = withData.reduce((s, r) => s + (r.currentValue || 0), 0);
  const totalPnl = withData.reduce((s, r) => s + r.pnl, 0);
  const totalAdjPnl = withData.reduce((s, r) => s + (r.adjPnl || 0), 0);
  const totalRet = totalEntry ? (totalAdjPnl / Math.abs(totalEntry)) * 100 : 0;
  const totalDelta = withData.reduce((s, r) => s + (r.netDelta || 0), 0);
  const totalOneSigmaPnl = withData.reduce((s, r) => s + (r.oneSigmaPnl || 0), 0);
  const totalTheta = withData.reduce((s, r) => s + (r.netThetaPerDay || 0), 0);
  const totalVega = withData.reduce((s, r) => s + (r.netVega || 0), 0);

  const setPnlText = (id, val, dp=0) => {
    const el = $(id);
    el.textContent = fmtSignedDollar(val, dp);
    el.className = 'stat-value mono ' + (val >= 0 ? 'pnl-pos' : 'pnl-neg');
  };

  $('sumCount').textContent = rows.length;
  $('sumEntry').textContent = '$' + totalEntry.toLocaleString('en-US', {maximumFractionDigits: 0});
  $('sumCurrent').textContent = '$' + totalCurrent.toLocaleString('en-US', {maximumFractionDigits: 0});
  setPnlText('sumPnl', totalPnl);
  const retEl = $('sumRet');
  retEl.textContent = (totalRet >= 0 ? '+' : '') + totalRet.toFixed(2) + '%';
  retEl.className = 'stat-value mono ' + (totalRet >= 0 ? 'pnl-pos' : 'pnl-neg');
  setPnlText('sumAdjPnl', totalAdjPnl);
  // Reflect the active haircut % on the label and column header
  const activeHc = currentHaircutPct().toFixed(0);
  $('sumAdjPnlLabel').textContent = `Adj P&L (${activeHc}%)`;
  const colHdr = $('colAdjPnlLabel');
  if (colHdr) colHdr.textContent = `Adj P&L (${activeHc}%)`;
  $('sumDelta').textContent = fmtSignedDollar(totalOneSigmaPnl, 0);
  $('sumDelta').className = 'stat-value mono ' + (totalOneSigmaPnl >= 0 ? 'pnl-pos' : 'pnl-neg');
  $('sumTheta').textContent = fmtSignedDollar(totalTheta, 2);
  $('sumTheta').className = 'stat-value mono ' + (totalTheta >= 0 ? 'pnl-pos' : 'pnl-neg');
  $('sumVega').textContent = fmtSignedDollar(totalVega, 2);
}

async function loadPositions() {
  try {
    const r = await fetch('/api/positions');
    positions = await r.json();
    if (positions.length === 0) {
      $('emptyState').style.display = 'block';
      $('posTable').style.display = 'none';
      return;
    }
    await refreshQuotes();
  } catch (e) {
    showErr('Failed to load positions: ' + e.message);
  }
}

function currentHaircutPct() {
  const raw = parseFloat($('haircutPct').value);
  if (!isFinite(raw)) return 80;
  return Math.max(0, Math.min(raw, 100));
}

async function refreshQuotes() {
  if (positions.length === 0) return;
  try {
    clearErr();
    const hc = currentHaircutPct();
    const r = await fetch('/api/positions/quotes?haircut=' + encodeURIComponent(hc));
    const data = await r.json();
    if (data.error) { showErr(data.error); return; }
    lastQuotes = data.positions;
    renderTable();
    $('lastUpdate').textContent = data.timestamp;
  } catch (e) {
    showErr('Quote refresh failed: ' + e.message);
  }
}

function renderTable() {
  const body = $('posBody');
  body.innerHTML = '';
  if (lastQuotes.length === 0) {
    $('emptyState').style.display = 'block';
    $('posTable').style.display = 'none';
    $('summaryPanel').style.display = 'none';
    return;
  }
  $('emptyState').style.display = 'none';
  $('posTable').style.display = 'table';

  for (const p of lastQuotes) {
    const tr = document.createElement('tr');
    const dte = dteFromExp(p.expiration);
    const legCell = (leg, strike) => {
      if (!leg) return `<td class="err-row">missing</td>`;
      const ivStr = leg.iv !== null && leg.iv !== undefined ? leg.iv.toFixed(1) + '%' : '--';
      return `<td><div class="leg-block">
        <span class="strike">${strike}</span>
        <span class="quote">${leg.bid.toFixed(2)} / ${leg.ask.toFixed(2)} / ${leg.last.toFixed(2)} / ${leg.volume.toLocaleString()} / ${ivStr}</span>
      </div></td>`;
    };

    const greeksCell = (p.netDelta === null || p.netDelta === undefined) ? '<td class="dim">--</td>' :
      `<td><div class="greek-block">
        <span class="${(p.oneSigmaPnl || 0) >= 0 ? 'pnl-pos' : 'pnl-neg'}">${fmtSignedDollar(p.oneSigmaPnl || 0, 0)}</span>
        <span class="${p.netThetaPerDay >= 0 ? 'pnl-pos' : 'pnl-neg'}">${p.netThetaPerDay >= 0 ? '+' : ''}$${p.netThetaPerDay.toFixed(2)}</span>
        <span>${p.netVega >= 0 ? '+' : ''}$${p.netVega.toFixed(2)}</span>
      </div></td>`;

    const chartCell = `<td>${sparkline(p.history || [])}</td>`;

    let pnlCell = '<td class="dim">--</td>';
    if (p.pnl !== null && p.pnl !== undefined) {
      const cls = p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      pnlCell = `<td><div class="pnl-block">
        <span class="${cls}">${sign(p.pnl)}${dollarFmt(p.pnl, 0)}</span>
        <span class="${cls} pct">${sign(p.pnlPct)}${p.pnlPct.toFixed(2)}%</span>
      </div></td>`;
    }

    let adjPnlCell = '<td class="dim">--</td>';
    if (p.adjPnl !== null && p.adjPnl !== undefined) {
      const cls = p.adjPnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      const totalComm = (p.totalCommission != null) ? p.totalCommission : 0;
      const hcLabel = (p.haircutPct != null) ? p.haircutPct.toFixed(0) : '80';
      const derivation = (p.pnl > 0)
        ? `pnl × ${hcLabel}% − $${totalComm.toFixed(2)} comm`
        : `pnl − $${totalComm.toFixed(2)} comm (no haircut on loss)`;
      adjPnlCell = `<td><div class="pnl-block">
        <span class="${cls}">${sign(p.adjPnl)}${dollarFmt(p.adjPnl, 0)}</span>
        <span class="${cls} pct">${sign(p.adjPnlPct)}${p.adjPnlPct.toFixed(2)}%</span>
        <span class="tt-dim">${derivation}</span>
      </div></td>`;
    }

    const label = p.label || `${p.symbol} ${p.longStrike}/${p.shortStrike}`;
    const spotCell = p.spot !== null && p.spot !== undefined ? `$${p.spot.toFixed(2)}` : '--';
    const midCell = p.spreadMid !== null ? `$${p.spreadMid.toFixed(2)}` : '--';
    const liqCell = p.spreadLiquidation !== null ? `$${p.spreadLiquidation.toFixed(2)}` : '--';
    const curCell = p.currentValue !== null ? dollarFmt(p.currentValue, 0) : '--';
    let entryCell = '--';
    if (p.entryCost !== null && p.entryCost !== undefined) {
      const entryComm = (p.entryCommission != null) ? p.entryCommission : 0;
      entryCell = `<div class="pnl-block">
        <span class="mono">${dollarFmt(p.entryCost, 0)}</span>
        <span class="tt-dim">entry comm $${entryComm.toFixed(2)}</span>
      </div>`;
    }

    tr.innerHTML = `
      <td><div class="leg-block"><span class="strike">${label}</span><span class="quote">${p.symbol}</span></div></td>
      <td class="mono">${p.expiration} <span class="dim">(${dte}d)</span></td>
      <td class="mono">${spotCell}</td>
      ${legCell(p.long, p.longStrike)}
      ${legCell(p.short, p.shortStrike)}
      <td class="mono">${p.contracts}</td>
      <td class="mono">${midCell}</td>
      <td class="mono">${liqCell}</td>
      <td class="mono">${entryCell}</td>
      <td class="mono">${curCell}</td>
      ${pnlCell}
      ${adjPnlCell}
      ${greeksCell}
      ${chartCell}
      <td>
        <button class="ghost" data-edit="${p.id}">Edit</button>
        <button class="danger" data-del="${p.id}">Delete</button>
      </td>
    `;
    if (p.error) {
      const errTd = document.createElement('tr');
      errTd.innerHTML = `<td colspan="15" class="err-row dim">${p.symbol} ${p.longStrike}/${p.shortStrike}: ${p.error}</td>`;
      body.appendChild(tr);
      body.appendChild(errTd);
    } else {
      body.appendChild(tr);
    }
  }

  body.querySelectorAll('[data-edit]').forEach(b => b.onclick = () => beginEdit(b.dataset.edit));
  body.querySelectorAll('[data-del]').forEach(b => b.onclick = () => deletePosition(b.dataset.del));

  updateSummary(lastQuotes);
}

function beginEdit(id) {
  const p = positions.find(x => x.id === id);
  if (!p) return;
  $('positionId').value = p.id;
  $('symbol').value = p.symbol;
  $('expiration').value = p.expiration;
  $('longStrike').value = p.longStrike;
  $('shortStrike').value = p.shortStrike;
  $('contracts').value = p.contracts;
  $('longEntryPrice').value = p.longEntryPrice;
  $('shortEntryPrice').value = p.shortEntryPrice;
  $('entryCommission').value = p.entryCommission || 0;
  $('label').value = p.label || '';
  $('formTitle').textContent = 'Edit Position';
  $('saveBtn').textContent = 'Update Position';
  $('cancelEditBtn').style.display = 'inline-block';
  window.scrollTo({top: 0, behavior: 'smooth'});
}

function resetForm() {
  $('positionForm').reset();
  $('positionId').value = '';
  $('formTitle').textContent = 'Add New Position';
  $('saveBtn').textContent = 'Save Position';
  $('cancelEditBtn').style.display = 'none';
  $('contracts').value = 1;
  $('entryCommission').value = 8.95;
}

async function deletePosition(id) {
  const p = positions.find(x => x.id === id);
  if (!confirm(`Delete ${p ? (p.label || p.symbol + ' ' + p.longStrike + '/' + p.shortStrike) : 'this position'}?`)) return;
  try {
    const r = await fetch('/api/positions/' + id, {method: 'DELETE'});
    const j = await r.json();
    if (j.error) { showErr(j.error); return; }
    await loadPositions();
  } catch (e) { showErr('Delete failed: ' + e.message); }
}

$('positionForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    symbol: $('symbol').value.trim().toUpperCase(),
    expiration: $('expiration').value.trim(),
    longStrike: parseFloat($('longStrike').value),
    shortStrike: parseFloat($('shortStrike').value),
    contracts: parseInt($('contracts').value, 10),
    longEntryPrice: parseFloat($('longEntryPrice').value),
    shortEntryPrice: parseFloat($('shortEntryPrice').value),
    entryCommission: parseFloat($('entryCommission').value) || 0,
    label: $('label').value.trim()
  };
  const id = $('positionId').value;
  try {
    const url = id ? '/api/positions/' + id : '/api/positions';
    const method = id ? 'PUT' : 'POST';
    const r = await fetch(url, {method, headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    const j = await r.json();
    if (j.error) { showErr(j.error); return; }
    resetForm();
    await loadPositions();
  } catch (e) { showErr('Save failed: ' + e.message); }
});

$('cancelEditBtn').addEventListener('click', resetForm);
$('refreshNowBtn').addEventListener('click', () => { refreshQuotes(); resetTimer(); });
$('refreshSec').addEventListener('change', resetTimer);

// Persist haircut % across sessions and re-refresh when it changes.
const _savedHaircut = localStorage.getItem('adjPnlHaircutPct');
if (_savedHaircut !== null && isFinite(parseFloat(_savedHaircut))) {
  $('haircutPct').value = _savedHaircut;
}
$('haircutPct').addEventListener('change', () => {
  localStorage.setItem('adjPnlHaircutPct', $('haircutPct').value);
  refreshQuotes();
});

function resetTimer() {
  if (refreshTimer) clearInterval(refreshTimer);
  if (countdownTimer) clearInterval(countdownTimer);
  const sec = Math.max(5, parseInt($('refreshSec').value, 10) || 30);
  nextRefreshAt = Date.now() + sec * 1000;
  refreshTimer = setInterval(() => {
    refreshQuotes();
    nextRefreshAt = Date.now() + sec * 1000;
  }, sec * 1000);
  countdownTimer = setInterval(updateCountdown, 250);
  updateCountdown();
}

function updateCountdown() {
  const remaining = Math.max(0, Math.ceil((nextRefreshAt - Date.now()) / 1000));
  $('countdown').textContent = remaining;
}

loadPositions();
resetTimer();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class SpreadHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            page = HTML_PAGE.replace("{RISK_FREE_RATE_PCT}", str(RISK_FREE_RATE_PCT))
            self.wfile.write(page.encode("utf-8"))

        elif parsed.path == "/api/spreads":
            params = parse_qs(parsed.query)
            min_premium = float(params.get("min_premium", [0])[0])
            max_premium = float(params.get("max_premium", [20])[0])
            min_leverage = float(params.get("min_leverage", [2])[0])
            max_width = float(params.get("max_width", [100])[0])
            max_otm = float(params.get("max_otm", [5.0])[0])
            risk_free_rate = float(params.get("risk_free_rate", [0.045])[0])
            expiration_filter = params.get("expiration", ["all"])[0]
            min_net_delta = float(params.get("min_net_delta", [0.33])[0])
            min_reward_risk = float(params.get("min_reward_risk", [0.5])[0])
            commission = float(params.get("commission", [35.80])[0])
            min_dte = int(float(params.get("min_dte", [30])[0]))
            max_leg_premium = float(params.get("max_leg_premium", [20000])[0])
            min_leg_premium = float(params.get("min_leg_premium", [0])[0])
            symbol = params.get("symbol", ["^SPX"])[0].strip().upper()
            move_pct = float(params.get("move_pct", [1.0])[0])

            try:
                print(f"\n{'='*60}")
                print(f"Searching {symbol}: premium=${min_premium}-${max_premium}, min_leverage={min_leverage}x, max_width={max_width}pts, max_otm={max_otm}%, r={risk_free_rate:.3f}, min_delta={min_net_delta}, min_rr={min_reward_risk}, commission=${commission}, min_dte={min_dte}, max_leg_premium=${max_leg_premium}, min_leg_premium=${min_leg_premium}, move_pct={move_pct}%, expiration={expiration_filter}")
                print(f"{'='*60}")
                result = fetch_and_find_spreads(min_premium, max_premium, min_leverage, max_width, max_otm, risk_free_rate, expiration_filter, min_net_delta, min_reward_risk, commission, min_dte, max_leg_premium, min_leg_premium, symbol, move_pct)
                print(f"Found {result['total_spreads']} matching spreads across {result['expirations_scanned']} expirations")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                self.send_response(200)  # 200 with error in body for clean frontend handling
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        elif parsed.path == "/positions":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(POSITIONS_PAGE.encode("utf-8"))

        elif parsed.path == "/api/positions":
            self._send_json(load_positions())

        elif parsed.path == "/api/positions/quotes":
            try:
                params = parse_qs(parsed.query)
                haircut_raw = params.get("haircut", ["80"])[0]
                try:
                    haircut_val = float(haircut_raw)
                except ValueError:
                    haircut_val = 80.0
                # Accept either 0.80 (fraction) or 80 (percent) — normalize to fraction.
                haircut_pct = haircut_val / 100.0 if haircut_val > 1.5 else haircut_val
                haircut_pct = max(0.0, min(haircut_pct, 1.0))
                positions = load_positions()
                quotes = fetch_position_quotes(positions, haircut_pct=haircut_pct)
                self._send_json({
                    "positions": quotes,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "haircutPct": round(haircut_pct * 100, 2),
                })
            except Exception as e:
                self._send_json({"error": str(e)})

        elif parsed.path == "/api/templates":
            self._send_json(load_templates())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/positions":
            try:
                body = self._read_json_body()
                self._validate_position_payload(body)
                positions = load_positions()
                entry = {
                    "id": uuid.uuid4().hex,
                    "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **body,
                }
                positions.append(entry)
                save_positions(positions)
                self._send_json(entry)
            except Exception as e:
                self._send_json({"error": str(e)})
        elif parsed.path == "/api/templates":
            try:
                body = self._read_json_body()
                self._validate_template_payload(body)
                templates = load_templates()
                entry = {
                    "id": uuid.uuid4().hex,
                    "createdAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **body,
                }
                templates.append(entry)
                save_templates(templates)
                self._send_json(entry)
            except Exception as e:
                self._send_json({"error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_PUT(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/positions/"):
            pos_id = parsed.path.rsplit("/", 1)[-1]
            try:
                body = self._read_json_body()
                self._validate_position_payload(body)
                positions = load_positions()
                for i, p in enumerate(positions):
                    if p.get("id") == pos_id:
                        positions[i] = {**p, **body}
                        save_positions(positions)
                        self._send_json(positions[i])
                        return
                self._send_json({"error": f"position {pos_id} not found"})
            except Exception as e:
                self._send_json({"error": str(e)})
        elif parsed.path.startswith("/api/templates/"):
            tpl_id = parsed.path.rsplit("/", 1)[-1]
            try:
                body = self._read_json_body()
                self._validate_template_payload(body)
                templates = load_templates()
                for i, t in enumerate(templates):
                    if t.get("id") == tpl_id:
                        templates[i] = {**t, **body}
                        save_templates(templates)
                        self._send_json(templates[i])
                        return
                self._send_json({"error": f"template {tpl_id} not found"})
            except Exception as e:
                self._send_json({"error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/positions/"):
            pos_id = parsed.path.rsplit("/", 1)[-1]
            try:
                positions = load_positions()
                new_positions = [p for p in positions if p.get("id") != pos_id]
                if len(new_positions) == len(positions):
                    self._send_json({"error": f"position {pos_id} not found"})
                    return
                save_positions(new_positions)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"error": str(e)})
        elif parsed.path.startswith("/api/templates/"):
            tpl_id = parsed.path.rsplit("/", 1)[-1]
            try:
                templates = load_templates()
                new_templates = [t for t in templates if t.get("id") != tpl_id]
                if len(new_templates) == len(templates):
                    self._send_json({"error": f"template {tpl_id} not found"})
                    return
                save_templates(new_templates)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    @staticmethod
    def _validate_template_payload(body):
        name = str(body.get("name") or "").strip()
        if not name:
            raise ValueError("template name required")
        params = body.get("params")
        if not isinstance(params, dict) or not params:
            raise ValueError("template params must be a non-empty object")
        body["name"] = name
        body["params"] = params

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    @staticmethod
    def _validate_position_payload(body):
        required = ["symbol", "expiration", "longStrike", "shortStrike", "contracts",
                    "longEntryPrice", "shortEntryPrice"]
        missing = [k for k in required if k not in body or body[k] in (None, "")]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        body["symbol"] = str(body["symbol"]).strip().upper()
        body["expiration"] = str(body["expiration"]).strip()
        body["longStrike"] = float(body["longStrike"])
        body["shortStrike"] = float(body["shortStrike"])
        body["contracts"] = int(body["contracts"])
        body["longEntryPrice"] = float(body["longEntryPrice"])
        body["shortEntryPrice"] = float(body["shortEntryPrice"])
        body["entryCommission"] = float(body.get("entryCommission") or 0)
        body["label"] = str(body.get("label") or "").strip()
        if body["contracts"] <= 0:
            raise ValueError("contracts must be positive")
        if body["longStrike"] >= body["shortStrike"]:
            raise ValueError("longStrike must be less than shortStrike for a bull call spread")

    def log_message(self, format, *args):
        # Suppress default request logging to keep console clean
        pass


def main():
    print(f"""
    ╔══════════════════════════════════════════════╗
    ║       Call Spread Finder                      ║
    ║                                              ║
    ║   Open: http://localhost:{PORT}                ║
    ║   Press Ctrl+C to stop                       ║
    ╚══════════════════════════════════════════════╝
    """)

    with socketserver.TCPServer(("", PORT), SpreadHandler) as httpd:
        httpd.allow_reuse_address = True
        # Open browser after a short delay
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            httpd.shutdown()


if __name__ == "__main__":
    main()
