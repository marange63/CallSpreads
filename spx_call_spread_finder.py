#!/usr/bin/env python3
"""
Call Spread Finder
==================
Finds bull call spread candidates using live Yahoo Finance option data.

Usage:
    python spx_call_spread_finder.py

Then open http://localhost:8765 in your browser.
"""

import errno
import http.server
import json
import math
import os
import pickle
import sys
import threading
import uuid
import webbrowser
import socketserver
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Market data: all vendor access goes through the pluggable PriceSource layer
# (price_sources.py, which also owns the yfinance import + auto-install).
# This file must contain no direct yf.* calls.
# ---------------------------------------------------------------------------
from price_sources import (_YAHOO_SOURCE, SOURCES, get_source,
                           set_active_source, register_sources)

import numpy as np  # ships with pandas/yfinance; used for the Monte Carlo path engine
import pandas as pd  # ships with yfinance; used for the beta regression on daily returns

PORT = 8765

# ---------------------------------------------------------------------------
# Fetch current risk-free rate (3-month Treasury bill yield from Yahoo Finance)
# ---------------------------------------------------------------------------

def fetch_risk_free_rate():
    """Fetch the current 3-month T-bill rate from Yahoo Finance (^IRX).
    Returns the rate as a percentage (e.g. 4.5 for 4.5%). Falls back to 4.5 on failure.

    Deliberately pinned to _YAHOO_SOURCE, not get_source(): this runs at module
    import, before sources.json is loaded, and the T-bill rate is reference
    data where Yahoo's delay is irrelevant."""
    try:
        hist = _YAHOO_SOURCE.get_daily_closes("^IRX", "5d")
        if not hist.empty:
            rate = float(hist.iloc[-1])
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
# Path-aware profit-target probability (Monte Carlo)
# ---------------------------------------------------------------------------
#
# Estimates P[ a spread's Adjusted P&L touches a profit target at any point before
# expiry ]. This is a first-passage problem: the target is a *constant* barrier in
# spread-VALUE space (V*, see spread_mc_stats) but a *moving* barrier in underlying
# space because the spread reprices as theta decays. We simulate zero-drift GBM paths
# for the underlying ONCE per expiration, invert V* into a per-timestep critical
# underlying level, then test how many paths ever cross it. Assumptions: zero drift
# (conservative "no edge"), per-leg IVs frozen along the path, ATM IV drives diffusion,
# and Black-Scholes *mid* prices the path (the haircut in Adjusted P&L stands in for the
# mid->liquidation bid/ask crossing). Daily monitoring slightly undercounts intraday
# touches; finer stepping for short-DTE is the mitigation.

# Haircut the Finder assumes when computing P(+X%), matching the Monitor's default so a
# spread's probability reads identically on both pages.
FINDER_HAIRCUT_PCT = 0.80

# Monte Carlo settings. Paths are shared across every spread of an expiration, so the
# Finder can afford a modest count; the Monitor has few positions so it uses more.
MC_PATHS_FINDER = 2000
MC_PATHS_MONITOR = 10000
MC_SEED = 12345  # fixed so the displayed probability doesn't jitter purely from the RNG


def _norm_cdf_np(x):
    """Vectorized standard-normal CDF (Zelen & Severo approximation, |err| < 7.5e-8)."""
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    d = 0.3989422804014327 * np.exp(-x * x / 2.0)
    p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
        + t * (-1.821255978 + t * 1.330274429))))
    return np.where(x >= 0.0, 1.0 - p, p)


def _bs_call_price_vec(S, K, T, r, sigma):
    """Black-Scholes call price, vectorized over same-shaped arrays S and T (K/r/sigma scalar)."""
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    pos = T > 0
    if sigma <= 0:
        return np.where(pos, np.maximum(S - K * np.exp(-r * np.where(pos, T, 0.0)), 0.0),
                        np.maximum(S - K, 0.0))
    Tp = np.where(pos, T, 1.0)  # dummy value where T<=0; masked out below
    sqrtT = np.sqrt(Tp)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * Tp) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    priced = S * _norm_cdf_np(d1) - K * np.exp(-r * Tp) * _norm_cdf_np(d2)
    return np.where(pos, priced, np.maximum(S - K, 0.0))


def simulate_paths(spot, atm_iv, T, n_paths, n_steps, drift=0.0, seed=MC_SEED):
    """Simulate zero-drift GBM underlying paths on a daily grid.

    Returns (S_matrix, T_remaining) where S_matrix has shape (n_paths, n_steps) holding
    the underlying price at each future step (step k is (k+1) days ahead; the last step is
    expiry) and T_remaining[k] is the time-to-expiry in years at that step. With drift=0
    the price has zero expected return (E[S_t] = spot); the -0.5*sigma^2 term is the Ito
    correction. Returns (None, None) for degenerate inputs.
    """
    if not atm_iv or atm_iv <= 0 or T <= 0 or n_steps < 1:
        return None, None
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift_term = (drift - 0.5 * atm_iv * atm_iv) * dt
    diff_term = atm_iv * math.sqrt(dt)
    z = rng.standard_normal((int(n_paths), int(n_steps)))
    log_paths = np.cumsum(drift_term + diff_term * z, axis=1)
    S = spot * np.exp(log_paths)
    steps = np.arange(1, n_steps + 1)
    T_remaining = T - steps * dt
    return S, T_remaining


def critical_levels(K1, K2, iv_l, iv_s, r, T_remaining, V_star, width):
    """Smallest underlying level whose bull-call-spread value >= V_star, per remaining time.

    Spread value is monotonically increasing in S (0 -> width*exp(-r*T)), so we bisect.
    Returns an array aligned with T_remaining; np.inf where V_star exceeds the reachable
    maximum at that step (the target simply cannot be hit then).
    """
    T_rem = np.asarray(T_remaining, dtype=float)
    S_star = np.full(T_rem.shape, np.inf)
    # Deep-ITM asymptote of the spread value: width*exp(-r*T) (=width at expiry).
    asymptote = np.where(T_rem > 0, width * np.exp(-r * np.maximum(T_rem, 0.0)), width)
    reachable = asymptote >= V_star
    if not np.any(reachable):
        return S_star
    idx = np.where(reachable)[0]
    Tr = T_rem[idx]
    lo = np.zeros_like(Tr)
    hi = np.full_like(Tr, max(K2, width) * 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = (_bs_call_price_vec(mid, K1, Tr, r, iv_l)
               - _bs_call_price_vec(mid, K2, Tr, r, iv_s))
        ge = val >= V_star
        hi = np.where(ge, mid, hi)
        lo = np.where(ge, lo, mid)
    S_star[idx] = hi
    return S_star


def spread_mc_stats(S_paths, T_remaining, K1, K2, iv_l, iv_s, r, T,
                    entry_debit, contracts, target_frac,
                    total_commission, haircut, width, value_offset=0.0):
    """Monte-Carlo statistics for a spread, all from the shared per-expiration paths.

    Returns None for degenerate inputs, else a dict:
      probTarget         — % of paths whose Adjusted P&L touches the profit target
                           before expiry (first-passage / early-exit probability).
      medianDaysToTarget — median first-touch day among the hitting paths (None if 0%).
      retVelocity        — target %-of-cost per day at the median touch (None if 0%).
      evPrem             — expected Adjusted P&L as a % of entry cost under the
                           exit-at-target strategy: hitting paths bank the target,
                           the rest hold to expiry at terminal intrinsic value.
      probProfitExp      — % of paths whose *terminal* Adjusted P&L is positive
                           (commission- and haircut-aware P(profit at expiration)).
      probLoss50         — % of paths whose terminal Adjusted P&L <= -50% of cost.

    target_frac is the target as a fraction of entry cost (e.g. 0.05 for 5%). Adjusted P&L
    is (liquidation - entry)*mult, haircut on gains, minus commission (the exact formula
    the position table displays), so the required per-share liquidation value is
    L* = entry_debit + (target_dollars + total_commission) / (haircut * 100 * contracts).

    Along the path we only have a Black-Scholes *mid*, which does not equal the current
    quoted liquidation (BS-vs-market model error plus the bid/ask crossing). `value_offset`
    = BS_mid(spot, T) - current_liquidation calibrates that gap so the barrier is anchored
    to the real current value the user sees: the underlying barrier is where
    BS_mid(S, t) reaches L* + value_offset.
    """
    if S_paths is None or T <= 0 or not iv_l or not iv_s or contracts <= 0 or entry_debit <= 0:
        return None
    if haircut <= 0 or width <= 0:
        return None
    mult = 100.0 * contracts
    entry_cost = entry_debit * mult
    target_dollars = target_frac * entry_cost
    # Liquidation-space barrier, then shift into BS-mid space via the calibration offset.
    L_star = entry_debit + (target_dollars + total_commission) / (haircut * mult)
    V_star = L_star + value_offset

    S = np.asarray(S_paths)

    # First-passage (touch) stats. V_star > width means the target is unreachable.
    prob_target = 0.0
    median_days = None
    ret_velocity = None
    hit = None
    if V_star <= width:
        S_star = critical_levels(K1, K2, iv_l, iv_s, r, T_remaining, V_star, width)
        touched = S >= S_star[None, :]
        hit = touched.any(axis=1)
        prob_target = float(hit.mean()) * 100.0
        if hit.any():
            # Step k is (k+1) days ahead (see simulate_paths).
            first_day = np.argmax(touched, axis=1)[hit] + 1
            median_days = float(np.median(first_day))
            if median_days > 0:
                ret_velocity = target_frac * 100.0 / median_days

    # Terminal (hold-to-expiry) stats: Adjusted P&L at expiration, same formula as
    # the position table — haircut on gains only, commissions netted in full.
    intrinsic = np.clip(S[:, -1] - K1, 0.0, width)
    raw_pnl = (intrinsic - entry_debit) * mult
    adj_pnl = np.where(raw_pnl > 0, raw_pnl * haircut, raw_pnl) - total_commission
    prob_profit_exp = float((adj_pnl > 0).mean()) * 100.0
    prob_loss50 = float((adj_pnl <= -0.5 * entry_cost).mean()) * 100.0
    # Exit-at-target expectancy: hitting paths bank the target, others ride to expiry.
    strat_pnl = adj_pnl if hit is None else np.where(hit, target_dollars, adj_pnl)
    ev_prem = float(strat_pnl.mean()) / entry_cost * 100.0

    return {
        "probTarget": round(prob_target, 1),
        "medianDaysToTarget": round(median_days, 1) if median_days is not None else None,
        "retVelocity": round(ret_velocity, 2) if ret_velocity is not None else None,
        "evPrem": round(ev_prem, 1),
        "probProfitExp": round(prob_profit_exp, 1),
        "probLoss50": round(prob_loss50, 1),
    }


# ---------------------------------------------------------------------------
# Data fetching & spread finding
# ---------------------------------------------------------------------------

# Test-mode option-chain cache — when Test mode is toggled on in the UI, each
# underlying's option chain and expiration list are fetched once and reused, so
# repeated searches against a static (after-hours) market don't re-hit the
# vendor or trip its rate limits. Keys are qualified by the active source's
# name, so frozen Yahoo data is never served while another source is active
# (and vice versa). Mirrored to disk (CHAIN_CACHE_FILE): chains with live
# two-sided quotes are persisted as they're fetched and reloaded at startup, so
# a Friday-session cache survives restarts and powers weekend Test-mode scans.
# Emptied (memory AND disk) only via the "Clear cache" button.
_TEST_CHAIN_CACHE = {}   # (source, symbol, exp) -> chain snapshot (SimpleNamespace)
_TEST_EXP_CACHE = {}     # (source, symbol) -> expirations tuple

CHAIN_CACHE_FILE = Path(__file__).parent / "chain_cache.pkl"
CHAIN_CACHE_VERSION = 2   # v1 keys had no source prefix (pre-PriceSource, all Yahoo)


def _chain_has_live_quotes(oc):
    """True if at least one call row shows a two-sided market (bid & ask > 0).

    Yahoo zeroes bids/asks outside market hours (worst on weekends); such a
    chain is useless for spread-building and must not overwrite a good cache.
    """
    try:
        c = oc.calls
        return bool(((c["bid"] > 0) & (c["ask"] > 0)).any())
    except Exception:
        return False


def _save_chain_cache():
    """Persist live-quality cached chains + expiration lists to disk, atomically."""
    try:
        payload = {
            "version": CHAIN_CACHE_VERSION,
            "chains": {k: v for k, v in _TEST_CHAIN_CACHE.items()
                       if _chain_has_live_quotes(v)},
            "exps": dict(_TEST_EXP_CACHE),
        }
        tmp = CHAIN_CACHE_FILE.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, CHAIN_CACHE_FILE)
    except Exception as e:
        print(f"  Warning: could not save chain cache ({e})")


def _load_chain_cache():
    """Load the disk chain cache into memory (startup). Returns chains loaded."""
    if not CHAIN_CACHE_FILE.exists():
        return 0
    try:
        with open(CHAIN_CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        chains = payload.get("chains", {})
        exps = payload.get("exps", {})
        if payload.get("version", 1) < 2:
            # v1 predates the PriceSource layer: keys were (symbol, exp) /
            # symbol and all data was Yahoo's by construction. Prefix in place
            # so a captured Friday cache survives the upgrade.
            chains = {("yahoo",) + k: v for k, v in chains.items()}
            exps = {("yahoo", k): v for k, v in exps.items()}
        _TEST_CHAIN_CACHE.update(chains)
        _TEST_EXP_CACHE.update(exps)
        return len(chains)
    except Exception as e:
        print(f"  Warning: could not load chain cache ({e}) — starting empty")
        return 0

# A leg whose last trade is older than this (relative to the scan / quote snapshot)
# gets a staleness flag in the UI — its bid/ask may not reflect a tradable market.
# One calendar day keeps legs traded during the most recent session "fresh" even
# when scanning after hours.
STALE_TRADE_AGE_SECS = 86400

# Yahoo reports a near-zero impliedVolatility for contracts with no live
# two-sided quote (market open, after hours, weekends). IVs below this floor
# are presumed bogus: ATM-IV selection prefers live-quoted strikes above it,
# and the monitor swaps untrusted leg IVs for a fallback (see _trusted_leg_iv).
MIN_TRUSTED_IV = 0.05


def _atm_iv_from_calls(calls, spot):
    """ATM IV from a calls DataFrame: the IV of the nearest-to-spot strike.

    Restricted to strikes with a live two-sided quote and an IV above
    MIN_TRUSTED_IV when any exist, so an unquoted contract sitting closest to
    spot can't hand back a bogus near-zero vol; falls back to any positive IV,
    else None.
    """
    if spot is None or calls is None or calls.empty:
        return None
    valid = calls[calls["impliedVolatility"] > 0]
    if valid.empty:
        return None
    live = valid[(valid["bid"] > 0) & (valid["ask"] > 0) &
                 (valid["impliedVolatility"] >= MIN_TRUSTED_IV)]
    pool = live if not live.empty else valid
    idx = (pool["strike"] - spot).abs().idxmin()
    return float(pool.loc[idx, "impliedVolatility"])


def _last_trade_epoch(row):
    """Epoch seconds of a chain row's lastTradeDate, or None if missing/NaT."""
    try:
        ts = row.get("lastTradeDate")
        if ts is None or not hasattr(ts, "timestamp"):
            return None
        epoch = ts.timestamp()  # NaT raises ValueError -> caught below
        if math.isnan(epoch):
            return None
        return int(epoch)
    except Exception:
        return None


def get_option_chain(symbol, exp, test_mode=False):
    src = get_source()
    if test_mode:
        key = (src.name, symbol, exp)
        oc = _TEST_CHAIN_CACHE.get(key)
        if oc is None:
            oc = src.get_option_chain(symbol, exp)
            _TEST_CHAIN_CACHE[key] = oc
            # Persist only chains with a live two-sided market so a weekend /
            # after-hours fetch (zeroed bid/ask) can't poison the disk cache.
            # Dead chains still cache in memory to spare the vendor in-session.
            if _chain_has_live_quotes(oc):
                _save_chain_cache()
        return oc
    return src.get_option_chain(symbol, exp)


def get_expirations(symbol, test_mode=False):
    src = get_source()
    if test_mode:
        key = (src.name, symbol)
        exps = _TEST_EXP_CACHE.get(key)
        if exps is None:
            exps = src.get_expirations(symbol)
            _TEST_EXP_CACHE[key] = exps
            _save_chain_cache()
        return exps
    return src.get_expirations(symbol)


def clear_test_cache():
    n = len(_TEST_CHAIN_CACHE) + len(_TEST_EXP_CACHE)
    _TEST_CHAIN_CACHE.clear()
    _TEST_EXP_CACHE.clear()
    try:
        CHAIN_CACHE_FILE.unlink(missing_ok=True)
    except Exception as e:
        print(f"  Warning: could not delete chain cache file ({e})")
    return n


def fetch_and_find_spreads(min_premium, max_premium, min_leverage, max_width=100, max_otm=5.0, risk_free_rate=None, expiration_filter="all", min_net_delta=0.33, min_reward_risk=0.5, commission=35.80, min_dte=30, max_leg_premium=20000, symbol="^SPX", move_pct=1.0, profit_target_pct=5.0, min_gamma=0.0, min_short_leg_delta=0.08, min_return_1sigma=0.0, test_mode=False):
    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE_PCT / 100.0
    """
    Fetch option chains and find bull call spreads matching criteria.

    Returns dict with 'spot', 'spreads', and metadata.
    """
    src = get_source()  # captured once so a mid-scan toggle can't mix sources

    # Get current price
    info = src.get_daily_closes(symbol, "1d")
    if info.empty:
        raise ValueError(f"Could not fetch {symbol} price. Market may be closed or the data source unavailable.")
    spot = float(info.iloc[-1])

    # Get all expiration dates
    expirations = get_expirations(symbol, test_mode)
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
    prev_close = src.get_previous_close(symbol)
    if prev_close is None:
        try:
            hist = src.get_daily_closes(symbol, "5d")
            if len(hist) >= 2:
                prev_close = float(hist.iloc[-2])
        except Exception:
            pass

    # ATM IV from the absolutely nearest expiration (independent of min_dte filter)
    atm_iv = None
    near_atm_exp = None
    try:
        all_exps = get_expirations(symbol, test_mode)
        if all_exps:
            near_atm_exp = all_exps[0]
            nc = get_option_chain(symbol, near_atm_exp, test_mode).calls
            atm_iv = _atm_iv_from_calls(nc, spot)
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

            chain = get_option_chain(symbol, exp_date_str, test_mode)
            calls = chain.calls

            if calls.empty:
                print(f"  {exp_date_str}: skipped (empty chain)")
                continue

            # ATM IV for this expiration: nearest-to-spot strike with a
            # trustworthy IV (live-quoted strikes preferred).
            exp_atm_iv = _atm_iv_from_calls(calls, spot)

            # Simulate underlying paths once per expiration; every spread below reuses
            # them for its profit-target probability (P(+X%)).
            target_paths, target_T_remaining = simulate_paths(
                spot, exp_atm_iv, T, MC_PATHS_FINDER, max(1, dte), drift=0.0, seed=MC_SEED)

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
            skipped_short_leg = 0
            skipped_leverage = 0
            skipped_delta = 0
            skipped_gamma = 0
            skipped_ret1s = 0
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

                    total_premium = net_premium * contracts

                    # Compute deltas using implied vol
                    iv_buy = float(row_buy["impliedVolatility"])
                    iv_sell = float(row_sell["impliedVolatility"])

                    delta_buy = bs_call_delta(spot, K1, T, risk_free_rate, iv_buy)
                    delta_sell = bs_call_delta(spot, K2, T, risk_free_rate, iv_sell)
                    net_delta = delta_buy - delta_sell

                    # Short-leg moneyness floor (raw delta): the short (higher) strike must be
                    # at least this delta — a normalized, cross-ticker way to keep it from being
                    # a too-far-OTM / illiquid token. Independent of contract count.
                    if min_short_leg_delta > 0 and delta_sell < min_short_leg_delta:
                        skipped_short_leg += 1
                        continue

                    gamma_buy = bs_gamma(spot, K1, T, risk_free_rate, iv_buy)
                    gamma_sell = bs_gamma(spot, K2, T, risk_free_rate, iv_sell)
                    net_gamma = gamma_buy - gamma_sell

                    theta_buy = bs_call_theta(spot, K1, T, risk_free_rate, iv_buy)
                    theta_sell = bs_call_theta(spot, K2, T, risk_free_rate, iv_sell)
                    net_theta = theta_buy - theta_sell  # annualized, per share

                    # Full Black-Scholes reprice of the spread at a shocked underlying
                    # (leg IVs and T held fixed). Exact, so it respects the value cap at
                    # width — the old local delta+gamma Taylor expansion overstated P&L
                    # on tight/near-cap spreads once a large move carried the short
                    # strike ITM.
                    def spread_value_at(S):
                        return (bs_call_price(S, K1, T, risk_free_rate, iv_buy)
                                - bs_call_price(S, K2, T, risk_free_rate, iv_sell))

                    value_now = spread_value_at(spot)
                    move_frac = move_pct / 100.0
                    pnl_move_per = spread_value_at(spot * (1.0 + move_frac)) - value_now

                    # P&L for a +1σ *one-day* underlying move using the expiration's ATM IV.
                    # Daily σ = spot * IV * sqrt(1/252) (trading-day convention).
                    if exp_atm_iv and T > 0:
                        one_sigma_dS = spot * exp_atm_iv * math.sqrt(1 / 252)
                        pnl_1sigma_per = spread_value_at(spot + one_sigma_dS) - value_now
                        # Per-premium greek contributions over a 1-day / 1σ move (% of premium),
                        # on a shared basis so they're comparable and additive:
                        #   deltaPrem + gammaPrem == return1sigma ; thetaPrem = one day's decay.
                        # deltaPrem is the linear part; gammaPrem is the full convexity
                        # residual of the exact reprice (not just ½Γ·ΔS²), so it too
                        # knows about the spread's value cap.
                        delta_prem = net_delta * one_sigma_dS / net_premium * 100
                        gamma_prem = pnl_1sigma_per / net_premium * 100 - delta_prem
                        theta_prem = (net_theta / 365.0) / net_premium * 100
                    else:
                        pnl_1sigma_per = 0.0
                        delta_prem = gamma_prem = theta_prem = None

                    # Leverage: exact reprice for a +1% move, per unit of premium, so it
                    # stays comparable regardless of move_pct.
                    leverage = ((spread_value_at(spot * 1.01) - value_now) / net_premium) / 0.01 if net_premium > 0 else 0

                    if leverage < min_leverage:
                        skipped_leverage += 1
                        continue

                    # Filter on per-contract net delta (Δ/Contract) — the spread's
                    # directional signature, independent of budget-driven sizing.
                    if net_delta < min_net_delta:
                        skipped_delta += 1
                        continue

                    # Gamma made intuitive: delta gained per 1% underlying move, at the
                    # position level (same units as netDelta). It's how fast the spread's
                    # directional exposure accelerates as the stock moves — higher = more
                    # convex/punchy (shorter-dated, near-money), lower = steadier/linear.
                    gamma_per_1pct = net_gamma * (spot * 0.01) * contracts
                    if gamma_per_1pct < min_gamma:
                        skipped_gamma += 1
                        continue

                    # Return on a +1σ one-day move, as a % of premium paid (the
                    # "Return 1σ 1d %" column). Per-contract ratio, size-independent.
                    return_1sigma = (pnl_1sigma_per / net_premium * 100) if net_premium > 0 else 0
                    if return_1sigma < min_return_1sigma:
                        skipped_ret1s += 1
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

                    # Path-aware MC stats: probability of the Adjusted P&L touching
                    # profit_target_pct of entry cost before expiry, plus expectancy /
                    # tail / time-to-target stats from the same paths. Round-trip
                    # commission = commission x contracts (finder `commission` is
                    # per-contract round-trip). Calibrate BS-mid to the freshly-entered
                    # spread's liquidation (sell long at bid, buy back short at ask) so
                    # the barrier is anchored to real quotes.
                    liq_now = float(row_buy["bid"]) - float(row_sell["ask"])
                    mc = spread_mc_stats(
                        target_paths, target_T_remaining, K1, K2, iv_buy, iv_sell,
                        risk_free_rate, T, net_premium, contracts,
                        profit_target_pct / 100.0, commission * contracts,
                        FINDER_HAIRCUT_PCT, spread_width, value_offset=value_now - liq_now)
                    mc = mc or {}

                    # Per-leg last-trade timestamps -> staleness flags (see
                    # STALE_TRADE_AGE_SECS). A leg that hasn't traded in over a day may
                    # be quoting a market you can't actually hit.
                    lt_buy = _last_trade_epoch(row_buy)
                    lt_sell = _last_trade_epoch(row_sell)
                    scan_epoch = now.timestamp()

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
                        "expAtmIv": round(exp_atm_iv * 100, 1) if exp_atm_iv else None,
                        "breakevenMovePct": round(breakeven_move_pct, 2),
                        "breakevenMoveSigma": round(be_move_sigma, 2),
                        "netDelta": round(net_delta * contracts, 4),
                        "netDeltaPer": round(net_delta, 4),
                        "netGamma": round(net_gamma * contracts, 6),
                        "gammaPer1pct": round(gamma_per_1pct, 3),
                        "deltaBuy": round(delta_buy, 4),
                        "deltaSell": round(delta_sell, 4),
                        "gammaBuy": round(gamma_buy, 5),
                        "gammaSell": round(gamma_sell, 5),
                        "ivBuy": round(iv_buy * 100, 1),
                        "ivSell": round(iv_sell * 100, 1),
                        "pctOtmBuy": round(pct_otm_buy, 2),
                        "pctOtmSell": round(pct_otm_sell, 2),
                        "rewardRisk": round(reward_risk, 2),
                        "rrPerSigma": round(rr_per_sigma, 2),
                        "returnAtMove": round(pnl_move_per / net_premium * 100, 1) if net_premium else None,
                        "return1sigma": round(return_1sigma, 1),
                        "deltaPrem": round(delta_prem, 1) if delta_prem is not None else None,
                        "gammaPrem": round(gamma_prem, 1) if gamma_prem is not None else None,
                        "thetaPrem": round(theta_prem, 1) if theta_prem is not None else None,
                        "oiMin": min(int(row_buy.get("openInterest", 0) or 0), int(row_sell.get("openInterest", 0) or 0)),
                        "probTarget": mc.get("probTarget"),
                        "medianDaysToTarget": mc.get("medianDaysToTarget"),
                        "retVelocity": mc.get("retVelocity"),
                        "evPrem": mc.get("evPrem"),
                        "probProfitExp": mc.get("probProfitExp"),
                        "probLoss50": mc.get("probLoss50"),
                        "lastTradeBuy": lt_buy,
                        "lastTradeSell": lt_sell,
                        "staleBuy": bool(lt_buy is not None and scan_epoch - lt_buy > STALE_TRADE_AGE_SECS),
                        "staleSell": bool(lt_sell is not None and scan_epoch - lt_sell > STALE_TRADE_AGE_SECS),
                        "volume_buy": int(row_buy.get("volume", 0) or 0),
                        "volume_sell": int(row_sell.get("volume", 0) or 0),
                        "oi_buy": int(row_buy.get("openInterest", 0) or 0),
                        "oi_sell": int(row_sell.get("openInterest", 0) or 0),
                        "commissionPerSpread": round(commission, 2),
                        "totalCommission": round(commission * contracts, 2),
                    })

            print(f"    => {found} matched | skipped: {skipped_premium_zero} zero/neg prem, {skipped_premium_high} over max prem, {skipped_leg_premium} over max leg prem, {skipped_short_leg} under min short-leg delta, {skipped_leverage} under min leverage, {skipped_delta} under min delta, {skipped_gamma} under min gamma, {skipped_ret1s} under min 1σ return, {skipped_rr} under min R/R")

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
        "profitTargetPct": profit_target_pct,
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


# ---------------------------------------------------------------------------
# Index beta + option-implied 1-sigma engine (for the "Idx ±1σ P&L (β)" column)
# ---------------------------------------------------------------------------
BETA_CACHE_FILE = Path(__file__).parent / "beta_cache.json"
BETA_LOOKBACK = "2y"          # daily-return window for the beta regression
DEFAULT_INDEX = "^GSPC"       # S&P 500 index
TRADING_DAYS = 252
# Each index's option-implied vol is read straight off its volatility index
# (annualized IV, in %). Far cheaper/more robust than parsing the index's own
# option chain every refresh; we fall back to the chain only for unmapped indices.
VOL_INDEX_MAP = {
    "^GSPC": "^VIX", "^SPX": "^VIX", "SPX": "^VIX", "SPY": "^VIX", "$SPX": "^VIX",
    "^NDX": "^VXN", "NDX": "^VXN", "QQQ": "^VXN",
    "^RUT": "^RVX", "RUT": "^RVX",
    "^DJI": "^VXD", "DJI": "^VXD",
}


def _load_json_dict(path):
    if not path.exists():
        return {}
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_json_dict(path, obj):
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Price-source config (sources.json, gitignored — holds vendor API keys).
# The PriceSource classes/registry live in price_sources.py; this file owns
# the config file and the server-global toggle persistence.
# ---------------------------------------------------------------------------
SOURCES_CONFIG_FILE = Path(__file__).parent / "sources.json"


def _load_sources_config():
    cfg = _load_json_dict(SOURCES_CONFIG_FILE)
    cfg.setdefault("active", "yahoo")
    cfg.setdefault("sources", {"yahoo": {}})
    return cfg


def init_sources():
    """Load sources.json, build the source registry, and activate the saved
    source (unknown/unusable names fall back to Yahoo with a warning).
    Called from main(); plain module import stays Yahoo-only with no config."""
    cfg = _load_sources_config()
    register_sources(cfg)
    name = cfg.get("active", "yahoo")
    try:
        return set_active_source(name)
    except KeyError:
        print(f"  Warning: configured data source '{name}' unavailable — using Yahoo")
        return set_active_source("yahoo")


def source_status():
    """Payload for GET /api/source: the active source + all usable sources,
    Yahoo first."""
    ordered = sorted(SOURCES.values(), key=lambda s: (s.name != "yahoo", s.name))
    return {
        "active": get_source().name,
        "sources": [{"name": s.name, "label": s.label, "realtime": s.realtime}
                    for s in ordered],
    }


def get_betas(symbols, index_symbol):
    """Return {symbol: beta} vs. index_symbol from BETA_LOOKBACK daily returns.

    Beta is slow-moving, so results are cached per calendar day on disk (keyed by
    index) — the 30s quote-refresh loop then does a pure dict lookup. Only symbols
    missing from today's cache trigger a *single batched* download of all stale
    tickers plus the index. Symbols with insufficient history default to 1.0.
    Cache stays keyed by index/day only (not by source): betas are statistical,
    not quote-fresh, so cross-source sharing is intentional.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    wanted = [s for s in dict.fromkeys(symbols) if s and s != index_symbol]
    cache = _load_json_dict(BETA_CACHE_FILE)
    bucket = cache.get(index_symbol) or {}
    betas = dict(bucket.get("betas") or {}) if bucket.get("date") == today else {}
    missing = [s for s in wanted if s not in betas]

    if missing:
        try:
            tickers = list(dict.fromkeys(missing + [index_symbol]))
            data = get_source().get_daily_closes_batch(tickers, BETA_LOOKBACK)
            idx_ret = data[index_symbol].pct_change()
            var_idx = float(idx_ret.var())
            for s in missing:
                if s not in data.columns or var_idx <= 0:
                    continue
                joined = pd.concat([data[s].pct_change(), idx_ret], axis=1).dropna()
                if len(joined) < 30:
                    continue
                cov = float(joined.iloc[:, 0].cov(joined.iloc[:, 1]))
                betas[s] = round(cov / var_idx, 4)
        except Exception as e:
            print(f"  Warning: beta fetch failed for {missing} vs {index_symbol} ({e})")
        cache[index_symbol] = {"date": today, "betas": betas}
        _save_json_dict(BETA_CACHE_FILE, cache)

    return {s: betas.get(s, 1.0) for s in wanted}


def get_index_sigma_1d(index_symbol, test_mode=False):
    """Return (sigma_1d, annual_iv_pct): the index's option-implied one-day return
    sigma and the annualized IV it came from. Prefers the mapped volatility index
    (e.g. ^GSPC -> ^VIX) as a single quote; falls back to the index's own ATM
    chain IV. Returns (None, None) if no implied vol is obtainable."""
    annual_iv = None
    vol_ticker = VOL_INDEX_MAP.get(index_symbol.upper())
    if vol_ticker:
        try:
            h = get_source().get_daily_closes(vol_ticker, "5d")
            if not h.empty:
                annual_iv = float(h.iloc[-1]) / 100.0
        except Exception:
            annual_iv = None
    if annual_iv is None:
        try:
            exps = get_expirations(index_symbol, test_mode)
            if exps:
                oc = get_option_chain(index_symbol, exps[0], test_mode)
                spot = _underlying_spot_and_time(oc, index_symbol)[0]
                annual_iv = _atm_iv_from_calls(oc.calls, spot)
        except Exception:
            annual_iv = None
    if not annual_iv or annual_iv <= 0:
        return None, None
    return annual_iv * math.sqrt(1.0 / TRADING_DAYS), round(annual_iv * 100, 2)


def get_atm_iv_30d(symbol, target_days=30, test_mode=False):
    """~30-day constant-maturity ATM implied vol (annualized fraction) for a
    symbol. Picks the listed expiration nearest target_days and reads the ATM
    IV (nearest-to-spot strike). Used to put the own-vol 1σ shock on the same
    ~30-day tenor as the index's VIX-based 1σ. Returns None if unavailable."""
    try:
        exps = get_expirations(symbol, test_mode)
        if not exps:
            return None
        now = datetime.now()
        best = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - now).days - target_days))
        oc = get_option_chain(symbol, best, test_mode)
        spot = _underlying_spot_and_time(oc, symbol)[0]
        return _atm_iv_from_calls(oc.calls, spot)
    except Exception:
        return None


def spread_shock_pnl(spot, dS, K1, K2, T, r, iv_l, iv_s, contracts):
    """(+shock, -shock) dollar P&L of the call spread for a ±dS move in spot,
    via full Black-Scholes reprice of both legs holding IV and time constant.
    Shared by the own-vol 1σ and beta-index 1σ columns so they use one engine."""
    def theo(s):
        return bs_call_price(s, K1, T, r, iv_l) - bs_call_price(s, K2, T, r, iv_s)
    base = theo(spot)
    up = (theo(spot + dS) - base) * 100 * contracts
    dn = (theo(max(spot - dS, 1e-6)) - base) * 100 * contracts
    return round(up, 2), round(dn, 2)


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
    # Today's mark: live bid/ask mid when both quoted (fresh even if untraded),
    # else the best of bid/ask/last. `quotedMid` flags a genuine two-sided market.
    quoted_mid = bid > 0 and ask > 0
    mid = (bid + ask) / 2 if quoted_mid else max(bid, ask, last)
    # Yesterday's close: Yahoo's official prior-day close. It equals lastPrice - change
    # and matches regularMarketPreviousClose, so it stays valid even when the last
    # TRADE is days stale (staleness sits in lastPrice, which cancels here).
    change_raw = float(row["change"]) if "change" in row else float("nan")
    prev_close = None if math.isnan(change_raw) else round(last - change_raw, 2)
    return {"bid": round(bid, 2), "ask": round(ask, 2), "last": round(last, 2),
            "mid": round(mid, 2), "volume": vol,
            "prevClose": prev_close, "quotedMid": quoted_mid,
            "iv": round(iv * 100, 1) if iv is not None else None,
            "lastTrade": _last_trade_epoch(row),
            "_iv_raw": iv}


def _trusted_leg_iv(leg, spot, strike, T, r, atm_iv):
    """Best-usable IV for one leg of a monitored spread.

    Yahoo reports a near-zero impliedVolatility for contracts with no live
    two-sided quote (market open, after hours, weekends); fed straight into
    Black-Scholes it prices the leg as worthless and zeroes every derived
    column (Net Delta, Daily Theo P&L, ±σ shocks, P(+X%)). Trust the leg's
    quoted IV only when it comes with a live two-sided market and clears
    MIN_TRUSTED_IV; otherwise fall back to the expiration's ATM IV, then to
    the IV re-implied from the leg's own mark, then to the raw value.
    """
    iv = leg.get("_iv_raw")
    if iv and iv >= MIN_TRUSTED_IV and leg.get("quotedMid"):
        return iv
    if atm_iv and atm_iv >= MIN_TRUSTED_IV:
        return atm_iv
    mark = leg.get("mid")
    if spot and mark and mark > 0 and T > 0:
        implied = implied_vol(mark, spot, strike, T, r)
        if implied and implied >= MIN_TRUSTED_IV:
            return implied
    return iv


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


# ---- Adjusted-P&L threshold alerts (ntfy.sh phone push) ----
# Each live /positions refresh checks every position's Adjusted P&L % against
# the configured thresholds and fires ONE push per (position, threshold) per
# calendar day via ntfy.sh — subscribe the ntfy phone app to the topic printed
# at startup. Config + latch state live in alerts.json (gitignored); the topic
# is a random secret generated on first run, since ntfy topics are
# world-readable by name.
ALERTS_FILE = Path(__file__).parent / "alerts.json"
DEFAULT_ALERT_THRESHOLDS_PCT = [2.5, 5.0]
_alerts_lock = threading.Lock()


def _load_alerts():
    state = {}
    if ALERTS_FILE.exists():
        try:
            state = json.loads(ALERTS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    if not state.get("topic"):
        state["topic"] = "callspreads-" + uuid.uuid4().hex[:10]
    state.setdefault("thresholds", list(DEFAULT_ALERT_THRESHOLDS_PCT))
    state.setdefault("date", "")
    state.setdefault("sent", [])
    return state


def _save_alerts(state):
    tmp = ALERTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, ALERTS_FILE)


def get_alert_topic():
    """The ntfy topic, creating alerts.json on first call — for the banner."""
    with _alerts_lock:
        state = _load_alerts()
        _save_alerts(state)
        return state["topic"]


def _send_ntfy(topic, title, body):
    """POST one push to ntfy.sh. Best-effort — alert delivery must never break
    a quotes refresh, and a failed send is dropped, not retried (the latch is
    taken before sending, so a flaky network can't cause duplicate pushes)."""
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}", data=body.encode("utf-8"),
            headers={"Title": title, "Priority": "high",
                     "Tags": "chart_with_upwards_trend"}, method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
        print(f"  Alert pushed: {body}")
    except Exception as e:
        print(f"  Alert push failed ({e}): {body}")


def _position_name(q):
    return q.get("label") or (
        f"{q.get('symbol')} {q.get('longStrike'):g}/{q.get('shortStrike'):g}"
        f" {q.get('expiration')}")


def check_pnl_alerts(quotes):
    """Push-alert the first time a position's Adjusted P&L % rises above each
    threshold. One push per (position, threshold) per calendar day — dipping
    back below and re-crossing doesn't re-fire until the date rolls over — so
    a spread can never send more than len(thresholds) pushes in a day. Called
    from /api/positions/quotes on live refreshes only (test-mode data is
    frozen, so a "crossing" there would be stale, not actionable)."""
    to_send = []
    with _alerts_lock:
        state = _load_alerts()
        today = datetime.now().strftime("%Y-%m-%d")
        if state["date"] != today:
            state["date"] = today
            state["sent"] = []
        for q in quotes:
            pct, pid = q.get("adjPnlPct"), q.get("id")
            if pct is None or not pid:
                continue
            for thr in state["thresholds"]:
                key = f"{pid}|{thr:g}"
                if pct > thr and key not in state["sent"]:
                    state["sent"].append(key)
                    body = (f"{_position_name(q)}: Adj P&L crossed +{thr:g}% "
                            f"(now {pct:+.2f}%, ${q.get('adjPnl') or 0:+,.0f})")
                    to_send.append((state["topic"], body))
        _save_alerts(state)
    for topic, body in to_send:
        threading.Thread(target=_send_ntfy, daemon=True,
                         args=(topic, "Call spread P&L alert", body)).start()


def _underlying_spot_and_time(oc, symbol):
    """Return (spot, quote_epoch_seconds, prev_close) for a chain snapshot.

    Prefers the underlying quote bundled with the chain (`oc.underlying`) because
    it comes from the *same* vendor response as the option quotes, so spot is
    time-aligned with the bids/asks/IVs used for P&L and greeks. Falls back
    through post/pre-market price, a bid/ask mid, and finally a separate daily
    close from the active source (which is NOT time-aligned, so quote_time is
    reported as None).

    prev_close is the underlying's official prior-day close (for the one-day
    theoretical P&L); may be None if the chain snapshot doesn't carry it.
    """
    u = getattr(oc, "underlying", None) or {}
    prev_close = u.get("regularMarketPreviousClose") or u.get("previousClose")
    prev_close = float(prev_close) if prev_close else None
    # Try each price source paired with its own timestamp.
    for price_key, time_key in (
        ("regularMarketPrice", "regularMarketTime"),
        ("postMarketPrice", "postMarketTime"),
        ("preMarketPrice", "preMarketTime"),
    ):
        px = u.get(price_key)
        if px:
            return float(px), u.get(time_key), prev_close
    # Bid/ask mid of the underlying, if quoted.
    bid, ask = u.get("bid") or 0, u.get("ask") or 0
    if bid and ask:
        return (float(bid) + float(ask)) / 2, u.get("regularMarketTime"), prev_close
    # Last resort: a separate daily-close fetch, not aligned with the chain.
    return float(get_source().get_daily_closes(symbol, "1d").iloc[-1]), None, prev_close


def fetch_position_quotes(positions, haircut_pct=0.80, profit_target_pct=15.0,
                          index_symbol=DEFAULT_INDEX, n_sigma=1.0, test_mode=False):
    """Fetch live leg data for every saved position, batching chain calls.

    haircut_pct: multiplier applied to (raw P&L − entry commission − exit commission)
    to produce Adjusted P&L. Exit commission is assumed equal to entry commission.
    profit_target_pct: profit target (as a % of entry cost) for the path-aware P(+X%)
    probability that the Adjusted P&L touches it before expiry.
    index_symbol: reference index for the beta-scaled Nσ index-move P&L column.
    n_sigma: number of standard deviations for the ±σ move columns (own-vol and
    beta-index). Defaults to 1; e.g. 2 shocks by ±2σ.
    """
    n_sigma = max(0.0, float(n_sigma))
    results = []
    chain_cache = {}   # (symbol, expiration) -> calls DataFrame
    spot_cache = {}    # symbol -> (spot: float, quote_epoch_seconds: int | None)
    paths_cache = {}   # (symbol, expiration) -> (S_matrix, T_remaining) for P(+X%)
    iv30_cache = {}    # symbol -> ~30-day ATM IV (annualized), for the own-vol 1σ shock

    # Beta (per-day cached) and the index's implied 1σ daily move, computed once
    # for the whole batch — not per position and not on the hot per-refresh path.
    index_symbol = (index_symbol or DEFAULT_INDEX).strip() or DEFAULT_INDEX
    betas = get_betas([p["symbol"] for p in positions], index_symbol)
    index_sigma_1d, index_iv_pct = get_index_sigma_1d(index_symbol, test_mode)

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
        result["quoteTime"] = None
        result["spreadMid"] = None
        result["spreadLiquidation"] = None
        result["currentValue"] = None
        result["liquidationValue"] = None
        # Entry cost is the pure premium outlay (commission-free) so raw P&L reflects the market move
        # only. Adjusted P&L nets out entry + mirrored exit commission and then applies haircut_pct.
        result["entryCost"] = round((long_entry - short_entry) * 100 * contracts, 2)
        # Net premium paid per spread (long entry − short entry); the per-spread
        # version of entryCost, shown in the "Entry Spread" column.
        result["entrySpread"] = round(long_entry - short_entry, 2)
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
        result["oneSigmaIvPct"] = None
        result["oneSigmaPnl"] = None
        result["oneSigmaPnlDown"] = None
        result["dailyTheoPnl"] = None
        result["dailyTheoMove"] = None
        result["beta"] = None
        result["betaDollarDeltaPer1Pct"] = None
        result["betaIndexMove"] = None
        result["betaIndexUpPnl"] = None
        result["betaIndexDownPnl"] = None
        result["indexSymbol"] = index_symbol
        result["indexIvPct"] = index_iv_pct
        result["probTarget"] = None
        result["dte"] = None
        result["history"] = _PNL_HISTORY.get(p.get("id"), [])

        try:
            key = (symbol, exp)
            if key not in chain_cache:
                oc = get_option_chain(symbol, exp, test_mode)
                chain_cache[key] = oc.calls
                if symbol not in spot_cache:
                    # Spot from the chain's own underlying snapshot => time-aligned
                    # with the option quotes below (see _underlying_spot_and_time).
                    spot_cache[symbol] = _underlying_spot_and_time(oc, symbol)
            calls = chain_cache[key]
            spot, quote_ts, prev_close = spot_cache.get(symbol, (None, None, None))
            result["spot"] = round(spot, 2) if spot is not None else None
            result["quoteTime"] = int(quote_ts) if quote_ts else None

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

                # ATM IV for this expiration: used for the 1σ greeks below, to
                # diffuse the P(+X%) paths, and as the fallback for untrusted
                # leg IVs.
                atm_iv = _atm_iv_from_calls(calls, spot)

                # Leg IVs for every BS-derived column, distrusting Yahoo's
                # bogus near-zero IVs on unquoted contracts (see _trusted_leg_iv).
                iv_l = _trusted_leg_iv(long_leg, spot, float(p["longStrike"]), T, r, atm_iv)
                iv_s = _trusted_leg_iv(short_leg, spot, float(p["shortStrike"]), T, r, atm_iv)

                if iv_l and iv_s and T > 0:
                    K1 = float(p["longStrike"])
                    K2 = float(p["shortStrike"])
                    delta_l = bs_call_delta(spot, K1, T, r, iv_l)
                    delta_s = bs_call_delta(spot, K2, T, r, iv_s)
                    theta_l = bs_call_theta(spot, K1, T, r, iv_l)
                    theta_s = bs_call_theta(spot, K2, T, r, iv_s)
                    vega_l = bs_vega(spot, K1, T, r, iv_l)
                    vega_s = bs_vega(spot, K2, T, r, iv_s)
                    net_delta_per = delta_l - delta_s
                    # Position-level: long minus short, scaled by contracts * 100 multiplier
                    result["netDelta"] = round(net_delta_per * 100 * contracts, 2)
                    # Theta per calendar day, in $ (annual / 365)
                    result["netThetaPerDay"] = round((theta_l - theta_s) / 365.0 * 100 * contracts, 2)
                    # Vega in $ per 1 vol-point (1%) move
                    result["netVega"] = round((vega_l - vega_s) * 100 * contracts * 0.01, 2)

                    # Own-vol ±1σ one-day P&L: full BS reprice at spot±ΔS (same engine
                    # as the beta column below). ΔS = spot · σ_own · √(1/252), with σ_own
                    # taken at a ~30-day tenor to match the index's VIX-based 1σ. Falls
                    # back to this expiration's ATM IV, then the leg-IV average.
                    if symbol not in iv30_cache:
                        iv30_cache[symbol] = get_atm_iv_30d(symbol, test_mode=test_mode)
                    sigma_own = iv30_cache[symbol] or (atm_iv if atm_iv and atm_iv > 0 else (iv_l + iv_s) / 2.0)
                    one_sigma_dS = spot * sigma_own * math.sqrt(1 / 252) * n_sigma
                    up_own, dn_own = spread_shock_pnl(spot, one_sigma_dS, K1, K2, T, r, iv_l, iv_s, contracts)
                    result["oneSigmaMove"] = round(one_sigma_dS, 2)
                    result["oneSigmaIvPct"] = round(sigma_own * 100, 1)
                    result["oneSigmaPnl"] = up_own          # +1σ (kept as the summary/legacy field)
                    result["oneSigmaPnlDown"] = dn_own      # -1σ

                    # Daily theoretical P&L from the underlying's one-day move only:
                    # reprice both legs with BS at current spot vs. the underlying's
                    # prior close, holding IV and time constant so only S changes.
                    # Uses the fresher spot to estimate the day's value change even
                    # when the option bids/asks are stale.
                    if prev_close and prev_close > 0:
                        theo_now = (bs_call_price(spot, K1, T, r, iv_l)
                                    - bs_call_price(spot, K2, T, r, iv_s))
                        theo_prev = (bs_call_price(prev_close, K1, T, r, iv_l)
                                     - bs_call_price(prev_close, K2, T, r, iv_s))
                        result["dailyTheoMove"] = round(spot - prev_close, 2)
                        result["dailyTheoPnl"] = round((theo_now - theo_prev) * 100 * contracts, 2)

                    # Beta-scaled 1σ index-move P&L: translate a ±1σ move in the
                    # reference index into this stock's systematic move via beta
                    # (ΔS = spot · β · σ_index_1d), then BS-reprice both legs at
                    # spot±ΔS vs. spot (IV and time held constant). Up and down
                    # differ in magnitude because of gamma.
                    beta = betas.get(symbol)
                    if beta is not None:
                        result["beta"] = round(beta, 3)
                        # Portfolio building block: this position's $ P&L for a +1% move
                        # in the reference index = position delta (per $1 of underlying)
                        # × spot × 1% × beta. Summed client-side into the β-weighted
                        # portfolio net delta in the summary.
                        result["betaDollarDeltaPer1Pct"] = round(
                            net_delta_per * 100 * contracts * spot * 0.01 * beta, 2)
                    if beta is not None and index_sigma_1d:
                        dS_beta = spot * beta * index_sigma_1d * n_sigma
                        up_beta, dn_beta = spread_shock_pnl(spot, dS_beta, K1, K2, T, r, iv_l, iv_s, contracts)
                        result["betaIndexMove"] = round(dS_beta, 2)
                        result["betaIndexUpPnl"] = up_beta
                        result["betaIndexDownPnl"] = dn_beta

                # Path-aware P(+X%): probability the Adjusted P&L touches profit_target_pct
                # of entry cost before expiry. Paths are simulated once per (symbol, exp).
                if (iv_l and iv_s and atm_iv and T > 0 and result["entryCost"]
                        and result["spreadLiquidation"] is not None):
                    if key not in paths_cache:
                        paths_cache[key] = simulate_paths(
                            spot, atm_iv, T, MC_PATHS_MONITOR, max(1, dte),
                            drift=0.0, seed=MC_SEED)
                    tgt_paths, tgt_T_remaining = paths_cache[key]
                    K1 = float(p["longStrike"])
                    K2 = float(p["shortStrike"])
                    # Calibrate BS-mid to the current quoted liquidation (the value the
                    # Adj P&L column is based on) so the barrier is anchored to reality.
                    bs_mid_now = (bs_call_price(spot, K1, T, r, iv_l)
                                  - bs_call_price(spot, K2, T, r, iv_s))
                    value_offset = bs_mid_now - result["spreadLiquidation"]
                    mc = spread_mc_stats(
                        tgt_paths, tgt_T_remaining, K1, K2, iv_l, iv_s, r, T,
                        long_entry - short_entry, contracts,
                        profit_target_pct / 100.0, result["totalCommission"],
                        haircut_pct, K2 - K1, value_offset=value_offset)
                    result["probTarget"] = mc.get("probTarget") if mc else None

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

  html { height: 100%; }
  body {
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    /* Fixed-viewport column: the top (header + form) stays locked; only the
       results table (.table-wrapper) scrolls. */
    height: 100vh;
    margin: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 32px;
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
    padding: 7px 32px;
    display: flex;
    align-items: flex-end;
    gap: 6px 14px;
    flex-wrap: wrap;
  }
  /* Model-inputs row: hidden until its toggle is opened. */
  .controls.collapsed { display: none; }
  /* Full-width heading that groups the fields by function (forces a flex wrap). */
  /* Inline group markers — pills that sit before each group's fields instead of
     each taking a full heading row, so the form stays dense. */
  .section-label {
    align-self: flex-end;
    margin: 0;
    padding: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--surface2);
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-dim);
    white-space: nowrap;
  }
  .section-label .sub { display: none; }  /* long subtitle dropped to save space; grouping shown by the pill */

  .input-group {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .input-group label {
    font-size: 11px;
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
    font-size: 13px;
    padding: 5px 9px;
    border-radius: 5px;
    width: 150px;
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

  /* Hints hidden to save vertical space — the same info lives in each field's
     hover tooltip. */
  .input-group .hint { display: none; }

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
    flex: 1 1 auto;
    min-height: 0;        /* allow the flex child to shrink so it can scroll */
    padding: 0 32px 24px;
    overflow: auto;       /* only this region scrolls */
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
    line-height: 1.45;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    pointer-events: none;
    max-height: calc(100vh - 16px);   /* never taller than the viewport */
    overflow: hidden;
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

  .stale-flag { color: #fbbf24; margin-left: 4px; cursor: help; }

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
  <h1>Call Spread Finder <span class="tag" id="modeTag">LIVE</span></h1>
  <a href="/positions" target="_blank" class="positions-link">My Positions &rarr;</a>
  <label id="testToggle" style="display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text-dim);cursor:pointer;" title="Test mode caches each underlying's option chain so repeated searches (e.g. after hours) don't re-fetch from the data source. Data stays frozen until you Clear cache or turn Test off.">
    <input type="checkbox" id="testMode" style="width:15px;height:15px;accent-color:var(--yellow);"> Test mode
  </label>
  <select id="dataSource" style="background:transparent;color:var(--text-dim);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;" title="Market data source (server-wide — affects all pages)"></select>
  <button type="button" id="clearCacheBtn" class="primary" style="background:transparent;color:var(--text-dim);border:1px solid var(--border);padding:5px 10px;font-size:12px;font-weight:500;" onclick="clearChainCache()">Clear cache</button>
  <div class="spot-display">
    <span class="label" id="spotLabel">SPX Last:</span>
    <span id="spotPrice">--</span>
  </div>
</div>

<div class="controls">
  <div class="section-label">Scan <span class="sub">— what universe to search</span></div>
  <div class="input-group tooltip-container">
    <label>Ticker</label>
    <input type="text" id="ticker" value="^SPX" style="width:120px;text-transform:uppercase;">
    <span class="hint">Yahoo Finance symbol</span>
    <span class="tooltip-text">Enter any optionable ticker — e.g. ^SPX, AAPL, TSLA, QQQ, SPY. Use ^ prefix for indices.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Expirations</label>
    <div id="expirationCheckboxes" style="display:grid;grid-template-columns:auto auto;gap:0 16px;padding:2px 0;font-size:12px;"></div>
    <span class="hint">Check one or more expirations</span>
    <span class="tooltip-text">3rd Friday of each month — check the expirations you want to scan, or "All" to scan everything</span>
  </div>

  <div class="section-label">Filters <span class="sub">— change which spreads come back</span></div>
  <div class="input-group tooltip-container">
    <label>Max Premium ($)</label>
    <input type="number" id="maxPremium" value="11000" min="100" step="100">
    <span class="hint">Max net dollars laid out (risk cap)</span>
    <span class="tooltip-text">Maximum net cash outlay (long call ask − short call bid) × 100 multiplier. This is your capital at risk per spread.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Leverage (x)</label>
    <input type="number" id="minLeverage" value="2" min="0.1" step="0.5">
    <span class="hint">Profit / premium for 1% move</span>
    <span class="tooltip-text">Minimum ratio of dollar profit from a 1% up move to premium paid — your P&amp;L per 1% recovery, per dollar risked.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Return 1&sigma; (%)</label>
    <input type="number" id="minReturn1sigma" value="0" step="5">
    <span class="hint">Return on a 1-day 1&sigma; move</span>
    <span class="tooltip-text">Minimum "Return 1&sigma; 1d %" — the P&amp;L for a +1&sigma; one-day move in the underlying, as a % of premium paid. Screens for spreads that pop meaningfully on a normal daily move. Set 0 to disable.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Reward/Risk</label>
    <input type="number" id="minRewardRisk" value="0.5" min="0" step="0.1">
    <span class="hint">Min max-profit / premium ratio</span>
    <span class="tooltip-text">Minimum ratio of max profit to premium paid — e.g. 1.0 means max profit ≥ premium (hold-to-expiry payoff).</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Max % OTM</label>
    <input type="number" id="maxOtm" value="5" min="0.1" step="0.5">
    <span class="hint">Buy strike max % above spot</span>
    <span class="tooltip-text">Maximum % the lower (buy) strike is above the current underlying price — controls where the long strike sits vs. spot (positioning for the bounce).</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Max Width (pts)</label>
    <input type="number" id="maxWidth" value="100" min="5" step="5">
    <span class="hint">Max strike spread in points</span>
    <span class="tooltip-text">Maximum distance between strikes — e.g. 50 pts = $5,000 max risk (×100 multiplier)</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Net Delta</label>
    <input type="number" id="minNetDelta" value="0.33" min="0" max="1" step="0.01">
    <span class="hint">Per-contract net delta (Δ/Contract) floor</span>
    <span class="tooltip-text">Minimum per-contract net delta (long delta − short delta) — the Δ/Contract column. Filters out low-directional spreads.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Gamma (&Delta;/1%)</label>
    <input type="number" id="minGamma" value="0.00" step="0.01">
    <span class="hint">&uarr; punchier/convex &middot; &darr; steadier/linear</span>
    <span class="tooltip-text">Gamma as delta gained per 1% move in the underlying — how fast the spread's directional exposure accelerates. Raise it to require more convexity (punchier, tends to shorter-dated / nearer-the-money spreads); lower it (or go negative) for steadier, more linear spreads (longer-dated / further OTM).</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Max Leg Premium ($)</label>
    <input type="number" id="maxLegPremium" value="20000" min="0" step="500">
    <span class="hint">Max $ per individual leg</span>
    <span class="tooltip-text">Maximum absolute dollar value (price × 100 × contracts) for either the long or short leg individually</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min Short-Leg Delta</label>
    <input type="number" id="minShortLegDelta" value="0.08" min="0" max="1" step="0.01">
    <span class="hint">Short-call moneyness floor (Δ)</span>
    <span class="tooltip-text">Minimum raw delta of the short (higher-strike) call — a normalized, cross-ticker measure of the short strike's moneyness (≈ probability of finishing ITM). Keeps the short leg from being a too-far-OTM / illiquid token; higher = short strike nearer the money (richer, upside capped sooner). Independent of contract count. Set 0 to disable. Default 0.08 ≈ an 8-delta short.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Min DTE</label>
    <input type="number" id="minDte" value="30" min="0" step="1">
    <span class="hint">Min days to expiration</span>
    <span class="tooltip-text">Skip expirations with fewer than this many days remaining</span>
  </div>

  <div class="section-label">Output &amp; scenario <span class="sub">— shape the result columns, not which spreads return</span></div>
  <div class="input-group tooltip-container">
    <label>Recovery move %</label>
    <input type="number" id="movePct" value="1" min="0.1" step="0.5">
    <span class="hint">Underlying bounce → "P&amp;L X% $" column</span>
    <span class="tooltip-text">The underlying % move you expect to recover — drives the "P&amp;L X% $" column (Δ·dS + ½·Γ·dS²). This is an UNDERLYING price move, not a P&amp;L target. It does not screen spreads (only a tiny 2nd-order effect on Leverage via gamma).</span>
  </div>
  <div class="input-group tooltip-container">
    <label>P&amp;L target (% of cost)</label>
    <input type="number" id="profitTarget" value="5" min="0.1" step="0.5">
    <span class="hint">Return on premium → "P(+X%)" column</span>
    <span class="tooltip-text">Profit target as a % of entry cost (return on premium PAID — not an underlying move), measured on Adjusted P&amp;L (80% haircut on gains + round-trip commission, matching the Positions monitor). Drives the path-aware "P(+X%)" probability column; it does not screen spreads.</span>
  </div>
  <div class="input-group">
    <label>Sort By</label>
    <select id="sortBy" onchange="applySortDropdown()">
      <option value="score">Score (high first)</option>
      <option value="leverage">Leverage (high first)</option>
      <option value="probTarget">P(+X%) (high first)</option>
      <option value="totalPremium">Premium (low first)</option>
      <option value="rewardRisk">Reward/Risk (high first)</option>
      <option value="dte">DTE (near first)</option>
      <option value="pctOtmBuy">% OTM (near first)</option>
    </select>
  </div>

  <div style="display:flex;align-items:flex-end;gap:12px;margin-left:auto;">
    <button type="button" class="primary" id="advancedToggle" style="background:transparent;color:var(--text-dim);border:1px solid var(--border);" onclick="toggleAdvanced()">Model inputs ▸</button>
    <button class="primary" id="searchBtn" onclick="doSearch()">Find Spreads</button>
    <button type="button" class="primary" id="scatterBtn" style="background:transparent;color:var(--accent);border:1px solid var(--accent);" onclick="openScatter()" title="Open a scatter plot of the current results in a new tab (pick any two columns for the axes).">&#128202; Scatter</button>
  </div>
</div>
<div class="controls collapsed" id="advancedFilters" style="border-top:none;padding-top:0;">
  <div class="section-label" style="border-top:none;padding-top:0;">Model inputs <span class="sub">— feed the Black-Scholes math &amp; economics, so they indirectly move the filters</span></div>
  <div class="input-group tooltip-container">
    <label>Risk-Free Rate (%)</label>
    <input type="number" id="riskFreeRate" value="{RISK_FREE_RATE_PCT}" min="0" max="20" step="0.1">
    <span class="hint">Feeds Black-Scholes delta/gamma</span>
    <span class="tooltip-text">Used in the delta/gamma calculation — approximate current Treasury yield. Changes the greeks, which feed the Delta/Gamma/Leverage filters.</span>
  </div>
  <div class="input-group tooltip-container">
    <label>Commission ($/spread)</label>
    <input type="number" id="commission" value="35.80" min="0" step="0.25">
    <span class="hint">Round-trip per spread</span>
    <span class="tooltip-text">Total commission per spread for opening + closing (all legs, both ways). Default $35.80 = $8.95/leg &times; 2 legs &times; 2 sides. Deducted from max profit, breakeven, and reward/risk — so it feeds the Reward/Risk filter.</span>
  </div>
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
  <button type="button" class="primary" id="dipBuyPresetBtn" style="margin-left:auto;" onclick="applyDipBuyPreset()" title="Load a starting parameter set for buying call spreads on dips to catch a recovery, then search. Tune and Save As New to keep your own.">Dip-buy preset</button>
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

<style>
  #scoreWeights { display:none; margin:0 0 8px; padding:6px 10px; background:var(--surface); border:1px solid var(--border); border-radius:8px; }
  #scoreWeights .sw-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  #swToggle { background:transparent; color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 10px; font-size:13px; font-weight:600; cursor:pointer; }
  #swToggle:hover { border-color:var(--accent); color:var(--accent); }
  #scoreWeights .sw-presets { display:flex; gap:6px; margin-left:auto; }
  #scoreWeights .sw-preset { background:transparent; color:var(--text-dim); border:1px solid var(--border); border-radius:6px; padding:3px 9px; font-size:12px; cursor:pointer; }
  #scoreWeights .sw-preset:hover { border-color:var(--accent); color:var(--accent); }
  #swBody { display:none; margin-top:8px; padding-top:8px; border-top:1px solid var(--border); }
  #swBody.open { display:flex; flex-wrap:wrap; gap:14px 22px; align-items:center; }
  #scoreWeights .sw-item { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-dim); }
  #scoreWeights .sw-item label { min-width:92px; }
  #scoreWeights .sw-item input[type=range] { width:120px; accent-color:var(--accent); }
  #scoreWeights .sw-item .sw-val { min-width:22px; text-align:right; color:var(--text); font-variant-numeric:tabular-nums; }
  #scoreWeights .sw-oi { display:flex; align-items:center; gap:6px; font-size:12px; color:var(--text-dim); }
  #scoreWeights .sw-oi input { width:70px; }
  #scoreWeights .sw-note { flex-basis:100%; font-size:11px; color:var(--text-dim); }
</style>
<div id="scoreWeights">
  <div class="sw-head">
    <button type="button" id="swToggle" onclick="toggleScoreWeights()">&#9881; Score weights &#9656;</button>
    <span style="font-size:11px;color:var(--text-dim);">Drag to re-rank instantly — no new scan.</span>
    <span class="sw-presets">
      <button type="button" class="sw-preset" onclick="applyScorePreset('balanced')" title="Balanced default: 30 / 25 / 20 / 15 / 10.">Reset</button>
      <button type="button" class="sw-preset" onclick="applyScorePreset('aggressive')" title="Tilt toward payoff &amp; convexity (leverage the bounce).">Aggressive</button>
      <button type="button" class="sw-preset" onclick="applyScorePreset('conservative')" title="Tilt toward probability &amp; reward/risk (higher hit-rate).">Conservative</button>
    </span>
  </div>
  <div id="swBody">
    <div class="sw-item"><label>Return @ +X%</label><input type="range" id="sw_returnAtMove" min="0" max="50" step="1" value="30" oninput="scoreWeightsChanged()"><span class="sw-val" id="swv_returnAtMove">30</span></div>
    <div class="sw-item"><label>P(+X%)</label><input type="range" id="sw_probTarget" min="0" max="50" step="1" value="25" oninput="scoreWeightsChanged()"><span class="sw-val" id="swv_probTarget">25</span></div>
    <div class="sw-item"><label>&Gamma;/Prem (convexity)</label><input type="range" id="sw_gammaPrem" min="0" max="50" step="1" value="20" oninput="scoreWeightsChanged()"><span class="sw-val" id="swv_gammaPrem">20</span></div>
    <div class="sw-item"><label>Reward/Risk</label><input type="range" id="sw_rewardRisk" min="0" max="50" step="1" value="15" oninput="scoreWeightsChanged()"><span class="sw-val" id="swv_rewardRisk">15</span></div>
    <div class="sw-item"><label>&Theta;/Prem (less decay)</label><input type="range" id="sw_thetaPrem" min="0" max="50" step="1" value="10" oninput="scoreWeightsChanged()"><span class="sw-val" id="swv_thetaPrem">10</span></div>
    <div class="sw-oi"><label>Liquidity target OI</label><input type="number" id="sw_oiTarget" min="0" step="50" value="500" onchange="scoreWeightsChanged()"><span title="Score is multiplied by min(1, worst-leg OI ÷ target), so thin spreads are discounted.">&#9432;</span></div>
    <div class="sw-note">Weights are relative (auto-normalized to sum 1). Score = 100 × Σ(weightᵢ × percentile-rankᵢ) × min(1, OI÷target), ranked within this result set.</div>
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
        <th data-col="score" onclick="sortTable('score')" title="Composite 0–100 score for risk-adjusted capital efficiency. Percentile-rank blend of Return@X%, P(+X%), Γ/Prem, Reward/Risk and Θ/Prem (weights below the form), discounted by a worst-leg liquidity gate. Ranks within the current result set. Drag the Score weights to re-rank instantly.">Score</th>
        <th data-col="expiration" onclick="sortTable('expiration')">Expiration</th>
        <th data-col="contracts" onclick="sortTable('contracts')">Contracts</th>
        <th data-col="buyStrike" onclick="sortTable('buyStrike')">Buy Strike</th>
        <th data-col="sellStrike" onclick="sortTable('sellStrike')">Sell Strike</th>
        <th data-col="pctOtmBuy" onclick="sortTable('pctOtmBuy')">% OTM</th>
        <th data-col="spreadWidth" onclick="sortTable('spreadWidth')">Width (pts)</th>
        <th data-col="netPremium" onclick="sortTable('netPremium')">Premium $</th>
        <th data-col="maxProfit" onclick="sortTable('maxProfit')">Max Profit $</th>
        <th data-col="rewardRisk" onclick="sortTable('rewardRisk')">Reward/Risk</th>
        <th data-col="leverage" onclick="sortTable('leverage')">Leverage</th>
        <th data-col="returnAtMove" onclick="sortTable('returnAtMove')" id="rocMoveHeader" title="Return on net premium if the underlying makes the Recovery move % — the capital-efficiency of your bounce thesis.">Return @ +5%</th>
        <th data-col="probTarget" onclick="sortTable('probTarget')" id="probTargetHeader">P(+5%)</th>
        <th data-col="probProfitExp" onclick="sortTable('probProfitExp')" title="Monte-Carlo probability the Adjusted P&amp;L (haircut on gains, commissions netted) is positive if held to expiration. Compare with P(+X%), the path-aware early-exit probability.">P(prof exp)</th>
        <th data-col="medianDaysToTarget" onclick="sortTable('medianDaysToTarget')" title="Median days until the profit target is first touched, among the Monte-Carlo paths that hit it. Lower = faster capital recycling.">Med d&rarr;tgt</th>
        <th data-col="evPrem" onclick="sortTable('evPrem')" title="Expected Adjusted P&amp;L as a % of premium under the exit-at-target strategy: paths that touch the target bank it, the rest hold to expiry. Zero-drift (no-edge) paths, so this is a conservative floor.">EV %</th>
        <th data-col="return1sigma" onclick="sortTable('return1sigma')" title="P&amp;L for a +1&sigma; one-day move in the underlying, as a % return on the premium paid.">Return 1&sigma; 1d %</th>
        <th data-col="breakevenMovePct" onclick="sortTable('breakevenMovePct')">BE Move %</th>
        <th data-col="breakevenMoveSigma" onclick="sortTable('breakevenMoveSigma')">BE Move &sigma;</th>
        <th data-col="netDelta" onclick="sortTable('netDelta')" title="Position dollar delta: $ P&amp;L per 1% move in the underlying (first-order directional exposure).">$&Delta; /1%</th>
        <th data-col="netDeltaPer" onclick="sortTable('netDeltaPer')">&Delta;/Contract</th>
        <th data-col="gammaPer1pct" onclick="sortTable('gammaPer1pct')" title="Gamma as delta gained per 1% move in the underlying — how fast the spread's directional exposure accelerates. Higher = more convex/punchy.">&Gamma; (&Delta;/1%)</th>
        <th data-col="deltaPrem" onclick="sortTable('deltaPrem')" title="Delta contribution to a +1&sigma; one-day move, as a % of premium paid. On the same 1-day/1&sigma; basis as &Gamma;/Prem and &Theta;/Prem; &Delta;/Prem + &Gamma;/Prem = Return 1&sigma;.">&Delta;/Prem %</th>
        <th data-col="gammaPrem" onclick="sortTable('gammaPrem')" title="Gamma (convexity) contribution to a +1&sigma; one-day move, as a % of premium paid. Same 1-day/1&sigma; basis as &Delta;/Prem and &Theta;/Prem.">&Gamma;/Prem %</th>
        <th data-col="thetaPrem" onclick="sortTable('thetaPrem')" title="One day's time decay as a % of premium paid (usually negative for a debit spread). Same per-premium, 1-day basis as &Delta;/Prem and &Gamma;/Prem.">&Theta;/Prem %</th>
        <th data-col="oiMin" onclick="sortTable('oiMin')" title="Worst-leg open interest = min(long OI, short OI) — the liquidity that gates execution.">Liq (OI)</th>
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
let currentSort = { col: 'score', asc: false };
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
  allLabel.innerHTML = '<input type="checkbox" id="exp_all" value="all" checked style="accent-color:var(--accent);width:13px;height:13px;vertical-align:middle;"> <span>All expirations</span>';
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
      lbl.innerHTML = `<input type="checkbox" class="exp-date-cb" value="${val}" style="accent-color:var(--accent);width:13px;height:13px;vertical-align:middle;"> <span>${label} (${dte}d)</span>`;
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

// ---------------- Advanced filters toggle ----------------
function toggleAdvanced(forceOpen) {
  const adv = document.getElementById('advancedFilters');
  const btn = document.getElementById('advancedToggle');
  const open = (forceOpen === undefined) ? adv.classList.contains('collapsed') : forceOpen;
  adv.classList.toggle('collapsed', !open);
  btn.textContent = open ? 'Model inputs ▾' : 'Model inputs ▸';
  try { localStorage.setItem('finderAdvancedOpen', open ? '1' : '0'); } catch (e) {}
}
// Restore prior open/closed state (default closed).
toggleAdvanced(localStorage.getItem('finderAdvancedOpen') === '1');

// ---------------- Dip-buy / recovery preset ----------------
// A tunable starting point for buying call spreads on dips to catch a recovery:
// modest risk cap, meaningful leverage per 1% bounce, long strike near spot, and
// enough time for the move. Leaves ticker & expirations as currently chosen.
const DIP_BUY_PRESET = {
  maxPremium: '3000', minLeverage: '3', minRewardRisk: '1', maxOtm: '3',
  maxWidth: '50', movePct: '5', minDte: '45'
};
function applyDipBuyPreset() {
  applyParams(DIP_BUY_PRESET);
  doSearch();
}

// ---------------- Templates (saved parameter sets) ----------------

// Input IDs to capture in a template (order matters for restoring)
const TEMPLATE_INPUT_IDS = [
  'ticker','maxPremium','minLeverage','minReturn1sigma','maxWidth','maxOtm','movePct',
  'profitTarget','riskFreeRate','minNetDelta','minGamma','minRewardRisk','commission','maxLegPremium',
  'minShortLegDelta','minDte','sortBy'
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

// ---- Test mode: cache each underlying's option chain so repeated searches
// (e.g. after hours) don't re-fetch from the data source. Sent as ?test=1. ----
let activeSourceName = '';   // short name of the active data source, for the badge
function updateModeTag() {
  const t = document.getElementById('testMode').checked;
  const tag = document.getElementById('modeTag');
  tag.textContent = t ? 'TEST · cached' : ('LIVE' + (activeSourceName ? ' · ' + activeSourceName : ''));
  tag.style.background = t ? 'var(--yellow)' : 'var(--accent)';
  tag.style.color = t ? '#0f1117' : '#fff';
}
function clearChainCache() {
  fetch('/api/clear_cache').then(r => r.json()).then(d => {
    const btn = document.getElementById('clearCacheBtn');
    const old = btn.textContent;
    btn.textContent = 'Cleared (' + (d.cleared || 0) + ')';
    setTimeout(() => { btn.textContent = old; }, 1500);
  }).catch(() => {});
}
(function initTestMode() {
  const cb = document.getElementById('testMode');
  if (localStorage.getItem('finderTestMode') === '1') cb.checked = true;
  updateModeTag();
  cb.addEventListener('change', () => {
    localStorage.setItem('finderTestMode', cb.checked ? '1' : '0');
    updateModeTag();
  });
})();

// ---- Data source selector: server-global (NOT per-browser/localStorage) —
// the server is the source of truth and every page must agree. ----
(function initDataSource() {
  const sel = document.getElementById('dataSource');
  function apply(d) {
    sel.innerHTML = '';
    (d.sources || []).forEach(s => {
      const o = document.createElement('option');
      o.value = s.name; o.textContent = s.label;
      sel.appendChild(o);
    });
    sel.value = d.active;
    const active = (d.sources || []).find(s => s.name === d.active);
    // Badge shows the short vendor name (label up to the first parenthesis).
    activeSourceName = active ? active.label.replace(/\s*\(.*$/, '') : '';
    updateModeTag();
  }
  fetch('/api/source').then(r => r.json()).then(apply).catch(() => { sel.style.display = 'none'; });
  sel.addEventListener('change', () => {
    const prev = activeSourceName;
    fetch('/api/source', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: sel.value}),
    }).then(r => r.json()).then(d => {
      apply(d);                      // server echoes the (possibly unchanged) state
      if (d.error) {                 // flash the select red; apply() already reverted it
        sel.style.borderColor = 'var(--red)';
        setTimeout(() => { sel.style.borderColor = 'var(--border)'; }, 1500);
      }
    }).catch(() => {                 // network failure: revert the visible choice
      fetch('/api/source').then(r => r.json()).then(apply).catch(() => {});
      activeSourceName = prev;
    });
  });
})();

// Hand the current results to a new scatter-plot tab via localStorage (no re-fetch).
// Stash the current results (incl. live Score) to localStorage. Writing this key
// fires a 'storage' event in any open /scatter tab, which re-reads and re-renders —
// so an open scatter tab tracks the finder as you re-search or re-weight. The ts
// field guarantees the value changes every write so the event always fires.
function stashScatterData() {
  if (!allSpreads || !allSpreads.length) return false;
  const rf = parseFloat(document.getElementById('riskFreeRate').value) / 100;
  const mp = parseFloat(document.getElementById('movePct').value) || 1;
  const pt = parseFloat(document.getElementById('profitTarget').value) || 5;
  const payload = { spreads: allSpreads, spot: currentSpot, symbol: currentSymbol,
                    rfRate: rf, movePct: mp, profitTargetPct: pt, ts: Date.now() };
  try { localStorage.setItem('finderScatterData', JSON.stringify(payload)); return true; }
  catch (e) { return false; }
}
let _scatterStashTimer = null;
function stashScatterDataSoon() {
  clearTimeout(_scatterStashTimer);
  _scatterStashTimer = setTimeout(stashScatterData, 200);
}

function openScatter() {
  if (!allSpreads || !allSpreads.length) { showError('Run a search first — no spreads to plot.'); return; }
  if (!stashScatterData()) { showError('Could not stash results for the scatter tab.'); return; }
  window.open('/scatter', '_blank');
}

async function doSearch() {
  const symbol = document.getElementById('ticker').value.trim().toUpperCase();
  if (!symbol) { showError('Please enter a ticker symbol.'); return; }
  const maxPremiumDollars = parseFloat(document.getElementById('maxPremium').value);
  const minLeverage = parseFloat(document.getElementById('minLeverage').value);
  const minReturn1sigma = parseFloat(document.getElementById('minReturn1sigma').value) || 0;
  const maxWidth = parseFloat(document.getElementById('maxWidth').value);
  const maxOtm = parseFloat(document.getElementById('maxOtm').value);
  const riskFreeRate = parseFloat(document.getElementById('riskFreeRate').value) / 100;
  const minNetDelta = parseFloat(document.getElementById('minNetDelta').value);
  const minGamma = parseFloat(document.getElementById('minGamma').value) || 0;
  const minRewardRisk = parseFloat(document.getElementById('minRewardRisk').value);
  const commission = parseFloat(document.getElementById('commission').value);
  const maxLegPremium = parseFloat(document.getElementById('maxLegPremium').value);
  const minShortLegDelta = parseFloat(document.getElementById('minShortLegDelta').value) || 0;
  const minDte = parseInt(document.getElementById('minDte').value) || 0;
  const movePct = parseFloat(document.getElementById('movePct').value) || 1.0;
  const profitTarget = parseFloat(document.getElementById('profitTarget').value) || 5.0;
  const allCb = document.getElementById('exp_all');
  let expiration = 'all';
  if (!allCb.checked) {
    const checked = [...document.querySelectorAll('.exp-date-cb:checked')].map(cb => cb.value);
    expiration = checked.length > 0 ? checked.join(',') : 'all';
  }

  // Convert actual dollars to quoted points (options multiplier = 100). No min
  // premium filter — send 0 so only the max-premium (risk) cap applies.
  const minPremium = 0;
  const maxPremium = maxPremiumDollars / 100;

  if (isNaN(maxPremiumDollars) || maxPremiumDollars <= 0) {
    showError('Please enter a valid max premium greater than 0.');
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
      min_return_1sigma: minReturn1sigma,
      max_width: maxWidth,
      max_otm: maxOtm,
      risk_free_rate: riskFreeRate,
      min_net_delta: minNetDelta,
      min_gamma: minGamma,
      min_reward_risk: minRewardRisk,
      commission: commission,
      max_leg_premium: maxLegPremium,
      min_short_leg_delta: minShortLegDelta,
      min_dte: minDte,
      move_pct: movePct,
      profit_target_pct: profitTarget,
      expiration: expiration,
      test: (document.getElementById('testMode').checked ? 1 : 0)
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
      document.getElementById('rocMoveHeader').innerHTML = 'Return @ +' + label;
    }
    if (data.profitTargetPct !== undefined && data.profitTargetPct !== null) {
      const pt = Number(data.profitTargetPct);
      const ptLabel = (Number.isInteger(pt) ? pt.toFixed(0) : pt.toString()) + '%';
      document.getElementById('probTargetHeader').innerHTML = 'P(+' + ptLabel + ')';
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
    computeScores();
    document.getElementById('scoreWeights').style.display = allSpreads.length ? 'block' : 'none';
    buildDteFilters();
    renderTable();
    stashScatterData();  // keep any open scatter tab in sync with this search

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

// ---- Composite Score: percentile-rank blend for risk-adjusted capital efficiency ----
// All five metrics are framed "higher = better" (incl. thetaPrem, where a less-negative
// decay ranks best), so every weight is positive. Ranks are within the current result set.
const SCORE_METRICS = ['returnAtMove', 'probTarget', 'gammaPrem', 'rewardRisk', 'thetaPrem'];
const SCORE_PRESETS = {
  balanced:     { returnAtMove: 30, probTarget: 25, gammaPrem: 20, rewardRisk: 15, thetaPrem: 10 },
  aggressive:   { returnAtMove: 40, probTarget: 15, gammaPrem: 25, rewardRisk: 10, thetaPrem: 10 },
  conservative: { returnAtMove: 25, probTarget: 35, gammaPrem: 10, rewardRisk: 20, thetaPrem: 10 },
};
let scoreWeights = Object.assign({}, SCORE_PRESETS.balanced);
let scoreOiTarget = 500;

// Percentile rank in [0,1]; null/NaN sorts to the bottom (worst). rank 0 = worst.
function pctRanks(vals) {
  const n = vals.length;
  if (n === 0) return [];
  if (n === 1) return [1];
  const idx = vals.map((v, i) => [i, (v == null || !isFinite(v)) ? -Infinity : v]);
  idx.sort((a, b) => a[1] - b[1]);
  const p = new Array(n);
  for (let r = 0; r < n; r++) p[idx[r][0]] = r / (n - 1);
  return p;
}

function computeScores() {
  const n = allSpreads.length;
  if (!n) return;
  const ranks = {};
  for (const mk of SCORE_METRICS) ranks[mk] = pctRanks(allSpreads.map(s => s[mk]));
  let wsum = 0;
  for (const mk of SCORE_METRICS) wsum += Math.max(0, scoreWeights[mk] || 0);
  const oiTarget = scoreOiTarget > 0 ? scoreOiTarget : 1;
  for (let i = 0; i < n; i++) {
    const s = allSpreads[i];
    let raw = 0;
    if (wsum > 0) {
      for (const mk of SCORE_METRICS) raw += (Math.max(0, scoreWeights[mk] || 0) / wsum) * ranks[mk][i];
    }
    const oi = (s.oiMin != null ? s.oiMin : Math.min(s.oi_buy || 0, s.oi_sell || 0)) || 0;
    const gate = Math.min(1, oi / oiTarget);
    s.score = Math.round(raw * gate * 1000) / 10; // 0-100, 1 decimal
  }
}

function scoreColor(v) {
  if (v == null) return 'var(--text-dim)';
  if (v >= 66) return 'var(--green)';
  if (v >= 33) return 'var(--yellow)';
  return 'var(--red)';
}

function scoreWeightsChanged() {
  for (const mk of SCORE_METRICS) {
    const el = document.getElementById('sw_' + mk);
    if (el) {
      scoreWeights[mk] = Number(el.value);
      const lab = document.getElementById('swv_' + mk);
      if (lab) lab.textContent = el.value;
    }
  }
  const oiEl = document.getElementById('sw_oiTarget');
  if (oiEl) scoreOiTarget = Math.max(0, Number(oiEl.value) || 0);
  saveScoreWeights();
  if (allSpreads.length) { computeScores(); renderTable(); stashScatterDataSoon(); }
}

function applyScorePreset(name) {
  const p = SCORE_PRESETS[name] || SCORE_PRESETS.balanced;
  scoreWeights = Object.assign({}, p);
  for (const mk of SCORE_METRICS) {
    const el = document.getElementById('sw_' + mk); if (el) el.value = p[mk];
    const lab = document.getElementById('swv_' + mk); if (lab) lab.textContent = p[mk];
  }
  scoreWeightsChanged();
}

function toggleScoreWeights() {
  const body = document.getElementById('swBody');
  const btn = document.getElementById('swToggle');
  const open = body.classList.toggle('open');
  btn.innerHTML = '&#9881; Score weights ' + (open ? '&#9662;' : '&#9656;');
  try { localStorage.setItem('finderScoreOpen', open ? '1' : '0'); } catch (e) {}
}

function saveScoreWeights() {
  try {
    localStorage.setItem('finderScoreWeights', JSON.stringify(scoreWeights));
    localStorage.setItem('finderScoreOiTarget', String(scoreOiTarget));
  } catch (e) {}
}

function initScoreWeights() {
  try {
    const w = JSON.parse(localStorage.getItem('finderScoreWeights') || 'null');
    if (w && typeof w === 'object') {
      for (const mk of SCORE_METRICS) if (w[mk] != null && isFinite(Number(w[mk]))) scoreWeights[mk] = Number(w[mk]);
    }
    const oi = localStorage.getItem('finderScoreOiTarget');
    if (oi != null && isFinite(Number(oi))) scoreOiTarget = Number(oi);
  } catch (e) {}
  for (const mk of SCORE_METRICS) {
    const el = document.getElementById('sw_' + mk); if (el) el.value = scoreWeights[mk];
    const lab = document.getElementById('swv_' + mk); if (lab) lab.textContent = scoreWeights[mk];
  }
  const oiEl = document.getElementById('sw_oiTarget'); if (oiEl) oiEl.value = scoreOiTarget;
  if (localStorage.getItem('finderScoreOpen') === '1') {
    const body = document.getElementById('swBody'); const btn = document.getElementById('swToggle');
    if (body) body.classList.add('open');
    if (btn) btn.innerHTML = '&#9881; Score weights &#9662;';
  }
}
initScoreWeights();

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

  // X range: include spot, K1, and K2 with padding, then widen the whole
  // horizontal range to 2x the default span (centered) for more context.
  const spot = currentSpot || K1;
  const xPad = width * 0.3;
  const xLo = Math.min(K1, spot) - xPad;
  const xHi = K2 + xPad;
  const xMid = (xLo + xHi) / 2;
  const xHalf = (xHi - xLo) / 2 * 2;   // 2x the default half-range
  const xMin = xMid - xHalf;
  const xMax = xMid + xHalf;
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

// Compact "how long ago" label for a last-trade epoch (seconds).
function tradeAgeStr(epoch) {
  if (!epoch) return 'n/a';
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epoch));
  if (secs < 3600) return Math.floor(secs / 60) + 'm';
  if (secs < 86400) return Math.floor(secs / 3600) + 'h';
  return Math.floor(secs / 86400) + 'd';
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
    const staleFlag = (s.staleBuy || s.staleSell)
      ? `<span class="stale-flag" title="Stale quote risk — last trade: long ${tradeAgeStr(s.lastTradeBuy)} ago, short ${tradeAgeStr(s.lastTradeSell)} ago">&#9888;</span>` : '';
    tr.innerHTML = `
      <td class="score-cell" style="font-weight:700;color:${scoreColor(s.score)};">${s.score != null ? s.score.toFixed(0) : '--'}</td>
      <td>${s.expiration}
        <div class="row-tooltip">
          <div class="tt-header">Spread Detail — ${s.expiration} (${s.dte}d) — ${c} contract${c > 1 ? 's' : ''}</div>
          <span class="tt-buy">BUY</span>  ${s.buyStrike.toFixed(0)} call &nbsp;×${c} &nbsp;@ $${s.buyAsk.toFixed(2)} ask <span class="tt-dim">&nbsp; &Delta; ${s.deltaBuy != null ? s.deltaBuy.toFixed(3) : '--'} &nbsp; &Gamma; ${s.gammaBuy != null ? s.gammaBuy.toFixed(4) : '--'} &nbsp; Vol: ${s.volume_buy.toLocaleString()} &nbsp; OI: ${s.oi_buy.toLocaleString()}</span><br>
          <span class="tt-sell">SELL</span> ${s.sellStrike.toFixed(0)} call ×${c} &nbsp;@ $${s.sellBid.toFixed(2)} bid <span class="tt-dim">&nbsp; &Delta; ${s.deltaSell != null ? s.deltaSell.toFixed(3) : '--'} &nbsp; &Gamma; ${s.gammaSell != null ? s.gammaSell.toFixed(4) : '--'} &nbsp; Vol: ${s.volume_sell.toLocaleString()} &nbsp; OI: ${s.oi_sell.toLocaleString()}</span>
          <div class="tt-sep"></div>
          <span class="tt-buy">Pay:</span> &nbsp;$${buyEach} × ${c} = $${(s.buyAsk * m * c).toLocaleString('en-US', {maximumFractionDigits: 0})}<br>
          <span class="tt-sell">Recv:</span> $${sellEach} × ${c} = $${(s.sellBid * m * c).toLocaleString('en-US', {maximumFractionDigits: 0})}<br>
          <span class="tt-net">Net:&nbsp; $${netEach} × ${c} = $${totalDollars}</span><br>
          <span class="tt-dim">Comm: $${s.commissionPerSpread.toFixed(2)} × ${c} = $${s.totalCommission.toFixed(2)} RT</span><br>
          <span class="tt-dim">Max Profit</span> $${(s.maxProfit * m).toLocaleString('en-US', {maximumFractionDigits: 0})} &nbsp;·&nbsp; <span class="tt-dim">R/R</span> ${s.rewardRisk.toFixed(1)}x &nbsp;·&nbsp; <span class="tt-dim">Lev</span> ${s.leverage.toFixed(1)}x<br>
          <span class="tt-dim">EV</span> ${s.evPrem != null ? (s.evPrem >= 0 ? '+' : '') + s.evPrem.toFixed(1) + '%' : '--'} &nbsp;·&nbsp; <span class="tt-dim">P(prof exp)</span> ${s.probProfitExp != null ? s.probProfitExp.toFixed(1) + '%' : '--'} &nbsp;·&nbsp; <span class="tt-dim">P(&minus;50%)</span> ${s.probLoss50 != null ? s.probLoss50.toFixed(1) + '%' : '--'}<br>
          <span class="tt-dim">Med d&rarr;tgt</span> ${s.medianDaysToTarget != null ? s.medianDaysToTarget.toFixed(0) + 'd' : '--'} &nbsp;·&nbsp; <span class="tt-dim">Velocity</span> ${s.retVelocity != null ? s.retVelocity.toFixed(2) + '%/d' : '--'} &nbsp;·&nbsp; <span class="tt-dim">Last trade</span> L ${tradeAgeStr(s.lastTradeBuy)} / S ${tradeAgeStr(s.lastTradeSell)}
          <div class="tt-sep"></div>
          ${buildPnlChart(s)}
        </div>
      </td>
      <td>${c}</td>
      <td>${s.buyStrike.toFixed(0)}</td>
      <td>${s.sellStrike.toFixed(0)}</td>
      <td class="dim">${s.pctOtmBuy.toFixed(1)}%</td>
      <td>${s.spreadWidth.toFixed(0)}</td>
      <td class="highlight">$${totalDollars}${staleFlag}</td>
      <td>$${(s.maxProfit * m).toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>${s.rewardRisk.toFixed(1)}x</td>
      <td class="highlight">${s.leverage.toFixed(1)}x</td>
      <td class="highlight">${s.returnAtMove != null ? (s.returnAtMove >= 0 ? '+' : '') + s.returnAtMove + '%' : '--'}</td>
      <td>${(s.probTarget === null || s.probTarget === undefined) ? '<span class="dim">--</span>' : s.probTarget.toFixed(1) + '%'}</td>
      <td>${(s.probProfitExp === null || s.probProfitExp === undefined) ? '<span class="dim">--</span>' : s.probProfitExp.toFixed(1) + '%'}</td>
      <td>${(s.medianDaysToTarget === null || s.medianDaysToTarget === undefined) ? '<span class="dim">--</span>' : s.medianDaysToTarget.toFixed(0) + 'd'}</td>
      <td>${(s.evPrem === null || s.evPrem === undefined) ? '<span class="dim">--</span>' : (s.evPrem >= 0 ? '+' : '') + s.evPrem.toFixed(1) + '%'}</td>
      <td>${s.return1sigma != null ? (s.return1sigma >= 0 ? '+' : '') + s.return1sigma + '%' : '--'}</td>
      <td>${s.breakevenMovePct >= 0 ? '+' : ''}${s.breakevenMovePct.toFixed(2)}%</td>
      <td>${s.breakevenMoveSigma >= 0 ? '+' : ''}${s.breakevenMoveSigma.toFixed(2)}σ</td>
      <td>$${(s.netDelta * (currentSpot || 0)).toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
      <td>${s.netDeltaPer.toFixed(4)}</td>
      <td>${(s.gammaPer1pct >= 0 ? '+' : '') + (s.gammaPer1pct != null ? s.gammaPer1pct.toFixed(2) : '--')}</td>
      <td>${s.deltaPrem != null ? (s.deltaPrem >= 0 ? '+' : '') + s.deltaPrem + '%' : '--'}</td>
      <td>${s.gammaPrem != null ? (s.gammaPrem >= 0 ? '+' : '') + s.gammaPrem + '%' : '--'}</td>
      <td>${s.thetaPrem != null ? (s.thetaPrem >= 0 ? '+' : '') + s.thetaPrem + '%' : '--'}</td>
      <td class="dim">${(s.oiMin != null ? s.oiMin : Math.min(s.oi_buy, s.oi_sell)).toLocaleString()}</td>
    `;
    tbody.appendChild(tr);
  }

  document.getElementById('matchCount').textContent =
    spreads.length + (spreads.length !== allSpreads.length ? ` / ${allSpreads.length}` : '');

  if (display.length < spreads.length) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="26" style="text-align:center;color:var(--text-dim);padding:16px;"><!-- 26 cols -->
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

  // Vertical: prefer above the row; flip below if it would clip the top; then
  // clamp so it never runs off the bottom (or top) of the viewport.
  let top = rect.top - tipRect.height - pad;
  if (top < pad) {
    top = rect.bottom + pad;
  }
  top = Math.min(top, window.innerHeight - tipRect.height - pad);
  top = Math.max(pad, top);

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
  th.adj-col, td.adj-col { border-left: 2px solid #fff; border-right: 2px solid #fff; }
  th.adj-col { border-top: 2px solid #fff; }
  tbody tr:last-child td.adj-col { border-bottom: 2px solid #fff; }
  .mono { font-family: var(--mono); }
  .dim { color: var(--text-dim); }
  .pnl-pos { color: var(--green); font-weight: 600; }
  .pnl-neg { color: var(--red); font-weight: 600; }
  .err-row { color: var(--yellow); }

  .quote-age { display: inline-block; font-family: var(--font); font-size: 10px; font-weight: 500;
               padding: 1px 6px; border-radius: 999px; margin-left: 6px; vertical-align: middle; }
  .quote-age.qa-fresh   { background: rgba(16, 185, 129, 0.15); color: var(--green); }
  .quote-age.qa-delayed { background: rgba(245, 158, 11, 0.15); color: var(--yellow); }
  .quote-age.qa-stale   { background: rgba(239, 68, 68, 0.15);  color: var(--red); }

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
  .stale-flag { color: #fbbf24; margin-left: 4px; cursor: help; }

  .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .stat { background: var(--surface2); border-radius: 6px; padding: 12px 14px;
          display: flex; flex-direction: column; gap: 4px; }
  .stat-label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-value { font-size: 18px; font-weight: 600; }
  .stat-sub { font-size: 11px; font-weight: 500; }
  .stat-sub.dim { color: var(--text-dim); }
  .stat-highlight { grid-column: span 2; display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
                    border: 2px solid #fff; border-radius: 8px; padding: 6px; }
  /* Combined scenario-risk box: own-vol ±σ, index-β ±σ, and Θ/Vega in one stat. */
  .stat-risk { grid-column: span 2; }
  .risk-rows { display: flex; flex-direction: column; gap: 4px; font-family: var(--mono); font-size: 14px; font-weight: 600; margin-top: 2px; }
  .risk-rows .rrow { display: grid; grid-template-columns: 96px 1fr 1fr; gap: 10px; align-items: baseline; }
  .risk-rows .rlbl { color: var(--text-dim); font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }

  .greek-block { display: flex; flex-direction: column; gap: 2px; font-family: var(--mono); font-size: 12px; }
  .greek-block .lbl { color: var(--text-dim); font-size: 10px; }
</style>
</head>
<body>

<div class="header">
  <h1>My Positions <span class="tag" id="posModeTag">LIVE</span></h1>
  <div class="right">
    <label id="posTestToggle" style="display:flex;align-items:center;gap:6px;font-size:13px;color:var(--text-dim);cursor:pointer;" title="Test mode caches each underlying's option chain so the 30s auto-refresh (e.g. after hours) doesn't re-fetch from the data source. Quotes stay frozen until you Clear cache or turn Test off.">
      <input type="checkbox" id="posTestMode" style="width:15px;height:15px;accent-color:var(--yellow);"> Test mode
    </label>
    <select id="dataSource" style="background:transparent;color:var(--text-dim);border:1px solid var(--border);border-radius:6px;padding:4px 8px;font-size:12px;" title="Market data source (server-wide — affects all pages)"></select>
    <button type="button" id="posClearCacheBtn" class="secondary" style="padding:4px 9px;font-size:12px;" onclick="clearChainCache()">Clear cache</button>
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
  <div class="field">
    <label for="profitTarget" title="Profit target as a % of entry cost, on Adjusted P&amp;L. The P(+X%) column is the path-aware probability the Adjusted P&amp;L touches this target before expiry.">Profit target</label>
    <input id="profitTarget" type="number" min="0.1" step="0.5" value="15">
    <span class="dim">%</span>
  </div>
  <div class="field">
    <label for="indexSymbol" title="Reference index for the beta-scaled ±σ index-move P&amp;L column. Beta uses a 2-year daily lookback; the index's σ is its option-implied vol (VIX-family). Default ^GSPC (S&amp;P 500).">Beta index</label>
    <input id="indexSymbol" type="text" style="width:70px;" value="^GSPC">
  </div>
  <div class="field">
    <label for="sigmaMult" title="Number of standard deviations for the ±σ move columns (own-vol Greeks and beta-index) and the summary. Default 1; e.g. set 2 to see ±2σ scenarios.">Std devs</label>
    <input id="sigmaMult" type="number" min="0.1" max="10" step="0.5" value="1">
    <span class="dim">&sigma;</span>
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
    <div class="stat-highlight">
      <div class="stat"><div class="stat-label" id="sumAdjPnlLabel">Adj P&amp;L (80%)</div><div class="stat-value mono" id="sumAdjPnl">--</div><div class="stat-sub mono" id="sumAdjPnlDay">--</div></div>
      <div class="stat"><div class="stat-label">Total Return</div><div class="stat-value mono" id="sumRet">--</div></div>
    </div>
    <div class="stat stat-risk">
      <div class="stat-label" id="sumRiskLabel">Scenario P&amp;L (&plusmn;1&sigma;)</div>
      <div class="risk-rows" id="sumRisk">--</div>
    </div>
  </div>
</div>

<div class="table-wrap">
  <div class="empty" id="emptyState">No positions saved. Add one above to start watching.</div>
  <table id="posTable" style="display:none;">
    <thead>
      <tr>
        <th title="Label, symbol, expiration (days to expiry) and current spot.">Position</th>
        <th title="Both legs — L = long, S = short — with strike, then bid / ask / last / vol / IV.">Quote</th>
        <th>Contracts</th>
        <th title="Per-spread prices: net premium paid at entry (long entry − short entry) over the current liquidation (long bid − short ask).">Entry / Liq</th>
        <th title="Total premium paid at entry (top) over the current total value / liquidation (bottom), with entry commission as subtext.">Cost / Value</th>
        <th>P&amp;L</th>
        <th id="colAdjPnlLabel" class="adj-col">Adj P&amp;L (80%)</th>
        <th id="colProbTarget">P(+15%)</th>
        <th>Daily Theo P&amp;L</th>
        <th id="colBetaIdx" title="Theoretical P&amp;L for a &plusmn;1&sigma; move in the reference index, scaled by each underlying's beta (2yr daily) to that index. Top = +1&sigma;, bottom = &minus;1&sigma;.">&plusmn;1&sigma; Idx P&amp;L (&beta;)</th>
        <th id="colGreeks" title="Own-vol &plusmn;&sigma; one-day P&amp;L (full BS reprice, ~30d ATM IV — same engine and tenor basis as the &beta; column), plus &Theta; per day and Vega per 1% IV.">Greeks (&plusmn;1&sigma; P&amp;L / &Theta;$/d / Vega)</th>
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
let curSigmas = 1;      // active std-dev multiple for the ±σ columns

function $(id) { return document.getElementById(id); }
// Compact σ label, e.g. 1 -> "1", 1.5 -> "1.5", 2 -> "2".
function sigStr() { return Number.isInteger(curSigmas) ? String(curSigmas) : String(curSigmas); }
function dollarFmt(v, dp=2) { return (v >= 0 ? '$' : '-$') + Math.abs(v).toFixed(dp); }
function sign(v) { return (v >= 0 ? '+' : ''); }

// Badge showing how stale the quote snapshot is. quoteTime is epoch seconds
// from the option chain's own underlying, so it dates BOTH spot and options.
function quoteAgeBadge(epochSec) {
  if (!epochSec) return '';
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epochSec));
  let label;
  if (secs < 60) label = secs + 's';
  else if (secs < 3600) label = Math.floor(secs / 60) + 'm';
  else if (secs < 86400) label = Math.floor(secs / 3600) + 'h';
  else label = Math.floor(secs / 86400) + 'd';
  // Yahoo option quotes are ~15 min delayed, so a couple of minutes is "fresh";
  // >=30 min usually means a stuck feed or a closed market.
  let cls = 'qa-fresh';
  if (secs >= 1800) cls = 'qa-stale';
  else if (secs >= 120) cls = 'qa-delayed';
  return `<span class="quote-age ${cls}" title="Quote snapshot age (spot + options)">${label} ago</span>`;
}

function showErr(msg) { const b = $('errBanner'); b.textContent = msg; b.classList.add('show'); }
function clearErr() { $('errBanner').classList.remove('show'); }

function dteFromExp(expStr) {
  const exp = new Date(expStr + 'T16:00:00');
  return Math.round((exp - new Date()) / 86400000);
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
  const totalDailyTheoPnl = withData.reduce((s, r) => s + (r.dailyTheoPnl || 0), 0);
  const totalRet = totalEntry ? (totalAdjPnl / Math.abs(totalEntry)) * 100 : 0;
  const totalDailyTheoPct = totalEntry ? (totalDailyTheoPnl / Math.abs(totalEntry)) * 100 : 0;
  const totalDelta = withData.reduce((s, r) => s + (r.netDelta || 0), 0);
  const totalOneSigmaPnl = withData.reduce((s, r) => s + (r.oneSigmaPnl || 0), 0);
  const totalOneSigmaDown = withData.reduce((s, r) => s + (r.oneSigmaPnlDown || 0), 0);
  const totalTheta = withData.reduce((s, r) => s + (r.netThetaPerDay || 0), 0);
  const totalVega = withData.reduce((s, r) => s + (r.netVega || 0), 0);
  const totalBetaUp = withData.reduce((s, r) => s + (r.betaIndexUpPnl || 0), 0);
  const totalBetaDown = withData.reduce((s, r) => s + (r.betaIndexDownPnl || 0), 0);
  const totalBetaDelta = withData.reduce((s, r) => s + (r.betaDollarDeltaPer1Pct || 0), 0);
  const idxSym = (withData.find(r => r.indexSymbol) || {}).indexSymbol || '^GSPC';

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
  // Daily theoretical P&L: sum of each position's BS reprice for the underlying's
  // one-day move (spot vs. prior close). Independent of the stale option quotes.
  const dayEl = $('sumAdjPnlDay');
  dayEl.textContent = 'Daily Theo ' + fmtSignedDollar(totalDailyTheoPnl, 0) +
    ' (' + sign(totalDailyTheoPct) + totalDailyTheoPct.toFixed(2) + '%)';
  dayEl.title = 'Sum of the per-position Daily Theo P&L (Black-Scholes reprice for the underlying’s one-day move).';
  dayEl.className = 'stat-sub mono ' + (totalDailyTheoPnl >= 0 ? 'pnl-pos' : 'pnl-neg');
  // Reflect the active haircut % on the label and column header
  const activeHc = currentHaircutPct().toFixed(0);
  $('sumAdjPnlLabel').textContent = `Adj P&L (${activeHc}%)`;
  const colHdr = $('colAdjPnlLabel');
  if (colHdr) colHdr.textContent = `Adj P&L (${activeHc}%)`;
  // Combined scenario-risk box: own-vol ±σ (Greeks column) and index-β ±σ
  // (beta column), plus the net Θ / Vega, all in one stat for the portfolio.
  $('sumRiskLabel').innerHTML = 'Scenario P&L (&plusmn;' + sigStr() + '&sigma;)';
  const cls = v => (v >= 0 ? 'pnl-pos' : 'pnl-neg');
  const scen = (lbl, up, dn) =>
    '<div class="rrow"><span class="rlbl">' + lbl + '</span>'
    + '<span class="' + cls(up) + '">+' + sigStr() + 'σ ' + fmtSignedDollar(up, 0) + '</span>'
    + '<span class="' + cls(dn) + '">−' + sigStr() + 'σ ' + fmtSignedDollar(dn, 0) + '</span></div>';
  $('sumRisk').innerHTML =
    scen('Own IV', totalOneSigmaPnl, totalOneSigmaDown)
    + scen(idxSym + ' β', totalBetaUp, totalBetaDown)
    + '<div class="rrow"><span class="rlbl">Θ / Vega</span>'
      + '<span class="' + cls(totalTheta) + '">Θ ' + fmtSignedDollar(totalTheta, 2) + '/d</span>'
      + '<span class="' + cls(totalVega) + '">V ' + fmtSignedDollar(totalVega, 2) + '/1%</span></div>'
    + '<div class="rrow" title="Portfolio beta-weighted net delta: total theoretical $ P&L for a +1% move in the reference index (Σ per-position $Δ × β). Your aggregate calibrated leverage in index terms.">'
      + '<span class="rlbl">β-wtd Δ</span>'
      + '<span class="' + cls(totalBetaDelta) + '">' + fmtSignedDollar(totalBetaDelta, 0) + ' /+1% ' + idxSym + '</span></div>';
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
    const tgt = parseFloat($('profitTarget').value);
    const tgtParam = (isFinite(tgt) && tgt > 0) ? tgt : 15;
    const idx = ($('indexSymbol').value || '^GSPC').trim() || '^GSPC';
    const sg = parseFloat($('sigmaMult').value);
    const sgParam = (isFinite(sg) && sg > 0) ? sg : 1;
    const testOn = $('posTestMode') && $('posTestMode').checked;
    const r = await fetch('/api/positions/quotes?haircut=' + encodeURIComponent(hc)
      + '&target=' + encodeURIComponent(tgtParam)
      + '&index=' + encodeURIComponent(idx)
      + '&sigmas=' + encodeURIComponent(sgParam)
      + '&test=' + (testOn ? 1 : 0));
    const data = await r.json();
    if (data.error) { showErr(data.error); return; }
    lastQuotes = data.positions;
    if (data.nSigma !== undefined && data.nSigma !== null) curSigmas = Number(data.nSigma);
    if (data.profitTargetPct !== undefined && data.profitTargetPct !== null) {
      const pt = Number(data.profitTargetPct);
      const ptLabel = (Number.isInteger(pt) ? pt.toFixed(0) : pt.toString());
      $('colProbTarget').textContent = 'P(+' + ptLabel + '%)';
    }
    $('colGreeks').innerHTML = 'Greeks (&plusmn;' + sigStr() + '&sigma; P&L / &Theta;$/d / Vega)';
    if (data.indexSymbol) {
      $('colBetaIdx').innerHTML = '&plusmn;' + sigStr() + '&sigma; ' + data.indexSymbol + ' P&L (&beta;)';
    }
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
    // Both legs collapsed into one "Quote" column: long then short, one line
    // each — side+strike, then bid / ask / last / vol / IV. A leg whose last
    // trade predates the quote snapshot by more than a day gets a staleness
    // flag — its bid/ask may not reflect a tradable market.
    const legStale = (leg) => {
      if (!leg || !leg.lastTrade || !p.quoteTime) return '';
      const age = p.quoteTime - leg.lastTrade;
      if (age <= 86400) return '';
      return `<span class="stale-flag" title="Last trade ${Math.floor(age / 86400)}d before this quote snapshot — bid/ask may be stale">&#9888;</span>`;
    };
    const legLine = (leg, side, strike) => {
      if (!leg) return `<span><span class="strike">${side} ${strike}</span> <span class="quote err-row">missing</span></span>`;
      const ivStr = leg.iv !== null && leg.iv !== undefined ? leg.iv.toFixed(1) + '%' : '--';
      return `<span><span class="strike">${side} ${strike}</span> <span class="quote">${leg.bid.toFixed(2)} / ${leg.ask.toFixed(2)} / ${leg.last.toFixed(2)} / ${leg.volume.toLocaleString()} / ${ivStr}</span>${legStale(leg)}</span>`;
    };
    const quoteCell = `<td><div class="leg-block">
      ${legLine(p.long, 'L', p.longStrike)}
      ${legLine(p.short, 'S', p.shortStrike)}
    </div></td>`;

    // Own-vol ±1σ P&L (full BS reprice, ~30d tenor) — same engine as the β column,
    // so the two differ only by the shock source (own total vol vs β·index vol).
    let greeksCell = '<td class="dim">--</td>';
    if (p.netDelta !== null && p.netDelta !== undefined) {
      const gUp = p.oneSigmaPnl, gDn = p.oneSigmaPnlDown;
      const upCls = (gUp || 0) >= 0 ? 'pnl-pos' : 'pnl-neg';
      const dnCls = (gDn || 0) >= 0 ? 'pnl-pos' : 'pnl-neg';
      const dS = (p.oneSigmaMove != null) ? `ΔS ±${Math.abs(p.oneSigmaMove).toFixed(2)}` : '';
      const ivPart = (p.oneSigmaIvPct != null) ? ` · IV ${p.oneSigmaIvPct.toFixed(1)}%` : '';
      greeksCell = `<td><div class="greek-block">
        <span class="${upCls}">+${sigStr()}σ ${fmtSignedDollar(gUp || 0, 0)}</span>
        <span class="${dnCls}">−${sigStr()}σ ${fmtSignedDollar(gDn || 0, 0)}</span>
        <span class="${p.netThetaPerDay >= 0 ? 'pnl-pos' : 'pnl-neg'}">Θ ${p.netThetaPerDay >= 0 ? '+' : ''}$${p.netThetaPerDay.toFixed(2)}</span>
        <span>V ${p.netVega >= 0 ? '+' : ''}$${p.netVega.toFixed(2)}</span>
        <span class="tt-dim">${dS}${ivPart}</span>
      </div></td>`;
    }

    let pnlCell = '<td class="dim">--</td>';
    if (p.pnl !== null && p.pnl !== undefined) {
      const cls = p.pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      pnlCell = `<td><div class="pnl-block">
        <span class="${cls}">${sign(p.pnl)}${dollarFmt(p.pnl, 0)}</span>
        <span class="${cls} pct">${sign(p.pnlPct)}${p.pnlPct.toFixed(2)}%</span>
      </div></td>`;
    }

    let adjPnlCell = '<td class="dim adj-col">--</td>';
    if (p.adjPnl !== null && p.adjPnl !== undefined) {
      const cls = p.adjPnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      const totalComm = (p.totalCommission != null) ? p.totalCommission : 0;
      const hcLabel = (p.haircutPct != null) ? p.haircutPct.toFixed(0) : '80';
      const derivation = (p.pnl > 0)
        ? `pnl × ${hcLabel}% − $${totalComm.toFixed(2)} comm`
        : `pnl − $${totalComm.toFixed(2)} comm (no haircut on loss)`;
      adjPnlCell = `<td class="adj-col"><div class="pnl-block">
        <span class="${cls}">${sign(p.adjPnl)}${dollarFmt(p.adjPnl, 0)}</span>
        <span class="${cls} pct">${sign(p.adjPnlPct)}${p.adjPnlPct.toFixed(2)}%</span>
        <span class="tt-dim">${derivation}</span>
      </div></td>`;
    }

    const probCell = (p.probTarget === null || p.probTarget === undefined)
      ? '<td class="dim">--</td>'
      : `<td class="mono">${p.probTarget.toFixed(1)}%</td>`;

    // Daily theoretical P&L: BS reprice of the spread for the underlying's
    // one-day move (spot vs. prior close), independent of the stale option quotes.
    let dailyTheoCell = '<td class="dim">--</td>';
    if (p.dailyTheoPnl !== null && p.dailyTheoPnl !== undefined) {
      const cls = p.dailyTheoPnl >= 0 ? 'pnl-pos' : 'pnl-neg';
      const moveStr = (p.dailyTheoMove !== null && p.dailyTheoMove !== undefined)
        ? `<span class="tt-dim">${p.symbol} ${sign(p.dailyTheoMove)}${Math.abs(p.dailyTheoMove).toFixed(2)}</span>` : '';
      dailyTheoCell = `<td><div class="pnl-block">
        <span class="${cls}">${sign(p.dailyTheoPnl)}${dollarFmt(p.dailyTheoPnl, 0)}</span>
        ${moveStr}
      </div></td>`;
    }

    // Beta-scaled ±1σ index-move P&L: top = +1σ index, bottom = -1σ index,
    // each colored by its own sign. Subtext shows beta and the implied ΔS.
    let betaCell = '<td class="dim">--</td>';
    if (p.betaIndexUpPnl !== null && p.betaIndexUpPnl !== undefined) {
      const up = p.betaIndexUpPnl, dn = p.betaIndexDownPnl;
      const upCls = up >= 0 ? 'pnl-pos' : 'pnl-neg';
      const dnCls = dn >= 0 ? 'pnl-pos' : 'pnl-neg';
      const beta = (p.beta != null) ? p.beta.toFixed(2) : '--';
      const dS = (p.betaIndexMove != null) ? `ΔS ${sign(p.betaIndexMove)}${Math.abs(p.betaIndexMove).toFixed(2)}` : '';
      betaCell = `<td><div class="pnl-block">
        <span class="${upCls}">+${sigStr()}σ ${sign(up)}${dollarFmt(up, 0)}</span>
        <span class="${dnCls}">−${sigStr()}σ ${sign(dn)}${dollarFmt(dn, 0)}</span>
        <span class="tt-dim">β ${beta} · ${dS}</span>
      </div></td>`;
    }

    const label = p.label || `${p.symbol} ${p.longStrike}/${p.shortStrike}`;
    const spotCell = (p.spot !== null && p.spot !== undefined ? `$${p.spot.toFixed(2)}` : '--') + quoteAgeBadge(p.quoteTime);
    // Entry Spread and Liquidation collapsed into one column: entry price paid
    // per spread over the current liquidation price per spread.
    const es = (p.entrySpread !== null && p.entrySpread !== undefined) ? `$${p.entrySpread.toFixed(2)}` : '--';
    const lq = (p.spreadLiquidation !== null && p.spreadLiquidation !== undefined) ? `$${p.spreadLiquidation.toFixed(2)}` : '--';
    const entryLiqCell = `<div class="pnl-block">
      <span class="mono">${es} <span class="tt-dim">entry</span></span>
      <span class="mono">${lq} <span class="tt-dim">liq</span></span>
    </div>`;
    // Entry Cost and Current Value collapsed into one column: total premium paid
    // over the current total value, with the entry commission as subtext.
    const cv = (p.currentValue !== null && p.currentValue !== undefined) ? dollarFmt(p.currentValue, 0) : '--';
    let costValCell = '<div class="pnl-block"><span class="dim">--</span></div>';
    if (p.entryCost !== null && p.entryCost !== undefined) {
      const entryComm = (p.entryCommission != null) ? p.entryCommission : 0;
      costValCell = `<div class="pnl-block">
        <span class="mono">${dollarFmt(p.entryCost, 0)} <span class="tt-dim">cost</span></span>
        <span class="mono">${cv} <span class="tt-dim">value</span></span>
        <span class="tt-dim">entry comm $${entryComm.toFixed(2)}</span>
      </div>`;
    }

    // Label/Symbol, Expiration (DTE) and Spot collapsed into one "Position" cell.
    const positionCell = `<td><div class="leg-block">
      <span class="strike">${label}</span>
      <span class="quote">${p.symbol} · ${p.expiration} <span class="dim">(${dte}d)</span></span>
      <span class="quote">spot ${spotCell}</span>
    </div></td>`;

    tr.innerHTML = `
      ${positionCell}
      ${quoteCell}
      <td class="mono">${p.contracts}</td>
      <td class="mono">${entryLiqCell}</td>
      <td class="mono">${costValCell}</td>
      ${pnlCell}
      ${adjPnlCell}
      ${probCell}
      ${dailyTheoCell}
      ${betaCell}
      ${greeksCell}
      <td>
        <button class="ghost" data-edit="${p.id}">Edit</button>
        <button class="danger" data-del="${p.id}">Delete</button>
      </td>
    `;
    if (p.error) {
      const errTd = document.createElement('tr');
      errTd.innerHTML = `<td colspan="12" class="err-row dim">${p.symbol} ${p.longStrike}/${p.shortStrike}: ${p.error}</td>`;
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

// Persist the beta reference index across sessions and re-refresh when it changes.
const _savedIndex = localStorage.getItem('betaIndexSymbol');
if (_savedIndex) { $('indexSymbol').value = _savedIndex; }
$('indexSymbol').addEventListener('change', () => {
  localStorage.setItem('betaIndexSymbol', $('indexSymbol').value);
  refreshQuotes();
});

// Persist the std-dev multiple across sessions and re-refresh when it changes.
const _savedSigmas = localStorage.getItem('sigmaMult');
if (_savedSigmas !== null && isFinite(parseFloat(_savedSigmas))) {
  $('sigmaMult').value = _savedSigmas;
}
$('sigmaMult').addEventListener('change', () => {
  localStorage.setItem('sigmaMult', $('sigmaMult').value);
  refreshQuotes();
});

// Test/Live mode: in Test mode the server caches each underlying's option chain,
// so the 30s auto-refresh reuses frozen quotes instead of re-fetching.
let activeSourceName = '';   // short name of the active data source, for the badge
function updateModeTag() {
  const t = $('posTestMode').checked;
  const tag = $('posModeTag');
  tag.textContent = t ? 'TEST · cached' : ('LIVE' + (activeSourceName ? ' · ' + activeSourceName : ''));
  tag.style.background = t ? 'var(--yellow)' : '';
  tag.style.color = t ? '#0f1117' : '';
}
function clearChainCache() {
  fetch('/api/clear_cache').then(r => r.json()).then(d => {
    const btn = $('posClearCacheBtn');
    const old = btn.textContent;
    btn.textContent = 'Cleared (' + (d.cleared || 0) + ')';
    setTimeout(() => { btn.textContent = old; }, 1500);
    refreshQuotes();
  }).catch(() => {});
}
(function initTestMode() {
  const cb = $('posTestMode');
  if (localStorage.getItem('posTestMode') === '1') cb.checked = true;
  updateModeTag();
  cb.addEventListener('change', () => {
    localStorage.setItem('posTestMode', cb.checked ? '1' : '0');
    updateModeTag();
    refreshQuotes();
  });
})();

// Data source selector: server-global (NOT per-browser/localStorage) — the
// server is the source of truth and every page must agree.
(function initDataSource() {
  const sel = $('dataSource');
  function apply(d) {
    sel.innerHTML = '';
    (d.sources || []).forEach(s => {
      const o = document.createElement('option');
      o.value = s.name; o.textContent = s.label;
      sel.appendChild(o);
    });
    sel.value = d.active;
    const active = (d.sources || []).find(s => s.name === d.active);
    activeSourceName = active ? active.label.replace(/\s*\(.*$/, '') : '';
    updateModeTag();
  }
  fetch('/api/source').then(r => r.json()).then(apply).catch(() => { sel.style.display = 'none'; });
  sel.addEventListener('change', () => {
    fetch('/api/source', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: sel.value}),
    }).then(r => r.json()).then(d => {
      apply(d);                      // server echoes the (possibly unchanged) state
      if (d.error) {                 // flash the select red; apply() already reverted it
        sel.style.borderColor = 'var(--red)';
        setTimeout(() => { sel.style.borderColor = 'var(--border)'; }, 1500);
      } else {
        refreshQuotes();             // reprice immediately off the new source
      }
    }).catch(() => {
      fetch('/api/source').then(r => r.json()).then(apply).catch(() => {});
    });
  });
})();

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
# Scatter-plot page (/scatter) — plots the finder's last results, two columns
# on the axes, one point per spread, with the SAME hover popup as the table.
# Data is handed over from the finder via localStorage (no re-fetch). This page
# is self-contained: it carries its own copy of the payoff chart + Spread Detail
# popup so the finder's working popup is left untouched.
# ---------------------------------------------------------------------------

# Copy of the finder's payoff chart, with the risk-free rate taken from the
# handed-over data (`rfRate`) instead of the finder's #riskFreeRate input.
PNL_CHART_JS = r'''
function buildPnlChart(s) {
  const m = 100;
  const c = s.contracts;
  const K1 = s.buyStrike;
  const K2 = s.sellStrike;
  const prem = s.netPremium * m * c;
  const totalComm = s.totalCommission;
  const totalCost = prem + totalComm;
  const maxProf = s.maxProfit * m;
  const width = s.spreadWidth;
  const maxLoss = -totalCost;
  const maxGain = maxProf;
  const be = s.breakeven;
  const W = 320, H = 300;
  const pad = {l: 55, r: 15, t: 15, b: 30};
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const spot = currentSpot || K1;
  const xPad = width * 0.3;
  const xLo = Math.min(K1, spot) - xPad;
  const xHi = K2 + xPad;
  const xMid = (xLo + xHi) / 2;
  const xHalf = (xHi - xLo) / 2 * 2;
  const xMin = xMid - xHalf;
  const xMax = xMid + xHalf;
  const xScale = (v) => pad.l + (v - xMin) / (xMax - xMin) * cw;
  const yPadding = Math.max(Math.abs(maxLoss), Math.abs(maxGain)) * 0.15;
  const yMin = maxLoss - yPadding;
  const yMax = maxGain + yPadding;
  const yScale = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * ch;
  const points = [
    {x: xMin, y: maxLoss}, {x: K1, y: maxLoss}, {x: K2, y: maxGain}, {x: xMax, y: maxGain},
  ];
  const line = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`).join(' ');
  const zeroY = yScale(0).toFixed(1);
  const lossFill = `M${xScale(xMin).toFixed(1)},${zeroY} ` +
    points.filter(p => p.x <= be + 0.1).map(p => `L${xScale(p.x).toFixed(1)},${yScale(Math.min(p.y, 0)).toFixed(1)}`).join(' ') +
    ` L${xScale(be).toFixed(1)},${zeroY} Z`;
  const profFill = `M${xScale(be).toFixed(1)},${zeroY} ` +
    `L${xScale(K2).toFixed(1)},${yScale(maxGain).toFixed(1)} ` +
    `L${xScale(xMax).toFixed(1)},${yScale(maxGain).toFixed(1)} ` +
    `L${xScale(xMax).toFixed(1)},${zeroY} Z`;
  const r = rfRate;
  const T = s.dte / 365;
  const ivB = s.ivBuy / 100;
  const ivS = s.ivSell / 100;
  const nSteps = 60;
  const theoPoints = [];
  for (let i = 0; i <= nSteps; i++) {
    const sx = xMin + (xMax - xMin) * i / nSteps;
    const callBuy = jsBsCallPrice(sx, K1, T, r, ivB);
    const callSell = jsBsCallPrice(sx, K2, T, r, ivS);
    const spreadVal = (callBuy - callSell) * m * c;
    const pnl = spreadVal - totalCost;
    theoPoints.push({x: sx, y: pnl});
  }
  const theoLine = theoPoints.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.x).toFixed(1)},${yScale(p.y).toFixed(1)}`).join(' ');
  const spotCallBuy = jsBsCallPrice(spot, K1, T, r, ivB);
  const spotCallSell = jsBsCallPrice(spot, K2, T, r, ivS);
  const spotPnl = (spotCallBuy - spotCallSell) * m * c - totalCost;
  const fmtK = (v) => v.toLocaleString('en-US', {maximumFractionDigits: 0});
  const fmtD = (v) => (v >= 0 ? '+' : '') + '$' + Math.abs(v).toLocaleString('en-US', {maximumFractionDigits: 0});
  const yTicks = [maxLoss, 0, maxGain];
  return `<svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin-top:6px;">
    <line x1="${pad.l}" y1="${zeroY}" x2="${W - pad.r}" y2="${zeroY}" stroke="#2e3348" stroke-width="1" stroke-dasharray="4,3"/>
    <path d="${lossFill}" fill="rgba(248,113,113,0.15)"/>
    <path d="${profFill}" fill="rgba(52,211,153,0.15)"/>
    <path d="${line}" fill="none" stroke="#e4e6f0" stroke-width="1.5" stroke-opacity="0.5"/>
    <path d="${theoLine}" fill="none" stroke="#c084fc" stroke-width="2"/>
    <circle cx="${xScale(spot).toFixed(1)}" cy="${yScale(spotPnl).toFixed(1)}" r="3.5" fill="#c084fc"/>
    <circle cx="${xScale(be).toFixed(1)}" cy="${zeroY}" r="3" fill="#fbbf24"/>
    <text x="${xScale(be).toFixed(1)}" y="${parseFloat(zeroY) - 7}" text-anchor="middle" fill="#fbbf24" font-size="9" font-family="sans-serif">BE ${fmtK(be)}</text>
    <text x="${xScale(K1).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#8b8fa3" font-size="9" font-family="sans-serif">${fmtK(K1)}</text>
    <text x="${xScale(K2).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#8b8fa3" font-size="9" font-family="sans-serif">${fmtK(K2)}</text>
    ${yTicks.map(v => `<text x="${pad.l - 5}" y="${(parseFloat(yScale(v)) + 3).toFixed(1)}" text-anchor="end" fill="${v > 0 ? '#34d399' : v < 0 ? '#f87171' : '#8b8fa3'}" font-size="9" font-family="sans-serif">${fmtD(v)}</text>`).join('')}
    <line x1="${xScale(K1).toFixed(1)}" y1="${pad.t}" x2="${xScale(K1).toFixed(1)}" y2="${H - pad.b}" stroke="#2e3348" stroke-width="1" stroke-dasharray="2,2"/>
    <line x1="${xScale(K2).toFixed(1)}" y1="${pad.t}" x2="${xScale(K2).toFixed(1)}" y2="${H - pad.b}" stroke="#2e3348" stroke-width="1" stroke-dasharray="2,2"/>
    <line x1="${xScale(spot).toFixed(1)}" y1="${pad.t}" x2="${xScale(spot).toFixed(1)}" y2="${H - pad.b}" stroke="#60a5fa" stroke-width="1.5" stroke-dasharray="4,2"/>
    <text x="${xScale(spot).toFixed(1)}" y="${H - 5}" text-anchor="middle" fill="#60a5fa" font-size="9" font-weight="600" font-family="sans-serif">${currentSymbol} ${fmtK(spot)}</text>
    <line x1="${W - 130}" y1="8" x2="${W - 115}" y2="8" stroke="#e4e6f0" stroke-width="1.5" stroke-opacity="0.5"/>
    <text x="${W - 112}" y="11" fill="#8b8fa3" font-size="8" font-family="sans-serif">At expiry</text>
    <line x1="${W - 65}" y1="8" x2="${W - 50}" y2="8" stroke="#c084fc" stroke-width="2"/>
    <text x="${W - 47}" y="11" fill="#c084fc" font-size="8" font-family="sans-serif">Now</text>
  </svg>`;
}
'''

# Copy of the finder's Spread Detail popup content (inner HTML only — the host
# element already has the .row-tooltip styling).
SPREAD_TOOLTIP_JS = r'''
// Compact "how long ago" label for a last-trade epoch (seconds).
function tradeAgeStr(epoch) {
  if (!epoch) return 'n/a';
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epoch));
  if (secs < 3600) return Math.floor(secs / 60) + 'm';
  if (secs < 86400) return Math.floor(secs / 3600) + 'h';
  return Math.floor(secs / 86400) + 'd';
}

function buildSpreadTooltip(s) {
  const m = 100, c = s.contracts;
  const buyEach = (s.buyAsk * m).toLocaleString('en-US', {maximumFractionDigits: 0});
  const sellEach = (s.sellBid * m).toLocaleString('en-US', {maximumFractionDigits: 0});
  const netEach = (s.netPremium * m).toLocaleString('en-US', {maximumFractionDigits: 0});
  const totalDollars = (s.totalPremium * m).toLocaleString('en-US', {maximumFractionDigits: 0});
  return `<div class="tt-header">Spread Detail — ${s.expiration} (${s.dte}d) — ${c} contract${c > 1 ? 's' : ''}</div>
    <span class="tt-buy">BUY</span>  ${s.buyStrike.toFixed(0)} call &nbsp;×${c} &nbsp;@ $${s.buyAsk.toFixed(2)} ask <span class="tt-dim">&nbsp; &Delta; ${s.deltaBuy != null ? s.deltaBuy.toFixed(3) : '--'} &nbsp; &Gamma; ${s.gammaBuy != null ? s.gammaBuy.toFixed(4) : '--'} &nbsp; Vol: ${s.volume_buy.toLocaleString()} &nbsp; OI: ${s.oi_buy.toLocaleString()}</span><br>
    <span class="tt-sell">SELL</span> ${s.sellStrike.toFixed(0)} call ×${c} &nbsp;@ $${s.sellBid.toFixed(2)} bid <span class="tt-dim">&nbsp; &Delta; ${s.deltaSell != null ? s.deltaSell.toFixed(3) : '--'} &nbsp; &Gamma; ${s.gammaSell != null ? s.gammaSell.toFixed(4) : '--'} &nbsp; Vol: ${s.volume_sell.toLocaleString()} &nbsp; OI: ${s.oi_sell.toLocaleString()}</span>
    <div class="tt-sep"></div>
    <span class="tt-buy">Pay:</span> &nbsp;$${buyEach} × ${c} = $${(s.buyAsk * m * c).toLocaleString('en-US', {maximumFractionDigits: 0})}<br>
    <span class="tt-sell">Recv:</span> $${sellEach} × ${c} = $${(s.sellBid * m * c).toLocaleString('en-US', {maximumFractionDigits: 0})}<br>
    <span class="tt-net">Net:&nbsp; $${netEach} × ${c} = $${totalDollars}</span><br>
    <span class="tt-dim">Comm: $${s.commissionPerSpread.toFixed(2)} × ${c} = $${s.totalCommission.toFixed(2)} RT</span><br>
    <span class="tt-dim">Max Profit</span> $${(s.maxProfit * m).toLocaleString('en-US', {maximumFractionDigits: 0})} &nbsp;·&nbsp; <span class="tt-dim">R/R</span> ${s.rewardRisk.toFixed(1)}x &nbsp;·&nbsp; <span class="tt-dim">Lev</span> ${s.leverage.toFixed(1)}x<br>
    <span class="tt-dim">EV</span> ${s.evPrem != null ? (s.evPrem >= 0 ? '+' : '') + s.evPrem.toFixed(1) + '%' : '--'} &nbsp;·&nbsp; <span class="tt-dim">P(prof exp)</span> ${s.probProfitExp != null ? s.probProfitExp.toFixed(1) + '%' : '--'} &nbsp;·&nbsp; <span class="tt-dim">P(&minus;50%)</span> ${s.probLoss50 != null ? s.probLoss50.toFixed(1) + '%' : '--'}<br>
    <span class="tt-dim">Med d&rarr;tgt</span> ${s.medianDaysToTarget != null ? s.medianDaysToTarget.toFixed(0) + 'd' : '--'} &nbsp;·&nbsp; <span class="tt-dim">Velocity</span> ${s.retVelocity != null ? s.retVelocity.toFixed(2) + '%/d' : '--'} &nbsp;·&nbsp; <span class="tt-dim">Last trade</span> L ${tradeAgeStr(s.lastTradeBuy)} / S ${tradeAgeStr(s.lastTradeSell)}
    <div class="tt-sep"></div>
    ${buildPnlChart(s)}`;
}
'''

SCATTER_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Spread Scatter</title>
<style>
  :root {
    --bg: #0f1117; --surface: #1a1d27; --surface2: #242837; --border: #2e3348;
    --text: #e4e6f0; --text-dim: #8b8fa3; --accent: #4f8ff7; --green: #34d399;
    --red: #f87171; --yellow: #fbbf24;
    --font: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { font-family: var(--font); background: var(--bg); color: var(--text);
         display: flex; flex-direction: column; overflow: hidden; }
  .bar { background: var(--surface); border-bottom: 1px solid var(--border);
         padding: 10px 20px; display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  .bar h1 { font-size: 15px; font-weight: 600; margin: 0; }
  .bar .field { display: flex; flex-direction: column; gap: 3px; }
  .bar label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); }
  .bar select { background: var(--bg); border: 1px solid var(--border); color: var(--text);
                font-family: var(--mono); font-size: 13px; padding: 5px 9px; border-radius: 5px; outline: none; }
  .bar a { color: var(--accent); text-decoration: none; font-size: 13px; margin-left: auto; }
  .bar a:hover { text-decoration: underline; }
  .bar .meta { color: var(--text-dim); font-size: 12px; font-family: var(--mono); }
  #panels { flex: 1 1 auto; min-height: 0; display: flex; }
  .panel { flex: 1 1 50%; min-width: 0; display: flex; flex-direction: column; }
  .panel + .panel { border-left: 1px solid var(--border); }
  .pctl { display: flex; align-items: flex-end; gap: 14px; flex-wrap: wrap;
          padding: 7px 16px; border-bottom: 1px solid var(--border); background: var(--surface); }
  .pctl .field { display: flex; flex-direction: column; gap: 3px; }
  .pctl label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); }
  .pctl select { background: var(--bg); border: 1px solid var(--border); color: var(--text);
                 font-family: var(--mono); font-size: 13px; padding: 4px 8px; border-radius: 5px; outline: none; }
  .plot { flex: 1 1 auto; min-height: 0; position: relative; }
  .plot svg { display: block; }
  .pt { cursor: pointer; transition: r 0.1s; }
  .empty { display: flex; align-items: center; justify-content: center; flex: 1 1 auto;
           color: var(--text-dim); text-align: center; padding: 40px; }
  .empty h2 { color: var(--text); font-weight: 600; }
  /* --- Spread Detail popup (copied from the finder so it looks identical) --- */
  .row-tooltip {
    display: none; position: fixed; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 18px; z-index: 20; white-space: nowrap;
    font-family: var(--mono); font-size: 13px; line-height: 1.45;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4); pointer-events: none;
    max-height: calc(100vh - 16px); overflow: hidden;
  }
  .row-tooltip .tt-header { font-family: var(--font); font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-dim); margin-bottom: 6px; }
  .row-tooltip .tt-buy { color: var(--green); }
  .row-tooltip .tt-sell { color: var(--red); }
  .row-tooltip .tt-net { color: var(--accent); font-weight: 600; }
  .row-tooltip .tt-dim { color: var(--text-dim); }
  .row-tooltip .tt-sep { border-top: 1px solid var(--border); margin: 6px 0; }
</style>
</head>
<body>
<div class="bar">
  <h1 id="title">Spread Scatter</h1>
  <span class="meta" id="meta"></span>
  <a href="/">&larr; Back to Finder</a>
</div>
<div class="empty" id="empty" style="display:none;"><div><h2>No data to plot</h2>
  <p>Run a search on the Finder, then click &ldquo;Scatter&rdquo;.</p></div></div>
<div id="panels">
  <div class="panel">
    <div class="pctl">
      <div class="field"><label>X axis</label><select id="x0"></select></div>
      <div class="field"><label>Y axis</label><select id="y0"></select></div>
      <div class="field"><label>Color</label><select id="z0"></select></div>
    </div>
    <div class="plot" id="plot0"></div>
  </div>
  <div class="panel">
    <div class="pctl">
      <div class="field"><label>X axis</label><select id="x1"></select></div>
      <div class="field"><label>Y axis</label><select id="y1"></select></div>
      <div class="field"><label>Color</label><select id="z1"></select></div>
    </div>
    <div class="plot" id="plot1"></div>
  </div>
</div>
<div class="row-tooltip" id="tip"></div>
<script>
// ---- data handed over from the finder via localStorage ----
let spreads = [], currentSpot = null, currentSymbol = '', rfRate = 0.045, movePct = 1, targetPct = 5;
let panelsReady = false;
function loadScatterData() {
  try {
    const raw = localStorage.getItem('finderScatterData');
    if (raw) {
      const d = JSON.parse(raw);
      spreads = d.spreads || [];
      currentSpot = d.spot; currentSymbol = d.symbol || '';
      rfRate = (d.rfRate != null ? d.rfRate : 0.045);
      movePct = (d.movePct != null ? d.movePct : 1);
      targetPct = (d.profitTargetPct != null ? d.profitTargetPct : 5);
    }
  } catch (e) {}
}
loadScatterData();

// ---- plottable columns (mirror the results table) ----
const mvLbl = (Number.isInteger(movePct) ? movePct : movePct) + '%';
const COLS = [
  {key:'score',        label:'Score (0-100)',      get:s=>s.score},
  {key:'leverage',     label:'Leverage (x)',       get:s=>s.leverage},
  {key:'returnAtMove', label:'Return @ +'+mvLbl,    get:s=>s.returnAtMove},
  {key:'rewardRisk',   label:'Reward/Risk',        get:s=>s.rewardRisk},
  {key:'probTarget',   label:'P(+'+targetPct+'%)',  get:s=>s.probTarget},
  {key:'probProfitExp', label:'P(profit @exp)',     get:s=>s.probProfitExp},
  {key:'medianDaysToTarget', label:'Median days→target', get:s=>s.medianDaysToTarget},
  {key:'retVelocity',  label:'Velocity %/day',     get:s=>s.retVelocity},
  {key:'evPrem',       label:'EV % of prem',       get:s=>s.evPrem},
  {key:'premium',      label:'Premium $',          get:s=>s.totalPremium*100},
  {key:'maxProfit',    label:'Max Profit $',       get:s=>s.maxProfit*100},
  {key:'pnlMove',      label:'P&L @'+mvLbl+' $',    get:s=>s.returnAtMove != null ? s.returnAtMove / 100 * s.totalPremium * 100 : null},
  {key:'return1sigma', label:'Return 1σ 1d %',      get:s=>s.return1sigma},
  {key:'beMovePct',    label:'BE Move %',          get:s=>s.breakevenMovePct},
  {key:'beMoveSigma',  label:'BE Move σ',          get:s=>s.breakevenMoveSigma},
  {key:'dollarDelta',  label:'$Δ /1%',             get:s=>s.netDelta*(currentSpot||0)},
  {key:'netDeltaPer',  label:'Δ/Contract',         get:s=>s.netDeltaPer},
  {key:'gammaPer1pct', label:'Γ (Δ/1%)',           get:s=>s.gammaPer1pct},
  {key:'deltaPrem',    label:'Δ/Prem %',            get:s=>s.deltaPrem},
  {key:'gammaPrem',    label:'Γ/Prem %',            get:s=>s.gammaPrem},
  {key:'thetaPrem',    label:'Θ/Prem %',            get:s=>s.thetaPrem},
  {key:'ivBuy',        label:'IV long %',          get:s=>s.ivBuy},
  {key:'oiMin',        label:'Liq (min OI)',       get:s=>s.oiMin != null ? s.oiMin : Math.min(s.oi_buy, s.oi_sell)},
  {key:'dte',          label:'DTE',                get:s=>s.dte},
  {key:'pctOtmBuy',    label:'% OTM',              get:s=>s.pctOtmBuy},
  {key:'spreadWidth',  label:'Width (pts)',        get:s=>s.spreadWidth},
  {key:'buyStrike',    label:'Buy Strike',         get:s=>s.buyStrike},
  {key:'sellStrike',   label:'Sell Strike',        get:s=>s.sellStrike},
  {key:'contracts',    label:'Contracts',          get:s=>s.contracts},
];
const colByKey = {};
COLS.forEach(c => colByKey[c.key] = c);

// ---- BS helpers + payoff chart + popup (self-contained copy of the finder's) ----
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
__PNL_CHART__
__SPREAD_TOOLTIP__

// ---- scatter rendering ----
const NF = (v, d=0) => (v==null||isNaN(v)) ? '--' : Number(v).toLocaleString('en-US', {maximumFractionDigits: d});
function niceRange(vals) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}
function render(idx) {
  if (!spreads.length) return;
  const plot = document.getElementById('plot'+idx);
  const xc = colByKey[document.getElementById('x'+idx).value];
  const yc = colByKey[document.getElementById('y'+idx).value];
  const pts = spreads.map(s => ({s, x: xc.get(s), y: yc.get(s)}))
                     .filter(p => p.x != null && p.y != null && isFinite(p.x) && isFinite(p.y));
  const W = plot.clientWidth, H = plot.clientHeight;
  const pad = {l: 72, r: 24, t: 20, b: 46};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const [xLo, xHi] = niceRange(pts.map(p => p.x));
  const [yLo, yHi] = niceRange(pts.map(p => p.y));
  const xS = v => pad.l + (v - xLo) / (xHi - xLo) * cw;
  const yS = v => pad.t + (1 - (v - yLo) / (yHi - yLo)) * ch;
  // Third axis = point color. Expiration keeps the DTE gradient; any numeric
  // dimension is split into thirds of its range: low=red, mid=yellow, high=green.
  const zKey = document.getElementById('z'+idx).value;
  let colorFor, legendText;
  if (zKey === '__dte__') {
    const dtes = pts.map(p => p.s.dte);
    const dLo = Math.min(...dtes), dHi = Math.max(...dtes);
    colorFor = s => { const t = dHi > dLo ? (s.dte - dLo) / (dHi - dLo) : 0; return 'hsl(' + (210 + 90*t).toFixed(0) + ', 72%, 60%)'; };
    legendText = 'color = DTE (' + dLo + 'd → ' + dHi + 'd)';
  } else {
    const zc = colByKey[zKey];
    const zv = pts.map(p => zc.get(p.s)).filter(v => v != null && isFinite(v));
    const zLo = Math.min(...zv), zHi = Math.max(...zv);
    colorFor = s => {
      const v = zc.get(s);
      if (v == null || !isFinite(v) || zHi === zLo) return '#fbbf24';
      const t = (v - zLo) / (zHi - zLo);
      return t < 1/3 ? '#f87171' : t < 2/3 ? '#fbbf24' : '#34d399';
    };
    legendText = 'color = ' + zc.label + '  (red low · yellow mid · green high)';
  }
  const tick = (lo, hi, sc, axis) => {
    let out = '';
    for (let i = 0; i <= 4; i++) {
      const v = lo + (hi - lo) * i / 4;
      if (axis === 'x') { const x = sc(v); out += '<line x1="'+x+'" y1="'+pad.t+'" x2="'+x+'" y2="'+(pad.t+ch)+'" stroke="#242837"/>'
        + '<text x="'+x+'" y="'+(pad.t+ch+16)+'" text-anchor="middle" fill="#8b8fa3" font-size="10">'+NF(v, Math.abs(hi-lo)<10?2:0)+'</text>'; }
      else { const y = sc(v); out += '<line x1="'+pad.l+'" y1="'+y+'" x2="'+(pad.l+cw)+'" y2="'+y+'" stroke="#242837"/>'
        + '<text x="'+(pad.l-8)+'" y="'+(y+3)+'" text-anchor="end" fill="#8b8fa3" font-size="10">'+NF(v, Math.abs(hi-lo)<10?2:0)+'</text>'; }
    }
    return out;
  };
  const circles = pts.map((p, i) =>
    '<circle class="pt" data-i="'+spreads.indexOf(p.s)+'" cx="'+xS(p.x).toFixed(1)+'" cy="'+yS(p.y).toFixed(1)+'" r="5" fill="'+colorFor(p.s)+'" fill-opacity="0.75" stroke="#0f1117" stroke-width="0.5"/>'
  ).join('');
  plot.querySelectorAll('svg').forEach(el => el.remove());
  const svg = '<svg width="'+W+'" height="'+H+'">'
    + tick(xLo, xHi, xS, 'x') + tick(yLo, yHi, yS, 'y')
    + '<text x="'+(pad.l+cw/2)+'" y="'+(H-8)+'" text-anchor="middle" fill="#e4e6f0" font-size="12" font-weight="600">'+xc.label+'</text>'
    + '<text transform="translate(16,'+(pad.t+ch/2)+') rotate(-90)" text-anchor="middle" fill="#e4e6f0" font-size="12" font-weight="600">'+yc.label+'</text>'
    + '<text x="'+(pad.l+cw)+'" y="'+(pad.t+2)+'" text-anchor="end" fill="#8b8fa3" font-size="10">'+legendText+'</text>'
    + circles + '</svg>';
  plot.insertAdjacentHTML('beforeend', svg);
}
function renderAll() { render(0); render(1); }

// ---- hover popup (single shared tooltip, positioned near the mouse) ----
const tip = document.getElementById('tip');
function positionTip(clientX, clientY) {
  const pad = 12;
  const r = tip.getBoundingClientRect();
  let left = clientX + 16;
  left = Math.max(pad, Math.min(left, window.innerWidth - r.width - pad));
  let top = clientY + 16;
  top = Math.min(top, window.innerHeight - r.height - pad);
  top = Math.max(pad, top);
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}
function wireHover(plot) {
  plot.addEventListener('mouseover', (e) => {
    if (!e.target.classList || !e.target.classList.contains('pt')) return;
    const s = spreads[parseInt(e.target.dataset.i, 10)];
    if (!s) return;
    e.target.setAttribute('r', '7');
    tip.innerHTML = buildSpreadTooltip(s);
    tip.style.display = 'block';
    positionTip(e.clientX, e.clientY);
  });
  plot.addEventListener('mousemove', (e) => {
    if (tip.style.display === 'block') positionTip(e.clientX, e.clientY);
  });
  plot.addEventListener('mouseout', (e) => {
    if (e.target.classList && e.target.classList.contains('pt')) {
      e.target.setAttribute('r', '5');
      tip.style.display = 'none';
    }
  });
}

// ---- init ----
function fillSelect(el, withDte) {
  if (withDte) el.insertAdjacentHTML('beforeend', '<option value="__dte__">Expiration (DTE)</option>');
  COLS.forEach(c => el.insertAdjacentHTML('beforeend', '<option value="'+c.key+'">'+c.label+'</option>'));
}
function initPanel(idx, defs) {
  const x = document.getElementById('x'+idx), y = document.getElementById('y'+idx), z = document.getElementById('z'+idx);
  fillSelect(x, false); fillSelect(y, false); fillSelect(z, true);
  x.value = defs.x; y.value = defs.y; z.value = defs.z;
  x.addEventListener('change', () => render(idx));
  y.addEventListener('change', () => render(idx));
  z.addEventListener('change', () => render(idx));
  wireHover(document.getElementById('plot'+idx));
}
// Recovery-move % and profit-target % are baked into a few axis labels; refresh
// them (and the already-rendered <option> text) whenever fresh data arrives.
function updateDynamicLabels() {
  const mv = movePct + '%';
  colByKey.returnAtMove.label = 'Return @ +' + mv;
  colByKey.pnlMove.label = 'P&L @' + mv + ' $';
  colByKey.probTarget.label = 'P(+' + targetPct + '%)';
  ['x0','y0','z0','x1','y1','z1'].forEach(id => {
    const el = document.getElementById(id); if (!el) return;
    [...el.options].forEach(o => { if (colByKey[o.value]) o.textContent = colByKey[o.value].label; });
  });
}
// Render/refresh the page from the current `spreads`. Safe to call repeatedly: the
// panels initialize once, then subsequent calls just re-render (preserving axis picks).
function applyData() {
  document.getElementById('title').textContent = 'Spread Scatter' + (currentSymbol ? ' — ' + currentSymbol : '');
  document.getElementById('meta').textContent = spreads.length ? (spreads.length + ' spreads' + (currentSpot ? ' · spot ' + currentSpot : '')) : '';
  if (spreads.length) {
    document.getElementById('empty').style.display = 'none';
    document.getElementById('panels').style.display = '';
    if (!panelsReady) {
      initPanel(0, {x:'leverage', y:'returnAtMove', z:'__dte__'});
      initPanel(1, {x:'rewardRisk', y:'probTarget', z:'__dte__'});
      panelsReady = true;
    }
    updateDynamicLabels();
    renderAll();
  } else {
    document.getElementById('empty').style.display = 'flex';
    document.getElementById('panels').style.display = 'none';
  }
}
applyData();
window.addEventListener('resize', renderAll);
// Live sync: the finder re-stashes results (with a fresh ts) on every search and
// weight change, which fires this 'storage' event in this tab — re-read & re-render.
window.addEventListener('storage', (e) => {
  if (e.key && e.key !== 'finderScatterData') return;
  loadScatterData();
  applyData();
});
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
            min_short_leg_delta = float(params.get("min_short_leg_delta", [0.08])[0])
            symbol = params.get("symbol", ["^SPX"])[0].strip().upper()
            move_pct = float(params.get("move_pct", [1.0])[0])
            profit_target_pct = float(params.get("profit_target_pct", [5.0])[0])
            min_gamma = float(params.get("min_gamma", [0.0])[0])
            min_return_1sigma = float(params.get("min_return_1sigma", [0.0])[0])
            test_mode = params.get("test", ["0"])[0].lower() in ("1", "true", "test", "on")

            try:
                print(f"\n{'='*60}")
                print(f"Searching {symbol}: premium=${min_premium}-${max_premium}, min_leverage={min_leverage}x, max_width={max_width}pts, max_otm={max_otm}%, r={risk_free_rate:.3f}, min_delta={min_net_delta}, min_rr={min_reward_risk}, commission=${commission}, min_dte={min_dte}, max_leg_premium=${max_leg_premium}, min_short_leg_delta=${min_short_leg_delta}, move_pct={move_pct}%, expiration={expiration_filter}")
                print(f"{'='*60}")
                result = fetch_and_find_spreads(min_premium, max_premium, min_leverage, max_width, max_otm, risk_free_rate, expiration_filter, min_net_delta, min_reward_risk, commission, min_dte, max_leg_premium, symbol, move_pct, profit_target_pct, min_gamma, min_short_leg_delta, min_return_1sigma, test_mode)
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

        elif parsed.path == "/scatter":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            page = (SCATTER_PAGE
                    .replace("__PNL_CHART__", PNL_CHART_JS)
                    .replace("__SPREAD_TOOLTIP__", SPREAD_TOOLTIP_JS))
            self.wfile.write(page.encode("utf-8"))

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
                try:
                    target_pct = float(params.get("target", ["15"])[0])
                except ValueError:
                    target_pct = 15.0
                target_pct = max(0.0, target_pct)
                index_symbol = (params.get("index", [DEFAULT_INDEX])[0] or DEFAULT_INDEX).strip() or DEFAULT_INDEX
                try:
                    n_sigma = float(params.get("sigmas", ["1"])[0])
                except ValueError:
                    n_sigma = 1.0
                n_sigma = max(0.1, min(n_sigma, 10.0))
                test_mode = params.get("test", ["0"])[0].lower() in ("1", "true", "test", "on")
                positions = load_positions()
                quotes = fetch_position_quotes(positions, haircut_pct=haircut_pct,
                                               profit_target_pct=target_pct,
                                               index_symbol=index_symbol, n_sigma=n_sigma,
                                               test_mode=test_mode)
                if not test_mode:
                    check_pnl_alerts(quotes)
                self._send_json({
                    "positions": quotes,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "haircutPct": round(haircut_pct * 100, 2),
                    "profitTargetPct": target_pct,
                    "indexSymbol": index_symbol,
                    "nSigma": n_sigma,
                })
            except Exception as e:
                self._send_json({"error": str(e)})

        elif parsed.path == "/api/templates":
            self._send_json(load_templates())

        elif parsed.path == "/api/clear_cache":
            n = clear_test_cache()
            self._send_json({"cleared": n})

        elif parsed.path == "/api/source":
            self._send_json(source_status())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/source":
            try:
                body = self._read_json_body()
                name = (body.get("name") or "").strip()
                if name not in SOURCES:
                    self._send_json({"error": f"Unknown data source: {name!r}",
                                     **source_status()})
                    return
                set_active_source(name)
                cfg = _load_sources_config()
                cfg["active"] = name
                _save_json_dict(SOURCES_CONFIG_FILE, cfg)
                self._send_json(source_status())
            except Exception as e:
                self._send_json({"error": str(e)})
        elif parsed.path == "/api/positions":
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


def start_server(preferred_port, max_tries=20):
    """Bind the server, auto-selecting the next free port if the preferred
    one is already in use (e.g. a previous instance is still running)."""
    for offset in range(max_tries):
        port = preferred_port + offset
        try:
            httpd = socketserver.TCPServer(("", port), SpreadHandler)
        except OSError as exc:
            # WinError 10048 / errno 98/48: address already in use -> try next port
            if getattr(exc, "winerror", None) == 10048 or exc.errno in (errno.EADDRINUSE, errno.EACCES):
                if offset == 0:
                    print(f"Port {preferred_port} is busy (another instance may be running). "
                          f"Looking for a free port...")
                continue
            raise
        return httpd, port
    raise SystemExit(
        f"Could not find a free port in range {preferred_port}-{preferred_port + max_tries - 1}. "
        f"Close the other server window and try again."
    )


def _print_banner(port):
    """Print the startup banner, tolerating non-UTF-8 stdout (e.g. a cp1252
    file when output is redirected) by falling back to an ASCII box."""
    banner = f"""
    ╔══════════════════════════════════════════════╗
    ║       Call Spread Finder                      ║
    ║                                              ║
    ║   Open: http://localhost:{port}                ║
    ║   Press Ctrl+C to stop                       ║
    ╚══════════════════════════════════════════════╝
    """
    try:
        print(banner)
    except UnicodeEncodeError:
        print(f"\n    Call Spread Finder\n"
              f"    Open: http://localhost:{port}\n"
              f"    Press Ctrl+C to stop\n")


def main():
    # Force UTF-8 on stdout/stderr so Unicode output (the banner box chars, greek
    # symbols, etc.) survives even when output is redirected to a cp1252 file
    # rather than a UTF-8 terminal. Best-effort: older streams may lack reconfigure.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    # Build the price-source registry from sources.json and activate the saved
    # source (falls back to Yahoo). Must precede the chain-cache load so cached
    # keys and the active source agree from the first request.
    active_src = init_sources()

    # Rehydrate the Test-mode chain cache from disk so a cache captured while
    # quotes were live (e.g. Friday's session) survives restarts. Cleared only
    # via the "Clear cache" button.
    n_cached = _load_chain_cache()
    if n_cached:
        print(f"  Chain cache: {n_cached} chain(s) loaded from {CHAIN_CACHE_FILE.name}")

    httpd, port = start_server(PORT)

    _print_banner(port)
    print(f"  Data source: {active_src.label}")
    topic = get_alert_topic()
    print(f"  Adj P&L alerts -> ntfy topic: {topic}")
    print(f"    (subscribe in the ntfy phone app, or watch https://ntfy.sh/{topic})")

    with httpd:
        httpd.allow_reuse_address = True
        # Open browser after a short delay
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            httpd.shutdown()


if __name__ == "__main__":
    main()
