"""
Pluggable market-data sources for the Call Spread Finder.

`PriceSource` defines exactly what the app consumes from a market-data vendor;
`YahooSource` (free, options ~15 min delayed) is the always-available default.
Integrating a real-time vendor = subclass `PriceSource`, implement the two
Tier-1 methods, and add one entry to `SOURCE_CLASSES` — nothing else changes.

Contract normalization: every method takes Yahoo-style symbols (^SPX, ^VIX,
^IRX — the app's canonical namespace; subclasses translate their vendor's
tickers via `map_symbol`) and returns data shaped like yfinance output (Yahoo
column/key names), so downstream code never changes per vendor.

The active source is a module-global selected by `set_active_source`; the main
file owns the config file (sources.json) and passes its dict to
`register_sources`. This module must not import from the main file.
"""

import abc
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Auto-install yfinance if missing (Yahoo is always available as a source)
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

import pandas as pd  # ships with yfinance


def _snap_chain(raw):
    """Freeze a yfinance option_chain result into a plain, picklable snapshot.

    Keeps exactly the attributes the app reads (.calls, .puts, .underlying) and
    stamps the fetch time, decoupled from yfinance's own result class so the
    disk cache survives yfinance upgrades. This IS the Yahoo->contract
    normalizer: every PriceSource returns snapshots of this shape.
    """
    return SimpleNamespace(
        calls=raw.calls,
        puts=getattr(raw, "puts", None),
        underlying=getattr(raw, "underlying", None),
        fetched_at=datetime.now().timestamp())


class PriceSource(abc.ABC):
    """What the app needs from a market-data vendor.

    Tier 1 (abstract — the realtime-sensitive surface a new vendor MUST
    implement): `get_expirations`, `get_option_chain`.

    Tier 2 (concrete defaults delegate to the always-available Yahoo source):
    `get_daily_closes`, `get_daily_closes_batch`, `get_previous_close`.
    These feed reference data — the ^IRX rate, VIX-family levels, prev-close
    fallbacks, and the 2y beta regression — where daily-close granularity makes
    Yahoo's delay irrelevant, and which many options vendors don't serve at
    all. A subclass may override any of them.

    Failures are surfaced, not hidden: Tier-1 methods and the daily-close
    methods may raise on transport/auth errors (callers already guard);
    there is deliberately no automatic fallback to Yahoo, so a "realtime"
    view can never silently contain delayed data.
    """

    name = "abstract"        # short id: config key, cache-key prefix, API value
    label = "Abstract"       # UI display label, e.g. "Yahoo (delayed ~15m)"
    realtime = False
    feed_options = ()        # runtime-switchable feed ids (empty = not switchable)

    def __init__(self, config=None):
        # Per-source dict from sources.json (API keys, account ids, ...).
        self.config = config or {}

    def map_symbol(self, symbol):
        """Translate a Yahoo-style symbol to this vendor's ticker.

        The app speaks Yahoo symbols everywhere (^SPX, ^VIX); vendors that
        name things differently (Tradier: SPX, Polygon: I:SPX) override this
        so Yahoo-isms never leak into their APIs. Default: identity.
        """
        return symbol

    def get_feed(self):
        """Current feed id, for sources whose feed is switchable at runtime
        (i.e. `feed_options` is non-empty); None otherwise."""
        return None

    def set_feed(self, feed):
        """Switch the active feed at runtime. Base sources have no feed."""
        raise NotImplementedError("this source has no switchable feed")

    # ---- Tier 1: options surface (must implement) ----

    @abc.abstractmethod
    def get_expirations(self, symbol):
        """Listed option expirations as a tuple of 'YYYY-MM-DD' strings,
        ascending. Empty tuple if the symbol has no options."""

    @abc.abstractmethod
    def get_option_chain(self, symbol, exp):
        """One expiration's chain as a plain, picklable snapshot:
        SimpleNamespace(calls, puts, underlying, fetched_at).

        .calls — DataFrame. REQUIRED columns (Yahoo names/semantics): strike,
          bid, ask, lastPrice, impliedVolatility (annualized fraction), volume,
          openInterest. OPTIONAL (the app degrades gracefully): change
          (lastPrice − prior official close; feeds per-leg prevClose),
          lastTradeDate (tz-aware Timestamp or NaT; feeds staleness flags).
        .puts — DataFrame or None (never read by the app; cache fidelity only).
        .underlying — dict with Yahoo quote keys. Supply at minimum
          regularMarketPrice + regularMarketTime (epoch secs) so spot is
          time-aligned with the option quotes; optional: postMarketPrice/Time,
          preMarketPrice/Time, regularMarketPreviousClose, previousClose,
          bid, ask. May be {} (the app falls back to a daily close).
        .fetched_at — epoch seconds."""

    # ---- Tier 2: reference data (Yahoo delegates by default) ----

    def get_daily_closes(self, symbol, period="5d", interval="1d"):
        """Close series (pd.Series, ascending DatetimeIndex) for a symbol.
        Covers the ^IRX rate, VIX-family levels, Finder spot, and prev-close /
        spot fallbacks."""
        return _YAHOO_SOURCE.get_daily_closes(symbol, period, interval)

    def get_daily_closes_batch(self, symbols, period, interval="1d"):
        """Multi-symbol daily Close DataFrame (one column per symbol — always
        a DataFrame, even for a single symbol). Feeds the beta regression."""
        return _YAHOO_SOURCE.get_daily_closes_batch(symbols, period, interval)

    def get_previous_close(self, symbol):
        """Official prior-day close as float, or None on any failure (never
        raises — it's a convenience quote with history fallbacks in callers)."""
        return _YAHOO_SOURCE.get_previous_close(symbol)

    def get_option_prev_close(self, symbol, exp, strike):
        """Prior-session closing mark of one CALL contract, or None.

        Preferably the closing bid/ask MIDPOINT (so the daily MTM is a true
        mid-to-mid change, matching how today's side is marked); a trade-based
        close is an acceptable fallback. Optional: feeds the monitor's daily
        mark-to-market P&L when a chain doesn't carry a per-contract `change`
        column (Yahoo's does, so its legs never need this). Callers memoize
        per day — implementations may fetch. Default: unavailable."""
        return None


class YahooSource(PriceSource):
    """Yahoo Finance via yfinance. Free; option quotes ~15 min delayed."""

    name = "yahoo"
    label = "Yahoo (delayed ~15m)"
    realtime = False

    _UQ_TTL = 5.0   # seconds to memoize a live underlying quote (hybrid spot)

    def __init__(self, config=None):
        super().__init__(config)
        self._uq_cache = {}   # symbol -> (epoch_fetched, quote_dict)

    def get_expirations(self, symbol):
        return tuple(yf.Ticker(self.map_symbol(symbol)).options)

    def get_option_chain(self, symbol, exp):
        return _snap_chain(yf.Ticker(self.map_symbol(symbol)).option_chain(exp))

    def get_daily_closes(self, symbol, period="5d", interval="1d"):
        return yf.Ticker(self.map_symbol(symbol)).history(
            period=period, interval=interval)["Close"]

    def get_daily_closes_batch(self, symbols, period, interval="1d"):
        symbols = [self.map_symbol(s) for s in symbols]
        data = yf.download(symbols, period=period, interval=interval,
                           auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):  # single ticker: promote to DataFrame
            data = data.to_frame(symbols[0])
        return data

    def get_previous_close(self, symbol):
        try:
            return float(yf.Ticker(self.map_symbol(symbol)).fast_info.previous_close)
        except Exception:
            return None

    def get_underlying_quote(self, symbol):
        """Live-ish underlying quote as a Yahoo-keyed dict: regularMarketPrice +
        regularMarketTime (from the last 1-minute intraday bar, so it carries a
        real timestamp) and regularMarketPreviousClose (from fast_info). Empty
        dict on failure. Used for HYBRID SPOT — when another vendor serves the
        option chain but only a delayed equity quote, that vendor can override
        the underlying with this live Yahoo price. Memoized per symbol for a few
        seconds so one scan/refresh touching many expirations of the same symbol
        makes a single Yahoo call (the repo minimizes Yahoo hits)."""
        now = datetime.now().timestamp()
        hit = self._uq_cache.get(symbol)
        if hit and now - hit[0] < self._UQ_TTL:
            return hit[1]
        sym = self.map_symbol(symbol)
        out = {}
        try:
            h = yf.Ticker(sym).history(period="1d", interval="1m")
            if len(h):
                out["regularMarketPrice"] = float(h["Close"].iloc[-1])
                out["regularMarketTime"] = int(h.index[-1].timestamp())
        except Exception:
            pass
        try:
            pc = yf.Ticker(sym).fast_info.previous_close
            if pc:
                out["regularMarketPreviousClose"] = float(pc)
        except Exception:
            pass
        self._uq_cache[symbol] = (now, out)
        return out


class MarketDataSource(PriceSource):
    """marketdata.app — near-real-time US stock/index option data.

    Config (sources.json): {"token": "<api token>", "feed": "cached"}.
    The feed matters because of the credit model (Trader plan = 100k/day):
      cached — 1 credit per CALL regardless of chain size; data seconds to a
               few minutes old. The default: effectively unlimited here.
      live   — 1 credit per CONTRACT returned; never fetch broad chains on it.
    We only request calls (side=call — the app never reads puts), which also
    halves any live-feed cost. Free/trial accounts always get delayed data
    and reject the feed parameter — that error is SURFACED, never silently
    downgraded to feedless (= live, per-contract billing); feedless requires
    an explicit "feed": "" in sources.json.

    Tier-2 reference data (^IRX rate, VIX level, beta history) stays on the
    Yahoo delegation — marketdata doesn't need to serve those.
    """

    name = "marketdata"
    label = "MarketData.app"
    realtime = True
    feed_options = ("cached", "live")   # switchable live (see set_feed)
    BASE = "https://api.marketdata.app/v1"

    def __init__(self, config=None):
        super().__init__(config)
        self.token = (self.config.get("token") or "").strip()
        if not self.token:
            raise ValueError("no token in sources.json")
        self.feed = (self.config.get("feed", "cached") or "").strip()
        # HYBRID SPOT: the Trader plan serves real-time OPTIONS but only a
        # ~15-min delayed equity quote, so the underlying spot lags the option
        # marks. Setting "spot_source": "yahoo" in sources.json overrides the
        # underlying with Yahoo's live-ish equity price (free, ~real-time),
        # while option quotes stay on marketdata. "" / absent = native quote.
        self.spot_source = (self.config.get("spot_source") or "").strip().lower()

    def map_symbol(self, symbol):
        # Yahoo's ^/$ index prefixes don't exist here (^SPX -> SPX, ^VIX -> VIX).
        return symbol.lstrip("^$")

    def get_feed(self):
        return self.feed

    def set_feed(self, feed):
        """Switch the options feed live (no restart): the next chain request
        reads `self.feed` (see `_chain_request`). 'cached' = ~15-min delayed,
        1 credit/call; 'live' = real-time, 1 credit PER CONTRACT; '' = feedless
        live. The caller persists the choice to sources.json."""
        self.feed = (feed or "").strip()
        self.config["feed"] = self.feed
        return self.feed

    def _get_json(self, path, params=None):
        query = urllib.parse.urlencode(params or {})
        url = f"{self.BASE}{path}{'?' + query if query else ''}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"})
        try:
            # 200 = real-time/consolidated, 203 = delayed (trial/free) — both fine.
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read().decode("utf-8")).get("errmsg", "")
            except Exception:
                detail = ""
            raise RuntimeError(
                f"marketdata {path}: HTTP {e.code} {detail}".strip()) from None

    def get_expirations(self, symbol):
        data = self._get_json(f"/options/expirations/{self.map_symbol(symbol)}/")
        if data.get("s") != "ok":
            return ()
        return tuple(data.get("expirations") or ())

    def _chain_request(self, sym, params):
        # BILLING SAFETY: a chain request WITHOUT a feed parameter defaults to
        # the live feed and bills 1 credit PER CONTRACT (a monitor refresh
        # cycle can burn thousands). So the configured feed is always sent,
        # and a rejection (trial/free plans answer feed control with a bare
        # HTTP 402) is surfaced as an error — never silently retried at
        # per-contract prices. Feedless live is an explicit opt-in:
        # set "feed": "" in sources.json.
        if not self.feed:
            return self._get_json(f"/options/chain/{sym}/", params)
        try:
            return self._get_json(f"/options/chain/{sym}/",
                                  {**params, "feed": self.feed})
        except RuntimeError as e:
            msg = str(e).lower()
            if "402" in msg or "feed" in msg:
                raise RuntimeError(
                    f"marketdata: plan rejected feed={self.feed!r} — real-time "
                    f"entitlement missing/pending. Not retrying without the "
                    f"feed (that bills per contract); use Yahoo meanwhile or "
                    f"set feed \"\" explicitly.") from None
            raise

    def get_option_chain(self, symbol, exp):
        data = self._chain_request(self.map_symbol(symbol),
                                   {"expiration": exp, "side": "call"})
        if data.get("s") != "ok":
            raise RuntimeError(
                f"marketdata: no chain data for {symbol} {exp} (s={data.get('s')})")
        n = len(data.get("strike") or [])

        def col(key):
            vals = data.get(key)
            return vals if isinstance(vals, list) and len(vals) == n else [None] * n

        calls = pd.DataFrame({
            "strike": col("strike"),
            "bid": col("bid"), "ask": col("ask"),
            "lastPrice": col("last"),
            "impliedVolatility": col("iv"),
            "volume": col("volume"),
            "openInterest": col("openInterest"),
        })
        # Null quotes -> 0.0, mirroring Yahoo's zeroed-out semantics (the app
        # treats bid/ask 0 as "no live two-sided market" and IV 0 as missing).
        for c in calls.columns:
            calls[c] = pd.to_numeric(calls[c], errors="coerce").fillna(0.0)
        # 'updated' is the per-contract quote time — not a trade time, but the
        # honest freshness signal the staleness flags exist to convey.
        calls["lastTradeDate"] = pd.to_datetime(col("updated"), unit="s", utc=True)

        underlying = {}
        upx, upd = data.get("underlyingPrice"), data.get("updated")
        if isinstance(upx, list) and upx and upx[0]:
            underlying["regularMarketPrice"] = float(upx[0])
        if isinstance(upd, list):
            ts = [t for t in upd if t]
            if ts:
                underlying["regularMarketTime"] = int(max(ts))
        try:
            if self.spot_source == "yahoo":
                # Hybrid spot: live Yahoo equity price overrides the delayed
                # marketdata quote (and saves the per-symbol quote credit).
                yq = _YAHOO_SOURCE.get_underlying_quote(symbol)
                if yq.get("regularMarketPrice"):
                    underlying.update(yq)
                else:                       # Yahoo empty -> fall back to native
                    underlying.update(self._underlying_quote(symbol))
            else:
                # One quotes call (1 credit): fresher price/time and the only way
                # to get the prior close (last − change) for Daily Theo P&L.
                underlying.update(self._underlying_quote(symbol))
        except Exception:
            pass  # chain-derived spot still works; candle fallback covers the rest
        return SimpleNamespace(calls=calls, puts=None, underlying=underlying,
                               fetched_at=datetime.now().timestamp())

    @staticmethod
    def _prior_business_day(day):
        day -= timedelta(days=1)
        while day.weekday() >= 5:   # Sat/Sun
            day -= timedelta(days=1)
        return day

    def get_option_prev_close(self, symbol, exp, strike):
        """Prior-session closing MID for one call contract — true mid-to-mid
        daily MTM, matching how today's side is marked.

        Historical EOD quotes exist even for contracts that didn't trade, so
        step back to the prior business day (one extra step for a holiday)
        and read the closing bid/ask midpoint. Falls back to the trade-based
        daily-candle close when no historical quote is available (which may
        predate yesterday on untraded legs). Billed as historical data —
        ~1 credit per call — and memoized per day by the caller."""
        occ = (f"{self.map_symbol(symbol)}"
               f"{datetime.strptime(exp, '%Y-%m-%d'):%y%m%d}"
               f"C{int(round(float(strike) * 1000)):08d}")
        day = datetime.now().date()
        for _ in range(2):
            day = self._prior_business_day(day)
            try:
                q = self._get_json(f"/options/quotes/{occ}/",
                                   {"date": day.strftime("%Y-%m-%d")})
            except RuntimeError:
                break                     # endpoint/plan issue -> candle fallback
            if q.get("s") != "ok":
                continue                  # holiday / no data -> step back once more
            bid = (q.get("bid") or [None])[-1]
            ask = (q.get("ask") or [None])[-1]
            if bid and ask:
                return round((float(bid) + float(ask)) / 2, 4)
            mid = (q.get("mid") or [None])[-1]
            if mid:
                return float(mid)
            break                         # quote row exists but is one-sided/empty
        # Fallback: trade-based daily candle close.
        data = self._get_json(f"/options/candles/D/{occ}/", {"countback": 2})
        if data.get("s") != "ok":
            return None
        today = datetime.now().date()
        for c, t in zip(reversed(data.get("c") or []), reversed(data.get("t") or [])):
            if c and t and datetime.fromtimestamp(t).date() < today:
                return float(c)
        return None

    def _underlying_quote(self, symbol):
        kind = "indices" if symbol.startswith(("^", "$")) else "stocks"
        data = self._get_json(f"/{kind}/quotes/{self.map_symbol(symbol)}/")
        if data.get("s") != "ok":
            return {}

        def first(key):
            vals = data.get(key)
            v = vals[0] if isinstance(vals, list) and vals else None
            return None if v is None else float(v)

        out = {}
        last, change = first("last"), first("change")
        if last:
            out["regularMarketPrice"] = last
        ts = first("updated")
        if ts:
            out["regularMarketTime"] = int(ts)
        bid, ask = first("bid"), first("ask")
        if bid:
            out["bid"] = bid
        if ask:
            out["ask"] = ask
        if last is not None and change is not None:
            out["regularMarketPreviousClose"] = round(last - change, 2)
        return out


# ---------------------------------------------------------------------------
# Registry + active source. Yahoo exists unconditionally (it is both the
# default source and the Tier-2 delegation target); other sources appear only
# if register_sources() can construct them from config.
# ---------------------------------------------------------------------------

_YAHOO_SOURCE = YahooSource()

SOURCE_CLASSES = {"yahoo": YahooSource,   # future vendor: add ONE entry here
                  "marketdata": MarketDataSource}
SOURCES = {"yahoo": _YAHOO_SOURCE}        # name -> instantiated, usable source
_ACTIVE_SOURCE = _YAHOO_SOURCE


def get_source():
    """The active PriceSource. Request handlers should call this once per
    request and use the captured instance throughout, so a concurrent toggle
    can never mix sources within one scan."""
    return _ACTIVE_SOURCE


def set_active_source(name):
    """Switch the active source (memory only — the caller persists config).
    Raises KeyError for a source that isn't registered/usable."""
    global _ACTIVE_SOURCE
    _ACTIVE_SOURCE = SOURCES[name]
    return _ACTIVE_SOURCE


def register_sources(config):
    """(Re)build SOURCES from a sources.json-shaped dict:
    {"active": ..., "sources": {name: {per-source config}, ...}}.

    Each registered class is constructed with its config sub-dict; a source
    that has no config entry at all is skipped silently (not offered), while
    one that is configured but fails to construct (bad API key) is skipped
    with a warning rather than killing startup. Yahoo is always
    (re)constructed and always present. Returns the usable source names."""
    cfg = (config or {}).get("sources", {})
    SOURCES.clear()
    for name, cls in SOURCE_CLASSES.items():
        conf = cfg.get(name)
        if conf is None and name != "yahoo":
            continue  # vendor not configured — don't construct, don't warn
        try:
            SOURCES[name] = cls(conf or {})
        except Exception as e:
            print(f"  Warning: price source '{name}' unavailable ({e})")
    if "yahoo" not in SOURCES:               # never let Yahoo drop out
        SOURCES["yahoo"] = _YAHOO_SOURCE
    return tuple(SOURCES)
