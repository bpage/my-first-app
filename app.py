from flask import Flask, jsonify, request, send_file
from datetime import datetime, timedelta
import math
import os
import logging
import traceback
import requests
import yfinance as yf
import concurrent.futures
from cachetools import TTLCache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

@app.errorhandler(429)
def ratelimit_error(e):
    retry_after = getattr(e, "retry_after", None)
    if retry_after:
        seconds = int(retry_after.total_seconds()) if hasattr(retry_after, "total_seconds") else int(retry_after)
        wait = f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60}s"
        message = f"Too many requests — try again in {wait}."
    else:
        message = "Too many requests — slow down and try again in a moment."
    return jsonify({"error": "rate_limited", "message": message}), 429

# TTL cache for options chain — prevents hammering Alpaca on every keystroke
_options_cache = TTLCache(maxsize=50, ttl=120)

# TTL cache for IV Rank — computed from 1-year price history, stable for an hour
_ivr_cache = TTLCache(maxsize=200, ttl=3600)

# ── MOMO INDEX blueprints ──
from momo_api import momo_bp
from reddit_scanner import reddit_bp
from x_scanner import x_bp
app.register_blueprint(momo_bp)
app.register_blueprint(reddit_bp)
app.register_blueprint(x_bp)


@app.route("/momo")
def momo_dashboard():
    return send_file("momo-index-v3.html")

# Alpaca API credentials (set via environment variables)
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_DATA_URL = "https://data.alpaca.markets"


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/covered-calls")
def covered_calls():
    return send_file("covered-calls.html")


@app.route("/cash-secured-puts")
def cash_secured_puts():
    return send_file("cash-secured-puts.html")


@app.route("/iv-hunter")
def iv_hunter():
    return send_file("iv-hunter.html")


@app.route("/wheel-tracker")
def wheel_tracker():
    return send_file("wheel-tracker.html")


@app.route("/wheel-strategy")
def wheel_strategy():
    return send_file("wheel-strategy.html")


@app.route("/api/status")
def status():
    alpaca_ok = False
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        try:
            resp = requests.get(
                f"{ALPACA_DATA_URL}/v2/stocks/AAPL/snapshot?feed=iex",
                headers={"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
                timeout=5,
            )
            alpaca_ok = resp.status_code == 200
        except Exception:
            alpaca_ok = False
    return jsonify({"status": "ok", "alpaca": alpaca_ok})


# ── Alpaca Data Fetching ──

def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }


def _bs_delta(S, K, T, r, sigma, opt_type="put"):
    """Black-Scholes delta. T in years, sigma as decimal (e.g. 0.50 = 50% IV).
    Uses math.erf for the normal CDF — no scipy needed."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if opt_type == "put":
            return -1.0 if S <= K else 0.0
        return 1.0 if S >= K else 0.0
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        return nd1 if opt_type == "call" else nd1 - 1.0
    except Exception:
        return None


def _alpaca_stock_price(symbol):
    """Get latest stock price from Alpaca."""
    url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/snapshot?feed=iex"
    resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    # Use latest trade price, fallback to daily bar close
    price = None
    if data.get("latestTrade"):
        price = data["latestTrade"].get("p")
    if not price and data.get("dailyBar"):
        price = data["dailyBar"].get("c")
    return float(price) if price else None


def _alpaca_options_chain(symbol, opt_type, exp_gte, exp_lte):
    """Fetch all options snapshots from Alpaca with pagination.

    Returns (snapshots_dict, truncated_bool). truncated is True if we hit
    the safety cap and stopped before the chain was exhausted.
    """
    all_snapshots = {}
    page_token = None
    MAX_PAGES = 100  # safety cap: 100 × 100 = 10,000 contracts per side
    truncated = False

    for page_num in range(MAX_PAGES):
        params = {
            "feed": "indicative",
            "type": opt_type,
            "expiration_date_gte": exp_gte,
            "expiration_date_lte": exp_lte,
            "limit": 100,
        }
        if page_token:
            params["page_token"] = page_token

        url = f"{ALPACA_DATA_URL}/v1beta1/options/snapshots/{symbol}"
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        snapshots = data.get("snapshots", {})
        all_snapshots.update(snapshots)

        page_token = data.get("next_page_token")
        if not page_token:
            break
        if page_num == MAX_PAGES - 1:
            # Hit safety cap while next_page_token still present
            truncated = True
            logger.warning(f"{symbol} {opt_type}: hit {MAX_PAGES}-page safety cap ({len(all_snapshots)} contracts fetched); chain may be truncated")

    return all_snapshots, truncated


def _parse_occ_symbol(occ_symbol):
    """Parse OCC option symbol like NVDA260320C00205000.
    Returns (underlying, expiration_date_str, option_type, strike)."""
    # Find where the date starts (6 digits after the underlying)
    # Format: SYMBOL + YYMMDD + C/P + 00000000 (strike * 1000)
    # Find the type character (C or P) by scanning from the end
    for i in range(len(occ_symbol) - 9, 0, -1):
        if occ_symbol[i] in ('C', 'P'):
            underlying = occ_symbol[:i - 6]
            date_str = occ_symbol[i - 6:i]
            opt_type = occ_symbol[i]
            strike = int(occ_symbol[i + 1:]) / 1000
            exp_date = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
            return underlying, exp_date, opt_type, strike
    return None, None, None, None


def _build_alpaca_response(symbol, stock_price, call_snapshots, put_snapshots):
    """Convert Alpaca snapshots into our standard response format."""
    # Group by expiration date
    expirations = {}

    for occ_sym, snap in call_snapshots.items():
        _, exp_date, _, strike = _parse_occ_symbol(occ_sym)
        if not exp_date:
            continue
        if exp_date not in expirations:
            expirations[exp_date] = {"calls": [], "puts": []}
        quote = snap.get("latestQuote", {})
        greeks = snap.get("greeks", {})
        bar = snap.get("dailyBar", {})
        expirations[exp_date]["calls"].append({
            "strike": strike,
            "expiration": int(datetime.strptime(exp_date, "%Y-%m-%d").timestamp()),
            "bid": quote.get("bp", 0) or 0,
            "ask": quote.get("ap", 0) or 0,
            "openInterest": None,
            "volume": bar.get("v", 0) or 0,
            "impliedVolatility": snap.get("impliedVolatility", 0) or 0,
            "delta": greeks.get("delta", 0) or 0,
            "gamma": greeks.get("gamma", 0) or 0,
            "theta": greeks.get("theta", 0) or 0,
        })

    for occ_sym, snap in put_snapshots.items():
        _, exp_date, _, strike = _parse_occ_symbol(occ_sym)
        if not exp_date:
            continue
        if exp_date not in expirations:
            expirations[exp_date] = {"calls": [], "puts": []}
        quote = snap.get("latestQuote", {})
        greeks = snap.get("greeks", {})
        bar = snap.get("dailyBar", {})
        expirations[exp_date]["puts"].append({
            "strike": strike,
            "expiration": int(datetime.strptime(exp_date, "%Y-%m-%d").timestamp()),
            "bid": quote.get("bp", 0) or 0,
            "ask": quote.get("ap", 0) or 0,
            "openInterest": None,
            "volume": bar.get("v", 0) or 0,
            "impliedVolatility": snap.get("impliedVolatility", 0) or 0,
            "delta": greeks.get("delta", 0) or 0,
            "gamma": greeks.get("gamma", 0) or 0,
            "theta": greeks.get("theta", 0) or 0,
        })

    # Build options array sorted by expiration
    all_options = []
    exp_timestamps = []
    for exp_date in sorted(expirations.keys()):
        exp_ts = int(datetime.strptime(exp_date, "%Y-%m-%d").timestamp())
        exp_timestamps.append(exp_ts)
        all_options.append({
            "expirationDate": exp_ts,
            "calls": sorted(expirations[exp_date]["calls"], key=lambda x: x["strike"]),
            "puts": sorted(expirations[exp_date]["puts"], key=lambda x: x["strike"]),
        })

    total_calls = sum(len(o["calls"]) for o in all_options)
    total_puts = sum(len(o["puts"]) for o in all_options)
    logger.info(f"{symbol}: Alpaca returning {len(all_options)} expirations, {total_calls} calls, {total_puts} puts")

    return {
        "source": "alpaca",
        "optionChain": {
            "result": [{
                "quote": {"regularMarketPrice": stock_price},
                "expirationDates": exp_timestamps,
                "options": all_options,
            }]
        }
    }



def _compute_ivr(symbol, current_iv_pct=None):
    """Compute IV Rank proxy using 1-year realized volatility as a historical IV range.

    IVR = (current_IV - 52wk_low_RV) / (52wk_high_RV - 52wk_low_RV) * 100

    current_iv_pct: current implied vol as percentage (e.g., 85.0). If None,
    uses the most recent 20-day realized vol as the current value.
    """
    cache_key = (symbol, current_iv_pct)
    if cache_key in _ivr_cache:
        return _ivr_cache[cache_key]
    try:
        hist = yf.Ticker(symbol).history(period="1y")
        if len(hist) < 30:
            _ivr_cache[cache_key] = None
            return None
        returns = hist["Close"].pct_change().dropna()
        rolling_vol = returns.rolling(20).std() * (252 ** 0.5) * 100
        rolling_vol = rolling_vol.dropna()
        if len(rolling_vol) < 20:
            _ivr_cache[cache_key] = None
            return None
        vol_min = float(rolling_vol.min())
        vol_max = float(rolling_vol.max())
        current = current_iv_pct if current_iv_pct is not None else float(rolling_vol.iloc[-1])
        if vol_max <= vol_min:
            ivr = 50.0
        else:
            ivr = (current - vol_min) / (vol_max - vol_min) * 100
            ivr = round(max(0.0, min(100.0, ivr)), 1)
        _ivr_cache[cache_key] = ivr
        return ivr
    except Exception as e:
        logger.warning(f"IVR compute failed for {symbol}: {e}")
        _ivr_cache[cache_key] = None
        return None


# Curated high-IV watchlist — stocks known for elevated implied volatility
# Covers: crypto-adjacent, meme stocks, biotech, energy/solar, high-growth tech, EV, fintech
HIGH_IV_WATCHLIST = [
    # Crypto / Bitcoin proxy
    "MSTR", "COIN", "RIOT", "MARA", "HUT", "BITF", "CLSK",
    # Meme / retail favorites
    "GME", "AMC", "HOOD", "SOFI", "LCID", "RIVN",
    # High-growth tech / volatile
    "SMCI", "PLTR", "SNOW", "CRWD", "NET", "DKNG", "ROKU", "SNAP", "SQ", "SHOP",
    "ARM", "IONQ", "RGTI", "QUBT",
    # Energy / solar / hydrogen
    "BE", "PLUG", "FSLR", "ENPH", "RUN", "SEDG",
    # Biotech / pharma
    "MRNA", "BNTX", "NVAX",
    # EV
    "TSLA", "NIO", "XPEV", "LI",
    # Large cap tech (high options volume)
    "NVDA", "AMD", "AAPL", "AMZN", "META", "GOOGL", "MSFT", "NFLX",
    # ETFs
    "SPY", "QQQ", "IWM", "ARKK", "XBI",
]


@app.route("/api/active-tickers")
@limiter.limit("6 per minute")
def api_active_tickers():
    """Get most active stock tickers from Alpaca screener, merged with high-IV watchlist."""
    if not ALPACA_API_KEY:
        return jsonify({"error": "Alpaca API not configured"}), 503

    try:
        url = f"{ALPACA_DATA_URL}/v1beta1/screener/stocks/most-actives"
        resp = requests.get(url, headers=_alpaca_headers(), params={"by": "trades", "top": 100}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        active_tickers = []
        for item in data.get("most_actives", []):
            sym = item.get("symbol", "")
            if not sym or "." in sym or len(sym) > 5:
                continue
            active_tickers.append(sym)

        # Merge: active tickers + high-IV watchlist (deduplicated, preserving order)
        seen = set()
        merged = []
        for sym in active_tickers + HIGH_IV_WATCHLIST:
            if sym not in seen:
                seen.add(sym)
                merged.append(sym)

        # Get snapshot prices to filter out penny stocks (< $15)
        # Process in batches of 50
        filtered = []
        for i in range(0, len(merged), 50):
            batch = merged[i:i+50]
            batch_url = f"{ALPACA_DATA_URL}/v2/stocks/snapshots"
            batch_resp = requests.get(batch_url, headers=_alpaca_headers(),
                                     params={"symbols": ",".join(batch), "feed": "iex"}, timeout=10)
            batch_resp.raise_for_status()
            prices = batch_resp.json()

            for sym in batch:
                snap = prices.get(sym)
                if not snap:
                    continue
                price = None
                if snap.get("latestTrade"):
                    price = snap["latestTrade"].get("p")
                if not price and snap.get("dailyBar"):
                    price = snap["dailyBar"].get("c")
                if price and price >= 15:
                    filtered.append(sym)

        logger.info(f"Active tickers: {len(filtered)} found (active + watchlist)")
        return jsonify({"tickers": filtered})

    except Exception as e:
        logger.error(f"Active tickers failed: {e}")
        return jsonify({"error": str(e)}), 502


@app.route("/api/iv-watchlist")
def api_iv_watchlist():
    """Return the curated high-IV watchlist tickers."""
    return jsonify({"tickers": HIGH_IV_WATCHLIST})


@app.route("/api/stock-price/<ticker>")
def api_stock_price(ticker):
    """Quick price lookup for a single ticker via Alpaca."""
    ticker = ticker.upper()
    if not ALPACA_API_KEY:
        return jsonify({"error": "Alpaca not configured"}), 503
    try:
        price = _alpaca_stock_price(ticker)
        if not price:
            return jsonify({"error": f"No price for {ticker}"}), 404
        return jsonify({"ticker": ticker, "price": round(price, 2)})
    except Exception as e:
        logger.error(f"stock-price failed for {ticker}: {e}")
        return jsonify({"error": str(e)}), 502


@app.route("/api/wheel-quote")
def api_wheel_quote():
    """Fetch live price + option greeks for a specific wheel position.

    Query params:
      ticker     — stock symbol (e.g. NVDA)
      type       — 'put' or 'call'
      strike     — option strike price (float)
      expiration — YYYY-MM-DD

    Returns price, dte, otm_pct, delta (from Alpaca greeks or B-S estimate), iv.
    """
    ticker     = request.args.get("ticker", "").upper()
    opt_type   = request.args.get("type", "put").lower()
    expiration = request.args.get("expiration", "")
    try:
        strike = float(request.args.get("strike", 0))
    except (ValueError, TypeError):
        strike = 0.0

    if not ticker:
        return jsonify({"error": "ticker required"}), 400
    if not ALPACA_API_KEY:
        return jsonify({"error": "Alpaca not configured"}), 503

    try:
        price = _alpaca_stock_price(ticker)
        if not price:
            return jsonify({"error": f"No price for {ticker}"}), 404

        # DTE
        dte = None
        if expiration:
            try:
                exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                dte = max(0, (exp_date - datetime.now().date()).days)
            except ValueError:
                pass

        result = {
            "ticker": ticker,
            "price": round(price, 2),
            "strike": strike,
            "expiration": expiration,
            "dte": dte,
            "otm_pct": None,
            "delta": None,
            "delta_estimated": False,
            "iv": None,
            "opt_bid": None,
            "opt_ask": None,
            "opt_mid": None,
        }

        if strike > 0:
            # OTM % — positive means out-of-the-money (good for premium sellers)
            if opt_type == "put":
                result["otm_pct"] = round((price - strike) / strike * 100, 2)
            else:
                result["otm_pct"] = round((strike - price) / strike * 100, 2)

        # Try Alpaca options snapshot for live greeks on the exact contract
        if strike > 0 and expiration:
            try:
                params = {
                    "feed": "indicative",
                    "type": opt_type,
                    "expiration_date_gte": expiration,
                    "expiration_date_lte": expiration,
                    "strike_price_gte": str(max(0, strike - 2.5)),
                    "strike_price_lte": str(strike + 2.5),
                    "limit": 20,
                }
                url = f"{ALPACA_DATA_URL}/v1beta1/options/snapshots/{ticker}"
                resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=8)
                if resp.status_code == 200:
                    snapshots = resp.json().get("snapshots", {})
                    best_snap = None
                    best_dist = float("inf")
                    for occ_sym, snap in snapshots.items():
                        _, _, _, snap_strike = _parse_occ_symbol(occ_sym)
                        if snap_strike and abs(snap_strike - strike) < best_dist:
                            best_dist = abs(snap_strike - strike)
                            best_snap = snap
                    if best_snap:
                        greeks = best_snap.get("greeks") or {}
                        iv_raw = best_snap.get("impliedVolatility")
                        quote = best_snap.get("latestQuote") or {}
                        bid = float(quote.get("bp", 0) or 0)
                        ask = float(quote.get("ap", 0) or 0)
                        if greeks.get("delta") is not None:
                            result["delta"] = round(float(greeks["delta"]), 4)
                        if iv_raw:
                            result["iv"] = round(float(iv_raw) * 100, 1)
                        result["opt_bid"] = round(bid, 2)
                        result["opt_ask"] = round(ask, 2)
                        result["opt_mid"] = round((bid + ask) / 2, 2)
            except Exception as e:
                logger.warning(f"wheel-quote greeks fetch failed for {ticker}: {e}")

        # Black-Scholes delta fallback when Alpaca didn't return greeks
        if result["delta"] is None and strike > 0 and dte is not None and dte > 0:
            iv_decimal = (result["iv"] / 100.0) if result["iv"] else 0.50
            bs = _bs_delta(price, strike, dte / 365.0, 0.045, iv_decimal, opt_type)
            if bs is not None:
                result["delta"] = round(bs, 4)
                result["delta_estimated"] = True

        logger.info(f"wheel-quote {ticker} {opt_type} ${strike} exp={expiration}: "
                    f"price=${price:.2f} dte={dte} otm={result['otm_pct']} delta={result['delta']}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"wheel-quote failed for {ticker}: {e}")
        return jsonify({"error": str(e)}), 502


def _iv_scan_one(symbol, exp_gte, exp_lte, now):
    """Fetch IV data for a single ticker. Returns a result dict or None."""
    try:
        price = _alpaca_stock_price(symbol)
        if not price:
            return None

        strike_lo = price * 0.90
        strike_hi = price * 1.02

        params = {
            "feed": "indicative",
            "type": "put",
            "expiration_date_gte": exp_gte,
            "expiration_date_lte": exp_lte,
            "strike_price_gte": f"{strike_lo:.0f}",
            "strike_price_lte": f"{strike_hi:.0f}",
            "limit": 200,  # SPY/QQQ can have 100+ strikes in this range across multiple weeklies
        }
        url = f"{ALPACA_DATA_URL}/v1beta1/options/snapshots/{symbol}"
        resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=10)
        resp.raise_for_status()
        snapshots = resp.json().get("snapshots", {})

        if not snapshots:
            return None

        # Group by expiration date, filter out zero/missing IV
        exp_groups: dict = {}
        for occ_sym, snap in snapshots.items():
            _, exp_date, _, strike = _parse_occ_symbol(occ_sym)
            iv = snap.get("impliedVolatility") or 0
            if not strike or iv <= 0:
                continue
            exp_groups.setdefault(exp_date, []).append((strike, snap))

        if not exp_groups:
            return None

        # Pick the expiration closest to 30 DTE, then find the ATM strike within it
        TARGET_DTE = 30
        best_exp = min(
            exp_groups,
            key=lambda d: abs((datetime.strptime(d, "%Y-%m-%d") - now).days - TARGET_DTE)
        )

        best = None
        best_distance = float("inf")
        for strike, snap in exp_groups[best_exp]:
            distance = abs(strike - price)
            if distance < best_distance:
                best_distance = distance
                greeks = snap.get("greeks", {})
                quote = snap.get("latestQuote", {})
                bid = quote.get("bp", 0) or 0
                ask = quote.get("ap", 0) or 0
                best = {
                    "strike": strike,
                    "expiration": best_exp,
                    "iv": round(snap["impliedVolatility"], 4),
                    "delta": round(greeks.get("delta", 0) or 0, 4),
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 2),
                }

        if not best:
            return None

        premium_pct = round((best["mid"] / price) * 100, 2)
        exp_dt = datetime.strptime(best["expiration"], "%Y-%m-%d")
        dte = (exp_dt - now).days
        ann_yield = round((best["mid"] / best["strike"]) * (365 / max(dte, 1)) * 100, 1) if dte > 0 else 0

        current_iv_pct = round(best["iv"] * 100, 1)
        ivr = _compute_ivr(symbol, current_iv_pct)

        logger.info(f"IV scan {symbol}: price=${price:.2f}, IV={current_iv_pct}%, IVR={ivr}, mid=${best['mid']}")
        return {
            "symbol": symbol,
            "price": round(price, 2),
            "iv30": current_iv_pct,
            "ivRank": ivr,
            "atmStrike": best["strike"],
            "expiration": best["expiration"],
            "dte": dte,
            "delta": best["delta"],
            "bid": best["bid"],
            "ask": best["ask"],
            "mid": best["mid"],
            "premiumPct": premium_pct,
            "annualizedYield": ann_yield,
        }
    except Exception as e:
        logger.warning(f"IV scan failed for {symbol}: {e}")
        return None


@app.route("/api/iv-scan")
@limiter.limit("10 per minute")
def api_iv_scan():
    """Scan multiple tickers for 30-day ATM implied volatility."""
    tickers_param = request.args.get("tickers", "")
    if not tickers_param:
        return jsonify({"error": "tickers parameter required"}), 400

    tickers = [t.strip().upper() for t in tickers_param.split(",") if t.strip()]
    if len(tickers) > 50:
        tickers = tickers[:50]

    if not ALPACA_API_KEY:
        return jsonify({"error": "Alpaca API not configured"}), 503

    now = datetime.now()
    exp_gte = (now + timedelta(days=20)).strftime("%Y-%m-%d")
    exp_lte = (now + timedelta(days=45)).strftime("%Y-%m-%d")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_iv_scan_one, sym, exp_gte, exp_lte, now): sym for sym in tickers}
        raw = [f.result() for f in concurrent.futures.as_completed(futures)]

    results = [r for r in raw if r is not None]
    return jsonify({"results": results})


@app.route("/api/yahoo/options/<symbol>")
@limiter.limit("30 per minute")
def api_yahoo_options(symbol):
    """Fetch stock price + options data. Uses Alpaca (primary) or Yahoo (fallback)."""
    symbol = symbol.upper()

    if symbol in _options_cache:
        return jsonify(_options_cache[symbol])

    # Try Alpaca first if credentials are available
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        try:
            resp = _fetch_alpaca(symbol)
            _options_cache[symbol] = resp.get_json()
            return resp
        except Exception as e:
            logger.warning(f"Alpaca failed for {symbol}, falling back to Yahoo: {e}")

    # Fallback to Yahoo
    resp = _fetch_yahoo(symbol)
    # Cache only successful responses (resp may be a (response, status) tuple on error)
    if not isinstance(resp, tuple) and resp.status_code == 200:
        _options_cache[symbol] = resp.get_json()
    return resp


def _fetch_alpaca(symbol):
    """Fetch from Alpaca API."""
    stock_price = _alpaca_stock_price(symbol)
    if not stock_price:
        raise ValueError(f"No Alpaca price for {symbol}")

    logger.info(f"{symbol} price (Alpaca): ${stock_price:.2f}")

    now = datetime.now()
    exp_gte = (now + timedelta(days=25)).strftime("%Y-%m-%d")
    exp_lte = (now + timedelta(days=90)).strftime("%Y-%m-%d")

    # Fetch calls and puts in parallel — cuts options chain fetch time roughly in half
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_calls = ex.submit(_alpaca_options_chain, symbol, "call", exp_gte, exp_lte)
        f_puts  = ex.submit(_alpaca_options_chain, symbol, "put",  exp_gte, exp_lte)
        call_snapshots, calls_truncated = f_calls.result()
        put_snapshots,  puts_truncated  = f_puts.result()

    chain_truncated = calls_truncated or puts_truncated
    logger.info(f"{symbol}: Alpaca returned {len(call_snapshots)} call snapshots, {len(put_snapshots)} put snapshots (truncated={chain_truncated})")

    # Extract ATM put IV for IVR — same methodology as IV Hunter scan for consistency
    atm_iv_pct = None
    best_dist = float("inf")
    for occ_sym, snap in put_snapshots.items():
        _, _, _, strike = _parse_occ_symbol(occ_sym)
        iv = snap.get("impliedVolatility")
        if not strike or not iv or iv <= 0:
            continue
        dist = abs(strike - stock_price)
        if dist < best_dist:
            best_dist = dist
            atm_iv_pct = round(iv * 100, 1)

    response_data = _build_alpaca_response(symbol, stock_price, call_snapshots, put_snapshots)
    response_data["ivRank"] = _compute_ivr(symbol, atm_iv_pct)
    if chain_truncated:
        response_data["warning"] = "Options chain may be incomplete — too many contracts to fetch in a single request. Results shown are partial."
    return jsonify(response_data)


def _fetch_yahoo(symbol):
    """Fallback: Fetch from Yahoo Finance via yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        stock_price = _get_yf_stock_price(ticker, symbol)
        if not stock_price:
            return jsonify({"error": f"No price data for {symbol}"}), 404

        logger.info(f"{symbol} price (Yahoo): ${stock_price:.2f}")

        exp_dates, valid_exps = _get_valid_expirations(ticker, symbol)
        if not valid_exps:
            return _empty_response(stock_price, "yfinance")

        all_options = []
        for exp_str in valid_exps[:8]:
            try:
                chain = ticker.option_chain(exp_str)
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
                exp_ts = int(exp_dt.timestamp())

                calls_list = _build_yf_option_list(chain.calls, exp_ts)
                puts_list = _build_yf_option_list(chain.puts, exp_ts)

                all_options.append({
                    "expirationDate": exp_ts,
                    "calls": calls_list,
                    "puts": puts_list,
                })
            except Exception as e:
                logger.warning(f"Failed fetching {symbol} options for {exp_str}: {e}")

        exp_timestamps = [int(datetime.strptime(e, "%Y-%m-%d").timestamp()) for e in valid_exps]

        return jsonify({
            "source": "yfinance",
            "ivRank": _compute_ivr(symbol),
            "optionChain": {
                "result": [{
                    "quote": {"regularMarketPrice": stock_price},
                    "expirationDates": exp_timestamps,
                    "options": all_options,
                }]
            }
        })

    except Exception as e:
        logger.error(f"Yahoo failed for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 502


# ── Yahoo/yfinance helpers ──

def _get_yf_stock_price(ticker, symbol):
    info = ticker.fast_info
    stock_price = getattr(info, "last_price", None)
    if not stock_price:
        stock_price = getattr(info, "previous_close", None)
    if not stock_price:
        hist = ticker.history(period="1d")
        if not hist.empty:
            stock_price = float(hist["Close"].iloc[-1])
    return float(stock_price) if stock_price else None


def _get_valid_expirations(ticker, symbol):
    try:
        exp_dates = ticker.options
    except Exception as e:
        logger.warning(f"No options available for {symbol}: {e}")
        return [], []

    if not exp_dates:
        return [], []

    now = datetime.now()
    valid_exps = []
    for exp_str in exp_dates:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - now).days
        if 25 <= dte <= 90:
            valid_exps.append(exp_str)

    return exp_dates, valid_exps


def _empty_response(stock_price, source="yfinance"):
    return jsonify({
        "source": source,
        "optionChain": {
            "result": [{
                "quote": {"regularMarketPrice": stock_price or 0},
                "expirationDates": [],
                "options": [],
            }]
        }
    })


def _build_yf_option_list(df, exp_ts):
    options_list = []
    for _, row in df.iterrows():
        opt = {
            "strike": float(row.get("strike", 0)),
            "expiration": exp_ts,
            "bid": float(row.get("bid", 0)),
            "ask": float(row.get("ask", 0)),
            "openInterest": int(row.get("openInterest", 0)) if not _is_nan(row.get("openInterest")) else 0,
            "volume": int(row.get("volume", 0)) if not _is_nan(row.get("volume")) else 0,
            "impliedVolatility": float(row.get("impliedVolatility", 0)),
        }
        options_list.append(opt)
    return options_list


def _is_nan(val):
    try:
        return val != val
    except Exception:
        return False


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
