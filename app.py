"""
Flask server for the SPX Put Hedge Ranker web UI.

Proxies CBOE delayed options data server-side (bypassing browser CORS),
runs the scoring engine from hedge.py, and returns ranked JSON to the
browser.  Results are cached for 60 seconds so parameter tuning is fast.

Usage:
    python app.py
    Open http://localhost:5000
"""

import math
import os
import threading
import time
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from cme import KNOWN_SYMBOLS, fetch_chain, parse_chain
from hedge import Params, filter_puts, score_puts
from scenarios import DEFAULT_MODEL, SCENARIO_MODELS, get_model

app = Flask(__name__, static_folder="static")

# Default EP view buckets — must match DEFAULT_BUCKETS in static/app.js
_DEFAULT_EP_BUCKETS = [(3475, 0.10), (4170, 0.25), (4860, 0.50), (5560, 0.70), (6250, 0.90)]
_DEFAULT_EP_CONFIDENCE = 0.75


@app.after_request
def no_cache_static(response):
    """Prevent browsers from caching static assets so JS/CSS changes take effect immediately."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# ---------------------------------------------------------------------------
# In-memory CBOE cache — 60-second TTL, keyed by symbol
# ---------------------------------------------------------------------------
_cache: dict = {}   # symbol -> {"df": ..., "underlying": ..., "fetched_at": ...}
_cache_lock = threading.Lock()
CACHE_TTL_SEC = 60

# ---------------------------------------------------------------------------
# GARCH singleton state
# ---------------------------------------------------------------------------
_garch_lock = threading.Lock()
_garch_state: dict = {
    "cache":    None,   # GARCHPathCache — built once
    "model":    None,   # GJREntropyModel — updated on each commit
    "meta":     None,   # dict of last-committed model params
    "loading":  False,
    "error":    None,
    "auto_reinit_done": False,  # True once boot-price correction has fired
}
_GARCH_DEFAULT_PRICE = 6500.0  # fallback if CBOE not yet fetched


def _garch_init_worker(price: float, annual_drift_pct: float | None = None) -> None:
    """Background thread: fit GARCH and simulate paths."""
    from spx_model import GARCHPathCache
    n_paths    = int(os.environ.get("GARCH_N_PATHS",    100_000))
    chunk_size = int(os.environ.get("GARCH_CHUNK_SIZE",  25_000))
    with _garch_lock:
        _garch_state["loading"] = True
        _garch_state["error"] = None
        _garch_state["cache"] = None
        _garch_state["model"] = None
    try:
        cache = GARCHPathCache(price, annual_drift_pct=annual_drift_pct,
                               n_paths=n_paths, chunk_size=chunk_size)
        # Auto-create a default EP model matching the UI's default parameters
        # so the grid shows the same results as clicking Apply without changes.
        default_model = cache.make_model(
            view_buckets=_DEFAULT_EP_BUCKETS,
            confidence_level=_DEFAULT_EP_CONFIDENCE,
        )
        default_meta = {
            "configured_at":  datetime.now().strftime("%H:%M:%S"),
            "spx_at_config":  cache.current_price,
            "confidence":     _DEFAULT_EP_CONFIDENCE,
            "buckets":        _DEFAULT_EP_BUCKETS,
            "n_buckets":      len(_DEFAULT_EP_BUCKETS),
        }
        with _garch_lock:
            _garch_state["cache"] = cache
            _garch_state["model"] = default_model
            _garch_state["meta"]  = default_meta
            _garch_state["loading"] = False
        label = (f"{annual_drift_pct:.1f}%/yr override"
                 if annual_drift_pct is not None
                 else f"{cache.fitted_annual_drift_pct:.1f}%/yr fitted")
        print(f"GARCH cache ready: SPX={price:,.0f}, drift={label}")
    except Exception as exc:
        with _garch_lock:
            _garch_state["loading"] = False
            _garch_state["error"] = str(exc)
        print(f"GARCH init failed: {exc}")


def _fetch_cboe_spot(symbol: str = "SPX") -> float | None:
    """
    Fetch the 15-min delayed spot price from CBOE's lightweight quote endpoint.
    Much faster than loading the full options chain.
    Returns None on any failure.
    """
    import requests as _req
    cdn_sym = KNOWN_SYMBOLS.get(symbol)
    if not cdn_sym:
        return None
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/quotes/{cdn_sym}.json"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.cboe.com/",
    }
    try:
        resp = _req.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        price = resp.json().get("data", {}).get("current_price")
        if price and float(price) > 1000:   # sanity: SPX is always > 1000
            return float(price)
    except Exception as exc:
        print(f"CBOE spot fetch failed: {exc}")
    return None


def start_garch_init(price: float | None = None, annual_drift_pct: float | None = None) -> None:
    """Kick off (or restart) GARCH simulation in a daemon thread."""
    p = price or _GARCH_DEFAULT_PRICE
    t = threading.Thread(target=_garch_init_worker, args=(p, annual_drift_pct), daemon=True)
    t.start()


def _start_garch_with_live_price(annual_drift_pct: float | None = None) -> None:
    """Fetch live SPX spot then kick off GARCH — runs in a daemon thread."""
    spot = _fetch_cboe_spot("SPX")
    if spot:
        print(f"Live SPX spot for GARCH init: {spot:,.2f}")
    else:
        spot = _GARCH_DEFAULT_PRICE
        print(f"Could not fetch live SPX spot; using default {spot:,.0f}")
    start_garch_init(price=spot, annual_drift_pct=annual_drift_pct)


# ---------------------------------------------------------------------------
# Heartbeat — auto-shutdown when browser closes
# ---------------------------------------------------------------------------
_last_heartbeat: float = time.time()
_HEARTBEAT_TIMEOUT = 90   # seconds without a ping before shutdown


def _heartbeat_watcher() -> None:
    """Daemon thread: shut down the process if no browser ping for 90 s."""
    while True:
        time.sleep(10)
        if time.time() - _last_heartbeat > _HEARTBEAT_TIMEOUT:
            print("No browser heartbeat for 90 s — shutting down.")
            os._exit(0)


if not os.environ.get("PORT"):   # local desktop mode only
    threading.Thread(target=_heartbeat_watcher, daemon=True).start()

# Start background simulation immediately when the server loads.
# Fetch live SPX spot first so paths start from the right price.
# Default drift: 7%/yr (conservative long-run estimate; fitted ~13% reflects
# survivorship-biased 30yr bull run and is overly optimistic for forward planning).
threading.Thread(target=_start_garch_with_live_price,
                 kwargs={"annual_drift_pct": 7.0}, daemon=True).start()

# Columns returned to the browser (order = left→right in grid)
RESPONSE_COLS = [
    "expiry", "session", "dte", "strike", "moneyness_pct",
    "mid", "spread_pct", "iv", "delta", "theta_daily",
    "open_interest", "volume",
    "cost_1c", "cost_Nc",
    "payoff_crash_25pct_1c", "payoff_crash_40pct_1c", "payoff_crash_55pct_1c",
    "e_payoff_roth_1c", "e_net_1c", "EPR", "crash_efficiency",
    "annual_cost_pct", "theo_vs_mid_pct",
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/options")
def api_options():
    # --- Parse & validate query params ---
    def _float(key, default, lo, hi):
        try:
            return max(lo, min(hi, float(request.args.get(key, default))))
        except (ValueError, TypeError):
            return default

    def _int(key, default, lo, hi):
        try:
            return max(lo, min(hi, int(request.args.get(key, default))))
        except (ValueError, TypeError):
            return default

    p_crash         = _float("p_crash",         0.55,  0.01, 0.99)
    annual_budget   = _float("budget",       75_000.0,  1000, 10_000_000)
    horizon_months  = _int  ("horizon",            18,     1, 60)
    roth_multiplier = _float("roth_multiplier",  1.25,   1.0,  3.0)
    contracts       = _int  ("contracts",           0,     0, 100)
    index_beta      = _float("index_beta",         1.0,   0.1,  5.0)
    force_refresh   = request.args.get("force_refresh", "false").lower() == "true"
    model_name      = request.args.get("model", "garch_ep")
    if model_name not in SCENARIO_MODELS:
        model_name = DEFAULT_MODEL
    symbol = request.args.get("symbol", "SPX").upper()
    if symbol not in KNOWN_SYMBOLS:
        symbol = "SPX"

    p = Params(
        p_crash=p_crash,
        annual_budget=annual_budget,
        horizon_months=horizon_months,
        roth_multiplier=roth_multiplier,
        contracts=contracts,
        symbol=symbol,
        index_beta=index_beta,
    )

    # --- Fetch CBOE data (cached per symbol) ---
    with _cache_lock:
        now   = time.time()
        entry = _cache.get(symbol, {})
        stale = (
            entry.get("fetched_at") is None
            or (now - entry["fetched_at"]) > CACHE_TTL_SEC
        )

        if stale or force_refresh:
            try:
                raw = fetch_chain(KNOWN_SYMBOLS[symbol])
                df_full, underlying = parse_chain(raw, base_symbol=symbol)
                _cache[symbol] = {"df": df_full, "underlying": underlying, "fetched_at": now}
            except Exception as exc:
                if entry.get("df") is None:
                    return jsonify({"error": "CBOE fetch failed", "detail": str(exc)}), 502
                # Fall through with stale data on transient errors

        entry      = _cache.get(symbol, {})
        df_full    = entry.get("df")
        underlying = entry.get("underlying")
        fetched_at = entry.get("fetched_at")

    # --- Auto-reinit GARCH if it used the fallback price ---
    # Fires at most once per server run: if CBOE was unavailable at boot, GARCH
    # started with _GARCH_DEFAULT_PRICE.  The first successful SPX fetch corrects
    # it.  The flag prevents this from re-firing every request when SPX happens
    # to be near the fallback value (which would clear any committed EP model).
    spx_spot = underlying["current_price"]
    if symbol == "SPX" and spx_spot:
        with _garch_lock:
            gc   = _garch_state["cache"]
            gl   = _garch_state["loading"]
            done = _garch_state["auto_reinit_done"]
        if not done and not gl and (gc is None or abs(gc.current_price - _GARCH_DEFAULT_PRICE) < 50):
            with _garch_lock:
                _garch_state["auto_reinit_done"] = True
            start_garch_init(price=spx_spot, annual_drift_pct=7.0)

    # --- Filter & score ---
    n_contracts = p.n_contracts(spx_spot)

    try:
        filtered, rejected = filter_puts(df_full, p, spx_spot)
    except Exception as exc:
        return jsonify({"error": "Filter failed", "detail": str(exc)}), 500

    if filtered.empty:
        return jsonify({
            "rows": [],
            "meta": _build_meta(underlying, spx_spot, n_contracts, p, 0, len(rejected), fetched_at),
        })

    # For garch_ep, use committed singleton; fall back to survival if not ready
    actual_model_key = model_name  # the clean dropdown key actually used
    if model_name == "garch_ep":
        with _garch_lock:
            active = _garch_state["model"]
        if active is None:
            scoring_model = get_model("survival")
            actual_model_key = "survival"
            model_name = "survival (garch_ep not configured)"
        else:
            scoring_model = active
    else:
        scoring_model = get_model(model_name)

    try:
        scored = score_puts(filtered, p, spx_spot, model=scoring_model)
    except Exception as exc:
        return jsonify({"error": "Scoring failed", "detail": str(exc)}), 500

    scored = scored.sort_values("EPR", ascending=False).reset_index(drop=True)

    # --- Serialize ---
    cols = [c for c in RESPONSE_COLS if c in scored.columns]
    out = scored[cols].copy()
    out["expiry"] = out["expiry"].astype(str)

    return jsonify({
        "rows": _to_records(out),
        "meta": _build_meta(underlying, spx_spot, n_contracts, p, len(scored), len(rejected), fetched_at, model_name, actual_model_key),
    })


@app.route("/api/model/status")
def api_model_status():
    with _garch_lock:
        s = _garch_state.copy()
    cache = s["cache"]
    meta  = s["meta"]
    stale = False
    if meta and cache:
        with _cache_lock:
            spot = (_cache.get("SPX", {}).get("underlying") or {}).get("current_price")
        if spot and abs(spot - cache.current_price) / cache.current_price > 0.02:
            stale = True
    return jsonify({
        "paths_ready":          cache is not None,
        "loading":              s["loading"],
        "error":                s["error"],
        "n_paths":              int(os.environ.get("GARCH_N_PATHS", 100_000)),
        "spx_at_init":          cache.current_price if cache else None,
        "fitted_annual_drift":  cache.fitted_annual_drift_pct if cache else None,
        "active_drift":         (cache.annual_drift_pct if cache.annual_drift_pct is not None
                                 else cache.fitted_annual_drift_pct) if cache else None,
        "active_model":         {**meta, "stale": stale} if meta else None,
    })


@app.route("/api/epay_breakdown", methods=["POST"])
def api_epay_breakdown():
    """E(Pay) breakdown for up to 5 options — shows bucketed GARCH terminal distribution."""
    body    = request.get_json(force=True) or {}
    options = body.get("options", [])[:5]

    with _garch_lock:
        model = _garch_state["model"]

    if model is None:
        return jsonify({"error": "GARCH model not ready — please wait for paths to finish loading."}), 503

    spot    = model._current_price
    results = []
    for opt in options:
        try:
            strike = float(opt["strike"])
            dte    = int(opt["dte"])
            expiry = str(opt.get("expiry", ""))
            label  = f"{expiry}  ·  {int(strike):,}P  ·  DTE {dte}"
            rows   = model.epay_breakdown(strike, dte)
            results.append({"label": label, "strike": strike, "dte": dte, "rows": rows})
        except Exception as exc:
            results.append({"label": str(opt.get("strike", "?")), "error": str(exc)})

    return jsonify({"options": results, "spot": spot})


@app.route("/api/model/reinit", methods=["POST"])
def api_model_reinit():
    """Re-run GARCH simulation with optional drift override. Uses current CBOE price."""
    body = request.get_json(force=True) or {}
    drift = body.get("annual_drift_pct")
    annual_drift_pct = float(drift) if drift is not None else None
    with _cache_lock:
        spot = (_cache.get("SPX", {}).get("underlying") or {}).get("current_price")
    start_garch_init(spot, annual_drift_pct=annual_drift_pct)
    return jsonify({"ok": True, "started_with_price": spot or _GARCH_DEFAULT_PRICE,
                    "annual_drift_pct": annual_drift_pct})


def _run_ep(buckets: list, confidence: float) -> "GJREntropyModel | None":
    """Build a GJREntropyModel from current cache + user buckets. Returns None if cache not ready."""
    from spx_model import GJREntropyModel
    with _garch_lock:
        cache = _garch_state["cache"]
    if cache is None:
        return None
    view_buckets = [(float(lv), float(cp)) for lv, cp in buckets]
    return cache.make_model(view_buckets=view_buckets, confidence_level=float(confidence))


def _ensure_cboe_cached(symbol: str) -> pd.DataFrame | None:
    """
    Return the cached CBOE options DataFrame for symbol, fetching it if needed.
    Useful when B-L is called before the main options tab has loaded.
    """
    with _cache_lock:
        entry = _cache.get(symbol, {})
        df = entry.get("df")
    if df is not None and not df.empty:
        return df
    # Cache miss — fetch synchronously (will take ~1-2s)
    try:
        print(f"B-L: CBOE cache empty for {symbol}, fetching now…")
        raw = fetch_chain(KNOWN_SYMBOLS[symbol])
        df_full, underlying = parse_chain(raw, base_symbol=symbol)
        with _cache_lock:
            _cache[symbol] = {"df": df_full, "underlying": underlying, "fetched_at": time.time()}
        return df_full
    except Exception as exc:
        print(f"B-L: CBOE fetch failed for {symbol}: {exc}")
        return None


def _bl_target_cal_dte(query_dte_trading: int) -> int:
    """Convert trading-day DTE to approximate calendar-day DTE for CBOE options lookup."""
    return int(round(query_dte_trading * 365.0 / 252.0))


def _compute_bl_market_running_min(
    symbol: str,
    cache,
    query_dte: int,
    drawdown_levels: list,
    r: float = 0.045,
    n_bins: int = 200,
) -> list:
    """
    Reweight GARCH paths by the B-L risk-neutral terminal density, then compute
    P(running min over 2yr ≤ level) under the reweighted measure.

    This makes the Market column directly comparable to GARCH Prior and Your View,
    which also show running-minimum probabilities.

    query_dte is in trading days (same scale as the model paths).
    Returns list of float, one per drawdown level. Returns [] on any error.
    """
    from spx_model import _compute_path_weights
    df = _ensure_cboe_cached(symbol)
    if df is None or df.empty:
        return []
    try:
        from datetime import date as _date
        from scipy.interpolate import interp1d
        from scipy.ndimage import gaussian_filter1d

        today = _date.today()
        cal_dte = _bl_target_cal_dte(query_dte)

        # --- Get GARCH terminal values at query_dte (trading days) ---
        paths = cache.paths           # (n_paths, n_weeks)
        week_idx = min((query_dte - 1) // 5, paths.shape[1] - 1)
        S_T = paths[:, week_idx].astype(float)   # (n_paths,)
        prior_pw = _compute_path_weights(
            cache.prior_weights, cache.bin_assignments, cache.n_paths
        )

        # --- Get B-L density at nearest expiry to cal_dte ---
        tmp = df[df["ask"] > 0].copy()
        if tmp.empty:
            return []
        tmp["dte_cal"] = tmp["expiry"].apply(lambda e: (e - today).days)
        available = tmp["dte_cal"].unique()
        best_cal = int(available[np.argmin(np.abs(available - cal_dte))])

        grp = tmp[tmp["dte_cal"] == best_cal]
        current_price = float(cache.current_price)

        # Quality filter
        oi_ok    = grp["open_interest"].isna() | (grp["open_interest"] > 0)
        ask_safe = grp["ask"].clip(lower=0.01)
        spread_ok = (grp["bid"] == 0) | ((grp["ask"] - grp["bid"]) / ask_safe <= 0.60)
        grp = grp[oi_ok & spread_ok]

        puts  = grp[(grp["type"] == "PUT")  & (grp["strike"] < current_price)].sort_values("strike")
        calls = grp[(grp["type"] == "CALL") & (grp["strike"] > current_price)].sort_values("strike")

        T    = best_cal / 365.0
        disc = np.exp(r * T)

        def _mids(df_):
            b = df_["bid"].values.astype(float)
            a = df_["ask"].values.astype(float)
            return np.where(b > 0, (b + a) / 2.0, a)

        ks_put, ds_put   = np.array([]), np.array([])
        ks_call, ds_call = np.array([]), np.array([])
        if len(puts) >= 3:
            ks_put, ds_put = _bl_fd_density(
                puts["strike"].values.astype(float), _mids(puts), disc)
        if len(calls) >= 3:
            ks_call, ds_call = _bl_fd_density(
                calls["strike"].values.astype(float), _mids(calls), disc)

        if len(ks_put) + len(ks_call) < 3:
            print(f"BL market running min: too few density points (puts={len(ks_put)}, calls={len(ks_call)})")
            return []

        all_ks = np.concatenate([ks_put, ks_call])
        all_ds = np.concatenate([ds_put, ds_call])
        sidx   = np.argsort(all_ks)
        all_ks = all_ks[sidx]
        all_ds = all_ds[sidx]
        all_ds = gaussian_filter1d(all_ds, sigma=2.0)
        all_ds = np.maximum(all_ds, 0.0)

        # Interpolate B-L density at each path's terminal value
        bl_interp = interp1d(all_ks, all_ds, kind='linear', bounds_error=False, fill_value=0.0)
        f_bl = np.maximum(bl_interp(S_T), 0.0)   # (n_paths,)

        # Estimate GARCH prior density from binned terminal distribution
        lo = float(np.percentile(S_T, 0.2))
        hi = float(np.percentile(S_T, 99.8))
        edges = np.linspace(lo, hi, n_bins + 1)
        bin_width = (hi - lo) / n_bins
        bin_assign = np.clip(np.searchsorted(edges[1:-1], S_T, side="left"), 0, n_bins - 1)
        bin_mass   = np.bincount(bin_assign, weights=prior_pw, minlength=n_bins)
        f_garch_bin = bin_mass / bin_width   # density per unit of SPX

        # Likelihood ratio per path (use bin-level GARCH density for stability)
        f_garch_path = f_garch_bin[bin_assign]
        lr = np.zeros(len(S_T))
        valid = f_garch_path > 0
        lr[valid] = f_bl[valid] / f_garch_path[valid]

        # Reweighted measure: w[i] ∝ prior_pw[i] * LR[i]
        new_weights = prior_pw * lr
        total = new_weights.sum()
        if total <= 0:
            print("BL market running min: zero total weight after reweighting")
            return []
        new_weights /= total

        # P(running min over 2yr ≤ level) under reweighted measure
        running_min = cache.running_min
        results = []
        for K in drawdown_levels:
            mask = running_min < K
            prob = float(np.clip(np.sum(new_weights[mask]), 0.0, 1.0))
            results.append(round(prob, 4))
        print(f"BL market running min: best_cal={best_cal}, results={results}")
        return results

    except Exception as exc:
        print(f"BL market running min error: {exc}")
        import traceback; traceback.print_exc()
        return []


def _bl_fd_density(strikes: np.ndarray, mids: np.ndarray, disc: float) -> tuple:
    """
    Breeden-Litzenberger density via direct finite differences on listed strikes.
    Returns (k_centers, densities) at the interior strike points.
    No interpolation — works directly on discrete price data.
    """
    n = len(strikes)
    k_out, d_out = [], []
    for i in range(1, n - 1):
        km, k0, kp = strikes[i-1], strikes[i], strikes[i+1]
        pm, p0, pp = mids[i-1],   mids[i],   mids[i+1]
        dkm = k0 - km   # left spacing
        dkp = kp - k0   # right spacing
        if dkm <= 0 or dkp <= 0:
            continue
        # Non-uniform second derivative
        d2 = 2.0 * (pp / dkp - p0 * (1.0/dkm + 1.0/dkp) + pm / dkm) / (dkm + dkp)
        density = d2 * disc
        if density > 0:
            k_out.append(k0)
            d_out.append(density)
    return np.array(k_out), np.array(d_out)


def _compute_bl_surface_slices(
    symbol: str,
    current_price: float,
    level_centers: list,
    r: float = 0.045,
) -> list:
    """
    Per-expiry risk-neutral PDF slices for the distribution surface chart.

    Uses direct finite differences on listed strikes (no interpolation),
    then linearly interpolates the density onto the GARCH level grid.

    Returns [{"dte": int, "density": list[float]}, ...] sorted by DTE.
    """
    df = _ensure_cboe_cached(symbol)
    if df is None or df.empty:
        print("BL surface slices: no CBOE data available")
        return []
    try:
        from datetime import date as _date, timedelta as _timedelta
        from scipy.interpolate import interp1d
        today = _date.today()
        tmp = df[df["ask"] > 0].copy()
        if tmp.empty:
            print("BL surface slices: no options with ask>0")
            return []
        tmp["dte"] = tmp["expiry"].apply(lambda e: (e - today).days)
        levels = np.array(level_centers, dtype=float)

        def _mids(df_):
            b = df_["bid"].values.astype(float)
            a = df_["ask"].values.astype(float)
            return np.where(b > 0, (b + a) / 2.0, a)

        slices = []
        expiry_groups = [(dte, grp) for dte, grp in tmp.groupby("dte")
                         if 30 <= dte <= 730]
        print(f"BL surface slices: {len(expiry_groups)} expiries in 30-730d range")

        for dte_val, grp in expiry_groups:
            T    = dte_val / 365.0   # dte_val is calendar days; T in years
            disc = np.exp(r * T)

            # Quality filter: drop zero-OI rows and excessively wide spreads.
            # Keep bid=0 rows (normal for deep OTM) — spread undefined, use ask as price.
            oi_ok     = grp["open_interest"].isna() | (grp["open_interest"] > 0)
            ask_safe  = grp["ask"].clip(lower=0.01)
            spread_ok = (grp["bid"] == 0) | ((grp["ask"] - grp["bid"]) / ask_safe <= 0.60)
            grp = grp[oi_ok & spread_ok]

            # OTM puts (below spot) + OTM calls (above spot), sorted by strike
            puts  = grp[(grp["type"] == "PUT")  & (grp["strike"] < current_price)].sort_values("strike")
            calls = grp[(grp["type"] == "CALL") & (grp["strike"] > current_price)].sort_values("strike")

            ks_put, ds_put   = (np.array([]), np.array([]))
            ks_call, ds_call = (np.array([]), np.array([]))

            if len(puts) >= 3:
                ks_put, ds_put = _bl_fd_density(
                    puts["strike"].values.astype(float), _mids(puts), disc)

            if len(calls) >= 3:
                ks_call, ds_call = _bl_fd_density(
                    calls["strike"].values.astype(float), _mids(calls), disc)

            # Require meaningful data from both sides.  Long-dated LEAPS can have
            # only 3-4 listed OTM call strikes → _bl_fd_density returns 1 interior
            # point.  Requiring ≥2 call points would drop all LEAPS; ≥1 is enough
            # since the median sanity-check below catches one-sided distributions.
            if len(ks_put) < 2 or len(ks_call) < 1:
                continue

            # Merge put and call density points
            all_ks = np.concatenate([ks_put, ks_call])
            all_ds = np.concatenate([ds_put, ds_call])
            sort_idx = np.argsort(all_ks)
            all_ks = all_ks[sort_idx]
            all_ds = all_ds[sort_idx]

            if len(all_ks) < 2:
                continue

            # Interpolate to the GARCH level grid (linear, zero outside support)
            interp_fn = interp1d(all_ks, all_ds, kind='linear',
                                 bounds_error=False, fill_value=0.0)
            pdf = np.maximum(interp_fn(levels), 0.0)

            # Gaussian smoothing to remove finite-difference noise
            from scipy.ndimage import gaussian_filter1d
            pdf = gaussian_filter1d(pdf, sigma=2.0)
            pdf = np.maximum(pdf, 0.0)

            total = pdf.sum()
            if total <= 0:
                continue
            pdf /= total
            expiry_label = (today + _timedelta(days=int(dte_val))).strftime('%b %Y')
            cum = np.cumsum(pdf)
            median_price = float(levels[np.searchsorted(cum, 0.5, side='left')])
            print(f"BL slice DTE={dte_val} ({expiry_label}): median={median_price:.0f} ({100*median_price/current_price:.0f}% of spot)")
            slices.append({"dte": int(dte_val), "expiry_label": expiry_label, "density": pdf.tolist()})

        slices.sort(key=lambda x: x["dte"])

        # Reduce to ~9 representative expiries at roughly 3-month intervals up to 24 months
        # (calendar days): avoids cluttered near-term weekly expiries
        if len(slices) > 9:
            targets_cal = [45, 91, 182, 273, 365, 456, 547, 638, 730]
            picked, used = [], set()
            for t in targets_cal:
                best = min(slices, key=lambda s: abs(s["dte"] - t))
                if best["dte"] not in used:
                    picked.append(best)
                    used.add(best["dte"])
            slices = sorted(picked, key=lambda x: x["dte"])

        print(f"BL surface slices: {len(slices)} expiries returned")
        return slices
    except Exception as exc:
        print(f"BL surface slices error: {exc}")
        return []


@app.route("/api/model/preview", methods=["POST"])
def api_model_preview():
    """Run EP with user inputs and return visualization data. Does NOT commit model."""
    body       = request.get_json(force=True) or {}
    buckets    = body.get("buckets", [])
    confidence = float(body.get("confidence", 1.0))
    query_dte  = int(body.get("query_dte", 252))

    with _garch_lock:
        cache = _garch_state["cache"]
    if cache is None:
        return jsonify({"error": "GARCH simulation not ready yet"}), 503

    try:
        prior_stats, _, _ = cache.prior_stats()
    except Exception as exc:
        return jsonify({"error": f"prior_stats failed: {exc}"}), 500

    if not buckets:
        # Pure GARCH — no EP
        try:
            model = cache.make_model()
        except Exception as exc:
            return jsonify({"error": f"make_model failed: {exc}"}), 500
    else:
        try:
            model = _run_ep(buckets, confidence)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    try:
        paths, groups, stats = model.quintile_paths()
        terminal             = model.terminal_distribution(query_dte)
    except Exception as exc:
        return jsonify({"error": f"path simulation failed: {exc}"}), 500

    # Market column: B-L-reweighted GARCH running-min probabilities
    # (same metric as GARCH Prior / Your View — directly comparable)
    try:
        market_priors = _compute_bl_market_running_min(
            "SPX", cache,
            query_dte,
            [s["level"] for s in prior_stats],
        )
    except Exception as exc:
        print(f"Market priors error: {exc}")
        market_priors = []

    # Distribution surface + B-L slices
    try:
        surface = model.distribution_surface()
        if surface:
            bl_slices = _compute_bl_surface_slices(
                "SPX", cache.current_price, surface["levels"]
            )
            surface["bl_slices"] = bl_slices
    except Exception as exc:
        print(f"Surface/BL error: {exc}")
        surface = {}

    return jsonify({
        "current_price":   cache.current_price,
        "prior_stats":     prior_stats,
        "quintile_paths":  paths,
        "quintile_groups": groups,
        "quintile_stats":  stats,
        "terminal":        terminal,
        "market_priors":   market_priors,
        "surface":         surface,
    })


@app.route("/api/model/commit", methods=["POST"])
def api_model_commit():
    """Run EP and store result as the active model for the options grid."""
    body       = request.get_json(force=True) or {}
    buckets    = body.get("buckets", [])
    confidence = float(body.get("confidence", 1.0))

    if not buckets:
        return jsonify({"error": "buckets required"}), 400

    try:
        model = _run_ep(buckets, confidence)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    if model is None:
        return jsonify({"error": "GARCH simulation not ready yet"}), 503

    meta = {
        "configured_at":  datetime.now().strftime("%H:%M:%S"),
        "spx_at_config":  model._cache.current_price,
        "confidence":     confidence,
        "buckets":        buckets,
        "n_buckets":      len(buckets),
    }
    with _garch_lock:
        _garch_state["model"] = model
        _garch_state["meta"]  = meta

    return jsonify({"ok": True, "meta": meta})


@app.route("/api/heartbeat", methods=["POST"])
def api_heartbeat():
    global _last_heartbeat
    _last_heartbeat = time.time()
    return jsonify({"ok": True})


@app.route("/api/health")
def api_health():
    with _cache_lock:
        spx_entry = _cache.get("SPX", {})
        fa  = spx_entry.get("fetched_at")
        age = round(time.time() - fa, 1) if fa else None
    return jsonify({"status": "ok", "cache_age_sec": age})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_meta(underlying, spx_spot, n_contracts, p, n_passed, n_filtered, fetched_at, model_name=DEFAULT_MODEL, model_key=None):
    with _garch_lock:
        garch_meta = _garch_state["meta"]
        garch_loading = _garch_state["loading"]
    return {
        "model": model_name,
        "model_key": model_key if model_key is not None else model_name,
        "garch_ep_meta": garch_meta,
        "garch_loading": garch_loading,
        "symbol":        p.symbol,
        "index_beta":    p.index_beta,
        "spx_spot":      _sf(spx_spot, 2),
        "n_contracts":   int(n_contracts),
        "total_budget":  _sf(p.total_budget, 0),
        "portfolio_beta": _sf(p.portfolio_beta, 3),
        "n_passed":      int(n_passed),
        "n_filtered":    int(n_filtered),
        "iv30":          _sf(underlying.get("iv30"), 4),
        "cboe_timestamp": underlying.get("timestamp"),
        "fetched_at":    datetime.fromtimestamp(fetched_at).isoformat() if fetched_at else None,
    }


def _sf(v, decimals=4):
    """Safe float conversion: None / NaN / inf → None, else round."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _to_records(df: pd.DataFrame) -> list:
    """Convert DataFrame to JSON-safe list of dicts."""
    records = []
    for row in df.itertuples(index=False):
        d = {}
        for col, val in zip(df.columns, row):
            if isinstance(val, (np.integer,)):
                d[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                d[col] = None if (math.isnan(float(val)) or math.isinf(float(val))) else round(float(val), 4)
            elif isinstance(val, (np.bool_,)):
                d[col] = bool(val)
            else:
                d[col] = val
        records.append(d)
    return records


# ---------------------------------------------------------------------------

def _free_port(port: int) -> None:
    """Kill any existing process listening on the given port (except ourselves)."""
    import subprocess
    try:
        result = subprocess.run(["netstat", "-ano"], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTEN" in line:
                pid = int(line.strip().split()[-1])
                if pid != os.getpid():
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                   capture_output=True)
                    print(f"Killed stale process PID {pid} on port {port}")
    except Exception as exc:
        print(f"Warning: could not free port {port}: {exc}")


if __name__ == "__main__":
    _free_port(5000)
    print("SPX Hedge Ranker  =>  http://localhost:5000")
    webbrowser.open("http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
