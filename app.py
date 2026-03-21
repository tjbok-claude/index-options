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
import threading
import time
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
# In-memory CBOE cache — 60-second TTL
# ---------------------------------------------------------------------------
_cache: dict = {
    "df":         None,
    "underlying": None,
    "fetched_at": None,
}
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
}
_GARCH_DEFAULT_PRICE = 6500.0  # fallback if CBOE not yet fetched


def _garch_init_worker(price: float, annual_drift_pct: float | None = None) -> None:
    """Background thread: fit GARCH and simulate paths."""
    from spx_model import GARCHPathCache
    with _garch_lock:
        _garch_state["loading"] = True
        _garch_state["error"] = None
        _garch_state["cache"] = None
        _garch_state["model"] = None
    try:
        cache = GARCHPathCache(price, annual_drift_pct=annual_drift_pct)
        # Auto-create a default EP model matching the UI's default parameters
        # so the grid shows the same results as clicking Apply without changes.
        default_model = cache.make_model(
            view_buckets=_DEFAULT_EP_BUCKETS,
            confidence_level=_DEFAULT_EP_CONFIDENCE,
        )
        with _garch_lock:
            _garch_state["cache"] = cache
            _garch_state["model"] = default_model
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


def start_garch_init(price: float | None = None, annual_drift_pct: float | None = None) -> None:
    """Kick off (or restart) GARCH simulation in a daemon thread."""
    p = price or _GARCH_DEFAULT_PRICE
    t = threading.Thread(target=_garch_init_worker, args=(p, annual_drift_pct), daemon=True)
    t.start()


# Start background simulation immediately when the server loads
start_garch_init()

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
    force_refresh   = request.args.get("force_refresh", "false").lower() == "true"
    model_name      = request.args.get("model", "garch_ep")
    if model_name not in SCENARIO_MODELS:
        model_name = DEFAULT_MODEL

    p = Params(
        p_crash=p_crash,
        annual_budget=annual_budget,
        horizon_months=horizon_months,
        roth_multiplier=roth_multiplier,
        contracts=contracts,
    )

    # --- Fetch CBOE data (cached) ---
    with _cache_lock:
        now = time.time()
        stale = (
            _cache["fetched_at"] is None
            or (now - _cache["fetched_at"]) > CACHE_TTL_SEC
        )

        if stale or force_refresh:
            try:
                raw = fetch_chain(KNOWN_SYMBOLS["SPX"])
                df_full, underlying = parse_chain(raw, base_symbol="SPX")
                _cache["df"]         = df_full
                _cache["underlying"] = underlying
                _cache["fetched_at"] = now
            except Exception as exc:
                if _cache["df"] is None:
                    return jsonify({"error": "CBOE fetch failed", "detail": str(exc)}), 502
                # Fall through with stale data on transient errors

        df_full    = _cache["df"]
        underlying = _cache["underlying"]
        fetched_at = _cache["fetched_at"]

    # --- Filter & score ---
    spx_spot    = underlying["current_price"]
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
    if model_name == "garch_ep":
        with _garch_lock:
            active = _garch_state["model"]
        if active is None:
            scoring_model = get_model("survival")
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
        "meta": _build_meta(underlying, spx_spot, n_contracts, p, len(scored), len(rejected), fetched_at, model_name),
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
            spot = (_cache["underlying"] or {}).get("current_price")
        if spot and abs(spot - cache.current_price) / cache.current_price > 0.02:
            stale = True
    return jsonify({
        "paths_ready":          cache is not None,
        "loading":              s["loading"],
        "error":                s["error"],
        "spx_at_init":          cache.current_price if cache else None,
        "fitted_annual_drift":  cache.fitted_annual_drift_pct if cache else None,
        "active_drift":         (cache.annual_drift_pct if cache.annual_drift_pct is not None
                                 else cache.fitted_annual_drift_pct) if cache else None,
        "active_model":         {**meta, "stale": stale} if meta else None,
    })


@app.route("/api/model/reinit", methods=["POST"])
def api_model_reinit():
    """Re-run GARCH simulation with optional drift override. Uses current CBOE price."""
    body = request.get_json(force=True) or {}
    drift = body.get("annual_drift_pct")
    annual_drift_pct = float(drift) if drift is not None else None
    with _cache_lock:
        spot = (_cache["underlying"] or {}).get("current_price")
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

    prior_stats, _, _ = cache.prior_stats()

    if not buckets:
        # Pure GARCH — no EP
        model = cache.make_model()
    else:
        try:
            model = _run_ep(buckets, confidence)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    paths, groups, stats = model.quintile_paths()
    terminal             = model.terminal_distribution(query_dte)

    return jsonify({
        "current_price":   cache.current_price,
        "prior_stats":     prior_stats,
        "quintile_paths":  paths,
        "quintile_groups": groups,
        "quintile_stats":  stats,
        "terminal":        terminal,
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


@app.route("/api/health")
def api_health():
    with _cache_lock:
        fa = _cache["fetched_at"]
        age = round(time.time() - fa, 1) if fa else None
    return jsonify({"status": "ok", "cache_age_sec": age})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_meta(underlying, spx_spot, n_contracts, p, n_passed, n_filtered, fetched_at, model_name=DEFAULT_MODEL):
    with _garch_lock:
        garch_meta = _garch_state["meta"]
        garch_loading = _garch_state["loading"]
    return {
        "model": model_name,
        "garch_ep_meta": garch_meta,
        "garch_loading": garch_loading,
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

if __name__ == "__main__":
    print("SPX Hedge Ranker  =>  http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
