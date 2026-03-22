"""
GJR-GARCH(1,1) + Entropy Pooling model for SPX terminal distribution.

Provides:
  predict_spx_distribution(current_price, target_date_days, ...)
  GJREntropyModel — implements ScenarioModel protocol (scenarios.py)
"""

from __future__ import annotations

import datetime
import os
import pickle
import warnings

import numpy as np
from arch import arch_model
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
import yfinance as yf

from scenarios import Scenario, _crash_name, _validate

# ---------------------------------------------------------------------------
# Disk-cache paths (same directory as this file)
# ---------------------------------------------------------------------------

_DIR         = os.path.dirname(os.path.abspath(__file__))
_PRICES_CSV  = os.path.join(_DIR, ".garch_prices_cache.csv")
_GARCH_PKL   = os.path.join(_DIR, ".garch_fit_cache.pkl")
_CACHE_DAYS  = 7   # refresh prices + refit at most once per week

# Module-level in-memory cache (survives multiple GARCHPathCache calls in one run)
_GARCH_CACHE: dict = {}


# ---------------------------------------------------------------------------
# GARCH fitting
# ---------------------------------------------------------------------------

def _prices_cache_stale() -> bool:
    """True if the prices CSV doesn't exist or is older than _CACHE_DAYS."""
    if not os.path.exists(_PRICES_CSV):
        return True
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(_PRICES_CSV))
    return age.days >= _CACHE_DAYS


def _fit_garch(ticker: str = "^GSPC", lookback_years: int = 30):
    """
    Fit GJR-GARCH(1,1) with Student-t to SPX returns.

    Caches downloaded prices to .garch_prices_cache.csv and the fitted model
    to .garch_fit_cache.pkl, refreshed at most once per week. This avoids a
    yfinance network call on every server restart.
    """
    cache_key = (ticker, lookback_years)
    if cache_key in _GARCH_CACHE:
        return _GARCH_CACHE[cache_key]

    # Try loading from disk cache first
    if not _prices_cache_stale() and os.path.exists(_GARCH_PKL):
        try:
            with open(_GARCH_PKL, "rb") as f:
                result = pickle.load(f)
            _GARCH_CACHE[cache_key] = result
            print(f"GARCH: loaded fitted model from disk cache ({_GARCH_PKL})")
            return result
        except Exception as e:
            print(f"GARCH: disk cache load failed ({e}), refitting…")

    # Download prices (or refresh stale cache)
    end   = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=lookback_years * 365)).isoformat()

    import pandas as pd
    if not _prices_cache_stale() and os.path.exists(_PRICES_CSV):
        print("GARCH: using cached prices (< 7 days old)")
        prices = pd.read_csv(_PRICES_CSV, index_col=0, parse_dates=True).squeeze()
    else:
        print(f"GARCH: downloading {lookback_years}yr SPX history from Yahoo Finance…")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(ticker, start=start, end=end, progress=False,
                               auto_adjust=True, timeout=15)
        prices = data["Close"].dropna()
        prices.to_csv(_PRICES_CSV)
        print(f"GARCH: prices saved to {_PRICES_CSV}")

    returns = prices.pct_change().dropna()
    am      = arch_model(returns * 100, vol="GARCH", p=1, q=1, o=1, dist="t", mean="Constant")
    result  = am.fit(disp="off", show_warning=False)

    try:
        with open(_GARCH_PKL, "wb") as f:
            pickle.dump(result, f)
        print(f"GARCH: fitted model saved to {_GARCH_PKL}")
    except Exception as e:
        print(f"GARCH: could not save fit cache ({e})")

    _GARCH_CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def _simulate_paths(
    garch_result,
    current_price: float,
    n_paths: int = 100_000,
    horizon: int = 504,
    burn: int = 500,
    chunk_size: int = 25_000,
    mu_override: float | None = None,
) -> np.ndarray:
    """
    Simulate SPX price paths via GJR-GARCH(1,1) with Student-t innovations.

    Processes paths in chunks of chunk_size to keep peak memory per chunk to
    ~(burn+horizon) × chunk_size × 16 bytes (~192 MiB at defaults) rather than
    allocating the entire innovation matrix at once.

    Returns array of shape (n_paths, horizon) with SPX price levels.
    """
    from scipy.stats import t as student_t

    p = garch_result.params
    # mu_override: annualised % return (e.g. 7.0 = 7%/yr); None = use fitted value
    if mu_override is not None:
        mu = float(mu_override) / 252.0   # convert annual % → daily %
    else:
        mu = float(p["mu"])
    omega = float(p["omega"])
    alpha = float(p["alpha[1]"])
    gamma = float(p["gamma[1]"])
    beta  = float(p["beta[1]"])
    nu    = float(p["nu"])

    total_steps = burn + horizon
    std_factor  = np.sqrt(nu / (nu - 2.0))
    denom       = max(1.0 - alpha - 0.5 * gamma - beta, 1e-8)
    h_init      = max(omega / denom, 1e-6)

    paths_out = np.empty((n_paths, horizon))

    for start in range(0, n_paths, chunk_size):
        end = min(start + chunk_size, n_paths)
        nc  = end - start

        # Generate and standardize in-place — avoids a second large allocation
        innov = student_t.rvs(nu, size=(total_steps, nc))
        innov /= std_factor

        h       = np.full(nc, h_init)
        returns = np.empty((total_steps, nc))

        for t in range(total_steps):
            sigma      = np.sqrt(np.maximum(h, 1e-8))
            eps        = sigma * innov[t]
            returns[t] = mu + eps
            indicator  = (eps < 0.0).astype(np.float64)
            h          = omega + (alpha + gamma * indicator) * eps ** 2 + beta * h

        pct = np.clip(returns[burn:], -99.0, None)
        paths_out[start:end] = (current_price * np.exp(np.cumsum(np.log1p(pct / 100.0), axis=0))).T

    return paths_out


# ---------------------------------------------------------------------------
# Running minimum
# ---------------------------------------------------------------------------

def _compute_running_min(paths: np.ndarray) -> np.ndarray:
    """2-year running minimum for each path. Returns (n_paths,)."""
    return np.minimum.accumulate(paths, axis=1)[:, -1]


# ---------------------------------------------------------------------------
# Bin construction
# ---------------------------------------------------------------------------

def _build_bins(
    running_min: np.ndarray,
    n_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Percentile-based bins from running minimum values.
    Returns (bin_edges, bin_centers, prior_weights, bin_assignments).
    """
    bin_edges = np.percentile(running_min, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)  # handle duplicate edges at tails
    n_bins = len(bin_edges) - 1

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    prior_weights = np.ones(n_bins) / n_bins

    bin_assignments = np.searchsorted(bin_edges[1:-1], running_min, side="left")
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    return bin_edges, bin_centers, prior_weights, bin_assignments


# ---------------------------------------------------------------------------
# Prior diagnostics
# ---------------------------------------------------------------------------

_DRAWDOWN_CHECKPOINTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def _compute_prior_stats(
    running_min: np.ndarray,
    current_price: float,
    tail_tilt: float,
    tilt_threshold: float,
) -> tuple[list[dict], float, float]:
    """
    Compute GARCH prior drawdown statistics.
    Returns (stats_list, p_prior_at_threshold, p_view).
    stats_list rows: {drawdown, level, prob, is_tilt_threshold}
    """
    stats = []
    p_prior = None
    for d in _DRAWDOWN_CHECKPOINTS:
        threshold = current_price * (1.0 - d)
        p = float(np.mean(running_min < threshold))
        is_tilt = abs(d - tilt_threshold) < 1e-9
        if is_tilt:
            p_prior = p
        stats.append({
            "drawdown": d,
            "level": int(round(threshold)),
            "prob": round(p, 4),
            "is_tilt_threshold": is_tilt,
        })
    if p_prior is None:
        p_prior = float(np.mean(running_min < current_price * (1.0 - tilt_threshold)))
    p_view = min(tail_tilt * p_prior, 0.99)
    return stats, p_prior, p_view


def _print_prior_stats(
    running_min: np.ndarray,
    current_price: float,
    tail_tilt: float,
    tilt_threshold: float,
) -> float:
    """Print GARCH prior drawdown probabilities. Returns p_prior at tilt_threshold."""
    stats, p_prior, p_view = _compute_prior_stats(
        running_min, current_price, tail_tilt, tilt_threshold
    )
    print(f"  GARCH prior — 2yr running min vs today ({current_price:,.0f}):")
    for s in stats:
        marker = "   ← tilt_threshold" if s["is_tilt_threshold"] else ""
        print(f"    P(ever dip >{s['drawdown']:.0%}) = {s['prob']:.0%}   [< {s['level']:,}]{marker}")
    print(f"  tail_tilt={tail_tilt:.1f} → P(ever dip >{tilt_threshold:.0%}) will be ≈ {p_view:.0%}")
    return p_prior


# ---------------------------------------------------------------------------
# View smoothing (PCHIP)
# ---------------------------------------------------------------------------

def _smooth_views(
    view_buckets: list[tuple[float, float]],
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    PCHIP interpolate view_buckets [(spx_level, cum_prob), ...] to bin weights.
    Returns target_weights of shape (n_bins,), normalized.
    """
    levels = np.array([v[0] for v in view_buckets], dtype=float)
    cdfs = np.array([v[1] for v in view_buckets], dtype=float)
    sort_idx = np.argsort(levels)
    levels, cdfs = levels[sort_idx], cdfs[sort_idx]

    # Augment with boundary points so we extrapolate to 0 / 1
    x = np.concatenate([[bin_edges[0] - 1.0], levels, [bin_edges[-1] + 1.0]])
    y = np.concatenate([[0.0], cdfs, [1.0]])

    interp = PchipInterpolator(x, y, extrapolate=True)
    cdf_at_edges = np.clip(interp(bin_edges), 0.0, 1.0)

    target_weights = np.diff(cdf_at_edges)
    target_weights = np.maximum(target_weights, 0.0)
    s = target_weights.sum()
    return target_weights / s if s > 0 else np.ones(len(target_weights)) / len(target_weights)


# ---------------------------------------------------------------------------
# Entropy pooling
# ---------------------------------------------------------------------------

def _entropy_pool(
    prior_weights: np.ndarray,
    bin_edges: np.ndarray,
    view_buckets: list[tuple[float, float]],
    confidence: float,
) -> np.ndarray:
    """
    Minimize KL(q||p) subject to CDF constraints from view_buckets.
    Returns posterior_weights of shape (n_bins,).
    """
    if confidence <= 0.0:
        return prior_weights.copy()

    n_bins = len(prior_weights)
    p = prior_weights.copy()

    # Map view bucket SPX levels to bin indices
    constraint_pairs: list[tuple[int, float]] = []
    seen_bins: set[int] = set()
    for spx_level, cum_prob in sorted(view_buckets, key=lambda x: x[0]):
        k = int(np.clip(np.searchsorted(bin_edges[1:], spx_level, side="right"), 0, n_bins - 1))
        if k not in seen_bins:
            seen_bins.add(k)
            constraint_pairs.append((k, float(cum_prob)))

    def kl_objective(q: np.ndarray) -> float:
        return float(np.sum(q * np.log(q / (p + 1e-300) + 1e-300)))

    def kl_gradient(q: np.ndarray) -> np.ndarray:
        return np.log(q / (p + 1e-300) + 1e-300) + 1.0

    constraints = [
        {
            "type": "eq",
            "fun": lambda q: q.sum() - 1.0,
            "jac": lambda q: np.ones(n_bins),
        }
    ]
    for k, cp in constraint_pairs:
        def make_con(k_: int = k, cp_: float = cp) -> dict:
            return {
                "type": "eq",
                "fun": lambda q, _k=k_, _cp=cp_: q[: _k + 1].sum() - _cp,
                "jac": lambda q, _k=k_: np.concatenate(
                    [np.ones(_k + 1), np.zeros(n_bins - _k - 1)]
                ),
            }
        constraints.append(make_con())

    result = minimize(
        kl_objective,
        p.copy(),
        jac=kl_gradient,
        method="SLSQP",
        bounds=[(1e-10, 1.0)] * n_bins,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    if result.success:
        q_ep = np.maximum(result.x, 0.0)
        q_ep /= q_ep.sum()
    else:
        # Fallback: direct PCHIP assignment
        q_ep = _smooth_views(view_buckets, bin_edges)

    # Blend with confidence
    posterior = confidence * q_ep + (1.0 - confidence) * p
    posterior = np.maximum(posterior, 0.0)
    posterior /= posterior.sum()
    return posterior


# ---------------------------------------------------------------------------
# Path weights and terminal distribution
# ---------------------------------------------------------------------------

def _compute_path_weights(
    posterior_bin_weights: np.ndarray,
    bin_assignments: np.ndarray,
    n_paths: int,
) -> np.ndarray:
    """Per-path probability weights from bin-level posterior."""
    counts = np.bincount(bin_assignments, minlength=len(posterior_bin_weights)).astype(float)
    bin_path_weights = np.where(
        counts > 0, posterior_bin_weights / np.maximum(counts, 1.0), 0.0
    )
    path_weights = bin_path_weights[bin_assignments]
    path_weights /= path_weights.sum()
    return path_weights


def _posterior_terminal(
    paths: np.ndarray,
    path_weights: np.ndarray,
    target_day: int,
    n_terminal_bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted terminal SPX distribution at target_day.
    Returns (terminal_levels, terminal_probs).
    """
    terminal_day = min(target_day - 1, paths.shape[1] - 1)
    terminal_values = paths[:, terminal_day]

    terminal_edges = np.percentile(terminal_values, np.linspace(0, 100, n_terminal_bins + 1))
    terminal_edges = np.unique(terminal_edges)
    n_t = len(terminal_edges) - 1

    t_assignments = np.searchsorted(terminal_edges[1:-1], terminal_values, side="left")
    t_assignments = np.clip(t_assignments, 0, n_t - 1)

    terminal_probs = np.bincount(t_assignments, weights=path_weights, minlength=n_t)
    wt_sum = np.bincount(t_assignments, weights=path_weights * terminal_values, minlength=n_t)
    bin_centers = (terminal_edges[:-1] + terminal_edges[1:]) / 2
    terminal_levels = np.where(terminal_probs > 0, wt_sum / terminal_probs, bin_centers)

    valid = terminal_probs > 0
    terminal_levels = terminal_levels[valid]
    terminal_probs = terminal_probs[valid]
    terminal_probs /= terminal_probs.sum()

    return terminal_levels, terminal_probs


# ---------------------------------------------------------------------------
# Public API: predict_spx_distribution
# ---------------------------------------------------------------------------

def predict_spx_distribution(
    current_price: float,
    target_date_days: int,
    tail_tilt: float = 1.0,
    tilt_threshold: float = 0.20,
    confidence_level: float = 1.0,
    view_buckets: list[tuple[float, float]] | None = None,
) -> dict:
    """
    Predict SPX terminal distribution at target_date_days trading days from now.

    Args:
        current_price:    current SPX level (e.g. 5500)
        target_date_days: trading days to target (1–504)
        tail_tilt:        multiplier on P(2yr dip ≥ tilt_threshold) vs GARCH prior.
                          1.0 = pure GARCH, 2.0 = crashes twice as likely, etc.
        tilt_threshold:   drawdown magnitude defining "crash" for the tilt (default 0.20)
        confidence_level: 0 = pure GARCH prior, 1 = views fully imposed
        view_buckets:     optional advanced override — [(spx_level, cum_prob), ...].
                          If provided, bypasses tail_tilt entirely.

    Returns dict:
        {spx_levels, probabilities, mean, std, percentiles}
    """
    print("Fitting GJR-GARCH(1,1)...")
    garch_result = _fit_garch()

    print("Simulating 100K paths × 504 days...")
    paths = _simulate_paths(garch_result, current_price)

    print("Computing 2-year running minima...")
    running_min = _compute_running_min(paths)

    print("Building percentile bins...")
    bin_edges, _, prior_weights, bin_assignments = _build_bins(running_min)

    # Resolve effective_buckets from tail_tilt or explicit override
    if view_buckets is not None:
        effective_buckets = view_buckets
    else:
        p_prior = _print_prior_stats(running_min, current_price, tail_tilt, tilt_threshold)
        if tail_tilt <= 1.0:
            effective_buckets = None
        else:
            p_view = min(tail_tilt * p_prior, 0.99)
            effective_buckets = [(current_price * (1.0 - tilt_threshold), p_view)]

    if effective_buckets is not None:
        print("Entropy pooling...")
        posterior_weights = _entropy_pool(prior_weights, bin_edges, effective_buckets, confidence_level)
    else:
        posterior_weights = prior_weights

    print("Computing terminal distribution...")
    path_weights = _compute_path_weights(posterior_weights, bin_assignments, len(paths))
    levels, probs = _posterior_terminal(paths, path_weights, target_date_days)

    mean = float(np.dot(levels, probs))
    variance = float(np.dot((levels - mean) ** 2, probs))
    std = float(np.sqrt(variance))

    sort_idx = np.argsort(levels)
    sorted_levels = levels[sort_idx]
    cum_probs = np.cumsum(probs[sort_idx])

    percentiles: dict[int, float] = {}
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        idx = min(int(np.searchsorted(cum_probs, pct / 100.0)), len(sorted_levels) - 1)
        percentiles[pct] = float(sorted_levels[idx])

    return {
        "spx_levels": levels,
        "probabilities": probs,
        "mean": mean,
        "std": std,
        "percentiles": percentiles,
    }


# ---------------------------------------------------------------------------
# GARCHPathCache — expensive simulation, computed once and reused
# ---------------------------------------------------------------------------

class GARCHPathCache:
    """
    Pre-computed GJR-GARCH simulation state (~10s to build).
    Pass as _cache= to GJREntropyModel to avoid re-simulating on each EP run.
    """

    def __init__(
        self,
        current_price: float,
        n_paths: int = 100_000,
        horizon: int = 504,
        n_bins: int = 200,
        annual_drift_pct: float | None = None,
    ):
        self.current_price = float(current_price)
        self.n_paths = n_paths
        self.horizon = horizon
        self.n_bins = n_bins
        self.annual_drift_pct = annual_drift_pct  # None = use fitted

        print("GARCHPathCache: Fitting GJR-GARCH(1,1)...")
        garch_result = _fit_garch()

        # Record the fitted drift so the UI can display it
        self.fitted_annual_drift_pct = round(float(garch_result.params["mu"]) * 252, 2)
        effective_drift = annual_drift_pct if annual_drift_pct is not None else self.fitted_annual_drift_pct
        print(f"GARCHPathCache: drift = {effective_drift:.2f}%/yr "
              f"({'override' if annual_drift_pct is not None else 'fitted'})")

        print(f"GARCHPathCache: Simulating {n_paths:,} paths × {horizon} days...")
        self.paths = _simulate_paths(
            garch_result, current_price, n_paths, horizon, mu_override=annual_drift_pct
        )

        print("GARCHPathCache: Computing running minima...")
        self.running_min = _compute_running_min(self.paths)

        print("GARCHPathCache: Building bins...")
        self.bin_edges, _, self.prior_weights, self.bin_assignments = _build_bins(
            self.running_min, n_bins
        )
        print("GARCHPathCache: Ready.")

    def make_model(
        self,
        tail_tilt: float = 1.0,
        tilt_threshold: float = 0.20,
        confidence_level: float = 1.0,
        view_buckets=None,
    ) -> "GJREntropyModel":
        return GJREntropyModel(
            self.current_price,
            tail_tilt=tail_tilt,
            tilt_threshold=tilt_threshold,
            confidence_level=confidence_level,
            view_buckets=view_buckets,
            _cache=self,
        )

    def prior_stats(
        self, tail_tilt: float = 1.0, tilt_threshold: float = 0.20
    ) -> tuple[list[dict], float, float]:
        """Return (stats_list, p_prior, p_view). Used by the API."""
        return _compute_prior_stats(
            self.running_min, self.current_price, tail_tilt, tilt_threshold
        )


# ---------------------------------------------------------------------------
# GJREntropyModel — ScenarioModel protocol implementation
# ---------------------------------------------------------------------------

class GJREntropyModel:
    """
    ScenarioModel using GJR-GARCH(1,1) simulation + Entropy Pooling.

    Initialize once (~5–10s for GARCH fit + simulation), call cheaply per DTE.
    Implements the ScenarioModel protocol from scenarios.py so it can be passed
    directly to score_puts() and picked up by hedge.py / app.py.

    Usage:
        model = GJREntropyModel(5500, tail_tilt=2.0)   # crashes 2× as likely
        model = GJREntropyModel(5500, tail_tilt=1.0)   # pure GARCH baseline
        model = GJREntropyModel(5500, view_buckets=[(4000, 0.15), (4500, 0.35)])  # advanced
        scenarios = model(params, dte=365)
    """

    def __init__(
        self,
        current_price: float,
        tail_tilt: float = 1.0,
        tilt_threshold: float = 0.20,
        confidence_level: float = 1.0,
        view_buckets: list[tuple[float, float]] | None = None,
        n_paths: int = 100_000,
        horizon: int = 504,
        n_bins: int = 200,
        _cache: "GARCHPathCache | None" = None,
    ):
        # Use injected cache (cheap) or build one from scratch (expensive)
        if _cache is not None:
            self._cache = _cache
        else:
            self._cache = GARCHPathCache(current_price, n_paths, horizon, n_bins)

        self._current_price = self._cache.current_price

        # Resolve effective_buckets from tail_tilt or explicit override
        if view_buckets is not None:
            effective_buckets = view_buckets
        else:
            p_prior = _print_prior_stats(
                self._cache.running_min, self._cache.current_price, tail_tilt, tilt_threshold
            )
            if tail_tilt <= 1.0:
                effective_buckets = None
            else:
                p_view = min(tail_tilt * p_prior, 0.99)
                effective_buckets = [(self._cache.current_price * (1.0 - tilt_threshold), p_view)]

        if effective_buckets is not None:
            print("GJREntropyModel: Entropy pooling...")
            posterior_weights = _entropy_pool(
                self._cache.prior_weights, self._cache.bin_edges,
                effective_buckets, confidence_level,
            )
        else:
            posterior_weights = self._cache.prior_weights.copy()

        print("GJREntropyModel: Computing path weights...")
        self._path_weights = _compute_path_weights(
            posterior_weights, self._cache.bin_assignments, self._cache.n_paths
        )
        print("GJREntropyModel: Ready.")

    def __call__(self, params, dte: int) -> list[Scenario]:
        """
        Generate scenarios at given DTE from the posterior terminal distribution.

        Crash scenario probabilities are derived from incremental CDF buckets.
        Non-crash probability is the complement, split by params.non_crash_splits shares.
        """
        levels, probs = _posterior_terminal(
            self._cache.paths,
            self._path_weights,
            target_day=dte,
        )

        sort_idx = np.argsort(levels)
        levels_sorted = levels[sort_idx]
        cum_probs = np.cumsum(probs[sort_idx])

        spot = self._cache.current_price

        def prob_below(threshold: float) -> float:
            idx = int(np.searchsorted(levels_sorted, threshold, side="right"))
            return float(cum_probs[min(idx - 1, len(cum_probs) - 1)]) if idx > 0 else 0.0

        # Sort crashes from most severe (most negative spx_ret) to least severe
        crash_splits_sorted = sorted(params.crash_splits, key=lambda x: x[1])

        scenarios: list[Scenario] = []
        prev_cum = 0.0

        for _crash_share, spx_ret in crash_splits_sorted:
            threshold = spot * (1.0 + spx_ret)
            p_below = prob_below(threshold)
            p_bucket = max(0.0, p_below - prev_cum)
            scenarios.append(
                Scenario(
                    name=_crash_name(spx_ret),
                    spx_return=spx_ret,
                    probability=p_bucket,
                    is_crash=True,
                )
            )
            prev_cum = p_below

        # Non-crash: probability above the least severe crash threshold
        least_severe_ret = max(spx_ret for _, spx_ret in params.crash_splits)
        p_no_crash = max(0.0, 1.0 - prob_below(spot * (1.0 + least_severe_ret)))

        nc_share_total = sum(share for _, _, share in params.non_crash_splits)
        for label, spx_ret, share in params.non_crash_splits:
            norm_share = (
                share / nc_share_total if nc_share_total > 0
                else 1.0 / len(params.non_crash_splits)
            )
            scenarios.append(
                Scenario(
                    name=label,
                    spx_return=spx_ret,
                    probability=p_no_crash * norm_share,
                    is_crash=False,
                )
            )

        _validate(scenarios, "garch_ep")
        return scenarios

    def decile_paths(
        self, n_deciles: int = 10, step: int = 5
    ) -> tuple[list[list[float]], list[float], list[dict]]:
        """
        Return (paths, ep_weights, stats) for n_deciles stylized representative paths.

        Each path is a smooth two-phase curve anchored to actual decile statistics:
          Phase 1 (day 0 → avg_bottom_day):  log-linear decline to avg bottom price
          Phase 2 (avg_bottom_day → day 504): log-linear recovery to avg terminal price

        This preserves the correct depth, timing, and recovery for each decile without
        the geometric-mean artifact of smoothing away the actual crash.

        stats[i]: {t_bottom, t_bottom_idx, v_bottom, v_terminal, drawdown_pct}
        """
        sorted_idx = np.argsort(self._cache.running_min)
        n         = len(sorted_idx)
        size      = n // n_deciles
        horizon   = self._cache.horizon
        days      = np.arange(0, horizon, step)
        v_start   = self._cache.current_price

        paths_out: list[list[float]] = []
        weights_out: list[float]     = []
        stats_out: list[dict]        = []

        for d in range(n_deciles):
            start = d * size
            end   = (d + 1) * size if d < n_deciles - 1 else n
            chunk = sorted_idx[start:end]
            chunk_paths = self._cache.paths[chunk]   # (chunk_size, horizon)

            # Average day of minimum (argmin per path); floor at 1 to avoid div/0
            t_bottom = max(1, int(np.mean(np.argmin(chunk_paths, axis=1))))

            # Geometric means (log-space averages) for bottom and terminal prices
            v_bottom   = float(np.exp(np.mean(np.log(self._cache.running_min[chunk]))))
            v_terminal = float(np.exp(np.mean(np.log(chunk_paths[:, -1]))))

            # Two-phase log-linear path
            log_s = np.log(v_start)
            log_b = np.log(v_bottom)
            log_t = np.log(v_terminal)
            t_tail = max(horizon - t_bottom, 1)

            log_path = np.where(
                days <= t_bottom,
                log_s + (log_b - log_s) * days / t_bottom,
                log_b + (log_t - log_b) * (days - t_bottom) / t_tail,
            )
            paths_out.append(np.exp(log_path).tolist())
            weights_out.append(round(float(np.sum(self._path_weights[chunk])), 4))
            stats_out.append({
                "t_bottom":     t_bottom,
                "t_bottom_idx": int(t_bottom // step),   # index in downsampled path
                "v_bottom":     round(v_bottom, 0),
                "v_terminal":   round(v_terminal, 0),
                "drawdown_pct": round((v_bottom / v_start - 1) * 100, 1),
            })

        return paths_out, weights_out, stats_out

    def quintile_paths(
        self, n_quintiles: int = 5, n_sample: int = 5, step: int = 5, seed: int | None = None,
    ) -> tuple[list[list[float]], list[int], list[dict]]:
        """
        Return (paths, groups, stats) for n_quintiles EP-weighted quintiles,
        n_sample randomly sampled actual paths each.

        Quintile boundaries are equal-posterior-weight intervals on the running-minimum
        distribution, so Q1 is the 20% of probability mass with the worst 2-year drawdown
        under the current EP posterior — not simply the bottom 20% of paths by count.

        paths:  list of n_quintiles*n_sample downsampled actual price paths
        groups: quintile index (0=most bearish) for each path, for chart coloring
        stats:  per-quintile avg drawdown_pct, v_bottom, v_terminal
        """
        rng = np.random.default_rng(seed)

        # Sort paths by running_min (most bearish first)
        sorted_idx    = np.argsort(self._cache.running_min)
        sorted_weights = self._path_weights[sorted_idx]
        cum_weights   = np.cumsum(sorted_weights)
        cum_weights   /= cum_weights[-1]   # normalize to [0, 1]

        days_idx   = np.arange(0, self._cache.horizon, step)
        boundaries = np.linspace(0, 1, n_quintiles + 1)

        paths_out:  list[list[float]] = []
        groups_out: list[int]         = []
        stats_out:  list[dict]        = []

        for q in range(n_quintiles):
            lo, hi = boundaries[q], boundaries[q + 1]
            mask = (cum_weights >= lo) & (cum_weights < hi) if q < n_quintiles - 1 else cum_weights >= lo
            q_indices = sorted_idx[mask]

            if len(q_indices) == 0:
                continue

            n = min(n_sample, len(q_indices))
            sampled = rng.choice(q_indices, size=n, replace=False)

            for idx in sampled:
                paths_out.append(self._cache.paths[idx][days_idx].tolist())
                groups_out.append(q)

            v_bottom   = float(np.exp(np.mean(np.log(self._cache.running_min[q_indices]))))
            v_terminal = float(np.exp(np.mean(np.log(self._cache.paths[q_indices, -1]))))
            stats_out.append({
                "v_bottom":     round(v_bottom, 0),
                "v_terminal":   round(v_terminal, 0),
                "drawdown_pct": round((v_bottom / self._cache.current_price - 1) * 100, 1),
            })

        return paths_out, groups_out, stats_out

    def terminal_distribution(
        self, target_day: int, n_bins: int = 80
    ) -> dict:
        """
        Prior vs posterior terminal SPX distribution at target_day.
        Returns {levels, prior_probs, posterior_probs} on a common bin grid.
        """
        prior_pw = _compute_path_weights(
            self._cache.prior_weights, self._cache.bin_assignments, self._cache.n_paths
        )
        terminal_day = min(target_day - 1, self._cache.paths.shape[1] - 1)
        all_vals = self._cache.paths[:, terminal_day]
        # Linear-width bins between 0.5th and 99.5th percentile.
        # Quantile-spaced bins would produce a flat chart under equal weights
        # because each bin contains the same number of paths by construction.
        lo = np.percentile(all_vals, 0.5)
        hi = np.percentile(all_vals, 99.5)
        edges = np.linspace(lo, hi, n_bins + 1)
        n_b = len(edges) - 1
        centers = ((edges[:-1] + edges[1:]) / 2).tolist()
        t_assign = np.clip(
            np.searchsorted(edges[1:-1], all_vals, side="left"), 0, n_b - 1
        )
        prior_probs = np.bincount(t_assign, weights=prior_pw, minlength=n_b).tolist()
        post_probs = np.bincount(
            t_assign, weights=self._path_weights, minlength=n_b
        ).tolist()
        return {"levels": centers, "prior_probs": prior_probs, "posterior_probs": post_probs}
