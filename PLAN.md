# Model improvement plan

Items accumulate here as we review each suggestion.
Priority: 🔴 Critical · 🟡 Notable · 🟢 Low / display-only

---

## 🔴 1. Fix DTE calendar/trading-day mismatch in `expected_payoffs()` (and `__call__`)

**Source**: Peer review §3 (found independently)
**Files**: `spx_model.py`
**Affects scoring**: Yes — EPR, CrEff, BEP for GARCH EP model

### Problem
`df["dte"]` from `filter_puts()` is in **calendar days** (`(expiry - today).days`).
Both `expected_payoffs()` and `_posterior_terminal()` index into the stored path array with:

```python
week_idx = min((dte - 1) // 5, self._cache.paths.shape[1] - 1)
```

The path array stores prices at every 5th **trading day** over a 504-trading-day horizon.
Treating calendar DTE as trading DTE causes a systematic ~1.41× time overestimate for all
options in the 180–500 calendar-day range.

| Calendar DTE | Correct trading days | Code uses (trading days) | Ratio |
|---|---|---|---|
| 180 (≈6mo) | 124 | 175 | 1.41× |
| 365 (1yr)  | 252 | 360 | 1.43× |
| 504 (≈16mo)| 348 | 500 | 1.44× |
| 730 (2yr)  | 504 | 500 (capped) | ≈1.0× |

Shorter-dated puts are systematically overvalued relative to longer-dated ones because their
terminal distribution is evaluated further into the future than their actual expiry.

### Fix
In `expected_payoffs()` (line 1056) and `_posterior_terminal()` (line 429), convert calendar
DTE to trading DTE before computing the weekly index:

```python
# spx_model.py — expected_payoffs(), line 1056
trading_dte = int(round(dte * 252 / 365))
week_idx = min((trading_dte - 1) // 5, self._cache.paths.shape[1] - 1)
```

```python
# spx_model.py — _posterior_terminal(), line 429
trading_days = int(round(target_day * 252 / 365))
terminal_day = min((trading_days - 1) // 5, paths.shape[1] - 1)
```

Note: `_posterior_terminal` is called with calendar DTE from `__call__`, and with trading
DTE from `_compute_bl_market_running_min` (which does its own conversion via
`_bl_target_cal_dte`). After this fix, callers must be consistent — document that `target_day`
expects **calendar days** throughout. The survival model is unaffected (uses a ratio).

---

## 🟡 2. Add EP transparency diagnostic: post-EP terminal probabilities

**Source**: Suggestion 1 (moderate concern)
**Files**: `spx_model.py`, `app.py`, `static/app.js`
**Affects scoring**: No — display only

### Problem
EP operates on the 2-year running-minimum distribution, not on the terminal distribution at
a specific DTE. Users tilting the crash probability see the running-minimum bars move but
have no direct feedback on whether P(terminal SPX < K) at their DTE of interest actually
shifted meaningfully. Some paths crash early and recover, so a running-minimum tilt may be
"wasted" on paths whose terminal value is near spot.

The EP design choice (views on running minimum) is correct and intentional — running-minimum
views are more natural for hedgers and avoid requiring DTE-specific view inputs. But users
should be able to verify the downstream effect.

### Fix
After EP is applied in `GJREntropyModel.__init__()` (and in the Model Builder preview
endpoint), compute and expose a small diagnostic table:

```python
# For each threshold in [0.70, 0.75, 0.80, 0.85, 0.90] × current_price
# and at the query DTE, report:
#   prior: P(terminal SPX < threshold)
#   posterior: P(terminal SPX < threshold)
```

Return this as `terminal_shift` in the `/api/model/preview` response (alongside the
existing `prior_stats`). Display it in the Model Builder panel as a compact table, e.g.:

```
SPX level  Prior    Posterior
< 4500     6.2%     11.4%
< 4000     2.1%      5.3%
< 3500     0.6%      2.1%
```

This lets users directly verify that their running-minimum view propagated into the terminal
distribution that drives option pricing.

---

## 🟢 3. Fix `__call__` to use conditional-mean spx_return per crash bucket

**Source**: Suggestion 2 (low, display-only)
**Files**: `spx_model.py`
**Affects scoring**: No — `expected_payoffs()` is used for all GARCH EP scoring; `__call__`
output is only used for (a) scenario names/labels in `payoff_*_1c` display columns and
(b) the UI scenario probability table.

### Problem
`GJREntropyModel.__call__()` reports each crash scenario with:
- `probability` = P(terminal ∈ bucket) from the GARCH posterior — correctly derived from paths
- `spx_return` = the **left boundary** of the bucket (e.g., -0.25 for the [-25%, -40%) bucket)

The conditional mean terminal return within the bucket may be significantly different from
the boundary (e.g., -31% when the boundary is -25%). Any caller using these scenarios for
payoff evaluation — including the fallback `else` branch in `score_puts()`, or direct API
callers — will underestimate payoffs for puts with strikes in that bucket's range.

Additionally, `__call__` shares the DTE calendar/trading-day bug from item 1, which
affects the scenario probability table in the UI.

### Fix

**Part A — conditional mean return** (lines 702–714 of spx_model.py):

After computing `p_below` at each bucket boundary, also compute the probability-weighted
mean terminal return of paths within the bucket using raw path data:

```python
week_idx  = min((trading_dte - 1) // 5, self._cache.paths.shape[1] - 1)
S_T       = self._cache.paths[:, week_idx]   # (n_paths,)
w         = self._path_weights               # (n_paths,)

for i, (_crash_share, spx_ret) in enumerate(crash_splits_sorted):
    upper_threshold = spot * (1.0 + prev_spx_ret) if i > 0 else np.inf
    lower_threshold = spot * (1.0 + spx_ret)
    in_bucket = (S_T >= lower_threshold) & (S_T < upper_threshold)
    w_bucket  = w * in_bucket
    w_sum     = w_bucket.sum()
    if w_sum > 0:
        mean_S_T  = np.dot(w_bucket, S_T) / w_sum
        cond_ret  = mean_S_T / spot - 1.0
    else:
        cond_ret  = spx_ret   # fallback to boundary if bucket is empty
    scenarios.append(Scenario(name=_crash_name(spx_ret), spx_return=cond_ret,
                               probability=p_bucket, is_crash=True))
```

The scenario `name` still uses the bucket label (e.g., `crash_25pct`) so display columns
remain consistently labeled. Only `spx_return` (the representative return) is updated.

**Part B — DTE fix**: Apply the same calendar→trading conversion from item 1 inside
`__call__` before the `_posterior_terminal` call (line 683).

---

## 🟢 4. Document arithmetic-vs-geometric drift convention for `annual_drift_pct`

**Source**: Suggestion 3 (labelled "moderate" by reviewer; assessed as low here)
**Files**: `spx_model.py` (docstring only), optionally `app.py` / UI tooltip
**Affects scoring**: No — documentation only

### Problem
The GARCH simulation generates simple daily percentage returns (`r = mu + eps`) and
converts them to log returns before compounding (`log1p(pct / 100)`). This is mathematically
correct but creates a drift convention that users and future developers should understand:

1. `mu_override / 252` interprets the annual input as an **arithmetic (simple) return**.
   The geometric compound rate the paths will actually exhibit is:

   ```
   r_geometric ≈ annual_drift_pct − (σ_daily² × 252 / 2)
                ≈ annual_drift_pct − 1.3%/year   (at typical 1%/day vol)
   ```

2. Setting `annual_drift_pct=0` does not produce zero expected geometric growth — it
   produces approximately −1.3%/year (the Jensen correction from daily variance).

The numerical difference is small (~1.3%/year vs the ~±20 percentage-point uncertainty
in crash probability), so option rankings are unaffected. The concern is conceptual
clarity for anyone using the drift override to represent a specific market view.

### Fix
Documentation only — no code change warranted:

1. Add a docstring note to `_simulate_paths()` and `GARCHPathCache.__init__()` explaining
   that `mu_override` (and by extension `annual_drift_pct`) is interpreted as an **arithmetic
   annual percentage return**, and that the realized geometric growth in paths will be
   approximately `annual_drift_pct − σ_annual²/2`.

2. Optionally: add a tooltip in the UI drift-override field clarifying the convention
   (e.g., "7% means expected arithmetic return of 7%/yr; geometric compound rate will be
   ~1–1.5% lower due to daily variance").

---

---

## 🟢 5. Fix KL divergence objective and gradient (minor bug)

**Source**: Suggestion 4
**Files**: `spx_model.py`, lines 353 and 356
**Affects scoring**: No — numerical difference is negligible; bounds prevent q reaching 0

### Problem
Current objective (line 353):
```python
np.sum(q * np.log(q / (p + 1e-300) + 1e-300))
```
Operator precedence: `q / (p + 1e-300) + 1e-300` evaluates the addition *before* the log,
so this computes `log(q/p + ε)` rather than the correct `log(q/p)`.

The gradient (line 356) has the identical mistake:
```python
np.log(q / (p + 1e-300) + 1e-300) + 1.0
```

The `+ 1e-300` was intended as a second guard against `log(0)`, but it is both redundant
(SLSQP bounds already floor `q` at `1e-10`; `p + 1e-300` guards the denominator) and
subtly wrong.

### Fix
Remove the outer `+ 1e-300` from both functions:

```python
def kl_objective(q: np.ndarray) -> float:
    return float(np.sum(q * np.log(q / (p + 1e-300))))

def kl_gradient(q: np.ndarray) -> np.ndarray:
    return np.log(q / (p + 1e-300)) + 1.0
```

---

---

## 🟡 6. Replace SLSQP entropy pooling with dual (Lagrange multiplier) formulation

**Source**: Suggestion 5
**Files**: `spx_model.py`, `_entropy_pool()` (~lines 327–397)
**Affects scoring**: No in normal operation; yes for aggressive views (current fallback gives
non-KL-optimal result silently)

### Problem
The primal SLSQP formulation optimizes over n_bins=200 variables with 2–4 equality
constraints and 200 bound constraints. For moderate view tilts this works fine. For
aggressive views (e.g., pushing crash probability from a 5% GARCH prior to 50%+), SLSQP
can fail to converge and the code silently falls back to PCHIP — a direct CDF interpolation
with no KL-minimality guarantee.

The current PCHIP fallback (line 391) was added as a safety net, but it means users with
aggressive view specifications may be getting a subtly wrong posterior without any warning.

### Fix
Replace the SLSQP block with the dual (Meucci 2008) formulation, which:
- Optimizes over dim = n_constraints (1–3 variables) instead of n_bins = 200
- Is provably concave → L-BFGS-B finds the global optimum
- Requires no bounds on q (non-negativity and normalization are automatic via the
  exponential family parameterization)
- Eliminates the PCHIP fallback path entirely

**Key identities:**

The optimal posterior has the closed form:
```
q*(i) = p(i) · exp((Aᵀλ)[i]) / Z(λ)
Z(λ)  = Σ_i p(i) · exp((Aᵀλ)[i])        # log-sum-exp for stability
```
where A is the (n_constraints × n_bins) indicator matrix of cumulative constraints,
b is the target CDF vector, and λ are the Lagrange multipliers.

**Dual objective** (concave, maximized over λ):
```
L(λ) = log Z(λ) − λᵀb
∇L    = E_q*[A]ᵀ − b                     # closed form
```

Minimize −L(λ) with L-BFGS-B:

```python
from scipy.special import logsumexp
from scipy.optimize import minimize as _minimize

def _entropy_pool_dual(
    prior_weights: np.ndarray,
    bin_edges: np.ndarray,
    view_buckets: list[tuple[float, float]],
    confidence: float,
) -> np.ndarray:
    n_bins = len(prior_weights)
    p = prior_weights.copy()
    log_p = np.log(p + 1e-300)

    # Build cumulative-indicator constraint matrix
    constraint_pairs: list[tuple[int, float]] = []
    seen_bins: set[int] = set()
    for spx_level, cum_prob in sorted(view_buckets, key=lambda x: x[0]):
        k = int(np.clip(np.searchsorted(bin_edges[1:], spx_level, side="right"), 0, n_bins - 1))
        if k not in seen_bins:
            seen_bins.add(k)
            constraint_pairs.append((k, float(cum_prob)))

    n_con = len(constraint_pairs)
    A = np.zeros((n_con, n_bins))
    b = np.zeros(n_con)
    for j, (k, cp) in enumerate(constraint_pairs):
        A[j, :k + 1] = 1.0
        b[j] = cp

    def neg_dual_and_grad(lam: np.ndarray):
        tilt   = A.T @ lam                          # (n_bins,)
        log_w  = log_p + tilt
        log_Z  = logsumexp(log_w)
        q      = np.exp(log_w - log_Z)              # posterior
        f      = -(log_Z - lam @ b)                 # negate → minimize
        g      = -(A @ q - b)                       # negate gradient
        return float(f), g

    result = _minimize(neg_dual_and_grad, np.zeros(n_con), jac=True, method="L-BFGS-B")

    tilt  = A.T @ result.x
    log_w = log_p + tilt
    q_ep  = np.exp(log_w - logsumexp(log_w))

    # Confidence blend (same as before)
    posterior = confidence * q_ep + (1.0 - confidence) * p
    posterior = np.maximum(posterior, 0.0)
    posterior /= posterior.sum()
    return posterior
```

The original `_entropy_pool()` can call this directly, or be replaced by it. The PCHIP
fallback (`_smooth_views`) is no longer needed as a correctness safety net (though it can
be retained as a last-resort emergency fallback with a logged warning).

---

---

## 🟢 7. Log warning when `np.unique` materially reduces bin count in `_build_bins`

**Source**: Suggestion 6 (minor)
**Files**: `spx_model.py`, `_build_bins()` (~line 225)
**Affects scoring**: No

### Problem
`_build_bins` requests `n_bins=200` equal-frequency percentile edges, then calls
`np.unique(bin_edges)` to remove duplicate edges (necessary — zero-width bins cause
division-by-zero in downstream path weighting). The effective bin count is silently
reduced whenever many paths share the same running-minimum value.

For SPX under calibrated GJR-GARCH dynamics with a 504-trading-day horizon, this is
near-theoretical: fitted daily vol (0.8–1.5%) means virtually no path goes 2 years without
a new low, so duplicates only arise at the deep-crash floor (the -99% clip) where collapsing
bins is correct behavior.

The concern is realistic for: (a) very low-volatility regimes (VIX < 10), (b) synthetic
testing with low-vol overrides, (c) any extension to non-equity asset classes.

### Fix
Add a single diagnostic log line after `np.unique`:

```python
bin_edges = np.unique(bin_edges)
actual_bins = len(bin_edges) - 1
if actual_bins < int(0.80 * n_bins):
    print(
        f"_build_bins: requested {n_bins} bins but only {actual_bins} unique edges "
        f"(np.unique collapsed {n_bins - actual_bins} duplicates). "
        f"Consider raising n_bins or investigating path distribution."
    )
n_bins = actual_bins
```

No change to the binning logic itself — `np.unique` is correct. The warning surfaces the
condition so it is visible in logs rather than silent.

---

---

## 🟢 8. Remove overreaching "Knightian uncertainty" language from BEP comment and tooltip

**Source**: Suggestion 7 (notation concern)
**Files**: `hedge.py` (~line 257), `static/app.js` (BEP tooltip)
**Affects scoring**: No — comment and tooltip only

### Problem
The comment in `score_puts()` attributes BEP to a "worst-case (Knightian uncertainty)
indifference condition." The Desmettre et al. paper's Knightian framework treats crash
*timing* as having no probability distribution at all, and derives optimal strategies via
BSDEs and min-max robust control. The BEP formula is not a result from that framework — it
is a plain expected-value break-even condition:

  BEP × E[payoff|crash] × roth_mult = cost
  → BEP = cost / (E[payoff|crash] × roth_mult) = p_crash / crash_efficiency

The legitimate connection to the paper is that CrEff (the denominator's driver) is a
useful metric under model uncertainty because it conditions on crash occurrence rather than
depending on P(crash) — a softer robustness argument. The BEP formula itself does not
derive from the paper's formalism.

### Fix

**`hedge.py`**, replace the comment block before `df["break_even_p"]`:

```python
# Break-even crash probability: the minimum P(crash) at which this put has
# positive expected value from crash paths alone.
# Derivation: at break-even, BEP × E[payoff|crash] × roth_mult = cost
# → BEP = cost / (E[payoff|crash] × roth_mult) = p_crash / crash_efficiency
# Lower BEP → put pays off under crash scenarios even if crash is unlikely.
```

**`static/app.js`**, BEP column tooltip: remove the sentence attributing BEP to
"portfolio crash-optimization theory" / "indifference probability" framing. Replace with
a plain description: "The minimum crash probability at which this put has positive
expected value from crash scenarios alone. Formula: BEP = P(crash) / CrEff."

---

---

## 🟡 9. Clarify "Market" column label: terminal distribution only, GARCH path dynamics assumed

**Source**: Suggestion 8 (main theoretical critique)
**Files**: `static/app.js`, optionally `app.py` response metadata, docs
**Affects scoring**: No — display only

### Problem
`_compute_bl_market_running_min` reweights GARCH paths so the *terminal marginal*
distribution matches the B-L risk-neutral density, then reads running-minimum probabilities
from the reweighted paths. This is correct as importance sampling for the terminal
distribution, but the likelihood ratio depends only on each path's terminal value S_T.
Two paths with identical S_T but different histories (one crashed -40% and recovered,
one drifted smoothly) get identical weights. The conditional distribution of
(running_min | terminal_value) is inherited entirely from the GARCH simulation.

This is the best available heuristic — doing better requires barrier-option prices or a
calibrated stochastic-vol model with analytic drawdown distributions, neither of which
are available from the CBOE chain. But the "Market" label overstates what the data can
support.

### Fix
Update UI labels and/or tooltip copy to make the assumption explicit:

- **Column label**: "Market" → "Market (terminal)" or keep "Market" with a tooltip
- **Tooltip / info box**: Add a note such as:
  "Derived from the B-L risk-neutral terminal density at the nearest expiry.
   Running-minimum probabilities are computed under the reweighted GARCH paths,
   so path dynamics (crash timing, recovery shape) still reflect the GARCH model —
   only the terminal distribution is anchored to market option prices."

No code change to the calculation itself.

---

## 🟢 10. Smooth B-L density in log-strike space, not index space

**Source**: Suggestion 8 (specific implementation concern 2); also flagged in peer review §9
**Files**: `app.py`, `_bl_fd_density()` call sites in `_compute_bl_market_running_min()`
and `_compute_bl_surface_slices()`
**Affects scoring**: No — affects the "Market" column visualization and running-min
reweighting density shape

### Problem
`gaussian_filter1d(all_ds, sigma=2.0)` smooths the B-L density array with a kernel of
2 adjacent *strike index* positions. SPX strikes are non-uniformly spaced (tighter near
ATM, wider OTM), so sigma=2 corresponds to:
- ~$10–20 of smoothing near ATM
- ~$50–100 of smoothing in the deep OTM tail

For a hedging tool focused on deep OTM puts, this systematically flattens the left tail
density more than the center — the region of greatest interest is smoothed most aggressively.

### Fix
Interpolate the B-L finite-difference density onto a uniform log-strike grid before
smoothing, then interpolate back to the original strike points:

```python
# Replace current smoothing block with:
from scipy.interpolate import interp1d as _interp1d

log_ks = np.log(all_ks)
uniform_log_ks = np.linspace(log_ks[0], log_ks[-1], 400)
to_uniform   = _interp1d(log_ks, all_ds, kind='linear', bounds_error=False, fill_value=0.0)
uniform_ds   = gaussian_filter1d(np.maximum(to_uniform(uniform_log_ks), 0.0), sigma=4.0)
from_uniform = _interp1d(uniform_log_ks, uniform_ds, kind='linear', bounds_error=False, fill_value=0.0)
all_ds = np.maximum(from_uniform(log_ks), 0.0)
```

(Sigma doubles approximately because the log-strike grid is ~2× denser than the original
strike grid in the tails. Tune to taste.) This gives translation-invariant smoothing in
log-strike space, consistent with how option pricing models are typically specified.

---

## 🟢 11. Improve GARCH tail density estimation for likelihood-ratio reweighting

**Source**: Suggestion 8 (specific implementation concern 1)
**Files**: `app.py`, `_compute_bl_market_running_min()`, lines 525–534
**Affects scoring**: No — affects stability of the "Market" running-min probabilities

### Problem
The GARCH prior terminal density is estimated via a uniform-width histogram with `n_bins=200`
bins between the 0.2th and 99.8th percentile of S_T. In the tails (crash paths), the bin
density `f_garch_bin = bin_mass / bin_width` is noisy because few paths fall in each tail
bin. The resulting likelihood ratio `f_bl / f_garch_path` is unstable in exactly the left
tail region where crash-put payoffs concentrate.

Note: the instability is bounded — B-L density is zero below the lowest listed put strike,
so `lr=0` there regardless. The concern applies to the region just above the lowest listed
strike where both densities are small and potentially mismatched.

### Fix
Replace the uniform histogram with a Gaussian KDE for the GARCH prior density:

```python
from scipy.stats import gaussian_kde

kde = gaussian_kde(S_T, weights=prior_pw, bw_method='scott')
f_garch_path = np.maximum(kde(S_T), 1e-300)   # (n_paths,) — direct per-path evaluation
```

This gives a smooth, per-path density estimate without binning artifacts, at the cost of
O(n_paths²) KDE evaluation — which is prohibitive at 100K paths. Use a subsample or a
fast approximate KDE (e.g., evaluate on a fine grid and interpolate):

```python
grid = np.linspace(float(np.percentile(S_T, 0.1)), float(np.percentile(S_T, 99.9)), 2000)
kde_grid = np.maximum(kde(grid), 1e-300)
from scipy.interpolate import interp1d as _interp1d
f_garch_path = np.maximum(
    _interp1d(grid, kde_grid, kind='linear', bounds_error=False, fill_value=1e-300)(S_T),
    1e-300,
)
```

The `bw_method='scott'` bandwidth is a reasonable default; `'silverman'` is an alternative.

---

*Next items to be added as review continues.*
