"""
Shared test fixtures.

The synthetic_cache fixture bypasses the ~15s GARCH fit so unit tests
run in <1s.  Path structure is deliberately simple so expected values
can be computed analytically and checked exactly (within float tolerance).

Layout:
  - n_crash paths: each path crashes to a level drawn from Uniform[3000, 4500]
    at week 5 and stays flat for all subsequent weeks.
  - n_bull  paths: each path rises to a level drawn from Uniform[5000, 7000]
    at week 10 and stays flat.

The crash half has running_min = crash_level (in [3000, 4500]).
The bull half has running_min = SPOT = 5000 (they never go below start).

This gives clean analytic answers for:
  - P(running_min < K)  for any K
  - E[max(K - S_T, 0)]  at any terminal week after week 10
"""

import types
import numpy as np
import pytest


SPOT     = 5_000.0
N_PATHS  = 10_000       # keep small so tests are fast
N_CRASH  = N_PATHS // 2
N_BULL   = N_PATHS - N_CRASH
N_WEEKS  = 104          # 2 years weekly (~104 weeks)
CRASH_LO = 3_000.0
CRASH_HI = 4_500.0
BULL_LO  = 5_000.0
BULL_HI  = 7_000.0


def _build_synthetic_paths(rng):
    """Return (paths, running_min) arrays for the synthetic fixture."""
    crash_levels = rng.uniform(CRASH_LO, CRASH_HI, N_CRASH)
    bull_levels  = rng.uniform(BULL_LO,  BULL_HI,  N_BULL)

    # crash paths: start at SPOT, drop to crash_level at week 5, stay
    crash_paths = np.empty((N_CRASH, N_WEEKS))
    crash_paths[:, :5]  = SPOT
    crash_paths[:, 5:]  = crash_levels[:, None]

    # bull paths: start at SPOT, rise to bull_level at week 10, stay
    bull_paths = np.empty((N_BULL, N_WEEKS))
    bull_paths[:, :10] = SPOT
    bull_paths[:, 10:] = bull_levels[:, None]

    paths = np.vstack([crash_paths, bull_paths])   # (N_PATHS, N_WEEKS)
    running_min = paths.min(axis=1)                # crash half: crash_level; bull half: SPOT
    return paths, running_min, crash_levels, bull_levels


@pytest.fixture(scope="session")
def synthetic_cache():
    """
    Session-scoped: built once, shared across all tests in the session.
    Returns (cache_ns, crash_levels, bull_levels) where cache_ns mimics
    the interface of GARCHPathCache.
    """
    from spx_model import _build_bins, _compute_path_weights

    rng = np.random.default_rng(42)
    paths, running_min, crash_levels, bull_levels = _build_synthetic_paths(rng)

    bin_edges, _, prior_weights, bin_assignments = _build_bins(running_min, n_bins=200)

    cache = types.SimpleNamespace(
        current_price   = SPOT,
        n_paths         = N_PATHS,
        paths           = paths,
        running_min     = running_min,
        bin_edges       = bin_edges,
        prior_weights   = prior_weights,
        bin_assignments = bin_assignments,
    )
    return cache, crash_levels, bull_levels
