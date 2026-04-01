"""
Tests for Entropy Pooling (_entropy_pool).

Focus areas:
  1. Constraint satisfaction — posterior CDF matches imposed view at each bucket.
  2. Monotone views — more bearish view → more weight on crash paths.
  3. Sign regression — the dual EP sign bug (crash paths should be UPWEIGHTED,
     not downweighted, when a bearish view is imposed).
  4. Confidence blending — confidence=0 → posterior = prior; confidence=1 → full tilt.
  5. Weights validity — non-negative, sum to 1.
"""

import numpy as np
import pytest
from spx_model import _entropy_pool, _compute_path_weights
from tests.conftest import SPOT, N_CRASH, N_BULL, N_PATHS, CRASH_LO, CRASH_HI


# ---------------------------------------------------------------------------
# Helper: compute P(running_min < K) from a set of path weights
# ---------------------------------------------------------------------------

def _p_below(cache, path_weights, threshold):
    mask = cache.running_min < threshold
    return float(np.sum(path_weights[mask]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEntropyPooling:

    def test_prior_weights_uniform(self, synthetic_cache):
        """Prior bin weights should be uniform (1/n_bins each)."""
        cache, _, _ = synthetic_cache
        pw = cache.prior_weights
        assert np.allclose(pw, pw[0]), "prior weights should be uniform"
        assert abs(pw.sum() - 1.0) < 1e-9

    def test_prior_path_weights_sum_to_one(self, synthetic_cache):
        """Per-path prior weights must sum to 1."""
        cache, _, _ = synthetic_cache
        ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)
        assert abs(ppw.sum() - 1.0) < 1e-6

    def test_constraint_satisfaction_single_bucket(self, synthetic_cache):
        """
        After EP, P(running_min < threshold) should match the imposed view
        to within 1 percentage point.
        """
        cache, crash_levels, _ = synthetic_cache

        threshold = 4_000.0
        # Prior P(running_min < 4000): fraction of crash paths with level < 4000
        # crash_levels ~ Uniform[3000, 4500], so P = (4000-3000)/(4500-3000) × 0.5
        # ≈ 0.333
        prior_ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)
        prior_p = _p_below(cache, prior_ppw, threshold)

        target = min(prior_p * 2.5, 0.80)  # push it up significantly
        view_buckets = [(threshold, target)]

        post_bw = _entropy_pool(cache.prior_weights, cache.bin_edges, view_buckets, confidence=1.0)
        post_ppw = _compute_path_weights(post_bw, cache.bin_assignments, cache.n_paths)
        post_p = _p_below(cache, post_ppw, threshold)

        assert abs(post_p - target) < 0.01, (
            f"EP constraint not satisfied: target={target:.3f}, got={post_p:.3f}"
        )

    def test_constraint_satisfaction_multiple_buckets(self, synthetic_cache):
        """EP should satisfy all imposed CDF constraints simultaneously."""
        cache, _, _ = synthetic_cache

        prior_ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)

        # Two constraints at different drawdown levels
        t1, t2 = 3_800.0, 4_200.0
        p1_prior = _p_below(cache, prior_ppw, t1)
        p2_prior = _p_below(cache, prior_ppw, t2)

        target1 = min(p1_prior * 2.0, 0.60)
        target2 = min(p2_prior * 1.8, 0.75)
        assert target1 < target2, "CDF must be monotone"

        view_buckets = [(t1, target1), (t2, target2)]
        post_bw  = _entropy_pool(cache.prior_weights, cache.bin_edges, view_buckets, confidence=1.0)
        post_ppw = _compute_path_weights(post_bw, cache.bin_assignments, cache.n_paths)

        assert abs(_p_below(cache, post_ppw, t1) - target1) < 0.015
        assert abs(_p_below(cache, post_ppw, t2) - target2) < 0.015

    def test_ep_upweights_crash_paths_sign_regression(self, synthetic_cache):
        """
        REGRESSION: The dual EP sign bug caused crash paths to be DOWNWEIGHTED
        when a bearish view was imposed.  This test catches that inversion.

        When we impose a view that P(running_min < threshold) is HIGHER than
        the prior, the total weight on crash paths must INCREASE.
        """
        cache, _, _ = synthetic_cache

        prior_ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)
        threshold = 4_000.0
        prior_crash_weight = float(prior_ppw[:N_CRASH].sum())   # weight on crash half

        # Impose bearish view: more probability mass below threshold
        target = min(_p_below(cache, prior_ppw, threshold) * 2.5, 0.80)
        view_buckets = [(threshold, target)]

        post_bw  = _entropy_pool(cache.prior_weights, cache.bin_edges, view_buckets, confidence=1.0)
        post_ppw = _compute_path_weights(post_bw, cache.bin_assignments, cache.n_paths)
        post_crash_weight = float(post_ppw[:N_CRASH].sum())

        assert post_crash_weight > prior_crash_weight, (
            f"Bearish EP view should INCREASE crash-path weight, "
            f"but prior={prior_crash_weight:.3f} → post={post_crash_weight:.3f}"
        )

    def test_confidence_zero_returns_prior(self, synthetic_cache):
        """confidence=0 means 'ignore views entirely' → posterior == prior."""
        cache, _, _ = synthetic_cache
        threshold = 4_000.0
        target = 0.70
        view_buckets = [(threshold, target)]

        post_bw = _entropy_pool(cache.prior_weights, cache.bin_edges, view_buckets, confidence=0.0)
        assert np.allclose(post_bw, cache.prior_weights, atol=1e-10), (
            "confidence=0 should return the prior unchanged"
        )

    def test_confidence_blend(self, synthetic_cache):
        """
        confidence=0.5 should produce a posterior halfway between prior and full-tilt.
        P(running_min < threshold) should be between prior_p and target.
        """
        cache, _, _ = synthetic_cache
        prior_ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)
        threshold = 4_000.0
        prior_p = _p_below(cache, prior_ppw, threshold)
        target  = min(prior_p * 2.5, 0.80)
        view_buckets = [(threshold, target)]

        post_bw  = _entropy_pool(cache.prior_weights, cache.bin_edges, view_buckets, confidence=0.5)
        post_ppw = _compute_path_weights(post_bw, cache.bin_assignments, cache.n_paths)
        post_p   = _p_below(cache, post_ppw, threshold)

        assert prior_p < post_p < target, (
            f"confidence=0.5 should give intermediate tilt: "
            f"prior={prior_p:.3f}, post={post_p:.3f}, target={target:.3f}"
        )

    def test_posterior_weights_valid(self, synthetic_cache):
        """Posterior bin weights must be non-negative and sum to 1."""
        cache, _, _ = synthetic_cache
        view_buckets = [(4_000.0, 0.65)]
        post_bw = _entropy_pool(cache.prior_weights, cache.bin_edges, view_buckets, confidence=1.0)

        assert (post_bw >= 0).all(), "posterior weights must be non-negative"
        assert abs(post_bw.sum() - 1.0) < 1e-8, "posterior weights must sum to 1"

    def test_posterior_path_weights_valid(self, synthetic_cache):
        """Per-path posterior weights must be non-negative and sum to 1."""
        cache, _, _ = synthetic_cache
        view_buckets = [(4_000.0, 0.65)]
        post_bw  = _entropy_pool(cache.prior_weights, cache.bin_edges, view_buckets, confidence=1.0)
        post_ppw = _compute_path_weights(post_bw, cache.bin_assignments, cache.n_paths)

        assert (post_ppw >= 0).all()
        assert abs(post_ppw.sum() - 1.0) < 1e-6
