"""
Tests for GJREntropyModel.expected_payoffs() and terminal_tail_probs().

Uses the synthetic cache so no GARCH fit is needed.

Key analytic results for the synthetic fixture
(terminal week ≥ 10, where all paths have reached their steady-state levels):

  crash paths: S_T = crash_level ~ Uniform[3000, 4500],   weight ≈ 0.5
  bull  paths: S_T = bull_level  ~ Uniform[5000, 7000],   weight ≈ 0.5

For a put at strike K (where 3000 ≤ K ≤ 4500):
  E[Pay] = Σ w_i × max(K - S_T_i, 0) × 100
         ≈ 0.5 × E[max(K - U[3000,4500], 0)] × 100
         = 0.5 × (K - 3000)² / (2 × 1500) × 100   (integration of uniform)

For K = 4000:
  E[Pay] ≈ 0.5 × (4000 - 3000)² / 3000 × 100
          = 0.5 × 1_000_000 / 3000 × 100 ≈ 16_667
"""

import numpy as np
import pytest
from spx_model import GJREntropyModel, _compute_path_weights
from tests.conftest import SPOT, N_CRASH, N_BULL, N_PATHS, CRASH_LO, CRASH_HI


def _make_model(synthetic_cache, view_buckets=None):
    cache, _, _ = synthetic_cache
    return GJREntropyModel(
        current_price   = SPOT,
        view_buckets    = view_buckets,
        confidence_level = 1.0,
        _cache          = cache,
    )


class TestExpectedPayoffs:

    # ------------------------------------------------------------------
    # DTE / week index conversion
    # ------------------------------------------------------------------

    def test_dte_conversion_one_year(self, synthetic_cache):
        """
        DTE=365 calendar days → trading_dte ≈ 252 → week_idx ≈ 50.
        Payoffs at week 50 (all paths steady-state) should be > 0 for a
        moderately OTM put.
        """
        model = _make_model(synthetic_cache)
        strikes = np.array([4_000.0])
        total_e, _ = model.expected_payoffs(strikes, dte=365)
        assert total_e[0] > 0, "E[Pay] should be positive for an ITM crash put"

    def test_dte_conversion_short(self, synthetic_cache):
        """
        DTE=25 calendar days → week_idx ≈ 4 (before paths reach steady state).
        At week 4, crash paths are still at SPOT (5000), so a put at 4000 pays 0.
        """
        model = _make_model(synthetic_cache)
        # crash paths don't crash until week 5, so at DTE≈4 weeks all at SPOT
        strikes = np.array([4_000.0])
        total_e, _ = model.expected_payoffs(strikes, dte=25)  # ~4 weeks
        # bull paths are also at SPOT at week 4, so nothing is ITM
        assert total_e[0] == pytest.approx(0.0, abs=100), (
            "Before paths diverge, a sub-SPOT put should have ~0 payoff"
        )

    # ------------------------------------------------------------------
    # Analytic payoff checks
    # ------------------------------------------------------------------

    def test_deep_otm_put_below_all_crash_levels(self, synthetic_cache):
        """A put BELOW the minimum possible crash level (CRASH_LO=3000) pays nothing."""
        model = _make_model(synthetic_cache)
        strikes = np.array([2_500.0])   # below CRASH_LO=3000, all paths OTM
        total_e, _ = model.expected_payoffs(strikes, dte=365)
        assert total_e[0] == pytest.approx(0.0, abs=1.0)

    def test_below_all_crash_levels_zero_payoff(self, synthetic_cache):
        """Strike below the minimum possible crash level → E[Pay] ≈ 0."""
        model = _make_model(synthetic_cache)
        strikes = np.array([2_000.0])   # below CRASH_LO=3000
        total_e, _ = model.expected_payoffs(strikes, dte=365)
        assert total_e[0] == pytest.approx(0.0, abs=1.0)

    def test_epay_matches_direct_dot_product(self, synthetic_cache):
        """
        expected_payoffs() should equal the direct weighted dot product
        Σ w_i × max(K - S_T_i, 0) × 100  computed from the same path weights.

        This is the core invariant — it doesn't depend on the bin structure.
        Tolerance ±0.1% (floating-point accumulation only).
        """
        cache, _, _ = synthetic_cache
        model = _make_model(synthetic_cache)

        week_idx = min((252 - 1) // 5, cache.paths.shape[1] - 1)
        S_T = cache.paths[:, week_idx]
        w   = model._path_weights   # access internal weights for cross-check

        strikes = np.array([3_500.0, 4_000.0, 4_250.0])
        total_e, _ = model.expected_payoffs(strikes, dte=365)

        for i, K in enumerate(strikes):
            direct = float(np.dot(w, np.maximum(K - S_T, 0.0) * 100.0))
            assert abs(total_e[i] - direct) / max(direct, 1.0) < 0.001, (
                f"K={K}: expected_payoffs={total_e[i]:.2f}, direct={direct:.2f}"
            )

    def test_crash_epay_subset_of_total(self, synthetic_cache):
        """crash_e must be ≤ total_e for all strikes."""
        model = _make_model(synthetic_cache)
        strikes = np.array([3_500.0, 4_000.0, 4_500.0, 5_000.0])
        total_e, crash_e = model.expected_payoffs(strikes, dte=365)
        assert (crash_e <= total_e + 1e-6).all(), "crash E[Pay] ≤ total E[Pay]"

    def test_payoffs_monotone_in_strike(self, synthetic_cache):
        """Higher strike → higher E[Pay] (put intrinsic increases with K)."""
        model = _make_model(synthetic_cache)
        strikes = np.array([3_500.0, 3_750.0, 4_000.0, 4_250.0, 4_500.0])
        total_e, _ = model.expected_payoffs(strikes, dte=365)
        assert np.all(np.diff(total_e) >= 0), (
            "E[Pay] should be non-decreasing in strike"
        )

    def test_ep_bearish_view_increases_epay(self, synthetic_cache):
        """
        A bearish EP view (upweighting crash paths) should increase E[Pay]
        for a crash put relative to the prior model.
        """
        cache, _, _ = synthetic_cache
        prior_ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)
        threshold = 4_000.0
        prior_p = float(np.sum(prior_ppw[cache.running_min < threshold]))
        target = min(prior_p * 2.5, 0.80)

        model_prior   = _make_model(synthetic_cache, view_buckets=None)
        model_bearish = _make_model(synthetic_cache, view_buckets=[(threshold, target)])

        strikes = np.array([4_000.0])
        total_prior, _   = model_prior.expected_payoffs(strikes, dte=365)
        total_bearish, _ = model_bearish.expected_payoffs(strikes, dte=365)

        assert total_bearish[0] > total_prior[0], (
            f"Bearish EP view should increase E[Pay]: "
            f"prior={total_prior[0]:.0f}, bearish={total_bearish[0]:.0f}"
        )


class TestTerminalTailProbs:

    def test_prior_probabilities_match_path_fraction(self, synthetic_cache):
        """
        terminal_tail_probs prior should equal the raw fraction of paths
        below each threshold (computed directly from path weights).
        """
        cache, _, _ = synthetic_cache
        model = _make_model(synthetic_cache, view_buckets=None)

        # At DTE=365 (week≈50), crash half is at crash_level, bull half at bull_level
        thresholds = [3_500.0, 4_000.0, 4_500.0]
        result = model.terminal_tail_probs(thresholds, query_dte=252)

        prior_ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)
        week_idx = min((252 - 1) // 5, cache.paths.shape[1] - 1)
        S_T = cache.paths[:, week_idx]

        for i, thr in enumerate(thresholds):
            expected = float(np.sum(prior_ppw[S_T < thr]))
            assert abs(result["prior"][i] - expected) < 1e-4, (
                f"threshold={thr}: prior={result['prior'][i]:.4f}, direct={expected:.4f}"
            )

    def test_posterior_more_bearish_than_prior(self, synthetic_cache):
        """After a bearish EP view, posterior tail probs should exceed prior."""
        cache, _, _ = synthetic_cache
        prior_ppw = _compute_path_weights(cache.prior_weights, cache.bin_assignments, cache.n_paths)
        threshold = 4_000.0
        prior_p = float(np.sum(prior_ppw[cache.running_min < threshold]))
        target  = min(prior_p * 2.5, 0.80)

        model = _make_model(synthetic_cache, view_buckets=[(threshold, target)])
        result = model.terminal_tail_probs([4_000.0], query_dte=252)

        assert result["posterior"][0] > result["prior"][0], (
            "Bearish EP view should make terminal posterior more bearish than prior"
        )

    def test_probabilities_bounded(self, synthetic_cache):
        """All terminal probabilities must be in [0, 1]."""
        model = _make_model(synthetic_cache, view_buckets=[(4_000.0, 0.65)])
        thresholds = [2_000.0, 3_500.0, 4_000.0, 5_000.0, 8_000.0]
        result = model.terminal_tail_probs(thresholds, query_dte=252)
        for p in result["prior"] + result["posterior"]:
            assert 0.0 <= p <= 1.0
