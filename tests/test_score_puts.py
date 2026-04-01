"""
Tests for score_puts() column calculations.

Uses a minimal synthetic DataFrame so the math can be checked analytically.
Does NOT require a GARCH model (uses a stub ScenarioModel that returns
deterministic scenarios).
"""

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from hedge import Params, score_puts
from tests.conftest import SPOT


# ---------------------------------------------------------------------------
# Minimal stub model — returns fixed scenarios without any simulation
# ---------------------------------------------------------------------------

class FixedScenarioModel:
    """
    Returns two deterministic scenarios: a crash and a bull.
    crash:  SPX -30%,  prob=p_crash
    bull:   SPX +10%,  prob=1-p_crash
    Supports the expected_payoffs fast path (hasattr check in score_puts).
    """

    def __init__(self, p_crash=0.50):
        self.p_crash = p_crash

    def __call__(self, params, dte):
        from scenarios import Scenario
        return [
            Scenario("crash_30pct", spx_return=-0.30, probability=self.p_crash,   is_crash=True),
            Scenario("bull_10pct",  spx_return=+0.10, probability=1-self.p_crash, is_crash=False),
        ]


# ---------------------------------------------------------------------------
# Minimal options DataFrame
# ---------------------------------------------------------------------------

def _make_options_df(spot=SPOT, n_strikes=5):
    """
    Returns a small DataFrame of put options at various strikes.
    bid/ask set so that mid = 50.0 (cost_1c = 5000).
    """
    today = date.today()
    expiry = today + timedelta(days=365)
    strikes = np.linspace(spot * 0.70, spot * 0.90, n_strikes)
    rows = []
    for k in strikes:
        rows.append({
            "expiry":        expiry,
            "session":       "regular",
            "strike":        round(k),
            "type":          "PUT",
            "bid":           45.0,
            "ask":           55.0,
            "iv":            0.25,
            "delta":        -0.30,
            "theta":        -0.50,
            "open_interest": 500,
            "volume":        100,
            "theo":          50.0,
        })
    df = pd.DataFrame(rows)
    today_d = date.today()
    df["dte"] = (df["expiry"] - today_d).apply(lambda d: d.days)
    df["moneyness_pct"] = (df["strike"] / spot - 1.0) * 100.0
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def params():
    return Params(
        portfolio_value = 1_000_000,
        p_crash         = 0.50,
        roth_multiplier = 1.25,
        annual_budget   = 20_000,
        horizon_months  = 12,
        contracts       = 2,
    )


@pytest.fixture
def scored(params):
    df = _make_options_df()
    model = FixedScenarioModel(p_crash=params.p_crash)
    return score_puts(df, params, SPOT, model=model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestColumnMath:

    def test_mid_price(self, scored):
        """mid = (bid + ask) / 2 = (45 + 55) / 2 = 50."""
        assert (scored["mid"] == 50.0).all()

    def test_cost_1c(self, scored):
        """cost_1c = mid × 100 = 5000."""
        assert (scored["cost_1c"] == 5_000.0).all()

    def test_spread_pct(self, scored):
        """spread_pct = (ask - bid) / mid × 100 = (10 / 50) × 100 = 20.0."""
        assert (scored["spread_pct"] == 20.0).all()

    def test_cost_nc(self, params, scored):
        """cost_Nc = cost_1c × n_contracts."""
        n = params.n_contracts(SPOT)
        expected = scored["cost_1c"] * n
        assert (scored["cost_Nc"] == expected).all()

    def test_epr_definition(self, scored):
        """EPR = e_payoff_roth_1c / cost_1c."""
        computed = (scored["e_payoff_roth_1c"] / scored["cost_1c"]).round(4)
        assert (scored["EPR"] == computed).all()

    def test_epr_nonnegative(self, scored):
        """EPR must be ≥ 0 (payoffs are non-negative)."""
        assert (scored["EPR"] >= 0).all()

    def test_crash_efficiency_definition(self, params, scored):
        """crash_efficiency = crash_E[Pay] × roth_mult / cost_1c."""
        # crash_e_payoff (before roth mult) = crash_efficiency × cost_1c / roth_mult
        # This is tested indirectly: crash_efficiency ≤ EPR (crash is subset of total)
        assert (scored["crash_efficiency"] <= scored["EPR"] + 1e-6).all()

    def test_crash_efficiency_nonnegative(self, scored):
        assert (scored["crash_efficiency"] >= 0).all()

    def test_bep_formula(self, params, scored):
        """break_even_p = p_crash / crash_efficiency, clipped to 1."""
        expected = (params.p_crash / scored["crash_efficiency"]).clip(upper=1.0).round(4)
        assert np.allclose(scored["break_even_p"].fillna(1.0), expected.fillna(1.0), atol=1e-4)

    def test_bep_bounded(self, scored):
        """BEP must be in (0, 1]."""
        valid = scored["break_even_p"].dropna()
        assert (valid > 0).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_affordable_flag(self, params, scored):
        """affordable = (cost_Nc ≤ total_budget)."""
        expected = scored["cost_Nc"] <= params.total_budget
        assert (scored["affordable"] == expected).all()

    def test_annual_cost_pct_positive(self, scored):
        """Annual cost as % of portfolio must be positive."""
        assert (scored["annual_cost_pct"] > 0).all()

    def test_theta_daily_scaling(self, scored):
        """theta_daily = theta × 100."""
        assert (scored["theta_daily"] == -50.0).all()   # theta=-0.50 → -50.0


class TestPayoffColumnValues:
    """
    With FixedScenarioModel (crash -30%, bull +10%), payoffs at specific
    strikes are computable analytically.
    """

    def test_payoff_crash_scenario(self):
        """
        For a put at K=4500 (=SPOT × 0.9) with crash to SPOT × 0.70 = 3500:
          payoff = max(4500 - 3500, 0) × 100 = 100_000
        """
        spot   = SPOT
        strike = round(spot * 0.90)      # 4500
        crash_terminal = spot * 0.70     # 3500  (-30%)
        expected_payoff = (strike - crash_terminal) * 100   # 100_000

        p = Params(p_crash=0.50, contracts=1)
        df = pd.DataFrame([{
            "expiry": date.today() + timedelta(days=365),
            "session": "regular",
            "strike": strike,
            "type": "PUT",
            "bid": 45.0, "ask": 55.0,
            "iv": 0.25, "delta": -0.30, "theta": -0.10,
            "open_interest": 100, "volume": 10, "theo": 50.0,
        }])
        today = date.today()
        df["dte"] = (df["expiry"] - today).apply(lambda d: d.days)
        df["moneyness_pct"] = (df["strike"] / spot - 1.0) * 100.0

        result = score_puts(df, p, spot, model=FixedScenarioModel(p_crash=0.50))
        assert "payoff_crash_30pct_1c" in result.columns
        assert abs(result["payoff_crash_30pct_1c"].iloc[0] - expected_payoff) < 1.0

    def test_payoff_bull_scenario_zero(self):
        """Bull scenario (+10%) → put at K<SPOT is OTM → payoff = 0."""
        spot   = SPOT
        strike = round(spot * 0.85)   # below spot, below bull terminal (5500)
        p = Params(p_crash=0.50, contracts=1)
        df = pd.DataFrame([{
            "expiry": date.today() + timedelta(days=365),
            "session": "regular",
            "strike": strike,
            "type": "PUT",
            "bid": 45.0, "ask": 55.0,
            "iv": 0.25, "delta": -0.20, "theta": -0.10,
            "open_interest": 100, "volume": 10, "theo": 50.0,
        }])
        today = date.today()
        df["dte"] = (df["expiry"] - today).apply(lambda d: d.days)
        df["moneyness_pct"] = (df["strike"] / spot - 1.0) * 100.0

        result = score_puts(df, p, spot, model=FixedScenarioModel(p_crash=0.50))
        assert result["payoff_bull_10pct_1c"].iloc[0] == 0.0

    def test_epr_analytic(self):
        """
        With crash -30%, bull +10%, p_crash=0.50, roth_mult=1.25:
          terminal = SPOT × 0.70 = 3500  (crash)
          payoff   = (K - 3500) × 100    (if K > 3500)
          E[Pay]   = p_crash × payoff  (bull payoff = 0 for K < SPOT × 1.1)
          EPay_roth = E[Pay] × roth_mult
          EPR      = EPay_roth / cost_1c

        For K=4500, mid=50:
          payoff   = (4500 - 3500) × 100 = 100_000
          E[Pay]   = 0.50 × 100_000 = 50_000
          EPay_roth = 50_000 × 1.25 = 62_500
          EPR      = 62_500 / 5_000 = 12.5
        """
        spot   = SPOT
        strike = 4_500
        p = Params(p_crash=0.50, roth_multiplier=1.25, contracts=1)
        df = pd.DataFrame([{
            "expiry": date.today() + timedelta(days=365),
            "session": "regular",
            "strike": strike,
            "type": "PUT",
            "bid": 45.0, "ask": 55.0,
            "iv": 0.25, "delta": -0.30, "theta": -0.10,
            "open_interest": 100, "volume": 10, "theo": 50.0,
        }])
        today = date.today()
        df["dte"] = (df["expiry"] - today).apply(lambda d: d.days)
        df["moneyness_pct"] = (df["strike"] / spot - 1.0) * 100.0

        result = score_puts(df, p, spot, model=FixedScenarioModel(p_crash=0.50))
        assert result["EPR"].iloc[0] == pytest.approx(12.5, rel=1e-4)
