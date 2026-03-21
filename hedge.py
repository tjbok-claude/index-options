"""
SPX put option utility ranker for $5M equity hedge.

Fetches the live CBOE SPX options chain, scores every put by expected
payoff relative to cost (EPR), and prints the top-ranked contracts.

Usage:
    python hedge.py                                # defaults
    python hedge.py --p-crash 0.65                # override crash probability
    python hedge.py --budget 50000                # tighter annual budget
    python hedge.py --horizon 12                  # 12-month horizon (months)
    python hedge.py --contracts 5                 # partial hedge
    python hedge.py --top 30                      # show 30 rows (default: 25)
    python hedge.py --save                        # save ranked output to CSV
    python hedge.py --show-all                    # include filtered-out rows
"""

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from cme import KNOWN_SYMBOLS, fetch_chain, parse_chain
from scenarios import DEFAULT_MODEL, SCENARIO_MODELS, ScenarioModel, get_model

# ---------------------------------------------------------------------------
# Parameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class Params:
    # Market / portfolio
    portfolio_value: float = 5_000_000
    beta_vti: float = 0.97
    beta_vxus_normal: float = 0.65
    beta_vxus_tail: float = 0.75
    weight_vti: float = 0.75
    weight_vxus: float = 0.25

    # Crash assumptions
    p_crash: float = 0.55          # P(≥20% SPX drawdown over horizon)
    crash_splits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.50, -0.25),   # 50% of crashes → -25% SPX
        (0.35, -0.40),   # 35% of crashes → -40% SPX
        (0.15, -0.55),   # 15% of crashes → -55% SPX
    ])

    # Non-crash scenarios (split of remaining 1-p_crash probability)
    # (label, spx_return, share_of_no_crash)
    non_crash_splits: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("bull", +0.20, 0.40),
        ("flat", +0.03, 0.35),
        ("bear", -0.12, 0.25),
    ])

    # Roth / tax
    roth_multiplier: float = 1.25

    # Budget
    annual_budget: float = 75_000
    horizon_months: int = 18

    # Sizing
    contracts: int = 0              # 0 = auto-compute from beta

    # Output
    top_n: int = 25
    save: bool = False
    show_all: bool = False

    # --- Derived ---
    @property
    def portfolio_beta(self) -> float:
        return self.weight_vti * self.beta_vti + self.weight_vxus * self.beta_vxus_normal

    @property
    def portfolio_beta_tail(self) -> float:
        return self.weight_vti * self.beta_vti + self.weight_vxus * self.beta_vxus_tail

    @property
    def total_budget(self) -> float:
        return self.annual_budget * (self.horizon_months / 12)

    def n_contracts(self, spx_spot: float) -> int:
        if self.contracts > 0:
            return self.contracts
        return max(1, round(self.portfolio_value * self.portfolio_beta / (spx_spot * 100)))


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def build_scenarios(p: Params) -> List[Tuple[str, float, float, bool]]:
    """
    Backward-compatible shim — delegates to flat_scenarios at horizon DTE.
    New code should call a ScenarioModel from scenarios.py directly.
    """
    from scenarios import flat_scenarios
    horizon_dte = int(p.horizon_months * 30.44)
    return [(s.name, s.spx_return, s.probability, s.is_crash)
            for s in flat_scenarios(p, dte=horizon_dte)]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_puts(df: pd.DataFrame, p: Params, spx_spot: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply pre-filters to the full chain.

    Returns (filtered_df, rejected_df_with_reason).
    rejected_df is only populated when p.show_all is True.
    """
    today = date.today()
    df = df.copy()

    # DTE
    df["dte"] = (df["expiry"] - today).apply(lambda d: d.days)
    # moneyness
    df["moneyness_pct"] = (df["strike"] / spx_spot - 1.0) * 100.0

    filters = [
        ("type == PUT",      df["type"] == "PUT"),
        ("bid > 0",          df["bid"].notna() & (df["bid"] > 0)),
        ("OI >= 100",        df["open_interest"].notna() & (df["open_interest"] >= 100)),
        ("DTE 180-540",      df["dte"].between(180, 540)),
        ("IV 0.05-1.50",     df["iv"].notna() & df["iv"].between(0.05, 1.50)),
        ("strike 55-98% spot", df["strike"].between(0.55 * spx_spot, 0.98 * spx_spot)),
        ("spread <= 25%",    _spread_ok(df)),
    ]

    mask = pd.Series(True, index=df.index)
    reject_reasons = pd.Series("", index=df.index)

    for reason, cond in filters:
        newly_rejected = mask & ~cond
        reject_reasons[newly_rejected] = reason
        mask = mask & cond

    passed = df[mask].copy()
    rejected = df[~mask].copy()
    rejected["filter_reason"] = reject_reasons[~mask]

    return passed, rejected


def _spread_ok(df: pd.DataFrame) -> pd.Series:
    mid = (df["bid"] + df["ask"]) / 2
    spread_pct = (df["ask"] - df["bid"]) / mid.replace(0, np.nan)
    return spread_pct <= 0.25


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_puts(
    df: pd.DataFrame,
    p: Params,
    spx_spot: float,
    model: "ScenarioModel | None" = None,
) -> pd.DataFrame:
    """
    Add payoff, utility, and sizing columns.

    model — a ScenarioModel callable from scenarios.py.
            Defaults to the DEFAULT_MODEL (currently 'survival').
            Pass any callable matching (params, dte: int) -> list[Scenario].
    """
    if model is None:
        model = SCENARIO_MODELS[DEFAULT_MODEL]

    df = df.copy()
    n = p.n_contracts(spx_spot)

    mid = (df["bid"] + df["ask"]) / 2
    df["mid"] = mid
    df["spread_pct"] = ((df["ask"] - df["bid"]) / mid * 100).round(1)
    df["cost_1c"] = mid * 100

    # ------------------------------------------------------------------
    # Step 1: compute gross payoffs for each scenario (vectorized, DTE-
    #         independent — only strike and terminal SPX matter).
    #
    # We call the model once at an arbitrary DTE just to learn the scenario
    # names and returns; probabilities from this call are NOT used here.
    # ------------------------------------------------------------------
    sample_dte = int(df["dte"].iloc[0])
    sample_scenarios = model(p, sample_dte)

    for s in sample_scenarios:
        spx_terminal = spx_spot * (1 + s.spx_return)
        payoff = (df["strike"] - spx_terminal).clip(lower=0) * 100
        df[f"payoff_{s.name}_1c"] = payoff.round(2)

    # ------------------------------------------------------------------
    # Step 2: compute probability-weighted expected payoffs.
    #
    # Call the model once per unique DTE so each row gets the correct
    # time-scaled probabilities for its expiry.
    # ------------------------------------------------------------------
    e_payoff_roth  = pd.Series(0.0, index=df.index)
    crash_e_payoff = pd.Series(0.0, index=df.index)

    for dte_val in df["dte"].unique():
        mask = df["dte"] == dte_val
        scenarios = model(p, int(dte_val))
        for s in scenarios:
            payoff = df.loc[mask, f"payoff_{s.name}_1c"]
            weighted = s.probability * payoff
            e_payoff_roth[mask]  += weighted
            if s.is_crash:
                crash_e_payoff[mask] += weighted

    e_payoff_roth *= p.roth_multiplier
    df["e_payoff_roth_1c"] = e_payoff_roth.round(2)
    df["e_net_1c"] = (e_payoff_roth - df["cost_1c"]).round(2)
    df["EPR"] = (e_payoff_roth / df["cost_1c"]).replace([np.inf, -np.inf], np.nan).round(4)

    # Crash efficiency (crash scenarios only, Roth-adjusted)
    df["crash_efficiency"] = (
        (crash_e_payoff * p.roth_multiplier) / df["cost_1c"]
    ).replace([np.inf, -np.inf], np.nan).round(4)

    # Sizing
    df["cost_Nc"] = (df["cost_1c"] * n).round(2)
    df["affordable"] = df["cost_Nc"] <= p.total_budget

    # Annual cost as % of portfolio
    df["annual_cost_pct"] = (
        df["cost_1c"] * n / p.portfolio_value * (365 / df["dte"]) * 100
    ).round(3)

    # Theo vs mid
    df["theo_vs_mid_pct"] = np.where(
        df["theo"].notna() & (df["theo"] > 0),
        ((mid - df["theo"]) / df["theo"] * 100).round(1),
        np.nan,
    )

    # Theta: CBOE theta is per day already (or per year divided by 365)
    # CBOE reports theta as daily $ per share, so theta_daily = theta * 100 per contract
    df["theta_daily"] = (df["theta"] * 100).round(2)

    return df


# ---------------------------------------------------------------------------
# Ranking and output
# ---------------------------------------------------------------------------

_DISPLAY_COLS = [
    "expiry", "session", "dte", "strike", "moneyness_pct",
    "mid", "spread_pct", "iv", "delta", "theta_daily", "open_interest", "volume",
    "cost_1c", "cost_Nc", "affordable",
    "payoff_crash_25pct_1c", "payoff_crash_40pct_1c", "payoff_crash_55pct_1c",
    "e_payoff_roth_1c", "e_net_1c", "EPR", "crash_efficiency",
    "annual_cost_pct", "theo_vs_mid_pct",
]


def rank_output(df: pd.DataFrame, p: Params, spx_spot: float, rejected: pd.DataFrame) -> pd.DataFrame:
    """Sort by EPR desc, select display columns, optionally save."""
    ranked = df.sort_values("EPR", ascending=False).reset_index(drop=True)

    # Select only columns that exist
    cols = [c for c in _DISPLAY_COLS if c in ranked.columns]
    ranked = ranked[cols]

    n = p.n_contracts(spx_spot)
    print(f"\n--- SPX Put Utility Ranker ---")
    print(f"  SPX spot : {spx_spot:,.2f}")
    print(f"  P(crash) : {p.p_crash:.0%}")
    print(f"  N        : {n} contracts  (${p.portfolio_value:,.0f} x beta={p.portfolio_beta:.3f})")
    print(f"  Budget   : ${p.annual_budget:,.0f}/yr  =>  ${p.total_budget:,.0f} over {p.horizon_months}mo")
    print(f"  Roth 1x  = taxable {p.roth_multiplier:.2f}x")
    print(f"  Passed filter: {len(df)} puts  |  Filtered out: {len(rejected)} rows")
    print()

    top = ranked.head(p.top_n)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", p.top_n + 5)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(top.to_string(index=True))

    if p.show_all and not rejected.empty:
        print(f"\n--- Filtered-out rows ({len(rejected)}) ---")
        rej_cols = ["expiry", "session", "dte", "strike", "bid", "ask", "iv",
                    "open_interest", "filter_reason"]
        rej_cols = [c for c in rej_cols if c in rejected.columns]
        print(rejected[rej_cols].to_string(index=False))

    if p.save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"SPX_hedge_ranked_{ts}.csv"
        ranked.to_csv(fname, index=True)
        print(f"\nSaved {len(ranked)} rows to {fname}")

    return ranked


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rank SPX puts by utility for a $5M equity hedge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--p-crash", type=float, default=0.55,
                        help="Probability of a ≥20%% SPX drawdown over horizon")
    parser.add_argument("--budget", type=float, default=75_000,
                        help="Annual premium budget in USD")
    parser.add_argument("--horizon", type=int, default=18,
                        help="Hold horizon in months (also sets DTE search range max)")
    parser.add_argument("--contracts", type=int, default=0,
                        help="Number of contracts (0 = auto from beta)")
    parser.add_argument("--top", type=int, default=25,
                        help="Number of rows to display")
    parser.add_argument("--save", action="store_true",
                        help="Save ranked output to CSV")
    parser.add_argument("--show-all", action="store_true",
                        help="Also show filtered-out rows with reason")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=list(SCENARIO_MODELS),
                        help="Scenario probability model")
    args = parser.parse_args()

    p = Params(
        p_crash=args.p_crash,
        annual_budget=args.budget,
        horizon_months=args.horizon,
        contracts=args.contracts,
        top_n=args.top,
        save=args.save,
        show_all=args.show_all,
    )

    print("Fetching SPX options chain from CBOE (delayed ~15 min)...")
    try:
        raw = fetch_chain(KNOWN_SYMBOLS["SPX"])
    except Exception as e:
        print(f"Failed to fetch chain: {e}")
        sys.exit(1)

    df, underlying = parse_chain(raw, base_symbol="SPX")

    if df.empty:
        print("No data parsed from CBOE response.")
        sys.exit(1)

    spx_spot = underlying["current_price"]
    print(f"SPX  price={spx_spot}  IV30={underlying['iv30']:.2f}  as-of={underlying['timestamp']}")

    filtered, rejected = filter_puts(df, p, spx_spot)

    if filtered.empty:
        print("No puts passed filters. Try --show-all to see what was filtered.")
        if p.show_all:
            rej_cols = ["expiry", "session", "dte", "strike", "bid", "ask", "iv",
                        "open_interest", "filter_reason"]
            rej_cols = [c for c in rej_cols if c in rejected.columns]
            print(rejected[rej_cols].to_string(index=False))
        sys.exit(1)

    model = get_model(args.model)
    scored = score_puts(filtered, p, spx_spot, model=model)
    rank_output(scored, p, spx_spot, rejected)


if __name__ == "__main__":
    main()
