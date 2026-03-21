"""
Scenario probability models for SPX option scoring.

A scenario model is any callable with signature:

    model(params, dte: int) -> list[Scenario]

where `params` is a Params-like object (needs p_crash, horizon_months,
crash_splits, non_crash_splits) and `dte` is the option's days to expiration.

The returned list must cover all scenarios with probabilities summing to 1.0.
Scenario names and SPX returns must be consistent across all DTE calls for a
given model — only probabilities may vary with DTE.

To add a new model:
    1. Write a function matching the signature above.
    2. Add it to SCENARIO_MODELS with a short key.
    That's it — hedge.py and app.py pick it up automatically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str           # e.g. "bull", "crash_25pct"
    spx_return: float   # terminal SPX return (e.g. -0.25)
    probability: float  # absolute probability for this DTE (sums to 1 across all scenarios)
    is_crash: bool      # whether this is a crash scenario


# A ScenarioModel is any callable matching this signature.
# Using a Protocol so type checkers can validate custom models.
@runtime_checkable
class ScenarioModel(Protocol):
    def __call__(self, params: Any, dte: int) -> list[Scenario]:
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _crash_name(spx_ret: float) -> str:
    return f"crash_{abs(int(round(spx_ret * 100)))}pct"


def _validate(scenarios: list[Scenario], model_name: str) -> None:
    total = sum(s.probability for s in scenarios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Scenario model '{model_name}' probabilities sum to {total:.6f}, not 1.0"
        )


# ---------------------------------------------------------------------------
# Model 1: flat (original behaviour)
# ---------------------------------------------------------------------------

def flat_scenarios(params: Any, dte: int) -> list[Scenario]:
    """
    Fixed probabilities regardless of DTE.

    P(crash) = params.p_crash for every option, regardless of how far out
    expiry is.  This was the original scoring behaviour — included for
    comparison and backward compatibility.
    """
    p_crash    = params.p_crash
    p_no_crash = 1.0 - p_crash

    result: list[Scenario] = []

    for label, spx_ret, share in params.non_crash_splits:
        result.append(Scenario(label, spx_ret, p_no_crash * share, is_crash=False))

    for crash_share, spx_ret in params.crash_splits:
        result.append(Scenario(_crash_name(spx_ret), spx_ret, p_crash * crash_share, is_crash=True))

    _validate(result, "flat")
    return result


# ---------------------------------------------------------------------------
# Model 2: survival (default)
# ---------------------------------------------------------------------------

def survival_scenarios(params: Any, dte: int) -> list[Scenario]:
    """
    Scale crash probability by DTE using the survival probability formula.

    Intuition: if you believe there is a 55% chance of a crash over 18 months,
    that same belief implies a lower probability for a 6-month window.

    Formula:
        P(no crash in dte days) = P(no crash in H days) ^ (dte / H)
        P(crash in dte days)    = 1 - P(no crash in dte days)

    where H = horizon_months × 30.44 (days).

    This is equivalent to a constant-hazard (Poisson) crash arrival process,
    which is the simplest defensible assumption.  The crash / non-crash
    relative splits are preserved — only the total mass shifts.

    At dte == horizon_days the result is identical to flat_scenarios.
    """
    horizon_days    = params.horizon_months * 30.44
    ratio           = dte / horizon_days
    p_no_crash_full = 1.0 - params.p_crash
    p_no_crash_dte  = p_no_crash_full ** ratio
    p_crash_dte     = 1.0 - p_no_crash_dte

    result: list[Scenario] = []

    for label, spx_ret, share in params.non_crash_splits:
        result.append(Scenario(label, spx_ret, p_no_crash_dte * share, is_crash=False))

    for crash_share, spx_ret in params.crash_splits:
        result.append(Scenario(_crash_name(spx_ret), spx_ret, p_crash_dte * crash_share, is_crash=True))

    _validate(result, "survival")
    return result


# ---------------------------------------------------------------------------
# Registry — add new models here
# ---------------------------------------------------------------------------

class _GarchEPPlaceholder:
    """Raises helpful error — use GJREntropyModel(current_price, view_buckets) directly."""

    def __call__(self, params: Any, dte: int) -> list[Scenario]:
        raise RuntimeError(
            "garch_ep model requires initialization with a current SPX price and view "
            "buckets. Import GJREntropyModel from spx_model and pass model= to "
            "score_puts() directly:\n\n"
            "    from spx_model import GJREntropyModel\n"
            "    model = GJREntropyModel(current_price, view_buckets)\n"
            "    score_puts(df, params, spx_spot, model=model)"
        )


SCENARIO_MODELS: dict[str, ScenarioModel] = {
    "flat":     flat_scenarios,
    "survival": survival_scenarios,
    "garch_ep": _GarchEPPlaceholder(),
}

DEFAULT_MODEL = "survival"


def get_model(name: str) -> ScenarioModel:
    """Look up a model by name, raising a clear error if unknown."""
    if name not in SCENARIO_MODELS:
        known = ", ".join(SCENARIO_MODELS)
        raise ValueError(f"Unknown scenario model '{name}'. Available: {known}")
    return SCENARIO_MODELS[name]
