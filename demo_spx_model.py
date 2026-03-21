"""
Verification script for spx_model.py

Run with:
    python demo_spx_model.py
"""

from spx_model import predict_spx_distribution, GJREntropyModel
from hedge import Params

SPOT = 5500

# --- 1. Pure GARCH baseline (tail_tilt=1.0 — prints prior stats, skips EP) ---
print("=" * 60)
print("Test 1: Pure GARCH baseline (tail_tilt=1.0)")
print("=" * 60)
model_base = GJREntropyModel(SPOT, tail_tilt=1.0)

# --- 2. Moderately bearish (tail_tilt=2.0 — prints prior stats + implied view) ---
print()
print("=" * 60)
print("Test 2: Moderately bearish (tail_tilt=2.0)")
print("=" * 60)
model_bear = GJREntropyModel(SPOT, tail_tilt=2.0)

# --- 3. Sanity: bearish model has strictly higher crash probability at every DTE ---
print()
print("=" * 60)
print("Test 3: Bearish model > base crash probability")
print("=" * 60)
params = Params()
for dte in [180, 365]:
    p_base = sum(s.probability for s in model_base(params, dte) if s.is_crash)
    p_bear = sum(s.probability for s in model_bear(params, dte) if s.is_crash)
    assert p_bear > p_base, f"DTE={dte}: bear crash={p_bear:.4f} not > base crash={p_base:.4f}"
    print(f"  DTE={dte}: base crash={p_base:.4f}  bear crash={p_bear:.4f}  PASS")

# --- 4. Probabilities sum to 1.0 ---
print()
print("=" * 60)
print("Test 4: Probabilities sum to 1.0")
print("=" * 60)
for dte in [90, 180, 365]:
    total = sum(s.probability for s in model_bear(params, dte))
    assert abs(total - 1.0) < 1e-6, f"DTE={dte}: probs sum to {total}"
    print(f"  DTE={dte}: total={total:.6f}  PASS")

# --- 5. Backward compat: view_buckets override still works ---
print()
print("=" * 60)
print("Test 5: Backward compat — view_buckets override")
print("=" * 60)
model_adv = GJREntropyModel(SPOT, view_buckets=[(4000, 0.15), (4500, 0.35)])
total = sum(s.probability for s in model_adv(params, 365))
assert abs(total - 1.0) < 1e-6, f"view_buckets model probs sum to {total}"
print(f"  probs sum to {total:.6f}  PASS")

print()
print("All scenarios at DTE=365 (bear model):")
for s in model_bear(params, dte=365):
    print(f"  {s.name:<20s}  spx_ret={s.spx_return:+.2f}  prob={s.probability:.4f}  crash={s.is_crash}")

print()
print("All tests passed.")
