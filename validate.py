"""
Validate payoff functions on deterministic boundary scenarios.

Hand-crafted path arrays replace Monte Carlo so the expected output is known
exactly. If any test fails, the payoff logic is wrong.

Single-underlier tests (4): autocall, knock-in, coupon-only, KI-then-recovery
Worst-of tests (4):         same boundary logic but using a 2-asset basket

Run with:   python validate.py
"""
import sys
import numpy as np
from pricer.payoff import autocallable_payoff, worst_of_payoff

# Shared note parameters used across all tests
FACE         = 1000.0
SPOT         = 100.0
R            = 0.0375
OBS_TIMES    = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]   # quarterly, 18 months
DFS          = np.exp(-R * np.array(OBS_TIMES))
COUPON_RATE  = 0.12
OBS_PER_YEAR = len(OBS_TIMES) / OBS_TIMES[-1]          # = 4.0
CPP          = COUPON_RATE / OBS_PER_YEAR               # coupon per period = 0.03

N = 500   # paths per test (all identical paths within each test)

PASS = 0
FAIL = 0


def check(label, got, expected, tol=1e-9):
    global PASS, FAIL
    diff = abs(got - expected)
    ok   = diff < tol
    mark = 'PASS' if ok else 'FAIL'
    print(f"  [{mark}] {label}")
    print(f"         got={got:.8f}  expected={expected:.8f}  diff={diff:.2e}")
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"         *** ASSERTION FAILED ***")


# ---------------------------------------------------------------------------
# Test 1 — Immediate autocall
# All paths sit at 110% of spot on every observation date.
# The autocall barrier is 100%, so the note calls on the FIRST date.
# Every path receives (face + coupon) discounted at t=0.25.
# ---------------------------------------------------------------------------
print("\nTest 1 — Immediate autocall (all paths above autocall barrier from date 1)")
paths_1 = np.full((N, len(OBS_TIMES)), SPOT * 1.10)
npv_1   = autocallable_payoff(paths_1, SPOT, FACE, 1.00, 0.75, 0.65, COUPON_RATE, OBS_PER_YEAR, DFS)
expected_1 = (1.0 + CPP) * DFS[0]
check("npv == (1 + coupon_per_period) * df[0]", npv_1, expected_1)

# ---------------------------------------------------------------------------
# Test 2 — Full knock-in with 50% recovery
# All paths sit at 50% of spot — permanently below the 65% knock-in barrier
# and below the 75% coupon barrier. No coupon is ever earned.
# At maturity: proportional principal = face * (50/100) = 500.
# NPV = df[-1] * 0.50
# ---------------------------------------------------------------------------
print("\nTest 2 — Permanent knock-in, 50% recovery (all paths at 50% of spot)")
paths_2 = np.full((N, len(OBS_TIMES)), SPOT * 0.50)
npv_2   = autocallable_payoff(paths_2, SPOT, FACE, 1.00, 0.75, 0.65, COUPON_RATE, OBS_PER_YEAR, DFS)
expected_2 = DFS[-1] * 0.50
check("npv == df[-1] * 0.50", npv_2, expected_2)

# ---------------------------------------------------------------------------
# Test 3 — Pure coupon machine (no autocall, no knock-in)
# All paths sit at 80% of spot — above the coupon barrier (75%) but below
# the autocall barrier (100%). No knock-in (80% > 65%).
# Note runs to maturity paying a coupon every period including the last,
# then returns full principal.
# NPV = coupon_per_period * sum(dfs) + df[-1]
# ---------------------------------------------------------------------------
print("\nTest 3 — Pure coupon + principal (paths in coupon zone, never autocall or knock-in)")
paths_3 = np.full((N, len(OBS_TIMES)), SPOT * 0.80)
npv_3   = autocallable_payoff(paths_3, SPOT, FACE, 1.00, 0.75, 0.65, COUPON_RATE, OBS_PER_YEAR, DFS)
expected_3 = CPP * float(np.sum(DFS)) + DFS[-1]
check("npv == coupon_per_period * sum(dfs) + df[-1]", npv_3, expected_3)

# ---------------------------------------------------------------------------
# Test 4 — Knock-in then recovery
# Paths breach knock-in during the term (drop to 60% at date 3),
# but recover above initial spot (105%) by maturity.
# Recovery condition: S_final >= S_initial → full principal returned.
# Plus full coupon at every date where stock >= coupon_barrier.
# ---------------------------------------------------------------------------
print("\nTest 4 — Knock-in occurs but stock recovers above initial spot at maturity")
levels = [0.80, 0.80, 0.60, 0.80, 0.80, 1.05]   # KI breach at date 3 (idx 2)
paths_4 = np.tile(np.array([SPOT * l for l in levels]), (N, 1))
npv_4   = autocallable_payoff(paths_4, SPOT, FACE, 1.00, 0.75, 0.65, COUPON_RATE, OBS_PER_YEAR, DFS)

# Dates 1,2: 80% — above coupon barrier, below autocall → coupon paid
# Date 3:    60% — below coupon barrier, below KI → no coupon, KI breached
# Dates 4,5: 80% — above coupon barrier → coupon paid
# Date 6:    105% — above coupon barrier → coupon paid; S >= spot → full principal
coupon_dates = [0, 1, 3, 4, 5]   # indices where coupon is earned
expected_4 = sum(CPP * DFS[i] for i in coupon_dates) + DFS[-1]
check("npv == coupons on qualifying dates + full principal (recovered)", npv_4, expected_4)

# ===========================================================================
# WORST-OF PAYOFF TESTS
# ===========================================================================
#
# Two underliers: Asset A (spot=100) and Asset B (spot=200).
# Worst performer at each date = min(S_A/100, S_B/200).
# Same note params as above (autocall=100%, coupon=75%, KI=65%).
#
# paths shape: (N, 2, n_obs)
#   paths[:, 0, :] = Asset A price paths
#   paths[:, 1, :] = Asset B price paths

SPOTS_WO = [100.0, 200.0]   # initial spots for the two underliers


# ---------------------------------------------------------------------------
# WO Test 1 — Immediate autocall (both assets above autocall barrier)
# A=110, B=220 → perf_A=1.10, perf_B=1.10 → worst=1.10 >= 1.00 → autocall t=0.25
# ---------------------------------------------------------------------------
print("\nWO Test 1 — Immediate autocall (both assets above 100% of initial)")
paths_wo1       = np.zeros((N, 2, len(OBS_TIMES)))
paths_wo1[:, 0, :] = 110.0   # Asset A at 110% of spot
paths_wo1[:, 1, :] = 220.0   # Asset B at 110% of spot

npv_wo1      = worst_of_payoff(paths_wo1, SPOTS_WO, FACE, 1.00, 0.75, 0.65,
                               COUPON_RATE, OBS_PER_YEAR, DFS)
expected_wo1 = (1.0 + CPP) * DFS[0]
check("wo npv == (1 + coupon_per_period) * df[0]", npv_wo1, expected_wo1)

# ---------------------------------------------------------------------------
# WO Test 2 — Worst performer drives barrier (A fine, B in knock-in zone)
# A=90 (90% of 100), B=100 (50% of 200) → worst=0.50 < 0.65 → KI breached every date
# At maturity: no coupon (0.50 < 0.75), KI breached, S_final < initial → proportional loss
# Expected: face * 0.50 discounted at maturity
# ---------------------------------------------------------------------------
print("\nWO Test 2 — Worst performer drives knock-in (A healthy, B at 50% of initial)")
paths_wo2        = np.zeros((N, 2, len(OBS_TIMES)))
paths_wo2[:, 0, :] = 90.0   # Asset A at 90% of spot (above coupon barrier)
paths_wo2[:, 1, :] = 100.0  # Asset B at 50% of spot (100/200 = 50%, below KI)

npv_wo2      = worst_of_payoff(paths_wo2, SPOTS_WO, FACE, 1.00, 0.75, 0.65,
                               COUPON_RATE, OBS_PER_YEAR, DFS)
expected_wo2 = DFS[-1] * 0.50   # worst perf = 0.50, proportional loss
check("wo npv == df[-1] * 0.50  (B drags worst perf into KI zone)", npv_wo2, expected_wo2)

# ---------------------------------------------------------------------------
# WO Test 3 — Both assets in coupon zone, note runs to maturity
# A=85 (85%), B=170 (85%) → worst=0.85 in [0.75, 1.00) → coupon every period, no autocall
# At maturity: coupon + full principal (no KI)
# Expected: coupon_per_period * sum(dfs) + df[-1]
# ---------------------------------------------------------------------------
print("\nWO Test 3 — Both in coupon zone, full term to maturity, full principal returned")
paths_wo3        = np.zeros((N, 2, len(OBS_TIMES)))
paths_wo3[:, 0, :] = 85.0   # Asset A at 85% of spot
paths_wo3[:, 1, :] = 170.0  # Asset B at 85% of spot

npv_wo3      = worst_of_payoff(paths_wo3, SPOTS_WO, FACE, 1.00, 0.75, 0.65,
                               COUPON_RATE, OBS_PER_YEAR, DFS)
expected_wo3 = CPP * float(np.sum(DFS)) + DFS[-1]
check("wo npv == coupon * sum(dfs) + df[-1]", npv_wo3, expected_wo3)

# ---------------------------------------------------------------------------
# WO Test 4 — KI breach then both assets recover above initial spot
# A stays at 105% throughout. B: [85%, 85%, 60%, 85%, 85%, 105%] of initial.
# Worst perf = min(A perf, B perf):
#   [min(1.05,0.85), min(1.05,0.85), min(1.05,0.60), min(1.05,0.85), min(1.05,0.85), min(1.05,1.05)]
#   = [0.85, 0.85, 0.60, 0.85, 0.85, 1.05]
# This mirrors single-underlier Test 4 exactly.
#   Date 1,2: worst=0.85 → coupon
#   Date 3:   worst=0.60 → no coupon, KI breached
#   Date 4,5: worst=0.85 → coupon
#   Date 6:   worst=1.05 >= 1.00 → coupon + full principal (recovered above 100%)
# Expected: coupons on dates [0,1,3,4,5] + full principal at maturity
# ---------------------------------------------------------------------------
print("\nWO Test 4 — Worst-of KI breach (B drops), both recover above initial spot at maturity")
b_levels = [0.85, 0.85, 0.60, 0.85, 0.85, 1.05]
paths_wo4           = np.zeros((N, 2, len(OBS_TIMES)))
paths_wo4[:, 0, :]  = 105.0                                            # Asset A: steady 105%
paths_wo4[:, 1, :]  = np.array([200.0 * l for l in b_levels])         # Asset B: varies

npv_wo4 = worst_of_payoff(paths_wo4, SPOTS_WO, FACE, 1.00, 0.75, 0.65,
                          COUPON_RATE, OBS_PER_YEAR, DFS)
coupon_dates_wo4 = [0, 1, 3, 4, 5]
expected_wo4 = sum(CPP * DFS[i] for i in coupon_dates_wo4) + DFS[-1]
check("wo npv == coupons on qualifying dates + full principal (worst-of recovered)",
      npv_wo4, expected_wo4)

# ===========================================================================
# Summary
# ===========================================================================
total = PASS + FAIL
print(f"\n{'=' * 40}")
print(f"  {PASS}/{total} tests passed")
print(f"{'=' * 40}\n")
sys.exit(0 if FAIL == 0 else 1)
