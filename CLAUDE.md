# Pricing Model — Claude Code Guide

## Primary reference document

All implementation decisions should align with the technical specification at:
`/Users/thabomahlaha/Downloads/structured_note_pricing_model_guide.pdf`

Read this document when making architectural decisions, adding new features, or
resolving ambiguity about pricing logic, calibration, or validation thresholds.

---

## Project purpose
Phoenix autocallable structured note pricer built for client Ryan Hysmith.
Supports single-underlier and worst-of-2 / worst-of-3 multi-asset structures.
Ryan runs the tool via the Streamlit dashboard (double-click `start.command` on Mac,
`start.bat` on Windows) — no terminal interaction required.

GitHub repo: https://github.com/Thabo178/pricing-model
Collaborative editing between the developer (Mac) and Ryan (Windows).


---

## How to run

```bash
# Launch the dashboard (the main interface)
./start.command          # Mac
start.bat                # Windows (Ryan)

# Price the mock NVDA note from the terminal (50k paths, ~5 seconds)
python run_pricer.py

# Quick run — 10k paths, less accurate but fast
python run_pricer.py data/sample_note.json 10000

# Calibrate all 12 underliers using live ORATS data
python calibrate.py

# Calibrate a single name (live)
python calibrate.py NVDA

# Calibrate using synthetic surface (correctness test — rmse = 0.00)
python calibrate.py NVDA --mock

# Validate payoff logic (must pass 4/4)
python validate.py
```

---

## Architecture

```
pricer/
  heston.py        — Heston parameter store + load_params() (calibrated → hardcoded fallback)
  monte_carlo.py   — Path simulation: single-underlier (generate_paths) and
                     multi-asset (generate_paths_multi, nearest_psd)
  payoff.py        — Payoff engines: autocallable_payoff (single) + worst_of_payoff (multi)
  calibration.py   — Vol surface generation, Heston calibration, JSON persistence
                     calibrate_heston()       — scipy L-BFGS-B (bounded, main path)
                     calibrate_heston_orats() — ORATS surface → scipy L-BFGS-B
                     calibrate_heston_ql()    — QuantLib LM (unbounded, avoid on live data)
  orats.py         — ORATS API client: live spot, vol surface, delta→strike conversion
  pricer.py        — Entry points: price_note(), price_note_dict(), price_worst_of()
  __init__.py      — Exposes price_note, price_note_dict, price_worst_of,
                     generate_paths, generate_paths_multi, nearest_psd,
                     autocallable_payoff, worst_of_payoff

calibrate.py       — CLI: calibrate one or all of the 12 underliers (live or mock)
run_pricer.py      — CLI: price a single-underlier note and print NPV output
validate.py        — 4 deterministic boundary tests for payoff correctness
dashboard.py       — Streamlit UI: Note Pricer tab, Vol Surface tab, Calibration tab

data/
  sample_note.json         — Mock NVDA term sheet
  calibrated/NVDA.json     — Calibrated Heston params (live ORATS, rmse = 0.00886)
```

---

## Note JSON schema — single underlier

```json
{
  "underlier":          "NVDA",
  "spot":               213.73,
  "face_value":         1000.0,
  "issue_date":         "2026-06-03",
  "maturity_date":      "2027-12-03",
  "observation_dates":  ["2026-09-03", "..."],
  "autocall_barrier":   1.00,
  "coupon_barrier":     0.75,
  "knockin_barrier":    0.65,
  "coupon_rate":        0.12,
  "risk_free_rate":     0.0375
}
```

## Note JSON schema — worst-of multi-asset

```json
{
  "underliers":          ["NVDA", "TSLA"],
  "spots":               [213.73, 180.00],
  "correlation_matrix":  [[1.00, 0.55], [0.55, 1.00]],
  "face_value":          1000.0,
  "issue_date":          "2026-06-03",
  "maturity_date":       "2027-12-03",
  "observation_dates":   ["2026-09-03", "..."],
  "autocall_barrier":    1.00,
  "coupon_barrier":      0.75,
  "knockin_barrier":     0.65,
  "coupon_rate":         0.12,
  "risk_free_rate":      0.0375
}
```

All barrier fields are fractions of spot (e.g. `0.75` = 75% of initial stock price).
`coupon_rate` is annual; the pricer divides by `obs_per_year` to get per-period amount.
For worst-of, all barrier checks use the worst-performing underlier's normalised level.

---

## Heston model — key facts

Five parameters per underlier: `v0` (initial variance), `kappa` (mean-reversion speed),
`theta` (long-run variance), `sigma` (vol of vol), `rho` (stock/vol correlation, always negative).

`load_params(ticker)` in `heston.py` checks `data/calibrated/{TICKER}.json` first, then
falls back to the hardcoded table in `HESTON_PARAMS`. Only NVDA has a calibrated file so far.

When adding a new underlier:
1. Add hardcoded defaults to `HESTON_PARAMS` in `heston.py`
2. Add spot and name to `UNDERLIERS` in `calibrate.py`
3. Run `python calibrate.py NEWTICKER`

---

## Simulation — implementation choices

### Single-underlier (`generate_paths`)
- **Log-Euler** for stock SDE: prevents negative prices.
- **Full truncation** for variance SDE: clamps `V` to `0` before each step so `sqrt(V)` is always real.
- **Correlated Brownian motions**: Cholesky — `Z2 = rho*Z1 + sqrt(1-rho²)*Z_independent`.
- Default: 50,000 paths, 252 steps/year, seed=42 (reproducible).

### Multi-asset (`generate_paths_multi`)
- Same Euler scheme as above, but vectorised across all underliers simultaneously.
- **Cross-asset correlation**: stock Brownians are jointly drawn via Cholesky of the asset
  correlation matrix. `Z1 = Z_raw @ L.T` where `L = chol(corr_matrix)`.
- **Variance Brownians**: each asset's variance Brownian is correlated only with its own
  stock Brownian (Heston rho_i). Independent across different assets.
- **Memory**: generates randoms step-by-step — O(n_paths × n_assets) per step, not
  O(n_paths × n_steps × n_assets) all at once. Prevents ~900MB allocation for 3-asset runs.
- **PSD enforcement** (`nearest_psd`): eigenvalue-clipping projection applied to the
  user-supplied correlation matrix before Cholesky, in case of rounding or misspecification.
- Output shape: `(n_paths, n_underliers, n_obs)`.

---

## Worst-of payoff logic

At each observation date, the **worst performer** across all underliers is identified:

    worst_perf[path, date] = min(S_i / S0_i  for each underlier i)

This normalised worst-performer level is then used identically to the single-underlier
stock level in the Phoenix payoff:
- autocall triggers if `worst_perf >= autocall_barrier_pct`
- coupon pays if `worst_perf >= coupon_barrier_pct`
- knock-in is breached if `worst_perf < knockin_barrier_pct` at any observation date
- at maturity: proportional loss = `face_value × worst_final_perf` if knock-in was breached

The worst-of discount vs single-underlier is significant — e.g. NVDA+TSLA worst-of prices
at ~91 cents vs ~97 cents for NVDA alone (6% discount for adding a second high-vol underlier
with 0.55 correlation). This is the extra risk the investor takes on for a higher coupon.

---

## Calibration workflow

### Live mode (default)
1. `live_spot(ticker)` in `orats.py` fetches the current stock price from ORATS SMV summary.
2. `build_calibration_set(ticker, ...)` fetches `monies/implied` from ORATS and converts
   each delta/IV pair to a strike using BSM call-delta inversion:
   `K = F × exp(−N⁻¹(Δ) × σ√T + ½σ²T)`
   Delta columns sampled: vol15, vol25, vol35, vol50, vol65, vol75, vol85.
   Filters: 7–730 days to expiry, confidence ≥ 0.5, sigma in [0.02, 3.0].
3. `calibrate_heston_orats()` converts the calibration set to `(strike, maturity_years, iv)`
   and delegates to `calibrate_heston()` (scipy L-BFGS-B with parameter bounds). The QL
   LevenbergMarquardt optimizer is NOT used on live data — it is unbounded and produces
   degenerate solutions (negative sigma, positive rho) on noisy short-dated ORATS points.
4. Result is saved to `data/calibrated/{TICKER}.json`.

NVDA live calibration result: `v0=0.1344, kappa=8.975, theta=0.2311, sigma=2.000, rho=-0.037, rmse=0.00886` (140 pts)

### Mock mode (`--mock` flag)
Generates a synthetic vol surface from the hardcoded default Heston params and calibrates
with scipy L-BFGS-B. Because the surface is built from the same params, calibration always
recovers them exactly (rmse = 0.00). Used as a correctness test, not for production.

---

## ORATS API

Token stored in `.env` as `ORATS_API_TOKEN`. Loaded via `python-dotenv`. Never hardcode
or commit the token.

Endpoints used:
- `/summaries`      — live spot price, ATM IV, IV rank
- `/strikes`        — full strike-level chain
- `/monies/implied` — constant-maturity smoothed IV at delta points (main calibration input)

ORATS `volX` columns use **call-delta convention**: `vol25` = 25Δ call (OTM call),
`vol50` = ATM, `vol75` = 75Δ call (ITM call / 25Δ put).

---

## Supported underliers

NVDA, TSLA, AMD, META, GOOGL, AMZN, HOOD, LULU, NOW, PLTR, WFC, SPY

---

## Dependencies

- Python 3.12
- `numpy >= 1.26`
- `QuantLib` 1.39 (must be installed; not on PyPI as plain `quantlib` — use `QuantLib-Python`)
- `scipy >= 1.14`
- `streamlit >= 1.35`
- `pandas`, `requests`, `python-dotenv`, `matplotlib`, `vollib`

Install: `pip install -r requirements.txt`

---

## Phase completion status

### Phase 1 — Single-underlier core (DONE)
- `python calibrate.py NVDA --mock` → converged, rmse = 0.00000
- `python validate.py` → 4/4 tests passed
- `python run_pricer.py` → NPV output for mock NVDA note

### Phase 2 — Live data + multi-asset (IN PROGRESS)
- [x] Step 1: ORATS wiring — live spot + vol surface → calibration pipeline
- [x] Step 2: Multi-asset Monte Carlo — `generate_paths_multi`, `nearest_psd`
- [x] Step 3: Worst-of payoff engine — `worst_of_payoff`, `price_worst_of`
- [ ] Dashboard UI for worst-of note entry (multi-underlier input fields)
- [ ] GitHub repo setup for Ryan

---

## Planned extensions

- Dashboard Phase 2 UI: worst-of underlier selector, correlation input, multi-asset pricing
- GitHub repo for Ryan to fork and run on his Windows PC

---

## TODO — Dividend yield from ORATS /cores  (§11 pitfall)

Currently all underliers use `q = 0.0` (zero dividend yield) in the Heston process.
This biases the forward price and causes calibration to absorb the error into `rho`.

**Fix when ready:**
1. Add `live_dividend_yield(ticker)` to `pricer/orats.py` using the `/cores` endpoint.
2. Store `q` in the calibrated JSON alongside the Heston params.
3. Pass `q` through `generate_paths()` and `generate_paths_multi()` into the Heston process.
4. Pass `q` into `calibrate_heston_orats()` so the forward prices used during calibration match.

Matters most for: SPY (~1.3%), META, AMZN, WFC. Negligible for zero-dividend growth names.
Reference: §11 — *"Pull divs from ORATS /cores; use continuous-div approximation."*
