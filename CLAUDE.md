# Pricing Model — Claude Code Guide

## Primary reference document

All implementation decisions should align with the technical specification at:
`/Users/thabomahlaha/Downloads/Summer Internship/structured_note_pricing_model_guide.pdf`

Read this document when making architectural decisions, adding new features, or
resolving ambiguity about pricing logic, calibration, or validation thresholds.

---

## Project purpose

Phoenix autocallable structured note pricer built for client Ryan Hysmith.
Supports single-underlier and worst-of-2 / worst-of-3 multi-asset structures.
Ryan runs the tool via the Streamlit dashboard (double-click `start.command` on Mac,
`start.bat` on Windows) — no terminal interaction required.

GitHub repo: https://github.com/Ryan1-consulting/pricing-model-main
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

# Validate payoff logic (must pass 8/8: 4 single-underlier + 4 worst-of)
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
  orats.py         — ORATS API client: live spot, dividend yield, vol surface,
                     delta→strike conversion (/summaries, /strikes, /monies/implied, /cores)
  pricer.py        — Entry points: price_note(), price_note_dict(), price_worst_of()
  greeks.py        — compute_greeks(): bump-and-reprice delta/gamma/vega/theta
                     (single-underlier and worst-of, per-underlier vectors for the latter)
  portfolio.py     — price_portfolio(): prices every note in a portfolio JSON file,
                     flags deviation vs issuer mark (OK / Review / Flag per §10 thresholds)
  offering.py      — evaluate_offering(): model fair value vs issuer offer price,
                     Buy/Skip/Gray Zone recommendation at ±1.5% (§6.4)
  correlation.py   — compute_historical_correlation(): 60-day daily / 250-day weekly
                     blended correlation matrix via yfinance (§9.1)
  vanilla.py       — VanillaEuropean (BSM + Heston pricing engines) + market_iv()
                     vollib-based implied-vol inversion (§4 Step 1.1/1.2)
  __init__.py      — Exposes price_note, price_note_dict, price_worst_of, price_portfolio,
                     compute_greeks, evaluate_offering, generate_paths, generate_paths_multi,
                     nearest_psd, autocallable_payoff, worst_of_payoff

calibrate.py       — CLI: calibrate one or all of the 12 underliers (live or mock)
run_pricer.py      — CLI: price a single-underlier note and print NPV output
validate.py        — 8 deterministic boundary tests for payoff correctness (4 single + 4 worst-of)
dashboard.py       — Streamlit UI, 6 tabs: Note Pricer, Vol Surface, Calibration,
                     Worst-Of, Portfolio, Offering Evaluator

data/
  sample_note.json              — Mock NVDA term sheet
  worstof_note_*.json           — Saved worst-of term sheets (2- and 3-asset)
  portfolio.json                — Mark-to-model portfolio (currently a sample —
                                   replace with Ryan's real 23-note book, see §10)
  calibrated/NVDA.json          — Calibrated Heston params (live ORATS, rmse = 0.00886)
                                   Only NVDA calibrated live so far; the other 11
                                   underliers still use hardcoded defaults from heston.py
```

---

## Note JSON schema — single underlier

```json
{
  "underlier": "NVDA",
  "spot": 213.73,
  "face_value": 1000.0,
  "issue_date": "2026-06-03",
  "maturity_date": "2027-12-03",
  "observation_dates": ["2026-09-03", "..."],
  "autocall_barrier": 1.0,
  "coupon_barrier": 0.75,
  "knockin_barrier": 0.65,
  "coupon_rate": 0.12,
  "risk_free_rate": 0.0375
}
```

## Note JSON schema — worst-of multi-asset

```json
{
  "underliers": ["NVDA", "TSLA"],
  "spots": [213.73, 180.0],
  "correlation_matrix": [
    [1.0, 0.55],
    [0.55, 1.0]
  ],
  "face_value": 1000.0,
  "issue_date": "2026-06-03",
  "maturity_date": "2027-12-03",
  "observation_dates": ["2026-09-03", "..."],
  "autocall_barrier": 1.0,
  "coupon_barrier": 0.75,
  "knockin_barrier": 0.65,
  "coupon_rate": 0.12,
  "risk_free_rate": 0.0375
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

- `/summaries` — live spot price, ATM IV, IV rank
- `/strikes` — full strike-level chain
- `/monies/implied` — constant-maturity smoothed IV at delta points (main calibration input)
- `/cores` — dividend yield (`live_dividend_yield()`), via `divYield`/`yield`/`dividendYield`/
  `forwardDivYield`/`annualDividendYield` (first match wins); falls back to `q=0.0`

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
- `yfinance >= 0.2` (historical price data for `correlation.py`)

Install: `pip install -r requirements.txt`

The only fully-tested environment is the `base` conda env (`/opt/anaconda3/bin/python`,
Python 3.12.2) — it has every package above installed. Other local envs (including the
default `agents` conda env) are missing several of these; if you hit import errors, check
`python -c "import streamlit, scipy, pandas, QuantLib, vollib"` in whichever interpreter
VS Code/the shell is actually using.

---

## Phase completion status

### Phase 1 — Single-underlier core (DONE)

- `python calibrate.py NVDA --mock` → converged, rmse = 0.00000
- `python validate.py` → 8/8 tests passed
- `python run_pricer.py` → NPV output for mock NVDA note

### Phase 2 — Live data + multi-asset (DONE)

- [x] Step 1: ORATS wiring — live spot + dividend yield + vol surface → calibration pipeline
- [x] Step 2: Multi-asset Monte Carlo — `generate_paths_multi`, `nearest_psd`
- [x] Step 3: Worst-of payoff engine — `worst_of_payoff`, `price_worst_of`
- [x] Dashboard UI for worst-of note entry (Worst-Of tab)
- [x] GitHub repo setup for Ryan (`Ryan1-consulting/pricing-model-main`)

### Phase 3 — Production quality (PARTIAL)

- [x] Issuer credit spread adjustment (§6.1) — flat add to discount rate, not a full curve
- [x] Per-note Greeks (§6.3) — `greeks.py`, wired into the dashboard
- [x] Portfolio mark-to-model (§6.3/§10) — `portfolio.py`, Portfolio tab
- [x] New offering evaluator (§6.4) — `offering.py`, Offering Evaluator tab
- [ ] Bates jump diffusion, calibration diagnostics, VIX-regime correlation, term-structure
      discount curve — see "Known gaps vs. the spec" below

---

## Planned extensions

- Load Ryan's real 23-note portfolio into `data/portfolio.json` (currently a sample)
- Calibrate the other 11 underliers live (only NVDA has a calibrated file so far)
- Close the gaps listed below, prioritizing the flat-discount-rate and MC time-grid items
  since the spec calls both out as anti-patterns rather than just missing features

---

## Dividend yield from ORATS /cores (§11 pitfall) — DONE

Previously all underliers used `q = 0.0`, biasing the forward price into `rho`. Fixed:

1. `live_dividend_yield(ticker)` in `pricer/orats.py` pulls `q` from the `/cores` endpoint.
2. `q` is stored as `dividend_yield` in the calibrated JSON alongside the Heston params.
3. `generate_paths()` and `generate_paths_multi()` both read `dividend_yield` from the
   Heston params dict and use it in the drift term.
4. `calibrate_heston_orats()` takes `q` so the forward prices used during calibration match.

Matters most for: SPY (~1.3%), META, AMZN, WFC. Negligible for zero-dividend growth names.

---

## Known gaps vs. the spec (§11 pitfalls not yet addressed)

Found during a full audit against `structured_note_pricing_model_guide.pdf` (2026-06-17).
None of these are correctness bugs — they're unimplemented Phase 3 "production quality"
items, plus two documented anti-patterns the spec explicitly warns against:

- **Flat discount rate** — `pricer.py` discounts with a single scalar
  `risk_free_rate + credit_spread` rather than a `ql.YieldTermStructureHandle` with
  multiple tenors. §11: "Hardcoding interest rate as scalar... term structure matters
  for multi-year products."
- **MC time grid not matched to observation frequency** — `monte_carlo.py` always
  defaults to `n_steps_per_year=252` (daily), even for quarterly-observation notes.
  §11: "10x compute cost; no accuracy gain... match time grid to observation frequency."
- **No calibration diagnostics (§8.2)** — only RMSE and a Feller boolean are surfaced
  (dashboard Calibration tab). No residual histogram or residual-by-strike/tenor heatmap.
- **No BSM/local-vol fallback (§8.3)** — when Heston calibration is unreliable (e.g. thin
  liquidity, extreme skew), there's no fallback path or "model-limited" flag.
- **No VIX-regime correlation shift / floor (§9.1)** — `correlation.py` always uses the
  static 60/40 blend; spec recommends shifting toward 100% 250-day weekly when VIX > 25,
  and flooring tech-tech pairs at 0.4.
- **No Bates jump diffusion (§6.2, optional)** — not implemented; relevant only if a
  calibrated Heston model leaves systematic residuals at deep OTM strikes.
- **Calibration always restarts from hardcoded defaults** — `calibrate_heston()`'s scipy
  x0 uses `get_heston_params()`, not the prior day's calibrated result. §11: "Recalibrating
  Heston every day from scratch... use prior day's params as initial guess."
