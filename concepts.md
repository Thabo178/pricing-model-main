# Core Concepts — Structured Note Pricer

## 1. What is a structured note?
A structured note is a debt instrument (like a bond) whose return is linked to the performance
of a stock or basket of stocks. Instead of a fixed interest rate, you get a conditional payoff
based on whether the stock stays above certain price levels. Banks create these and sell them
to investors.

## 2. What is an autocallable?
The specific type we are pricing. It has three key mechanisms:

- **Autocall barrier** — checked on each observation date (e.g. quarterly). If the stock is
  above this level (usually 100% of starting price), the note "calls" early: the investor gets
  their principal back plus a coupon. The note ends.
- **Coupon barrier** — if the stock hasn't triggered the autocall but is still above this lower
  level (e.g. 75% of starting price), the investor still earns their coupon for that period.
- **Knock-in barrier** — only matters at maturity if the note never called. If the stock is
  below this level (e.g. 65% of starting price) at any observation date, the investor loses
  principal proportionally to how far the stock fell at maturity. This is the "principal at risk"
  feature.

So the investor is essentially selling downside protection on the stock in exchange for an
above-market coupon.

## 3. Why does this need a pricer?
When a bank offers Ryan a note at, say, 97 cents on the dollar, is that fair? Too expensive?
Too cheap? Without a model, he can't tell. The pricer gives him an independent fair value —
if the model says 94 cents, the bank is charging him 3 cents too much.

## 4. Black-Scholes vs Heston — why it matters
Black-Scholes assumes volatility is constant. That's fine for a simple option expiring in 35
days, but structured notes have 12-24 month terms and path-dependent payoffs (the autocall
depends on what happens on each observation date along the way). Over that time horizon,
volatility itself moves — it spikes during crashes and compresses during calm markets.

**Heston** models volatility as a random process with its own behavior. It has 5 parameters:
- `v0`    — volatility today (technically, initial *variance* = vol²)
- `theta` — where volatility gravitates to long-term (the "mean" variance)
- `kappa` — how fast it mean-reverts to theta
- `rho`   — correlation between stock moves and vol moves (typically negative: when the stock
             drops, vol spikes — this is the "leverage effect")
- `sigma` — how "jumpy" vol itself is (vol-of-vol)

This makes the model significantly more realistic for long-dated, path-dependent products.

## 5. Why Monte Carlo?
The autocallable payoff is **path-dependent** — whether the note calls on Month 3 vs Month 6
vs maturity completely changes the cash flows. There's no closed-form formula for this.
Monte Carlo solves it by simulating thousands of possible price paths for the stock, computing
what the note would pay out on each path, and averaging the results. That average (discounted
back to today) is the fair value.

## 6. The key number the pricer outputs
**NPV (Net Present Value)** — expressed as a dollar amount or as a percentage of face value
(e.g. 96.2 cents on the dollar). This is what Ryan compares against whatever the bank is
charging him.

---

## 7. What is a vol surface, and why do we need one for calibration?
A single option has one implied volatility. But if you look at all the options trading on a
stock across different strikes and expiries, each one implies a slightly different volatility.
Plotted together, this is the **implied volatility surface** — a 2D grid of (strike, expiry) → IV.

The surface has two well-known features:
- **Skew** — OTM puts (protection against crashes) cost more than OTM calls, so their IV is
  higher. A plot of IV vs strike at a fixed expiry looks like a smirk.
- **Term structure** — short-dated options usually have higher IV than long-dated ones in
  calm markets; the relationship flips during stress.

Heston calibration means finding the 5 parameters that best reproduce this surface. Once
calibrated, the model correctly prices the structured note (which depends on the vol surface
over its full 12-24 month life).

## 8. How ORATS provides the vol surface
ORATS is a data vendor that publishes **constant-maturity implied volatilities at standard
delta points** via their `monies/implied` API endpoint. Instead of giving you IV at specific
strikes and expiries (which change every day), they interpolate to give you a clean, stable
surface that's easy to work with.

The delta points they use are call-delta convention:
- `vol15` = 15Δ call (deep OTM call, high IV due to skew)
- `vol25` = 25Δ call (OTM call)
- `vol50` = 50Δ call (ATM — the "at-the-money" vol)
- `vol75` = 75Δ call (ITM call, equivalent to 25Δ put)
- `vol85` = 85Δ call (deep ITM call)

To use these in calibration, we convert each delta/IV pair to a strike via the
**BSM call-delta inversion formula**:

    K = F × exp(−N⁻¹(Δ) × σ√T + ½σ²T)

where F is the forward price, N⁻¹ is the inverse normal CDF, and σ is the IV at that delta.

## 9. What is "worst-of" and why does it change the pricing?
A **worst-of** note references a basket of 2 or 3 stocks. At each observation date, all
barrier checks (autocall, coupon, knock-in) are determined by the **worst-performing** stock
in the basket — the one that has fallen the most relative to its starting price.

From the investor's perspective: you're now exposed to the weakest link. If NVDA is doing
great but TSLA has dropped 30%, the note behaves as if you only owned TSLA. This makes the
product significantly more risky than a single-underlier note, which is why banks can offer
a higher coupon to compensate.

**Pricing impact**: A NVDA+TSLA worst-of note priced at ~91 cents vs ~97 cents for NVDA alone
in our model — a 6% discount for adding a second high-volatility underlier with ~0.55 correlation.
The lower the correlation between the two stocks, the bigger this discount gets (more
independent chances for one to crash).

## 10. Cross-asset correlation and how we model it
When simulating multiple stocks simultaneously, their price moves need to be correlated
realistically — two tech stocks generally move together, not independently.

We model this with a **correlation matrix**. For two stocks with correlation ρ = 0.55:

    [[1.00, 0.55],
     [0.55, 1.00]]

To generate correlated random moves in the simulation, we use **Cholesky decomposition**:
find matrix L such that L × Lᵀ = correlation matrix, then multiply independent random
draws by L. This produces draws with the correct correlations.

One subtlety: the user-supplied correlation matrix must be **positive semi-definite (PSD)** —
otherwise Cholesky fails. We apply `nearest_psd()` first, which clips any negative eigenvalues
to a small positive floor. This handles rounding errors or misspecified inputs gracefully.

## 11. Why the QuantLib optimizer fails on live data
QuantLib's built-in `LevenbergMarquardt` optimizer has no parameter bounds. On clean, synthetic
data it converges fine. On real ORATS data (which has noise in short-dated expiries), it can
wander into economically impossible regions:
- Negative sigma (vol-of-vol can't be negative)
- Positive rho (equities almost always have negative rho — crash → vol spike)
- Theta > 1 (implied long-run vol > 100%, absurd for equities)

Our fix: use `scipy.optimize.minimize` with `L-BFGS-B` and explicit bounds:
- kappa: [0.10, 10.0]
- theta: [0.01, 1.0]
- sigma: [0.01, 2.0]
- rho:   [−0.99, −0.01]
- v0:    [0.01, 1.0]

This keeps the optimizer in the economically valid region at all times.

---

## How the concepts map to the code files
| File               | Concept it implements                              |
|--------------------|----------------------------------------------------|
| `heston.py`        | Heston stochastic vol model, parameter store       |
| `monte_carlo.py`   | Monte Carlo path simulation (single + multi-asset) |
| `payoff.py`        | Autocallable payoff (single + worst-of)            |
| `calibration.py`   | Vol surface calibration (mock + ORATS live)        |
| `orats.py`         | ORATS API, delta→strike conversion                 |
| `pricer.py`        | NPV entry points (single, dict, worst-of)          |
| `dashboard.py`     | Streamlit UI for Ryan                              |

---

## Things to understand deeper (study list)

### Finance
- **Feller condition**: `2κθ > σ²` — when this holds, the variance process can't reach zero.
  Our model doesn't enforce it as a hard constraint (only via bounds), so you may see it
  violated in calibrated params. Worth understanding the implications.
- **Put-call parity**: `C = P + DF × (F − K)`. Used in calibration to convert put prices
  to call prices before implied vol inversion.
- **Risk-neutral vs real-world measure**: The model prices under the risk-neutral measure
  (drift = risk-free rate). Real stock drift is irrelevant for option pricing.
- **Path-dependency vs European payoffs**: European options only care about the final price.
  Autocallables care about the price at every observation date — that's what makes closed-form
  solutions impossible and Monte Carlo necessary.
- **Discount factor**: `exp(−r × T)` — the present value of $1 received at time T. Every
  cash flow in the simulation is multiplied by this before averaging.
- **Worst-of discount**: The gap between worst-of and single-underlier pricing widens as
  correlation decreases. At ρ = 0 (perfectly uncorrelated), the basket has maximum worst-of
  risk. At ρ = 1 (perfectly correlated), worst-of = single-underlier. Understand why intuitively.

### Mathematics
- **Stochastic differential equations (SDEs)**: The Heston model is two coupled SDEs.
  Understanding how Euler-Maruyama discretises them is core to understanding the simulation.
- **Cholesky decomposition**: The standard way to draw correlated normals. If `L @ Lᵀ = Σ`
  and `z ~ N(0, I)`, then `Lz ~ N(0, Σ)`.
- **Eigenvalue decomposition and PSD**: A matrix is PSD if all eigenvalues ≥ 0. Nearest-PSD
  projection clips negative eigenvalues to zero/small values. Understand why Cholesky requires
  PSD.
- **Log-normal vs normal returns**: Stock prices are log-normal (log returns are normal). This
  is why we simulate `log(S)` and exponentiate — it naturally prevents negative prices.
- **Implied volatility inversion**: Given an option price, implied vol is the σ that makes
  Black-Scholes equal that price. There's no closed form — it's solved numerically (bisection
  or Newton's method). QuantLib does this for us.

### Python / Engineering
- **NumPy broadcasting**: The multi-asset payoff uses shapes `(n_paths, n_assets, n_obs)`.
  Understanding how `arr / spots[np.newaxis, :, np.newaxis]` broadcasts is essential for
  reading and extending the payoff code.
- **Memory layout**: Pre-allocating `(50000, 378, 3)` float64 arrays uses ~450MB per array.
  That's why `generate_paths_multi` generates randoms step-by-step instead.
- **scipy L-BFGS-B**: A bounded quasi-Newton optimizer. L-BFGS stores a low-rank approximation
  of the Hessian. `ftol` and `gtol` control convergence — tighter values = more iterations.
- **python-dotenv**: Loads key=value pairs from `.env` into environment variables at import time.
  The API token must never be hardcoded or committed to git.
