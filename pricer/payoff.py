import numpy as np


def autocallable_payoff(
    paths: np.ndarray,
    spot: float,
    face_value: float,
    autocall_barrier_pct: float,
    coupon_barrier_pct: float,
    knockin_barrier_pct: float,
    coupon_rate: float,
    obs_per_year: float,
    discount_factors: np.ndarray,
    memory: bool = False,
    _return_se: bool = False,
):
    """
    Compute the fair value of a single-underlier Phoenix autocallable note.

    Phoenix structure — at each observation date (§5.1):
      1. Coupon: if S >= coupon_barrier, pay periodic coupon.
         With memory=True, also pay any coupons unpaid from prior periods where
         the coupon barrier was not met.
      2. Autocall: if S >= autocall_barrier (non-final dates only), the note
         redeems at par. Investor receives principal; note ends for this path.
      3. No payment: if S < coupon_barrier, skip coupon (or accrue under memory).

    At maturity (final observation date):
      - Coupon paid if S >= coupon_barrier (including any accrued under memory)
      - Full principal returned if knock-in was NEVER breached, OR S_final >= S_initial
      - Proportional principal loss if knock-in was breached AND S_final < S_initial:
            principal = face_value × (S_final / S_initial)

    Knock-in is monitored at each observation date (discrete, not continuous).

    Parameters
    ----------
    paths               : (n_paths, n_obs) stock prices at observation dates
    spot                : initial stock price (= S_initial)
    face_value          : notional per note (e.g. 1000.0)
    autocall_barrier_pct: fraction of initial spot (e.g. 1.00 = 100%)
    coupon_barrier_pct  : fraction of initial spot (e.g. 0.75 = 75%)
    knockin_barrier_pct : fraction of initial spot (e.g. 0.65 = 65%)
    coupon_rate         : annual coupon rate (e.g. 0.12 = 12%)
    obs_per_year        : observation frequency (e.g. 4.0 for quarterly)
    discount_factors    : (n_obs,) risk-free discount factor for each date
    memory              : if True, unpaid coupons from periods below the coupon
                          barrier are carried forward and paid at the next
                          observation where the barrier is met (§5.1 memory feature)
    _return_se          : if True, return (npv_fraction, se_fraction) tuple

    Returns
    -------
    npv_fraction : fair value as a fraction of face_value (scalar)
                   or (npv_fraction, se_fraction) tuple if _return_se=True
    """
    autocall_barrier = autocall_barrier_pct * spot
    coupon_barrier   = coupon_barrier_pct   * spot
    knockin_barrier  = knockin_barrier_pct  * spot
    coupon_amount    = (coupon_rate / obs_per_year) * face_value

    n_paths, n_obs   = paths.shape
    payoffs          = np.zeros(n_paths)
    active           = np.ones(n_paths, dtype=bool)
    knockin_breached = np.zeros(n_paths, dtype=bool)
    accrued          = np.zeros(n_paths)   # memory: unpaid coupon amounts

    for i in range(n_obs):
        S       = paths[:, i]
        df      = discount_factors[i]
        is_last = (i == n_obs - 1)

        knockin_breached |= active & (S < knockin_barrier)

        # — Coupon step —
        at_coupon    = active & (S >= coupon_barrier)
        below_coupon = active & ~at_coupon

        if memory:
            # Pay current + all accrued when above barrier; reset accrual
            payoffs[at_coupon]    += df * (coupon_amount + accrued[at_coupon])
            accrued[at_coupon]     = 0.0
            accrued[below_coupon] += coupon_amount
        else:
            payoffs[at_coupon] += df * coupon_amount

        if not is_last:
            # — Autocall step (non-final observations only) —
            autocalled = active & (S >= autocall_barrier)
            payoffs[autocalled] += df * face_value
            active[autocalled]   = False
        else:
            # — Maturity: settle principal —
            full_principal = active & (~knockin_breached | (S >= spot))
            partial_loss   = active & knockin_breached & (S < spot)

            payoffs[full_principal] += df * face_value
            payoffs[partial_loss]   += df * face_value * (S[partial_loss] / spot)

    npv_fraction = float(payoffs.mean() / face_value)
    if _return_se:
        se_fraction = float((payoffs / face_value).std() / np.sqrt(n_paths))
        return npv_fraction, se_fraction
    return npv_fraction


def worst_of_payoff(
    paths: np.ndarray,
    spots: list,
    face_value: float,
    autocall_barrier_pct: float,
    coupon_barrier_pct: float,
    knockin_barrier_pct: float,
    coupon_rate: float,
    obs_per_year: float,
    discount_factors: np.ndarray,
    memory: bool = False,
    _return_se: bool = False,
):
    """
    Compute the fair value of a worst-of Phoenix autocallable note.

    Identical Phoenix logic to autocallable_payoff(), but all barrier checks
    apply to the worst-performing underlier at each observation date (§5.3):

        worst_perf[path, date] = min(S_i / S0_i  for each underlier i)

    This single normalised level is then compared to the barrier percentages,
    so the payoff logic is exactly the same as the single-underlier case.

    Parameters
    ----------
    paths  : (n_paths, n_underliers, n_obs) absolute stock prices
    spots  : [S0_asset1, S0_asset2, ...] initial prices for each underlier
    ...    : same barrier / coupon / discount parameters as autocallable_payoff()
    memory : same memory coupon feature as autocallable_payoff()

    Returns
    -------
    npv_fraction : fair value as a fraction of face_value (scalar)
                   or (npv_fraction, se_fraction) tuple if _return_se=True
    """
    spots_arr  = np.array(spots, dtype=float)
    perf       = paths / spots_arr[np.newaxis, :, np.newaxis]  # normalise by S0
    worst_perf = perf.min(axis=1)                               # (n_paths, n_obs)

    n_paths, n_obs   = worst_perf.shape
    coupon_amount    = (coupon_rate / obs_per_year) * face_value
    payoffs          = np.zeros(n_paths)
    active           = np.ones(n_paths, dtype=bool)
    knockin_breached = np.zeros(n_paths, dtype=bool)
    accrued          = np.zeros(n_paths)

    for i in range(n_obs):
        wp      = worst_perf[:, i]   # worst performer level (fraction of initial)
        df      = discount_factors[i]
        is_last = (i == n_obs - 1)

        knockin_breached |= active & (wp < knockin_barrier_pct)

        at_coupon    = active & (wp >= coupon_barrier_pct)
        below_coupon = active & ~at_coupon

        if memory:
            payoffs[at_coupon]    += df * (coupon_amount + accrued[at_coupon])
            accrued[at_coupon]     = 0.0
            accrued[below_coupon] += coupon_amount
        else:
            payoffs[at_coupon] += df * coupon_amount

        if not is_last:
            autocalled = active & (wp >= autocall_barrier_pct)
            payoffs[autocalled] += df * face_value
            active[autocalled]   = False
        else:
            full_principal = active & (~knockin_breached | (wp >= 1.0))
            partial_loss   = active & knockin_breached & (wp < 1.0)

            payoffs[full_principal] += df * face_value
            payoffs[partial_loss]   += df * face_value * wp[partial_loss]

    npv_fraction = float(payoffs.mean() / face_value)
    if _return_se:
        se_fraction = float((payoffs / face_value).std() / np.sqrt(n_paths))
        return npv_fraction, se_fraction
    return npv_fraction
