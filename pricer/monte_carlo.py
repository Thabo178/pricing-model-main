import numpy as np


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix.

    Clips negative eigenvalues to a small positive floor, then re-normalises
    so the diagonal stays exactly 1 (i.e. the result is a valid correlation matrix).
    Required before Cholesky-decomposing user-supplied correlation matrices, which
    may be slightly indefinite due to rounding or misspecification.
    """
    m = (matrix + matrix.T) / 2.0          # enforce symmetry
    vals, vecs = np.linalg.eigh(m)
    vals = np.maximum(vals, 1e-8)
    m = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(m))
    return m / np.outer(d, d)              # re-normalise to unit diagonal


def generate_paths_multi(
    spots: list,
    heston_params_list: list,
    correlation_matrix: np.ndarray,
    observation_times: list,
    n_paths: int = 50_000,
    n_steps_per_year: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate correlated stock price paths for 2 or 3 underliers under Heston SV.

    Each underlier has its own Heston parameters.  Cross-asset correlation is
    applied only to the stock Brownians (W1_i) via Cholesky decomposition of
    `correlation_matrix`.  Each asset's variance Brownian (W2_i) is correlated
    with its own stock Brownian through the Heston rho_i parameter but is
    independent of other assets' processes — the standard multi-asset Heston setup.

    Random numbers are generated step-by-step (not pre-allocated) to keep
    peak memory at O(n_paths × n_assets) per step rather than O(n_paths × n_steps × n_assets).

    Parameters
    ----------
    spots                : [S0_asset1, S0_asset2, ...] initial prices
    heston_params_list   : [params_asset1, params_asset2, ...] — each dict has
                           v0, kappa, theta, sigma, rho, risk_free_rate
    correlation_matrix   : (n_assets, n_assets) correlation matrix between stock processes.
                           Projected to PSD automatically if not already.
    observation_times    : year-fractions at which to record prices
    n_paths              : number of Monte Carlo paths
    n_steps_per_year     : Euler steps per year (252 = daily)
    seed                 : random seed for reproducibility

    Returns
    -------
    paths : np.ndarray of shape (n_paths, n_underliers, n_obs)
            Absolute stock price at each observation date for each path and underlier
    """
    n_assets  = len(spots)
    spots_arr = np.array(spots, dtype=float)

    r_arr         = np.array([p['risk_free_rate'] for p in heston_params_list])
    kappa_arr     = np.array([p['kappa']          for p in heston_params_list])
    theta_arr     = np.array([p['theta']          for p in heston_params_list])
    sigma_arr     = np.array([p['sigma']          for p in heston_params_list])
    rho_arr       = np.array([p['rho']            for p in heston_params_list])
    v0_arr        = np.array([p['v0']             for p in heston_params_list])
    sqrt_1m_rho2  = np.sqrt(1.0 - rho_arr ** 2)

    T       = max(observation_times)
    n_steps = max(int(T * n_steps_per_year), len(observation_times) * 20)
    dt      = T / n_steps
    sqrt_dt = np.sqrt(dt)

    corr_psd = nearest_psd(np.asarray(correlation_matrix, dtype=float))
    L = np.linalg.cholesky(corr_psd)  # lower triangular: L @ L.T = corr_psd

    obs_step_indices = [round(t / dt) for t in observation_times]

    rng   = np.random.default_rng(seed)
    log_S = np.tile(np.log(spots_arr), (n_paths, 1))  # (n_paths, n_assets)
    V     = np.tile(v0_arr,            (n_paths, 1))   # (n_paths, n_assets)

    paths   = np.zeros((n_paths, n_assets, len(observation_times)))
    obs_ptr = 0

    for step in range(n_steps):
        # Correlated stock Brownians: Z1 rows are jointly distributed with cov = corr_psd
        Z1 = rng.standard_normal((n_paths, n_assets)) @ L.T   # (n_paths, n_assets)

        # Variance Brownians: correlated with own stock Brownian (Heston rho), not others
        Z2 = rho_arr * Z1 + sqrt_1m_rho2 * rng.standard_normal((n_paths, n_assets))

        V_pos  = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)

        log_S += (r_arr - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z1
        V     += kappa_arr * (theta_arr - V_pos) * dt + sigma_arr * sqrt_V * sqrt_dt * Z2

        if obs_ptr < len(obs_step_indices) and (step + 1) == obs_step_indices[obs_ptr]:
            paths[:, :, obs_ptr] = np.exp(log_S)
            obs_ptr += 1

    return paths


def generate_paths(
    spot: float,
    heston_params: dict,
    observation_times: list[float],
    n_paths: int = 50_000,
    n_steps_per_year: int = 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate stock price paths under the Heston stochastic volatility model.

    The Heston model uses two coupled SDEs (stochastic differential equations):

        dS = r * S * dt  +  sqrt(V) * S * dW1          (stock price)
        dV = k * (θ - V) * dt  +  σ * sqrt(V) * dW2   (variance)
        corr(dW1, dW2) = ρ

    We discretise these using Euler-Maruyama:
      - Stock:    log-Euler step to keep prices positive
      - Variance: full truncation (clamp V to 0 before each step) to avoid sqrt of negatives

    Parameters
    ----------
    spot               : current stock price
    heston_params      : dict with keys v0, kappa, theta, sigma, rho, risk_free_rate
    observation_times  : list of year-fractions at which we record the stock price
    n_paths            : number of Monte Carlo simulations (more = more accurate, slower)
    n_steps_per_year   : time-steps per year (252 = daily)
    seed               : random seed for reproducibility

    Returns
    -------
    paths : np.ndarray of shape (n_paths, len(observation_times))
            Stock price at each observation date for each simulated path
    """
    kappa = heston_params['kappa']
    theta = heston_params['theta']
    v0    = heston_params['v0']
    sigma = heston_params['sigma']
    rho   = heston_params['rho']
    r     = heston_params['risk_free_rate']

    T       = max(observation_times)
    n_steps = max(int(T * n_steps_per_year), len(observation_times) * 20)
    dt      = T / n_steps
    sqrt_dt = np.sqrt(dt)

    # Two correlated Brownian motions:
    # dW2 = rho * dW1 + sqrt(1 - rho^2) * dZ  (Cholesky decomposition)
    rng = np.random.default_rng(seed)
    Z1  = rng.standard_normal((n_paths, n_steps))
    Z2  = rho * Z1 + np.sqrt(1.0 - rho ** 2) * rng.standard_normal((n_paths, n_steps))

    # Map each observation time to a step index
    obs_step_indices = [round(t / dt) for t in observation_times]

    log_S = np.full(n_paths, np.log(spot))
    V     = np.full(n_paths, v0)

    paths   = np.zeros((n_paths, len(observation_times)))
    obs_ptr = 0

    for step in range(n_steps):
        V_pos  = np.maximum(V, 0.0)   # full truncation: no negative variance
        sqrt_V = np.sqrt(V_pos)

        # Log-Euler step for stock (exact for GBM, avoids negative prices)
        log_S += (r - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z1[:, step]

        # Euler step for variance
        V += kappa * (theta - V_pos) * dt + sigma * sqrt_V * sqrt_dt * Z2[:, step]

        if obs_ptr < len(obs_step_indices) and (step + 1) == obs_step_indices[obs_ptr]:
            paths[:, obs_ptr] = np.exp(log_S)
            obs_ptr += 1

    return paths
