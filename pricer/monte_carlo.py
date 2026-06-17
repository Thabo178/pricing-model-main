import numpy as np


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest positive semi-definite matrix.

    Clips negative eigenvalues to a small positive floor, then re-normalises
    so the diagonal stays exactly 1 (i.e. the result is a valid correlation matrix).
    Required before Cholesky-decomposing user-supplied correlation matrices, which
    may be slightly indefinite due to rounding or misspecification.
    """
    m = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(m)
    vals = np.maximum(vals, 1e-8)
    m = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.diag(m))
    return m / np.outer(d, d)


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

    Each underlier has its own Heston parameters. Cross-asset correlation is
    applied only to the stock Brownians (W1_i) via Cholesky decomposition of
    `correlation_matrix`. Each asset's variance Brownian (W2_i) is correlated
    with its own stock Brownian through the Heston rho_i parameter but is
    independent of other assets' processes.

    heston_params_list entries should contain:
      v0, kappa, theta, sigma, rho, risk_free_rate, dividend_yield
    """
    n_assets = len(spots)
    spots_arr = np.array(spots, dtype=float)

    r_arr = np.array([p["risk_free_rate"] for p in heston_params_list], dtype=float)
    q_arr = np.array([p.get("dividend_yield", 0.0) for p in heston_params_list], dtype=float)
    kappa_arr = np.array([p["kappa"] for p in heston_params_list], dtype=float)
    theta_arr = np.array([p["theta"] for p in heston_params_list], dtype=float)
    sigma_arr = np.array([p["sigma"] for p in heston_params_list], dtype=float)
    rho_arr = np.array([p["rho"] for p in heston_params_list], dtype=float)
    v0_arr = np.array([p["v0"] for p in heston_params_list], dtype=float)
    sqrt_1m_rho2 = np.sqrt(1.0 - rho_arr**2)

    T = max(observation_times)
    n_steps = max(int(T * n_steps_per_year), len(observation_times) * 20)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    corr_psd = nearest_psd(np.asarray(correlation_matrix, dtype=float))
    L = np.linalg.cholesky(corr_psd)

    obs_step_indices = [round(t / dt) for t in observation_times]

    rng = np.random.default_rng(seed)
    log_S = np.tile(np.log(spots_arr), (n_paths, 1))
    V = np.tile(v0_arr, (n_paths, 1))

    paths = np.zeros((n_paths, n_assets, len(observation_times)))
    obs_ptr = 0

    for step in range(n_steps):
        Z1 = rng.standard_normal((n_paths, n_assets)) @ L.T
        Z2 = rho_arr * Z1 + sqrt_1m_rho2 * rng.standard_normal((n_paths, n_assets))

        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)

        log_S += ((r_arr - q_arr) - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z1
        V += kappa_arr * (theta_arr - V_pos) * dt + sigma_arr * sqrt_V * sqrt_dt * Z2

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

    Under the risk-neutral measure with dividends:
        dS = (r - q) * S * dt + sqrt(V) * S * dW1
        dV = kappa * (theta - V) * dt + sigma * sqrt(V) * dW2
        corr(dW1, dW2) = rho

    We discretise these using Euler-Maruyama:
      - Stock:    log-Euler step to keep prices positive
      - Variance: full truncation (clamp V to 0 before each step)
    """
    kappa = heston_params["kappa"]
    theta = heston_params["theta"]
    v0 = heston_params["v0"]
    sigma = heston_params["sigma"]
    rho = heston_params["rho"]
    r = heston_params["risk_free_rate"]
    q = heston_params.get("dividend_yield", 0.0)

    T = max(observation_times)
    n_steps = max(int(T * n_steps_per_year), len(observation_times) * 20)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + np.sqrt(1.0 - rho**2) * rng.standard_normal((n_paths, n_steps))

    obs_step_indices = [round(t / dt) for t in observation_times]

    log_S = np.full(n_paths, np.log(spot))
    V = np.full(n_paths, v0)

    paths = np.zeros((n_paths, len(observation_times)))
    obs_ptr = 0

    for step in range(n_steps):
        V_pos = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)

        log_S += ((r - q) - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z1[:, step]
        V += kappa * (theta - V_pos) * dt + sigma * sqrt_V * sqrt_dt * Z2[:, step]

        if obs_ptr < len(obs_step_indices) and (step + 1) == obs_step_indices[obs_ptr]:
            paths[:, obs_ptr] = np.exp(log_S)
            obs_ptr += 1

    return paths