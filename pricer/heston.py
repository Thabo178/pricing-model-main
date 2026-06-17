# Heston stochastic volatility model parameters for each underlier.
#
# The five Heston parameters:
#   v0    - initial variance (e.g. 0.1225 = 35% vol squared)
#   kappa - mean-reversion speed (how fast vol snaps back to theta)
#   theta - long-run variance (where vol gravitates over time)
#   sigma - vol of vol (how volatile the volatility itself is)
#   rho   - correlation between stock returns and vol changes
#           (negative because stocks usually spike vol when they fall)
#
# These are reasonable starting estimates. Once real ORATS vol surface
# data is available, they should be calibrated properly.

HESTON_PARAMS = {
    'NVDA': {'v0': 0.1225, 'kappa': 2.0, 'theta': 0.0900, 'sigma': 0.50, 'rho': -0.70, 'dividend_yield': 0.0},
    'TSLA': {'v0': 0.1600, 'kappa': 1.8, 'theta': 0.1200, 'sigma': 0.60, 'rho': -0.65, 'dividend_yield': 0.0},
    'AMD':  {'v0': 0.1100, 'kappa': 2.0, 'theta': 0.0800, 'sigma': 0.45, 'rho': -0.70, 'dividend_yield': 0.0},
    'META': {'v0': 0.0900, 'kappa': 2.2, 'theta': 0.0700, 'sigma': 0.40, 'rho': -0.65, 'dividend_yield': 0.0},
    'GOOGL':{'v0': 0.0700, 'kappa': 2.5, 'theta': 0.0600, 'sigma': 0.35, 'rho': -0.60, 'dividend_yield': 0.0},
    'AMZN': {'v0': 0.0800, 'kappa': 2.2, 'theta': 0.0650, 'sigma': 0.40, 'rho': -0.65, 'dividend_yield': 0.0},
    'HOOD': {'v0': 0.2500, 'kappa': 1.5, 'theta': 0.1800, 'sigma': 0.70, 'rho': -0.60, 'dividend_yield': 0.0},
    'LULU': {'v0': 0.1400, 'kappa': 1.8, 'theta': 0.1000, 'sigma': 0.50, 'rho': -0.65, 'dividend_yield': 0.0},
    'NOW':  {'v0': 0.0900, 'kappa': 2.0, 'theta': 0.0700, 'sigma': 0.42, 'rho': -0.62, 'dividend_yield': 0.0},
    'PLTR': {'v0': 0.1800, 'kappa': 1.6, 'theta': 0.1400, 'sigma': 0.65, 'rho': -0.60, 'dividend_yield': 0.0},
    'WFC':  {'v0': 0.0600, 'kappa': 2.5, 'theta': 0.0500, 'sigma': 0.30, 'rho': -0.55, 'dividend_yield': 0.0},
    'SPY':  {'v0': 0.0400, 'kappa': 3.0, 'theta': 0.0350, 'sigma': 0.25, 'rho': -0.75, 'dividend_yield': 0.0},
}

_DEFAULT = {'v0': 0.1000, 'kappa': 2.0, 'theta': 0.0800, 'sigma': 0.45, 'rho': -0.65, 'dividend_yield': 0.0}


def get_heston_params(underlier: str) -> dict:
    """Return default (hardcoded) Heston parameters. Used as calibration starting point."""
    return HESTON_PARAMS.get(underlier.upper(), _DEFAULT).copy()


def load_params(underlier: str, calibrated_dir: str = None) -> dict:
    """
    Return the best available Heston parameters for an underlier.

    Checks data/calibrated/{TICKER}.json first (produced by calibrate.py).
    Falls back to hardcoded defaults when no calibration file exists.

    Returns:
        {
            'v0': ...,
            'kappa': ...,
            'theta': ...,
            'sigma': ...,
            'rho': ...,
            'dividend_yield': ...
        }
    """
    import json
    from pathlib import Path

    if calibrated_dir is None:
        calibrated_dir = Path(__file__).parent.parent / 'data' / 'calibrated'

    path = Path(calibrated_dir) / f"{underlier.upper()}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return {
            'v0': float(data['v0']),
            'kappa': float(data['kappa']),
            'theta': float(data['theta']),
            'sigma': float(data['sigma']),
            'rho': float(data['rho']),
            'dividend_yield': float(data.get('dividend_yield', 0.0)),
        }

    params = get_heston_params(underlier)
    params.setdefault('dividend_yield', 0.0)
    return params