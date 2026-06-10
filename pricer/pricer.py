import json
import numpy as np
from datetime import datetime
from pathlib import Path

from .heston import load_params
from .monte_carlo import generate_paths, generate_paths_multi
from .payoff import autocallable_payoff, worst_of_payoff



def _year_fraction(start: datetime.date, end: datetime.date) -> float:
    return (end - start).days / 365.0


def _discount_factors(rate: float, times: list[float]) -> np.ndarray:
    return np.exp(-rate * np.array(times))


def price_note_dict(note: dict, n_paths: int = 50_000, seed: int = 42,
                    memory: bool = False,
                    _heston_params: dict = None) -> dict:
    """Price a note from a dict instead of a JSON file path.

    _heston_params : optional override — bypasses load_params(); used by
                     greeks.py to bump individual Heston parameters without
                     modifying calibrated files on disk.
    """
    return _price_note_core(note, n_paths=n_paths, seed=seed, memory=memory,
                            heston_override=_heston_params)


def _price_note_core(note: dict, n_paths: int = 50_000, seed: int = 42,
                     memory: bool = False, heston_override: dict = None) -> dict:
    """Shared implementation used by both price_note and price_note_dict."""
    issue_date = datetime.strptime(note['issue_date'], '%Y-%m-%d').date()
    obs_dates  = [datetime.strptime(d, '%Y-%m-%d').date() for d in note['observation_dates']]

    obs_times = [_year_fraction(issue_date, d) for d in obs_dates]

    # §6.1 — credit-adjusted discount curve: r_treasury + issuer_credit_spread
    effective_rate = note['risk_free_rate'] + note.get('credit_spread', 0.0)
    discount_factors = _discount_factors(effective_rate, obs_times)

    obs_per_year = len(obs_dates) / obs_times[-1]

    heston_params = heston_override if heston_override is not None else load_params(note['underlier'])
    heston_params = dict(heston_params)
    heston_params['risk_free_rate'] = note['risk_free_rate']

    paths = generate_paths(
        spot=note['spot'],
        heston_params=heston_params,
        observation_times=obs_times,
        n_paths=n_paths,
        seed=seed,
    )

    npv_fraction, se_fraction = autocallable_payoff(
        paths=paths,
        spot=note['spot'],
        face_value=note['face_value'],
        autocall_barrier_pct=note['autocall_barrier'],
        coupon_barrier_pct=note['coupon_barrier'],
        knockin_barrier_pct=note['knockin_barrier'],
        coupon_rate=note['coupon_rate'],
        obs_per_year=obs_per_year,
        discount_factors=discount_factors,
        memory=memory,
        _return_se=True,
    )

    return {
        'underlier':  note['underlier'],
        'npv_pct':    round(npv_fraction * 100, 2),
        'npv_dollar': round(npv_fraction * note['face_value'], 2),
        'se_pct':     round(se_fraction * 100, 3),
        'se_bps':     round(se_fraction * 10000, 1),
        'face_value': note['face_value'],
        'n_paths':    n_paths,
    }


def price_note(note_path: str, n_paths: int = 50_000, seed: int = 42,
               memory: bool = False) -> dict:
    """
    Price a single-underlier Phoenix autocallable note from a JSON term sheet.

    Parameters
    ----------
    note_path : path to a JSON file following the data/sample_note.json schema
    n_paths   : Monte Carlo paths — higher = more accurate, but slower
                50,000 is a good balance; use 10,000 for quick runs
    seed      : random seed so results are reproducible

    Returns
    -------
    dict with keys:
        underlier   - ticker symbol
        npv_pct     - fair value as % of face (e.g. 96.20)
        npv_dollar  - fair value in dollars per $1,000 face (e.g. 962.00)
        face_value  - note face value
        n_paths     - paths used
    """
    note = json.loads(Path(note_path).read_text())
    return _price_note_core(note, n_paths=n_paths, seed=seed, memory=memory)


def price_worst_of(note: dict, n_paths: int = 50_000, seed: int = 42,
                   memory: bool = False) -> dict:
    """
    Price a worst-of Phoenix autocallable note.

    note dict schema — same barrier / coupon fields as the single-underlier note,
    plus these multi-asset fields:

        underliers         : ["NVDA", "TSLA"]          — 2 or 3 tickers
        spots              : [213.73, 180.00]           — initial prices
        correlation_matrix : [[1.0, 0.55], [0.55, 1.0]] — asset × asset

    All other fields (face_value, issue_date, maturity_date, observation_dates,
    autocall_barrier, coupon_barrier, knockin_barrier, coupon_rate, risk_free_rate)
    carry the same meaning as in the single-underlier schema.

    Returns
    -------
    dict with keys:
        underliers  - list of tickers
        npv_pct     - fair value as % of face
        npv_dollar  - fair value in dollars per face unit
        face_value  - note face value
        n_paths     - paths used
    """
    issue_date = datetime.strptime(note['issue_date'], '%Y-%m-%d').date()
    obs_dates  = [datetime.strptime(d, '%Y-%m-%d').date() for d in note['observation_dates']]

    obs_times = [_year_fraction(issue_date, d) for d in obs_dates]
    effective_rate   = note['risk_free_rate'] + note.get('credit_spread', 0.0)
    discount_factors = _discount_factors(effective_rate, obs_times)
    obs_per_year     = len(obs_dates) / obs_times[-1]

    tickers = note['underliers']
    spots   = note['spots']
    corr    = np.array(note['correlation_matrix'], dtype=float)

    heston_params_list = []
    for ticker in tickers:
        p = load_params(ticker)
        p['risk_free_rate'] = note['risk_free_rate']
        heston_params_list.append(p)

    paths = generate_paths_multi(
        spots=spots,
        heston_params_list=heston_params_list,
        correlation_matrix=corr,
        observation_times=obs_times,
        n_paths=n_paths,
        seed=seed,
    )

    npv_fraction, se_fraction = worst_of_payoff(
        paths=paths,
        spots=spots,
        face_value=note['face_value'],
        autocall_barrier_pct=note['autocall_barrier'],
        coupon_barrier_pct=note['coupon_barrier'],
        knockin_barrier_pct=note['knockin_barrier'],
        coupon_rate=note['coupon_rate'],
        obs_per_year=obs_per_year,
        discount_factors=discount_factors,
        memory=memory,
        _return_se=True,
    )

    return {
        'underliers': tickers,
        'npv_pct':    round(npv_fraction * 100, 2),
        'npv_dollar': round(npv_fraction * note['face_value'], 2),
        'se_pct':     round(se_fraction * 100, 3),
        'se_bps':     round(se_fraction * 10000, 1),
        'face_value': note['face_value'],
        'n_paths':    n_paths,
    }
