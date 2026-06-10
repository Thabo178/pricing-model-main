"""
Step 1.3 — ORATS vol surface ingestion

Endpoints used:
  /summaries       — SMV summary: ATM IV, slope, curvature, IV rank
  /strikes         — Full strike-level chain with ORATS theoretical prices and IVs
  /monies/implied  — Constant-maturity smoothed IV at standard delta points (main calibration input)

Source: Structured Note Pricing Model Technical Reference, §4 and §7
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv
from scipy.stats import norm

load_dotenv(Path(__file__).parent.parent / '.env')

ORATS_TOKEN = os.getenv('ORATS_API_TOKEN')
ORATS_BASE = 'https://api.orats.io/datav2'

# Call-delta columns to sample from monies/implied (fraction → column name)
_DELTA_COLS = {
    'vol15': 0.15,
    'vol25': 0.25,
    'vol35': 0.35,
    'vol50': 0.50,
    'vol65': 0.65,
    'vol75': 0.75,
    'vol85': 0.85,
}


def get_smv_summary(ticker):
    """ORATS Smoothed Market Values summary — constant-maturity IV at delta points."""
    r = requests.get(
        f'{ORATS_BASE}/summaries',
        params={'token': ORATS_TOKEN, 'ticker': ticker})
    r.raise_for_status()
    return pd.DataFrame(r.json()['data'])


def get_strikes(ticker):
    """Full strike-level data including ORATS theoretical prices."""
    r = requests.get(
        f'{ORATS_BASE}/strikes',
        params={'token': ORATS_TOKEN, 'ticker': ticker})
    r.raise_for_status()
    return pd.DataFrame(r.json()['data'])


def get_monies_implied(ticker):
    """Constant-maturity smoothed IV at standard delta points — main calibration input."""
    r = requests.get(
        f'{ORATS_BASE}/monies/implied',
        params={'token': ORATS_TOKEN, 'ticker': ticker})
    r.raise_for_status()
    return pd.DataFrame(r.json()['data'])


def live_spot(ticker: str) -> float:
    """Return the current spot price from ORATS SMV summary."""
    smv = get_smv_summary(ticker)
    return float(smv.iloc[0]['stockPrice'])


def build_calibration_set(
    ticker: str,
    eval_date,          # ql.Date
    spot: float,
    r: float = 0.0375,
    q: float = 0.0,
    min_days: int = 7,
    max_days: int = 730,
    min_confidence: float = 0.5,
) -> list:
    """
    Convert ORATS monies/implied data into a Heston calibration set.

    Each entry: {'expiry': ql.Date, 'strike': float, 'iv': float}

    Delta-to-strike conversion (call delta convention):
        K = F * exp(-N⁻¹(Δ) * σ√T + ½σ²T)
    where F = S * exp((r - q) * T) is the forward price.

    Source: Structured Note Pricing Model Technical Reference, §4 Step 1.3
    """
    import QuantLib as ql

    df = get_monies_implied(ticker)
    cal_set = []

    for _, row in df.iterrows():
        try:
            expiry_dt = datetime.strptime(str(row['expirDate']), '%Y-%m-%d')
            expiry_ql = ql.Date(expiry_dt.day, expiry_dt.month, expiry_dt.year)
            days = int(expiry_ql - eval_date)
        except Exception:
            continue

        if days < min_days or days > max_days:
            continue

        if float(row.get('confidence', 1.0)) < min_confidence:
            continue

        t = days / 365.0
        S = float(row.get('spotPrice', spot))
        rf = float(row.get('riskFreeRate', r))
        F = S * np.exp((rf - q) * t)

        for col, delta in _DELTA_COLS.items():
            if col not in row or pd.isna(row[col]):
                continue
            sigma = float(row[col])
            if sigma <= 0.02 or sigma > 3.0:
                continue

            # Call delta → strike
            d1 = norm.ppf(delta)
            K = F * np.exp(-d1 * sigma * np.sqrt(t) + 0.5 * sigma ** 2 * t)
            if K <= 0:
                continue

            cal_set.append({'expiry': expiry_ql, 'strike': K, 'iv': sigma})

    return cal_set
