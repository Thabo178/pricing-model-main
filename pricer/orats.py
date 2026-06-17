"""
Step 1.3 — ORATS vol surface ingestion

Endpoints used:
  /summaries       — SMV summary: ATM IV, slope, curvature, IV rank
  /strikes         — Full strike-level chain with ORATS theoretical prices and IVs
  /monies/implied  — Constant-maturity smoothed IV at standard delta points (main calibration input)
  /cores           — Core metadata including dividend / carry-related fields

Source: Structured Note Pricing Model Technical Reference, §4, §7, §11
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import norm


load_dotenv(Path(__file__).parent.parent / ".env")

ORATS_TOKEN = os.getenv("ORATS_API_TOKEN")
ORATS_BASE = "https://api.orats.io/datav2"
REQUEST_TIMEOUT = 20

# Call-delta columns to sample from monies/implied (fraction → column name)
_DELTA_COLS = {
    "vol15": 0.15,
    "vol25": 0.25,
    "vol35": 0.35,
    "vol50": 0.50,
    "vol65": 0.65,
    "vol75": 0.75,
    "vol85": 0.85,
}


def _require_token():
    if not ORATS_TOKEN:
        raise RuntimeError("ORATS_API_TOKEN is not set in .env")


def _get_orats(endpoint: str, ticker: str) -> pd.DataFrame:
    """
    Generic ORATS GET helper.
    Returns the 'data' payload as a pandas DataFrame.
    """
    _require_token()

    r = requests.get(
        f"{ORATS_BASE}/{endpoint}",
        params={"token": ORATS_TOKEN, "ticker": ticker},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()

    payload = r.json()
    data = payload.get("data", [])
    return pd.DataFrame(data)


def get_smv_summary(ticker: str) -> pd.DataFrame:
    """ORATS Smoothed Market Values summary — constant-maturity IV at delta points."""
    return _get_orats("summaries", ticker)


def get_strikes(ticker: str) -> pd.DataFrame:
    """Full strike-level data including ORATS theoretical prices."""
    return _get_orats("strikes", ticker)


def get_monies_implied(ticker: str) -> pd.DataFrame:
    """Constant-maturity smoothed IV at standard delta points — main calibration input."""
    return _get_orats("monies/implied", ticker)


def get_cores(ticker: str) -> pd.DataFrame:
    """
    ORATS cores endpoint — core metadata, often including dividend/carry fields.
    """
    return _get_orats("cores", ticker)


def live_spot(ticker: str) -> float:
    """Return the current spot price from ORATS SMV summary."""
    smv = get_smv_summary(ticker)
    if smv.empty:
        raise RuntimeError(f"No ORATS summary data returned for {ticker}")

    row = smv.iloc[0]

    for col in ["stockPrice", "spotPrice", "price"]:
        if col in row and pd.notna(row[col]):
            return float(row[col])

    raise RuntimeError(f"Could not find spot price field in ORATS summary for {ticker}")


def live_dividend_yield(ticker: str) -> float:
    """
    Return annual continuous dividend yield q from ORATS /cores.

    Falls back to 0.0 if the endpoint is empty or no recognized dividend field exists.
    If ORATS returns a percent-like value (> 1), convert to decimal.
    """
    cores = get_cores(ticker)
    if cores.empty:
        return 0.0

    row = cores.iloc[0]

    candidate_cols = [
        "divYield",
        "yield",
        "dividendYield",
        "forwardDivYield",
        "annualDividendYield",
    ]

    for col in candidate_cols:
        if col in row and pd.notna(row[col]):
            q = float(row[col])
            if q > 1.0:
                q /= 100.0
            return max(0.0, q)

    return 0.0


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
        K = F * exp(-N⁻¹(Δ) * σ√T + 0.5σ²T)
    where F = S * exp((r - q) * T) is the forward price.

    Source: Structured Note Pricing Model Technical Reference, §4 Step 1.3
    """
    import QuantLib as ql

    df = get_monies_implied(ticker)
    cal_set = []

    if df.empty:
        return cal_set

    for _, row in df.iterrows():
        try:
            expiry_raw = row.get("expirDate")
            expiry_dt = datetime.strptime(str(expiry_raw), "%Y-%m-%d")
            expiry_ql = ql.Date(expiry_dt.day, expiry_dt.month, expiry_dt.year)
            days = int(expiry_ql - eval_date)
        except Exception:
            continue

        if days < min_days or days > max_days:
            continue

        confidence = float(row.get("confidence", 1.0))
        if confidence < min_confidence:
            continue

        t = days / 365.0
        if t <= 0:
            continue

        s = float(row.get("spotPrice", spot))
        rf = float(row.get("riskFreeRate", r))
        forward = s * np.exp((rf - q) * t)

        for col, delta in _DELTA_COLS.items():
            if col not in row or pd.isna(row[col]):
                continue

            sigma = float(row[col])
            if sigma <= 0.02 or sigma > 3.0:
                continue

            d1 = norm.ppf(delta)
            strike = forward * np.exp(-d1 * sigma * np.sqrt(t) + 0.5 * sigma**2 * t)

            if strike <= 0:
                continue

            cal_set.append(
                {
                    "expiry": expiry_ql,
                    "strike": float(strike),
                    "iv": float(sigma),
                }
            )

    return cal_set