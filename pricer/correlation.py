"""
Historical correlation estimation for worst-of basket construction (§9.1).

Recommended blend: 60% weight on 60-day daily returns + 40% on 250-day weekly returns.
During high-vol regimes (VIX > 25), shift toward 100% 250-day weekly to capture
stress correlation.

Requires: pip install yfinance
"""
import numpy as np
import pandas as pd


def compute_historical_correlation(
    tickers: list,
    blend_60d_weight: float = 0.60,
) -> dict:
    """
    Compute blended pairwise asset correlation matrix (§9.1).

    Method:
      1. Download 1 year of daily closing prices via yfinance
      2. Compute 60-day daily return correlation (short-term, reactive)
      3. Compute ~52-week weekly return correlation (stable, regime-aware)
      4. Blend: blend_60d_weight × corr_60d + (1 - blend_60d_weight) × corr_250d

    Parameters
    ----------
    tickers         : list of ticker strings, e.g. ['NVDA', 'TSLA']
    blend_60d_weight: weight on 60-day daily component (default 0.60 per §9.1)

    Returns
    -------
    dict with keys:
        matrix  : correlation matrix as list of lists (n × n)
        tickers : ordered ticker list
        method  : description of computation
        n_obs   : number of daily observations downloaded
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for historical correlation. "
            "Install with: pip install yfinance"
        )

    raw = yf.download(tickers, period='1y', progress=False, auto_adjust=True)

    # Handle single-ticker case
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close'][tickers]
    else:
        prices = raw[['Close']].rename(columns={'Close': tickers[0]})

    prices = prices.dropna()

    if len(prices) < 65:
        raise ValueError(
            f"Only {len(prices)} trading days of data — need at least 65 for 60-day correlation."
        )

    # 60-day daily returns
    daily_rets = prices.pct_change().dropna()
    corr_60d   = daily_rets.tail(60).corr()

    # ~52-week weekly returns
    weekly_prices = prices.resample('W').last()
    weekly_rets   = weekly_prices.pct_change().dropna()
    corr_250d     = weekly_rets.tail(52).corr()

    # Blend
    corr_raw = blend_60d_weight * corr_60d + (1 - blend_60d_weight) * corr_250d

    # Clip to valid range and fix diagonal
    vals = np.clip(corr_raw.values, -1.0, 1.0)
    np.fill_diagonal(vals, 1.0)

    w60  = int(blend_60d_weight * 100)
    w250 = 100 - w60

    return {
        'matrix':  vals.tolist(),
        'tickers': list(tickers),
        'method':  f'{w60}% 60-day daily / {w250}% 52-week weekly',
        'n_obs':   len(prices),
    }
