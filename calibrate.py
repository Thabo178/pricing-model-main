"""
Calibrate Heston parameters for one or all supported underliers.

Usage:
python calibrate.py
python calibrate.py NVDA
python calibrate.py NVDA TSLA
python calibrate.py --mock
python calibrate.py NVDA --mock

Behavior:
- Live mode:
  1. Fetch spot from ORATS if available, otherwise fall back to a hardcoded spot.
  2. Build the ORATS calibration set.
  3. Delegate calibration to pricer.calibration.calibrate_heston_orats().
  4. Save the calibrated parameter JSON.

- Mock mode:
  1. Build a synthetic surface from default Heston params.
  2. Calibrate against that surface as a correctness test.
  3. Save the calibrated parameter JSON.

Important:
This script does not itself implement the optimizer. The actual live-vs-mock
calibration logic lives in pricer.calibration. This wrapper is intentionally
agnostic so the CLI and dashboard stay aligned with the underlying engine.
"""

import sys
import time
import QuantLib as ql

from pricer.calibration import (
    generate_mock_surface,
    calibrate_heston,
    calibrate_heston_orats,
    save_calibrated,
)
from pricer.orats import build_calibration_set, live_spot

UNDERLIERS = {
    "NVDA": 219.16,
    "TSLA": 180.00,
    "AMD": 160.00,
    "META": 510.00,
    "GOOGL": 175.00,
    "AMZN": 190.00,
    "HOOD": 22.00,
    "LULU": 85.00,
    "NOW": 820.00,
    "PLTR": 25.00,
    "WFC": 57.00,
    "SPY": 525.00,
}

RISK_FREE_RATE = 0.0375
TODAY = ql.Date.todaysDate()


def calibrate_name_live(ticker: str, fallback_spot: float) -> dict | None:
    t0 = time.time()

    used_fallback_spot = False
    try:
        spot = live_spot(ticker)
    except Exception as exc:
        print(f" {ticker:<6} !! ORATS live spot failed: {exc}")
        print(f" {ticker:<6} .. Falling back to hardcoded spot: {fallback_spot:.2f}")
        spot = fallback_spot
        used_fallback_spot = True

    try:
        cal_set = build_calibration_set(ticker, TODAY, spot, r=RISK_FREE_RATE)
    except Exception as exc:
        print(f" {ticker:<6} !! Failed to build ORATS calibration set: {exc}")
        return None

    if not cal_set:
        print(f" {ticker:<6} !! No ORATS surface data — skipping")
        return None

    try:
        result = calibrate_heston_orats(ticker, TODAY, spot, RISK_FREE_RATE, cal_set)
    except Exception as exc:
        print(f" {ticker:<6} !! Calibration failed: {exc}")
        return None

    result["spot_used"] = float(spot)
    result["used_fallback_spot"] = used_fallback_spot
    result["calibration_mode"] = "live_orats"

    save_calibrated(result)
    elapsed = time.time() - t0

    status = "OK" if result.get("converged", False) else "~~"
    feller = "✓" if result.get("feller_satisfied", True) else "✗"

    print(
        f" {ticker:<6} {status} "
        f"v0={result['v0']:.4f} "
        f"kappa={result['kappa']:.3f} "
        f"theta={result['theta']:.4f} "
        f"sigma={result['sigma']:.3f} "
        f"rho={result['rho']:.3f} "
        f"rmse={result.get('rmse', float('nan')):.5f} "
        f"({elapsed:.1f}s) "
        f"[{result.get('n_points', 0)} pts] "
        f"Feller:{feller} "
        f"{'[fallback spot]' if used_fallback_spot else '[live spot]'}"
    )

    return result


def calibrate_name_mock(ticker: str, spot: float) -> dict | None:
    t0 = time.time()

    try:
        surface = generate_mock_surface(ticker, spot, RISK_FREE_RATE, today=TODAY)
        result = calibrate_heston(ticker, spot, RISK_FREE_RATE, surface, today=TODAY)
    except Exception as exc:
        print(f" {ticker:<6} !! Mock calibration failed: {exc}")
        return None

    result["spot_used"] = float(spot)
    result["used_fallback_spot"] = True
    result["calibration_mode"] = "mock_surface"

    save_calibrated(result)
    elapsed = time.time() - t0

    status = "OK" if result.get("converged", False) else "~~"
    feller = "✓" if result.get("feller_satisfied", True) else "✗"

    print(
        f" {ticker:<6} {status} "
        f"v0={result['v0']:.4f} "
        f"kappa={result['kappa']:.3f} "
        f"theta={result['theta']:.4f} "
        f"sigma={result['sigma']:.3f} "
        f"rho={result['rho']:.3f} "
        f"rmse={result.get('rmse', float('nan')):.5f} "
        f"({elapsed:.1f}s) "
        f"Feller:{feller} [mock]"
    )

    return result


def main():
    raw_args = list(sys.argv[1:])
    use_mock = "--mock" in raw_args or "--MOCK" in raw_args
    tickers = [a.upper() for a in raw_args if not a.startswith("--")]

    if not tickers:
        tickers = list(UNDERLIERS)

    unknown = [t for t in tickers if t not in UNDERLIERS]
    if unknown:
        print(f"Unknown tickers: {unknown}")
        print(f"Valid options: {list(UNDERLIERS)}")
        sys.exit(1)

    mode = "synthetic surface (mock)" if use_mock else "live ORATS surface"
    print(f"\nCalibrating {len(tickers)} underlier(s) [{mode}] ...")
    print(f" {'Ticker':<6} {'v0':>7} {'kappa':>7} {'theta':>7} {'sigma':>7} {'rho':>7} {'rmse':>9}")
    print(f" {'-' * 78}")

    calibrate_fn = calibrate_name_mock if use_mock else calibrate_name_live

    successes = 0
    failures = 0

    for ticker in tickers:
        result = calibrate_fn(ticker, UNDERLIERS[ticker])
        if result is None:
            failures += 1
        else:
            successes += 1

    print(f"\nCompleted: {successes} succeeded, {failures} failed.")
    print("Calibrated parameters saved via pricer.calibration.save_calibrated().")


if __name__ == "__main__":
    main()