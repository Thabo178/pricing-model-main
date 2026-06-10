"""
New offering evaluator (§6.4).

Workflow:
  1. Accept a note dict (same schema as price_note_dict / price_worst_of)
  2. Price it with the Heston MC pricer
  3. Compare model fair value to the issuer's offer price (par for new issuance)
  4. Return recommendation based on the document's ±1.5% threshold

Recommendation logic (§6.4):
    model_fv > offer + 1.5%  →  "Buy"       (model says note is cheap)
    model_fv < offer - 1.5%  →  "Skip"      (model says note is expensive)
    otherwise                →  "Gray Zone"  (within model noise band)

Output also includes:
    confidence : "High" if 2·SE < |deviation|, "Low" otherwise
                  — flags when the signal is smaller than MC noise
"""


def evaluate_offering(
    note: dict,
    offer_pct: float = 100.0,
    n_paths: int = 50_000,
    seed: int = 42,
) -> dict:
    """
    Evaluate a new note offering against the model fair value.

    Parameters
    ----------
    note      : term sheet dict (single-underlier or worst-of schema)
    offer_pct : issuer's offer price as % of face (default 100.0 = par)
    n_paths   : Monte Carlo paths
    seed      : random seed

    Returns
    -------
    dict with keys:
        model_fv      — model fair value (% of face)
        model_dollar  — model fair value per face unit ($)
        offer_pct     — issuer offer price (% of face)
        offer_dollar  — issuer offer price in $ per face unit
        deviation_pct — model_fv − offer_pct  (positive = note is cheap)
        deviation_bps — deviation in basis points
        se_bps        — Monte Carlo standard error (1σ, bps)
        recommendation— "Buy", "Skip", or "Gray Zone"
        confidence    — "High" if |deviation| > 2·SE, else "Low"
        underliers    — ticker(s) priced
    """
    is_worst_of = 'underliers' in note

    if is_worst_of:
        from .pricer import price_worst_of
        result = price_worst_of(note, n_paths=n_paths, seed=seed)
        underliers = ' / '.join(note['underliers'])
    else:
        from .pricer import price_note_dict
        result = price_note_dict(note, n_paths=n_paths, seed=seed)
        underliers = note['underlier']

    model_fv     = result['npv_pct']
    se_bps       = result['se_bps']
    face         = result['face_value']
    deviation    = model_fv - offer_pct          # % of face
    deviation_bps = round(deviation * 100, 1)

    # §6.4 recommendation thresholds
    if deviation > 1.5:
        recommendation = 'Buy'
    elif deviation < -1.5:
        recommendation = 'Skip'
    else:
        recommendation = 'Gray Zone'

    # Signal-to-noise: is the deviation distinguishable from MC noise?
    confidence = 'High' if abs(deviation_bps) > 2 * se_bps else 'Low'

    return {
        'underliers':     underliers,
        'model_fv':       round(model_fv, 2),
        'model_dollar':   round(model_fv * face / 100, 2),
        'offer_pct':      round(offer_pct, 2),
        'offer_dollar':   round(offer_pct * face / 100, 2),
        'deviation_pct':  round(deviation, 3),
        'deviation_bps':  deviation_bps,
        'se_bps':         se_bps,
        'recommendation': recommendation,
        'confidence':     confidence,
        'face_value':     face,
        'n_paths':        n_paths,
    }
