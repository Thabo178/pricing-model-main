"""
Portfolio mark-to-model pricer (§10).

Loads a JSON portfolio file and prices every note — single-underlier or worst-of —
comparing model fair value to the issuer's mark and flagging large deviations.

Deviation thresholds (§10):
  |dev| ≤ 100 bps  — within model noise; OK
  100 < |dev| ≤ 300 bps — review; likely surface or correlation issue
  |dev| > 300 bps  — flag; model misspecification or issuer mark off

Portfolio JSON schema — each note contains the standard term sheet fields plus:
  cusip          : identifier string
  issuer         : issuer name
  purchase_price : price paid (% of face; 100.0 = par)
  issuer_mark    : current issuer mark (% of face)
  memory         : bool (optional, default false)
  For worst-of notes: underliers, spots, correlation_matrix
  For single-underlier: underlier, spot
"""
import json
from pathlib import Path


def price_portfolio(
    portfolio_path: str,
    n_paths: int = 30_000,
    seed: int = 42,
) -> list:
    """
    Price all notes in a portfolio JSON file.

    Parameters
    ----------
    portfolio_path : path to JSON file with a 'notes' array
    n_paths        : MC paths per note (30k is a good balance for portfolios)
    seed           : random seed

    Returns
    -------
    List of result dicts, one per note, with keys:
        cusip, issuer, structure, underliers, face_value,
        purchase_price, issuer_mark, model_fv, model_dollar,
        deviation_bps, pnl_vs_purchase, flag, error (if failed)
    """
    data  = json.loads(Path(portfolio_path).read_text())
    notes = data.get('notes', [])

    from .pricer import price_note_dict, price_worst_of

    results = []
    for note in notes:
        try:
            memory = note.get('memory', False)

            if 'underliers' in note:
                result      = price_worst_of(note, n_paths=n_paths, seed=seed, memory=memory)
                structure   = 'Worst-Of'
                underliers  = ' / '.join(note['underliers'])
            else:
                result      = price_note_dict(note, n_paths=n_paths, seed=seed, memory=memory)
                structure   = 'Single' + (' + Memory' if memory else '')
                underliers  = note['underlier']

            model_fv     = result['npv_pct']
            model_dollar = result['npv_dollar']
            se_bps       = result['se_bps']
            issuer_mark  = note.get('issuer_mark')
            purch_price  = note.get('purchase_price')

            deviation_bps = (
                round((model_fv - issuer_mark) * 100, 1)
                if issuer_mark is not None else None
            )
            pnl_vs_purchase = (
                round((model_fv - purch_price) * note['face_value'] / 100, 2)
                if purch_price is not None else None
            )

            if deviation_bps is None:
                flag = 'N/A'
            elif abs(deviation_bps) <= 100:
                flag = 'OK'
            elif abs(deviation_bps) <= 300:
                flag = 'Review'
            else:
                flag = 'Flag ⚠'

            results.append({
                'cusip':          note.get('cusip', 'N/A'),
                'issuer':         note.get('issuer', 'N/A'),
                'structure':      structure,
                'underliers':     underliers,
                'face_value':     note['face_value'],
                'purchase_price': purch_price,
                'issuer_mark':    issuer_mark,
                'model_fv':       round(model_fv, 2),
                'model_dollar':   model_dollar,
                'se_bps':         se_bps,
                'deviation_bps':  deviation_bps,
                'pnl_vs_purchase': pnl_vs_purchase,
                'flag':           flag,
            })

        except Exception as e:
            results.append({
                'cusip':     note.get('cusip', 'N/A'),
                'issuer':    note.get('issuer', 'N/A'),
                'structure': 'ERROR',
                'underliers': note.get('underlier',
                              ' / '.join(note.get('underliers', ['?']))),
                'flag':      'ERROR',
                'error':     str(e),
            })

    return results
