"""
Entry point — run the autocallable pricer from the command line.

Usage:
    python run_pricer.py                          # prices data/sample_note.json
    python run_pricer.py data/my_note.json        # prices a custom note file
    python run_pricer.py data/sample_note.json 10000  # fast run with 10k paths
"""
import sys
from pricer import price_note

note_path = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_note.json'
n_paths   = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000

print(f"\nPricing note: {note_path}  ({n_paths:,} paths) ...")
result = price_note(note_path, n_paths=n_paths)

print(f"\n{'=' * 48}")
print(f"  Autocallable Pricer  —  {result['underlier']}")
print(f"{'=' * 48}")
print(f"  Fair Value :  {result['npv_pct']:.2f}%  of face")
print(f"  NPV        :  ${result['npv_dollar']:.2f}  per ${result['face_value']:.0f} face")
print(f"  MC Std Err :  ±{result['se_bps']:.1f} bps  (2σ = ±{result['se_bps']*2:.1f} bps)")
print(f"  Paths used :  {result['n_paths']:,}")
print(f"{'=' * 48}")
print()
print("  How to read this:")
print(f"  If a bank offers this note at 100.00% of face, you are")
print(f"  {'overpaying' if result['npv_pct'] < 100 else 'getting a fair deal'}.")
print(f"  Model says fair value is {result['npv_pct']:.2f}% — that is a")
print(f"  {abs(100 - result['npv_pct']):.2f}% {'discount to fair value' if result['npv_pct'] > 100 else 'premium over fair value'}.")
print()
