"""
run_eval.py — Score the full pipeline against your labeled eval set.

Run this to get a baseline before changing anything.
Then run it again after each optimization (chunk size, top-k, threshold)
and keep changes only when the numbers improve.

Usage:
  python run_eval.py
"""

import json
from eval.metrics import run_full_eval

if __name__ == "__main__":
    results = run_full_eval()
    print("\n=== Eval Results ===")
    print(json.dumps(results, indent=2))
    print("\nPaste these into the eval table in your README.")
