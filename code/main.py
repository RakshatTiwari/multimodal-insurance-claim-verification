#!/usr/bin/env python3
"""
main.py — Main entrypoint for the Damage Claim Verification System.

Usage:
    python main.py                          # process claims.csv → output.csv
    python main.py --evaluate               # run evaluation on sample_claims.csv first
    python main.py --evaluate --no-test     # evaluate only, skip test set
    python main.py --claims path/to/file    # custom claims file
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "verification"))
sys.path.insert(0, str(Path(__file__).parent / "evaluation"))

from verification.verify import process_claims, load_csv

BASE = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(
        description="Damage Claim Verification System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--claims", default=str(BASE / "dataset" / "claims.csv"),
                        help="Input claims CSV (default: dataset/claims.csv)")
    parser.add_argument("--sample", default=str(BASE / "dataset" / "sample_claims.csv"),
                        help="Sample claims CSV for evaluation")
    parser.add_argument("--evidence", default=str(BASE / "dataset" / "evidence_requirements.csv"))
    parser.add_argument("--history", default=str(BASE / "dataset" / "user_history.csv"))
    parser.add_argument("--output", default=str(BASE / "dataset" / "output.csv"),
                        help="Output predictions CSV (default: dataset/output.csv)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation on sample_claims.csv before test set")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip processing claims.csv (evaluate only)")
    parser.add_argument("--base-dir", default=str(BASE),
                        help="Base directory for image path resolution")

    args = parser.parse_args()
    base = Path(args.base_dir)

    # ── Step 1: Evaluate on sample set ───────────────────────────────────────
    if args.evaluate:
        sample_path = base / args.sample
        if not sample_path.exists():
            print(f"✗ Sample file not found: {sample_path}", file=sys.stderr)
            sys.exit(1)

        print("=" * 60)
        print("STEP 1: Running evaluation on sample_claims.csv")
        print("=" * 60)

        import evaluation.evaluate
        evaluation.evaluate.main()

    # ── Step 2: Process test claims ───────────────────────────────────────────
    if not args.no_test:
        claims_path = base / args.claims
        if not claims_path.exists():
            print(f"✗ Claims file not found: {claims_path}", file=sys.stderr)
            sys.exit(1)

        print("\n" + "=" * 60)
        print("STEP 2: Processing test claims → output.csv")
        print("=" * 60)

        t0 = time.time()
        process_claims(
            claims_path=str(claims_path),
            evidence_path=str(base / args.evidence),
            history_path=str(base / args.history),
            output_path=args.output,
            base_dir=str(base),
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"\n✓ Completed in {elapsed:.1f}s — output saved to {args.output}")


if __name__ == "__main__":
    main()
