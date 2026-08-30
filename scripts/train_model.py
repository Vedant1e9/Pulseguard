"""
PatientTriage.ai — Model Training Entry Point

Trains on real NHAMCS emergency department visits, splits by hospital so the
test set is made of departments the model has never seen, calibrates its
probabilities, fits conformal thresholds, and selects an operating point
against an explicit escalation budget.

Usage:
    python -m scripts.train_model
    python -m scripts.train_model --years 2021 2022 --alpha 0.10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.real.nhamcs_loader import load_clean
from models.decision_policy import SITE_PROFILES
from models.triage_model import train_bundle, save_bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022])
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--max-lane-load", type=float, default=0.35)
    ap.add_argument("--min-critical-recall", type=float, default=0.65)
    ap.add_argument("--profile", default="urban_trauma_center",
                    choices=list(SITE_PROFILES.keys()))
    ap.add_argument("--out", default="saved_models/triage_bundle.joblib")
    args = ap.parse_args()

    print("=" * 72)
    print("PatientTriage.ai — Training on real emergency department data")
    print("=" * 72)

    df = load_clean(years=tuple(args.years))
    print(f"\nCohort: {len(df):,} visits with a nurse-assigned triage level "
          f"across {df.hospital_id.nunique()} hospitals")

    bundle, report = train_bundle(
        df,
        alpha=args.alpha,
        cost_matrix=SITE_PROFILES[args.profile],
        max_critical_lane_load=args.max_lane_load,
        min_critical_recall=args.min_critical_recall,
        verbose=True,
    )

    save_bundle(bundle, args.out)
    os.makedirs("evaluation/saved_results", exist_ok=True)
    with open("evaluation/saved_results/training_report.json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "test_fold_indices"},
                  f, indent=2, default=str)

    print(f"\n✓ Bundle saved to {args.out}")
    print("✓ Training report saved to evaluation/saved_results/training_report.json")


if __name__ == "__main__":
    main()
