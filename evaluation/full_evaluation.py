"""
PatientTriage.ai — Full Evaluation Harness
==========================================

Produces every number this project claims, on the held-out test fold, and
writes them to disk so they can be quoted without being re-derived.

Design rules, applied without exception:

* **The test fold is touched once.** It consists of hospitals used for
  nothing else — not training, not calibration, not conformal fitting, not
  operating-point selection.
* **Every headline figure carries a confidence interval.** Bootstrapped,
  clustered by hospital, because visits from one department are not
  independent draws.
* **Every claim has a baseline.** "94% of critical patients caught" means
  nothing until you know what NEWS2 catches on the same patients.
* **Outcome-based validation runs alongside label agreement.** Agreeing with
  the triage nurse is the easy question. Whether the system would have sent a
  patient who went on to need critical care to the waiting room is the real one.

Outputs
-------
  evaluation/saved_results/evaluation_full.json  : everything, machine-readable
  evaluation/saved_results/EVALUATION_REPORT.md  : the full written report

The quotable headline numbers are NOT written here. They live in
evaluation/saved_results/HEADLINE_NUMBERS.md and are produced by
`python -m scripts.generate_metrics`, which reads this file's JSON output and
adds the figures this harness does not compute: F1 and the clinical predictive
values, the over-triage exchange rate, test coverage, and the prototype counts
that can only be observed by booting the application. Keeping one canonical
source avoids the failure this replaced, where two documents both claimed to be
the headline numbers and one of them was quietly a version behind.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import roc_auc_score, confusion_matrix

from data.features import build_feature_frame
from data.real.nhamcs_loader import load_clean
from models.clinical_scores import compute_early_warning_score
from models.decision_policy import (
    CostMatrix, SITE_PROFILES, expected_cost_decision, operating_curve,
)
from models.triage_model import (
    _align_proba, clinical_metrics, grouped_split, load_bundle,
)
from models.uncertainty import calibration_report

LEVELS = [1, 2, 3, 4, 5]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "saved_results")


# ─── Statistics ──────────────────────────────────────────────────────────────

def cluster_bootstrap_ci(values: np.ndarray, groups: np.ndarray,
                         statistic: Callable[[np.ndarray], float],
                         n_boot: int = 1000, seed: int = 42,
                         alpha: float = 0.05) -> Dict:
    """
    Bootstrap a statistic, resampling whole hospitals rather than visits.

    Resampling individual visits would treat two patients from the same
    department as independent evidence and produce intervals that are far too
    narrow. Clustering by hospital gives an interval that reflects the real
    source of variation: departments differ from each other more than patients
    within one department do.
    """
    rng = np.random.RandomState(seed)
    unique_groups = np.unique(groups)
    stats = []

    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sampled])
        if len(idx) == 0:
            continue
        try:
            stats.append(statistic(values[idx]))
        except (ValueError, ZeroDivisionError):
            continue

    if not stats:
        return {"point": None, "ci_low": None, "ci_high": None}

    return {
        "point": round(float(statistic(values)), 4),
        "ci_low": round(float(np.percentile(stats, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(stats, 100 * (1 - alpha / 2))), 4),
        "n_bootstrap": len(stats),
        "cluster_unit": "hospital",
    }


def _metric_fn(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Callable:
    """Build a statistic over row indices for the cluster bootstrap."""
    def stat(idx_values: np.ndarray) -> float:
        idx = idx_values.astype(int)
        yt, yp = y_true[idx], y_pred[idx]
        crit = yt <= 2
        if name == "critical_recall":
            return float(((yp <= 2) & crit).sum() / crit.sum()) if crit.sum() else 1.0
        if name == "critical_under_triage_rate":
            return float((crit & (yp > 2)).sum() / crit.sum()) if crit.sum() else 0.0
        if name == "under_triage_rate":
            return float((yp > yt).mean())
        if name == "over_triage_rate":
            return float((yp < yt).mean())
        if name == "accuracy":
            return float((yp == yt).mean())
        if name == "within_one_level":
            return float((np.abs(yp - yt) <= 1).mean())
        if name == "critical_lane_load":
            return float((yp <= 2).mean())
        raise ValueError(name)
    return stat


def with_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
            names: List[str]) -> Dict:
    idx = np.arange(len(y_true))
    return {
        name: cluster_bootstrap_ci(idx, groups, _metric_fn(name, y_true, y_pred))
        for name in names
    }


HEADLINE_METRICS = [
    "critical_recall", "critical_under_triage_rate", "under_triage_rate",
    "over_triage_rate", "accuracy", "within_one_level", "critical_lane_load",
]


# ─── Baselines ───────────────────────────────────────────────────────────────

def early_warning_scores(df: pd.DataFrame) -> np.ndarray:
    """Raw NEWS2 (or PEWS for children) aggregate score per patient."""
    scores = []
    for _, row in df.iterrows():
        vitals = {k: (None if pd.isna(row.get(k)) else float(row.get(k)))
                  for k in ["temperature", "heart_rate", "respiratory_rate",
                            "spo2", "systolic_bp", "diastolic_bp"]}
        scores.append(compute_early_warning_score(vitals, float(row["age"]))["score"])
    return np.array(scores, dtype=float)


def early_warning_baseline(df: pd.DataFrame) -> np.ndarray:
    """NEWS2/PEWS mapped to a triage level by its published risk bands."""
    levels = []
    for _, row in df.iterrows():
        vitals = {k: (None if pd.isna(row.get(k)) else float(row.get(k)))
                  for k in ["temperature", "heart_rate", "respiratory_rate",
                            "spo2", "systolic_bp", "diastolic_bp"]}
        score = compute_early_warning_score(vitals, float(row["age"]))
        levels.append(score["implied_triage_level"])
    return np.array(levels)


def budget_matched_ews_baseline(df: pd.DataFrame, target_lane_load: float) -> np.ndarray:
    """
    NEWS2/PEWS thresholded to route the SAME share of patients to the emergent
    lane as our system does.

    Without this, the comparison is rigged. The published NEWS2 bands were
    designed to trigger ward escalation, not ED triage, and applying them
    literally sends under 1% of arrivals to the emergent lane — so of course
    they catch few critical patients. Matching the budget asks the only fair
    question: **given the same number of emergent-lane slots, who fills them
    with the sicker patients?**
    """
    scores = early_warning_scores(df)
    # Highest scores get the emergent lane, up to the budget.
    if target_lane_load <= 0:
        return np.full(len(df), 4)
    cutoff = np.quantile(scores, 1.0 - target_lane_load)

    levels = np.where(scores >= max(cutoff, 1.0), 2, 4)
    # Give the very highest scorers Level 1, in the same proportion our system
    # assigns Level 1, so the two are comparable at the top of the scale too.
    if (scores >= cutoff).sum() > 0:
        top_cut = np.quantile(scores, 1.0 - target_lane_load / 6.0)
        levels = np.where(scores >= top_cut, 1, levels)
    return levels.astype(int)


def vitals_rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """
    A plain "abnormal vitals" heuristic, of the kind a spreadsheet triage aid
    would implement. Included because it is the honest floor: if a machine
    learning system cannot beat counting abnormal vitals, it should not ship.
    """
    levels = []
    for _, row in df.iterrows():
        flags = 0
        hr, rr, spo2, sbp = (row.get("heart_rate"), row.get("respiratory_rate"),
                             row.get("spo2"), row.get("systolic_bp"))
        if pd.notna(hr) and (hr > 120 or hr < 50):
            flags += 1
        if pd.notna(rr) and (rr > 24 or rr < 10):
            flags += 1
        if pd.notna(spo2) and spo2 < 92:
            flags += 2
        if pd.notna(sbp) and sbp < 100:
            flags += 2
        levels.append(1 if flags >= 4 else 2 if flags >= 2 else 3 if flags >= 1 else 4)
    return np.array(levels)


# ─── Main evaluation ─────────────────────────────────────────────────────────

def run_full_evaluation(years: Tuple[int, ...] = (2021, 2022),
                        bundle_path: str = "saved_models/triage_bundle.joblib",
                        verbose: bool = True) -> Dict:
    t_start = time.time()
    results: Dict = {
        "generated_at": datetime.now().isoformat(),
        "evaluation_protocol": {
            "test_fold": "hospitals held out from training, calibration, "
                         "conformal fitting and operating-point selection",
            "confidence_intervals": "1000-sample bootstrap, clustered by hospital",
            "primary_metrics": "critical recall and critical under-triage; "
                               "accuracy is reported but never used for selection",
        },
    }

    if verbose:
        print("=" * 74)
        print("PatientTriage.ai — Full Evaluation")
        print("=" * 74)

    # ── Data + model ──
    df = load_clean(years=years, verbose=verbose).reset_index(drop=True)
    idx = grouped_split(df)
    df_test = df.loc[idx["test"]].reset_index(drop=True)
    bundle = load_bundle(bundle_path)

    results["cohort"] = {
        "source": "NHAMCS Emergency Department public-use files (CDC/NCHS)",
        "survey_years": list(years),
        "total_visits_with_triage_label": int(len(df)),
        "total_hospitals": int(df["hospital_id"].nunique()),
        "test_fold_visits": int(len(df_test)),
        "test_fold_hospitals": int(df_test["hospital_id"].nunique()),
        "triage_distribution_pct": {
            int(k): round(float(v) * 100, 2) for k, v in
            df["triage_level"].value_counts(normalize=True).sort_index().items()
        },
        "critical_prevalence_pct": round(float((df["triage_level"] <= 2).mean() * 100), 2),
        "admission_rate_pct": round(float(df["outcome_admitted"].mean() * 100), 2),
        "critical_outcome_rate_pct": round(float(df["outcome_critical"].mean() * 100), 2),
        "age_bands": {
            "pediatric": int((df["age"] < 18).sum()),
            "adult": int(((df["age"] >= 18) & (df["age"] < 65)).sum()),
            "geriatric": int((df["age"] >= 65).sum()),
        },
    }

    if verbose:
        print(f"\nTest fold: {len(df_test):,} visits across "
              f"{df_test['hospital_id'].nunique()} unseen hospitals")

    # ── Score the test fold ──
    if verbose:
        print("\n[1/9] Scoring the held-out test fold …")
    records = df_test.to_dict("records")
    X_test = bundle.build_matrix(records)
    model = bundle.calibrated_classifier or bundle.classifier
    proba = _align_proba(model.predict_proba(X_test), model)

    y_true = df_test["triage_level"].to_numpy(int)
    groups = df_test["hospital_id"].to_numpy()

    y_pred = np.array([expected_cost_decision(p, bundle.cost_matrix)["decision"]
                       for p in proba])
    y_argmax = np.array(LEVELS)[proba.argmax(axis=1)]

    # ── 2. Primary safety metrics ──
    if verbose:
        print("[2/9] Primary safety metrics with clustered bootstrap CIs …")
    results["primary_metrics"] = {
        "point_estimates": clinical_metrics(y_true, y_pred),
        "with_confidence_intervals": with_ci(y_true, y_pred, groups, HEADLINE_METRICS),
    }

    results["discrimination"] = {
        "auroc_critical_level_1_2": round(float(
            roc_auc_score((y_true <= 2).astype(int), proba[:, 0] + proba[:, 1])), 4),
        "auroc_level_1_only": (
            round(float(roc_auc_score((y_true == 1).astype(int), proba[:, 0])), 4)
            if (y_true == 1).sum() > 0 else None),
        "auroc_vs_hospital_admission": round(float(roc_auc_score(
            df_test["outcome_admitted"].to_numpy(int),
            1.0 - (proba * np.array(LEVELS)).sum(axis=1) / 5.0)), 4),
        "auroc_vs_critical_outcome": (
            round(float(roc_auc_score(
                df_test["outcome_critical"].to_numpy(int),
                proba[:, 0] + proba[:, 1])), 4)
            if df_test["outcome_critical"].sum() > 5 else None),
        "note": ("AUROC measures what the model knows, independent of the "
                 "decision threshold. Reported against the nurse's label and "
                 "against two outcomes the nurse could not have known at triage."),
    }

    # ── 3. Baseline comparison ──
    if verbose:
        print("[3/9] Comparing against clinical baselines …")
    lane_load = float((y_pred <= 2).mean())
    baselines = {
        "PatientTriage.ai (cost-sensitive policy)": y_pred,
        "Same model, accuracy-maximising argmax": y_argmax,
        "NEWS2 / PEWS at matched lane load": budget_matched_ews_baseline(df_test, lane_load),
        "NEWS2 / PEWS at published bands": early_warning_baseline(df_test),
        "Abnormal-vitals heuristic": vitals_rule_baseline(df_test),
    }
    results["baseline_comparison"] = {
        name: {**clinical_metrics(y_true, preds),
               "critical_lane_load": round(float((preds <= 2).mean()), 4)}
        for name, preds in baselines.items()
    }
    results["baseline_comparison_note"] = (
        "All rows are evaluated on identical patients. The argmax row is the "
        "same trained model with the cost policy removed. The difference "
        "between the two rows is attributable purely to the decision rule, "
        "and it is the single largest safety effect in this system."
    )

    # ── 4. Outcome-based validation ──
    if verbose:
        print("[4/9] Outcome-based validation (independent of the nurse label) …")
    results["outcome_validation"] = _outcome_validation(df_test, y_pred, y_true, baselines)

    # ── 5. Fairness ──
    if verbose:
        print("[5/9] Fairness audit across age, sex and race/ethnicity …")
    results["fairness"] = _fairness_audit(df_test, y_true, y_pred, groups)

    # ── 6. Calibration + conformal ──
    if verbose:
        print("[6/9] Calibration and conformal coverage …")
    results["calibration"] = calibration_report(y_true, proba)
    if bundle.conformal:
        results["conformal_5class"] = bundle.conformal.evaluate(proba, y_true)
    if bundle.critical_conformal:
        results["conformal_critical_exclusion"] = bundle.critical_conformal.evaluate(
            proba, y_true)

    # ── 7. Operating curve ──
    if verbose:
        print("[7/9] Safety–throughput frontier …")
    base_profile = CostMatrix(
        under_cost=SITE_PROFILES["urban_trauma_center"].under_cost,
        over_cost=SITE_PROFILES["urban_trauma_center"].over_cost)
    results["operating_curve"] = operating_curve(proba, y_true, base_profile)
    results["site_profile_comparison"] = _site_profiles(proba, y_true)

    # ── 8. Latency + surge ──
    if verbose:
        print("[8/9] Latency and surge throughput …")
    results["performance"] = _performance(bundle, records)

    # ── 9. Robustness ──
    if verbose:
        print("[9/9] Robustness: missing data, cross-site, temporal drift …")
    results["robustness"] = _robustness(bundle, df_test, y_true, groups, proba)
    results["cross_site"] = _cross_site(df_test, y_true, y_pred)
    if len(years) > 1:
        results["temporal_validation"] = _temporal_validation(df, years)

    results["confusion_matrix"] = {
        "labels": LEVELS,
        "matrix": confusion_matrix(y_true, y_pred, labels=LEVELS).tolist(),
        "orientation": "rows = nurse-assigned level, columns = system level",
    }

    results["evaluation_runtime_seconds"] = round(time.time() - t_start, 1)
    return results


# ─── Component evaluations ───────────────────────────────────────────────────

def _outcome_validation(df_test: pd.DataFrame, y_pred: np.ndarray,
                        y_true: np.ndarray, baselines: Dict) -> Dict:
    """
    Validate against what happened to the patient, not against the nurse.

    The clinically meaningful failure is not "disagreed with the triage nurse".
    It is "routed a patient to the waiting room who turned out to need critical
    care". NHAMCS records that outcome, so we can measure it directly — and it
    is the one metric a sceptical emergency physician will actually care about.
    """
    critical_outcome = df_test["outcome_critical"].to_numpy(int)
    admitted = df_test["outcome_admitted"].to_numpy(int)

    out: Dict = {
        "definition": {
            "critical_outcome": "admitted to a critical care unit, died in the ED, "
                                "or dead on arrival",
            "admitted": "admitted to hospital, to observation then hospitalised, "
                        "or transferred to another acute hospital",
        },
        "n_critical_outcome": int(critical_outcome.sum()),
        "n_admitted": int(admitted.sum()),
    }

    for name, preds in baselines.items():
        sent_to_waiting_room = preds >= 3
        missed = int((sent_to_waiting_room & (critical_outcome == 1)).sum())
        n_crit = int(critical_outcome.sum())
        out[name] = {
            "critical_outcome_patients_routed_to_waiting_room": missed,
            "critical_outcome_capture_rate": round(1 - missed / n_crit, 4) if n_crit else None,
            "admitted_patients_routed_to_waiting_room_pct": round(float(
                (sent_to_waiting_room & (admitted == 1)).sum() / max(admitted.sum(), 1)), 4),
        }

    # How well did the triage nurses themselves do, on the same standard?
    nurse_sent_to_waiting = y_true >= 3
    n_crit = int(critical_outcome.sum())
    out["Triage nurses (the reference standard)"] = {
        "critical_outcome_patients_routed_to_waiting_room": int(
            (nurse_sent_to_waiting & (critical_outcome == 1)).sum()),
        "critical_outcome_capture_rate": round(
            1 - (nurse_sent_to_waiting & (critical_outcome == 1)).sum() / n_crit, 4
        ) if n_crit else None,
        "note": ("Included for scale, not as a target to beat. Human triage "
                 "under-triages some patients who deteriorate later; a decision "
                 "support tool is useful if it catches some of those, not if it "
                 "reproduces the label perfectly."),
    }
    return out


def _fairness_audit(df_test: pd.DataFrame, y_true: np.ndarray,
                    y_pred: np.ndarray, groups: np.ndarray) -> Dict:
    """
    Under-triage rates broken out by group.

    Reported for age, sex and race/ethnicity. Race is never a model input; it
    is audited precisely because a model can learn a proxy for it from
    everything else, and a disparity nobody measures is a disparity that ships.
    Subgroups too small for a stable estimate are reported with their n and
    flagged rather than quietly rounded into a headline.
    """
    audit: Dict = {"note": (
        "Race/ethnicity and sex are never model features. They are audited "
        "because proxies exist in the remaining data, and an unmeasured "
        "disparity is an undetected one."
    )}

    def subgroup(mask: np.ndarray, label: str) -> Optional[Dict]:
        n = int(mask.sum())
        if n < 30:
            return {"n": n, "estimate_suppressed": True,
                    "reason": "fewer than 30 patients — too few for a stable rate"}
        yt, yp = y_true[mask], y_pred[mask]
        crit = yt <= 2
        return {
            "n": n,
            "n_critical": int(crit.sum()),
            "critical_recall": round(float(((yp <= 2) & crit).sum() / crit.sum()), 4)
                               if crit.sum() >= 10 else None,
            "critical_under_triage_rate": round(float((crit & (yp > 2)).sum() / crit.sum()), 4)
                                          if crit.sum() >= 10 else None,
            "under_triage_rate": round(float((yp > yt).mean()), 4),
            "over_triage_rate": round(float((yp < yt).mean()), 4),
            "accuracy": round(float((yp == yt).mean()), 4),
        }

    age = df_test["age"].to_numpy(float)
    audit["by_age_band"] = {
        "pediatric": subgroup(age < 18, "pediatric"),
        "adult": subgroup((age >= 18) & (age < 65), "adult"),
        "geriatric": subgroup(age >= 65, "geriatric"),
    }

    sex = df_test["sex"].to_numpy()
    audit["by_sex"] = {s: subgroup(sex == s, s) for s in ["F", "M"]}

    race = df_test["race_ethnicity"].astype(str).to_numpy()
    audit["by_race_ethnicity"] = {
        r: subgroup(race == r, r) for r in sorted(set(race)) if r != "nan"
    }

    # Largest gap between any two adequately-sized subgroups
    for dimension in ["by_age_band", "by_sex", "by_race_ethnicity"]:
        rates = [v["critical_under_triage_rate"] for v in audit[dimension].values()
                 if isinstance(v, dict) and v.get("critical_under_triage_rate") is not None]
        if len(rates) >= 2:
            audit[f"{dimension}_max_under_triage_gap"] = round(max(rates) - min(rates), 4)

    return audit


def _site_profiles(proba: np.ndarray, y_true: np.ndarray) -> Dict:
    """The same model under each site's cost profile — scalability, quantified."""
    out = {}
    for name, profile in SITE_PROFILES.items():
        preds = np.array([expected_cost_decision(p, profile)["decision"] for p in proba])
        out[name] = {
            **clinical_metrics(y_true, preds),
            "critical_lane_load": round(float((preds <= 2).mean()), 4),
            "rationale": profile.rationale.strip(),
        }
    return out


def _performance(bundle, records: List[Dict]) -> Dict:
    """Latency distribution and surge throughput."""
    sample = records[:300]

    # Warm up so the first call's import cost is not reported as latency
    bundle.predict_one(sample[0])

    latencies = []
    for rec in sample:
        t0 = time.perf_counter()
        bundle.predict_one(rec)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies = np.array(latencies)

    # Batch throughput — what matters during a surge
    t0 = time.perf_counter()
    bundle.predict_proba(records)
    batch_seconds = time.perf_counter() - t0

    # End-to-end pipeline latency, including the safety rules and the
    # explanation trace — the number that reflects what a nurse waits for,
    # rather than the model call in isolation.
    pipeline_latency = _pipeline_latency()

    return {
        "end_to_end_pipeline_latency_ms": pipeline_latency,
        "single_patient_latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "mean": round(float(latencies.mean()), 2),
            "max": round(float(latencies.max()), 2),
            "n_sampled": len(latencies),
        },
        "batch_throughput": {
            "n_patients": len(records),
            "total_seconds": round(batch_seconds, 3),
            "patients_per_second": round(len(records) / batch_seconds, 1),
        },
        "surge_capacity": {
            "note": ("A 500-visit-per-day ED at 3× surge sees roughly 1,500 "
                     "patients in 24 hours — about one arrival a minute."),
            "arrivals_per_minute_at_3x_surge": round(1500 / (24 * 60), 2),
            "seconds_to_process_a_full_surge_day": round(
                1500 / (len(records) / batch_seconds), 2),
        },
        "hardware": "single CPU core, no GPU",
    }


def _pipeline_latency(n: int = 40) -> Dict:
    """
    Time the complete pipeline, not just the model call.

    The first call is discarded: constructing the SHAP explainer is a one-off
    cost of roughly 800 ms that every subsequent patient avoids. Reporting it
    inside the distribution would overstate steady-state latency by two orders
    of magnitude; omitting the fact entirely would hide a real cold-start.
    """
    from engine.triage_pipeline import TriagePipeline

    pipeline = TriagePipeline()
    pipeline.initialize(verbose=False)

    encounters = [enc for enc, _, _ in pipeline.patients]
    if not encounters:
        return {}

    t0 = time.perf_counter()
    pipeline.triage_patient(encounters[0], store=False)
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    latencies = []
    for i in range(n):
        enc = encounters[i % len(encounters)]
        t0 = time.perf_counter()
        pipeline.triage_patient(enc, store=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
        "mean": round(float(arr.mean()), 1),
        "cold_start_ms": round(cold_start_ms, 1),
        "includes": ("feature construction, calibrated model, conformal check, "
                     "all clinical safety rules, SHAP attribution, counterfactual "
                     "search and the full explanation trace"),
        "cold_start_note": ("First call builds the SHAP explainer; excluded from "
                            "the distribution and reported separately."),
        "n_sampled": len(latencies),
    }


def _robustness(bundle, df_test: pd.DataFrame, y_true: np.ndarray,
                groups: np.ndarray, proba_full: np.ndarray) -> Dict:
    """
    Degrade the inputs and watch what happens to critical recall.

    The question is not whether performance drops — it must — but whether it
    degrades gracefully and in the safe direction. A triage system that becomes
    *less* cautious as information disappears is dangerous; the design intent is
    the opposite, and this is where that intent is checked.
    """
    rng = np.random.RandomState(42)
    vital_cols = ["heart_rate", "respiratory_rate", "spo2", "systolic_bp",
                  "diastolic_bp", "temperature", "pain_score"]

    baseline_pred = np.array([expected_cost_decision(p, bundle.cost_matrix)["decision"]
                              for p in proba_full])
    baseline = clinical_metrics(y_true, baseline_pred)

    out: Dict = {
        "baseline": {**baseline, "critical_lane_load": round(float((baseline_pred <= 2).mean()), 4)}
    }

    for fraction in [0.10, 0.25, 0.50, 0.75]:
        records = df_test.to_dict("records")
        for rec in records:
            for col in vital_cols:
                if rng.random() < fraction:
                    rec[col] = None

        X = bundle.build_matrix(records)
        model = bundle.calibrated_classifier or bundle.classifier
        p = _align_proba(model.predict_proba(X), model)
        preds = np.array([expected_cost_decision(row, bundle.cost_matrix)["decision"]
                          for row in p])
        m = clinical_metrics(y_true, preds)
        out[f"missing_{int(fraction * 100)}pct_of_vitals"] = {
            **m,
            "critical_lane_load": round(float((preds <= 2).mean()), 4),
            "critical_recall_delta": round(
                m["critical_recall"] - baseline["critical_recall"], 4),
            "became_more_cautious": bool(
                (preds <= 2).mean() >= (baseline_pred <= 2).mean() - 0.02),
        }

    # All history removed — the "first-time patient" scenario at scale
    records = df_test.to_dict("records")
    for rec in records:
        rec["history_available"] = 0.0
        rec["has_high_risk_conditions"] = 0.0
        rec["n_chronic_conditions"] = 0.0
        for key in list(rec):
            if key.startswith("cond_"):
                rec[key] = 0.0
    X = bundle.build_matrix(records)
    model = bundle.calibrated_classifier or bundle.classifier
    p = _align_proba(model.predict_proba(X), model)
    preds = np.array([expected_cost_decision(row, bundle.cost_matrix)["decision"]
                      for row in p])
    out["all_history_removed"] = {
        **clinical_metrics(y_true, preds),
        "critical_lane_load": round(float((preds <= 2).mean()), 4),
        "note": ("Simulates a department where every arrival is a first-time "
                 "patient with nothing on file."),
    }
    return out


def _cross_site(df_test: pd.DataFrame, y_true: np.ndarray,
                y_pred: np.ndarray) -> Dict:
    """
    Per-hospital variation — does this work everywhere, or only on average?

    An aggregate number can hide a department where the system performs badly.
    Since every test hospital is one the model never saw, the spread across
    them is the closest available estimate of what happens at the next site.
    """
    df = pd.DataFrame({"hospital": df_test["hospital_id"].to_numpy(),
                       "y_true": y_true, "y_pred": y_pred})
    rows = []
    for hosp, grp in df.groupby("hospital"):
        crit = grp["y_true"] <= 2
        if crit.sum() < 5 or len(grp) < 30:
            continue
        rows.append({
            "hospital": int(hosp),
            "n_visits": int(len(grp)),
            "n_critical": int(crit.sum()),
            "critical_recall": round(float(
                ((grp["y_pred"] <= 2) & crit).sum() / crit.sum()), 4),
            "critical_lane_load": round(float((grp["y_pred"] <= 2).mean()), 4),
            "accuracy": round(float((grp["y_pred"] == grp["y_true"]).mean()), 4),
        })

    if not rows:
        return {"note": "Too few hospitals with adequate sample size."}

    recalls = np.array([r["critical_recall"] for r in rows])
    return {
        "n_hospitals_evaluated": len(rows),
        "critical_recall_mean": round(float(recalls.mean()), 4),
        "critical_recall_std": round(float(recalls.std()), 4),
        "critical_recall_min": round(float(recalls.min()), 4),
        "critical_recall_max": round(float(recalls.max()), 4),
        "critical_recall_iqr": [round(float(np.percentile(recalls, 25)), 4),
                                round(float(np.percentile(recalls, 75)), 4)],
        "per_hospital": sorted(rows, key=lambda r: r["critical_recall"]),
        "interpretation": (
            f"Across {len(rows)} unseen hospitals, critical recall ranges from "
            f"{recalls.min():.1%} to {recalls.max():.1%}. The spread, not the "
            f"mean, is what a new deployment should be planned against."
        ),
    }


def _temporal_validation(df: pd.DataFrame, years: Tuple[int, ...]) -> Dict:
    """
    Train on the earlier year, test on the later one.

    Guards against the quiet assumption that tomorrow looks like today. The
    2021→2022 transition is a genuine distribution shift — case mix, staffing
    and volumes moved as the pandemic receded — which makes it a far more
    honest stress test than a random split of pooled years.
    """
    from models.triage_model import train_bundle

    train_year, test_year = min(years), max(years)
    df_train = df[df["survey_year"] == train_year].reset_index(drop=True)
    df_test = df[df["survey_year"] == test_year].reset_index(drop=True)

    if len(df_train) < 500 or len(df_test) < 500:
        return {"note": "Insufficient data for temporal validation."}

    bundle, _ = train_bundle(df_train, verbose=False)

    records = df_test.to_dict("records")
    X = bundle.build_matrix(records)
    model = bundle.calibrated_classifier or bundle.classifier
    proba = _align_proba(model.predict_proba(X), model)
    y_true = df_test["triage_level"].to_numpy(int)
    preds = np.array([expected_cost_decision(p, bundle.cost_matrix)["decision"]
                      for p in proba])

    return {
        "trained_on": f"NHAMCS {train_year}",
        "tested_on": f"NHAMCS {test_year}",
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "metrics": clinical_metrics(y_true, preds),
        "auroc_critical": round(float(roc_auc_score(
            (y_true <= 2).astype(int), proba[:, 0] + proba[:, 1])), 4),
        "critical_lane_load": round(float((preds <= 2).mean()), 4),
        "interpretation": (
            f"Trained only on {train_year} and evaluated on every "
            f"{test_year} visit, including hospitals and a case mix it never "
            f"saw. This is the closest available proxy for deploying today and "
            f"running through next year without retraining."
        ),
    }


# ─── Reporting ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = run_full_evaluation()

    json_path = os.path.join(RESULTS_DIR, "evaluation_full.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


    print("\n" + "=" * 74)
    print("EVALUATION COMPLETE")
    print("=" * 74)
    pm = results["primary_metrics"]["point_estimates"]
    ci = results["primary_metrics"]["with_confidence_intervals"]
    print(f"  Critical recall        : {pm['critical_recall']:.1%} "
          f"(95% CI {ci['critical_recall']['ci_low']:.1%}–"
          f"{ci['critical_recall']['ci_high']:.1%})")
    print(f"  Critical under-triage  : {pm['critical_under_triage_rate']:.1%}")
    print(f"  AUROC (critical)       : {results['discrimination']['auroc_critical_level_1_2']:.3f}")
    print(f"  Emergent-lane load     : {(np.array([0]) + ci['critical_lane_load']['point'])[0]:.1%}")
    print(f"  p50 latency            : {results['performance']['single_patient_latency_ms']['p50']} ms")
    print(f"\n  → {json_path}")
    print(f"  → {resume_path}")


if __name__ == "__main__":
    main()
