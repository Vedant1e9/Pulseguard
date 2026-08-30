"""
Tests for the calibration, conformal and cost-sensitive decision layers.

These cover the machinery that turns a probability into an action — the part
of the system where a subtle error produces confident, plausible, wrong
triage decisions rather than a crash.
"""
import numpy as np
import pytest

from models.decision_policy import (
    CostMatrix, SITE_PROFILES, expected_cost_decision, operating_curve,
    scaled_profile, select_operating_point,
)
from models.uncertainty import (
    ConformalTriage, CriticalRiskConformal, brier_score_multiclass,
    calibration_report, expected_calibration_error,
)

LEVELS = [1, 2, 3, 4, 5]


# ─── Cost-sensitive decisions ────────────────────────────────────────────────

def test_under_triage_costs_more_than_over_triage():
    cm = CostMatrix()
    # Sending a Level 1 patient to Level 2 vs sending a Level 2 to Level 1
    assert cm.cost(predicted=2, true=1) > cm.cost(predicted=1, true=2)


def test_perfect_prediction_is_free():
    cm = CostMatrix()
    for level in LEVELS:
        assert cm.cost(level, level) == 0.0


def test_small_probability_of_critical_can_outweigh_a_likely_minor_case():
    """
    The whole point of the cost policy: a 6% chance of a Level 1 patient should
    beat a 70% chance of a Level 4, because the two errors are not equally bad.
    """
    proba = np.array([0.06, 0.04, 0.10, 0.70, 0.10])
    argmax_level = LEVELS[int(np.argmax(proba))]
    decision = expected_cost_decision(proba, CostMatrix())

    assert argmax_level == 4
    assert decision["decision"] < argmax_level
    assert decision["differs_from_argmax"] is True
    assert "critical" in decision["rationale"].lower()


def test_symmetric_distance_costs_yield_the_median_not_the_mode():
    """
    A guard against a tempting misreading of the symmetric profile.

    Because costs scale with the distance between assigned and true level,
    symmetric costs minimise expected absolute error — which gives the
    probability-weighted MEDIAN, not the most likely level. Anyone assuming
    "symmetric costs = argmax" would misreport what the baseline measures.
    """
    flat = SITE_PROFILES["symmetric_baseline"]
    rng = np.random.RandomState(5)
    for _ in range(100):
        proba = rng.dirichlet(np.ones(5))
        decision = expected_cost_decision(proba, flat)["decision"]
        cumulative = np.cumsum(proba)
        median_level = LEVELS[int(np.searchsorted(cumulative, 0.5))]
        assert decision == median_level


def test_raising_lambda_never_reduces_critical_recall():
    """The frontier must be monotone — otherwise λ is not a safety dial."""
    rng = np.random.RandomState(0)
    proba = rng.dirichlet(np.ones(5), size=400)
    y = rng.choice(LEVELS, size=400)

    curve = operating_curve(proba, y, CostMatrix(),
                            lambdas=[0.0, 0.1, 0.5, 1.0, 5.0, 20.0])
    recalls = [p["critical_recall"] for p in curve]
    assert recalls == sorted(recalls), "Critical recall fell as λ rose"


def test_operating_point_respects_the_lane_budget_when_feasible():
    curve = [
        {"lambda": 0.1, "critical_recall": 0.50, "pct_routed_level_1_2": 0.10,
         "critical_under_triage_rate": 0.5, "over_triage_rate": 0.1,
         "under_triage_rate": 0.2, "accuracy": 0.6},
        {"lambda": 0.5, "critical_recall": 0.80, "pct_routed_level_1_2": 0.30,
         "critical_under_triage_rate": 0.2, "over_triage_rate": 0.3,
         "under_triage_rate": 0.1, "accuracy": 0.5},
        {"lambda": 2.0, "critical_recall": 0.95, "pct_routed_level_1_2": 0.80,
         "critical_under_triage_rate": 0.05, "over_triage_rate": 0.8,
         "under_triage_rate": 0.0, "accuracy": 0.2},
    ]
    chosen = select_operating_point(curve, max_critical_lane_load=0.35,
                                    min_critical_recall=0.65)
    assert chosen["selected_lambda"] == 0.5
    assert chosen["status"] == "met_both_constraints"


def test_operating_point_reports_when_the_budget_cannot_be_met():
    """Failing the budget must be surfaced, not silently absorbed."""
    curve = [
        {"lambda": 1.0, "critical_recall": 0.90, "pct_routed_level_1_2": 0.70,
         "critical_under_triage_rate": 0.1, "over_triage_rate": 0.7,
         "under_triage_rate": 0.0, "accuracy": 0.3},
    ]
    chosen = select_operating_point(curve, max_critical_lane_load=0.30,
                                    min_critical_recall=0.65)
    assert chosen["status"] != "met_both_constraints"


def test_site_profiles_all_preserve_the_asymmetry():
    for name, profile in SITE_PROFILES.items():
        if name == "symmetric_baseline":   # comparison baseline, not a deployment profile
            continue
        assert profile.under_cost[1] > profile.over_cost[1], (
            f"{name} does not price under-triage above over-triage")


# ─── Conformal prediction ────────────────────────────────────────────────────

def _synthetic_probabilities(n=2000, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.choice(LEVELS, size=n, p=[0.02, 0.16, 0.52, 0.27, 0.03])
    proba = np.zeros((n, 5))
    for i, label in enumerate(y):
        base = rng.dirichlet(np.ones(5) * 0.8)
        base[LEVELS.index(label)] += 1.2      # informative but far from perfect
        proba[i] = base / base.sum()
    return proba, y


def test_conformal_coverage_meets_its_guarantee():
    proba, y = _synthetic_probabilities()
    cal_proba, cal_y = proba[:1000], y[:1000]
    test_proba, test_y = proba[1000:], y[1000:]

    conformal = ConformalTriage(alpha=0.10).calibrate(cal_proba, cal_y)
    evaluation = conformal.evaluate(test_proba, test_y)

    assert evaluation["empirical_coverage"] >= 0.85
    assert evaluation["guarantee_holds"]


def test_conformal_sets_are_contiguous_and_never_empty():
    proba, y = _synthetic_probabilities()
    conformal = ConformalTriage(alpha=0.10).calibrate(proba[:1000], y[:1000])
    for row in proba[1000:1200]:
        pred_set = conformal.predict_set(row)
        assert pred_set, "Empty prediction set gives a nurse nothing to act on"
        assert pred_set == list(range(min(pred_set), max(pred_set) + 1))


def test_critical_exclusion_respects_its_miss_budget():
    proba, y = _synthetic_probabilities(n=4000, seed=3)
    predictor = CriticalRiskConformal(alpha=0.05).calibrate(proba[:2000], y[:2000])
    evaluation = predictor.evaluate(proba[2000:], y[2000:])
    assert evaluation["guarantee_holds"], (
        f"Missed {evaluation['empirical_miss_rate']:.1%} of critical patients, "
        f"target ≤ {evaluation['target_max_miss_rate']:.0%}")


def test_conformal_round_trips_through_serialisation():
    proba, y = _synthetic_probabilities()
    original = ConformalTriage(alpha=0.10).calibrate(proba, y)
    restored = ConformalTriage.from_dict(original.to_dict())
    for row in proba[:100]:
        assert original.predict_set(row) == restored.predict_set(row)


# ─── Calibration metrics ─────────────────────────────────────────────────────

def test_brier_score_rewards_a_confident_correct_prediction():
    perfect = np.array([[1.0, 0, 0, 0, 0]])
    hedged = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    assert brier_score_multiclass(np.array([1]), perfect) < \
           brier_score_multiclass(np.array([1]), hedged)


def test_calibration_error_detects_overconfidence():
    """A model that always says 99% while being right half the time."""
    n = 1000
    proba = np.tile([0.99, 0.0025, 0.0025, 0.0025, 0.0025], (n, 1))
    y = np.array([1] * (n // 2) + [3] * (n // 2))
    result = expected_calibration_error(y, proba)
    assert result["ece"] > 0.4


def test_calibration_report_flags_the_direction_of_miscalibration():
    n = 500
    proba = np.tile([0.95, 0.02, 0.01, 0.01, 0.01], (n, 1))
    y = np.array([1] * 250 + [4] * 250)
    report = calibration_report(y, proba)
    assert "overconfident" in report["interpretation"].lower()
