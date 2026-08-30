"""
PatientTriage.ai — Cost-Sensitive Decision Policy
=================================================

The brief is explicit that under-triage and over-triage carry asymmetric
costs, and that a solution must be *deliberately tuned* to escalate under
uncertainty rather than optimised for average accuracy.

Most systems answer that with a rule of thumb ("if unsure, bump it up one").
We answer it with the decision-theoretic version, which is both stronger and
easier to defend in a governance review: pick the level that minimises
**expected clinical cost**, not the level with the highest probability.

    chosen_level = argmin_p  Σ_t  P(true = t | x) · Cost(p, t)

Everything contestable about the system's risk appetite is therefore isolated
in one auditable object — the cost matrix. A rural ED with no on-site surgeon
and a large urban trauma centre can run identical code with different matrices,
and the difference in behaviour is inspectable, versioned, and explainable to a
regulator. That is what "tuned rather than solved away" looks like in practice.

Design of the default matrix:

* Missing a Level 1 patient is catastrophic and is priced accordingly (50 per
  level of under-triage). Missing a Level 5 is an inconvenience (1).
* Over-triage costs are real but flat and small (1.0–1.5 per level) — a wasted
  resuscitation bay, a bed unavailable to someone else.
* The ratio between the two is what produces escalation bias. At the default
  weights, the system escalates whenever the probability of a critical case
  exceeds roughly 2–4%, which is far below the ~50% an accuracy-maximising
  argmax would require.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

LEVELS = [1, 2, 3, 4, 5]


# ─── Cost model ──────────────────────────────────────────────────────────────

# Cost of under-triaging a patient whose true level is `t`, per level of
# under-triage. A Level 1 patient sent to the Level 3 queue incurs 2 × 50.
UNDER_TRIAGE_COST_BY_TRUE_LEVEL = {1: 50.0, 2: 20.0, 3: 6.0, 4: 2.0, 5: 1.0}

# Cost of over-triaging, per level. Resource cost, not safety cost.
OVER_TRIAGE_COST_BY_TRUE_LEVEL = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.2, 5: 1.5}


class CostMatrix:
    """
    An explicit, versioned statement of a department's risk appetite.

    Kept as a first-class object rather than constants scattered through the
    codebase because this is the thing a medical director should be able to
    read, argue with, and sign off.
    """

    def __init__(self,
                 under_cost: Optional[Dict[int, float]] = None,
                 over_cost: Optional[Dict[int, float]] = None,
                 name: str = "default",
                 version: str = "1.0",
                 rationale: str = ""):
        self.under_cost = dict(under_cost or UNDER_TRIAGE_COST_BY_TRUE_LEVEL)
        self.over_cost = dict(over_cost or OVER_TRIAGE_COST_BY_TRUE_LEVEL)
        self.name = name
        self.version = version
        self.rationale = rationale or (
            "Default urban ED profile: under-triage of a Level 1 patient is "
            "priced 50× a single level of over-triage, reflecting that a missed "
            "critical case can be fatal while an over-prioritised minor case "
            "costs a bed."
        )

    def cost(self, predicted: int, true: int) -> float:
        """Clinical cost of assigning `predicted` when the truth is `true`."""
        if predicted == true:
            return 0.0
        if predicted > true:            # less urgent than reality → under-triage
            return (predicted - true) * self.under_cost.get(true, 5.0)
        return (true - predicted) * self.over_cost.get(true, 1.0)

    def matrix(self, classes: Sequence[int] = LEVELS) -> np.ndarray:
        return np.array([[self.cost(p, t) for t in classes] for p in classes])

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "rationale": self.rationale,
            "under_triage_cost_by_true_level": self.under_cost,
            "over_triage_cost_by_true_level": self.over_cost,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CostMatrix":
        return cls(
            under_cost={int(k): float(v) for k, v in
                        d.get("under_triage_cost_by_true_level", {}).items()},
            over_cost={int(k): float(v) for k, v in
                       d.get("over_triage_cost_by_true_level", {}).items()},
            name=d.get("name", "custom"),
            version=d.get("version", "1.0"),
            rationale=d.get("rationale", ""),
        )


# ─── Site profiles ───────────────────────────────────────────────────────────
# The same assistant, flexed across hospitals of very different size and
# capability — one of the brief's explicit scalability asks.

SITE_PROFILES = {
    "urban_trauma_center": CostMatrix(
        name="urban_trauma_center",
        rationale=(
            "Large urban trauma centre with on-site surgical and critical care "
            "cover around the clock. Over-triage is comparatively cheap because "
            "surge capacity exists, so the default asymmetry is used unchanged."
        ),
    ),
    "rural_community_ed": CostMatrix(
        under_cost={1: 80.0, 2: 30.0, 3: 8.0, 4: 2.0, 5: 1.0},
        over_cost={1: 2.0, 2: 2.0, 3: 2.0, 4: 2.5, 5: 3.0},
        name="rural_community_ed",
        rationale=(
            "Small rural ED with no on-site specialist cover and a transfer time "
            "measured in hours. Missing a critical case is even costlier than in "
            "an urban centre because rescue is slower, but over-triage is also "
            "costlier, since escalating consumes a large share of a small "
            "department. Both sides rise; the asymmetry is preserved."
        ),
    ),
    "pediatric_specialty": CostMatrix(
        under_cost={1: 90.0, 2: 35.0, 3: 8.0, 4: 2.0, 5: 1.0},
        over_cost={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.2},
        name="pediatric_specialty",
        rationale=(
            "Paediatric ED. Children compensate well and then decompensate "
            "abruptly, so the window in which under-triage is recoverable is "
            "shorter. Under-triage costs are raised and over-triage costs held "
            "low to reflect that asymmetry."
        ),
    ),
    "symmetric_baseline": CostMatrix(
        under_cost={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
        over_cost={1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
        name="symmetric_baseline",
        rationale=(
            "Symmetric costs. A comparison baseline only, never a deployment "
            "profile. Note what it actually optimises: because costs scale with "
            "the DISTANCE between the assigned and true level, minimising "
            "expected cost here yields the probability-weighted median, not the "
            "most likely level. It is therefore a 'minimise average levels of "
            "error' baseline rather than an accuracy-maximising one. The "
            "accuracy-maximising comparison in the evaluation is computed as a "
            "direct argmax instead, which is a different rule again. Treating "
            "the two as interchangeable is an easy and consequential mistake."
        ),
    ),
}


# ─── Decision rule ───────────────────────────────────────────────────────────

def expected_cost_decision(proba: np.ndarray,
                           cost_matrix: Optional[CostMatrix] = None,
                           classes: Sequence[int] = LEVELS) -> Dict:
    """
    Choose the triage level minimising expected clinical cost for one patient.

    Returns the decision plus the full expected-cost vector, because showing a
    clinician *why* the system passed over the most-likely level ("Level 3 was
    most likely at 61%, but the 7% chance of Level 1 costs more than the
    certain price of escalating") is far more persuasive than the level alone.
    """
    cost_matrix = cost_matrix or CostMatrix()
    classes = list(classes)
    proba = np.asarray(proba, dtype=float)

    expected = []
    for p in classes:
        expected.append(sum(proba[i] * cost_matrix.cost(p, t)
                            for i, t in enumerate(classes)))
    expected = np.array(expected)

    chosen_idx = int(np.argmin(expected))
    chosen = classes[chosen_idx]
    argmax_idx = int(np.argmax(proba))
    most_likely = classes[argmax_idx]

    return {
        "decision": chosen,
        "most_likely_level": most_likely,
        "most_likely_probability": round(float(proba[argmax_idx]), 4),
        "expected_costs": {int(c): round(float(e), 3) for c, e in zip(classes, expected)},
        "expected_cost_of_decision": round(float(expected[chosen_idx]), 3),
        "differs_from_argmax": chosen != most_likely,
        "escalation_from_argmax": int(most_likely - chosen),
        "cost_profile": cost_matrix.name,
        "rationale": _decision_rationale(chosen, most_likely, proba, classes, expected),
    }


def _decision_rationale(chosen: int, most_likely: int, proba: np.ndarray,
                        classes: Sequence[int], expected: np.ndarray) -> str:
    if chosen == most_likely:
        return (f"Level {chosen} is both the most likely level "
                f"({proba[list(classes).index(most_likely)]:.0%}) and the lowest-risk "
                f"choice given the cost of being wrong.")
    crit_p = sum(proba[i] for i, c in enumerate(classes) if c <= 2)
    return (
        f"Level {most_likely} was the single most likely outcome "
        f"({proba[list(classes).index(most_likely)]:.0%}), but there is a "
        f"{crit_p:.1%} chance this patient is Level 1 or 2. Because missing a "
        f"critical patient is priced far above the cost of an unnecessary "
        f"escalation, the lowest-expected-harm action is Level {chosen}."
    )


def scaled_profile(base: CostMatrix, lam: float, name: Optional[str] = None) -> CostMatrix:
    """
    Scale the under-triage side of a cost matrix by λ.

    λ is the single knob that moves the system along the safety–throughput
    frontier. λ → 0 collapses to accuracy-maximising argmax; large λ escalates
    everyone. Neither extreme is usable, which is the point: the operating
    point has to be *chosen*, against a stated budget, and that choice has to
    be visible.
    """
    return CostMatrix(
        under_cost={k: v * lam for k, v in base.under_cost.items()},
        over_cost=dict(base.over_cost),
        name=name or f"{base.name}_lambda{lam:g}",
        version=base.version,
        rationale=base.rationale,
    )


def operating_curve(proba: np.ndarray, y_true: np.ndarray,
                    base: Optional[CostMatrix] = None,
                    lambdas: Optional[Sequence[float]] = None,
                    classes: Sequence[int] = LEVELS) -> List[Dict]:
    """
    Trace the safety–throughput frontier by sweeping λ.

    Each point answers: "if we accept this much over-triage, how many critical
    patients do we catch?" Publishing the whole curve — rather than a single
    number — is what lets a medical director pick the point their department
    can actually staff, instead of accepting ours.
    """
    base = base or CostMatrix()
    y_true = np.asarray(y_true)
    if lambdas is None:
        lambdas = [0.0, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25,
                   0.28, 0.3, 0.35, 0.4, 0.5, 0.65, 0.75, 1.0, 1.5, 2.0,
                   3.0, 5.0, 10.0]

    curve = []
    for lam in lambdas:
        profile = scaled_profile(base, lam)
        preds = np.array([expected_cost_decision(p, profile, classes)["decision"]
                          for p in proba])
        critical = y_true <= 2
        n_crit = int(critical.sum())
        curve.append({
            "lambda": lam,
            "critical_recall": round(
                float(((preds <= 2) & critical).sum() / n_crit) if n_crit else 1.0, 4),
            "critical_under_triage_rate": round(
                float((critical & (preds > 2)).sum() / n_crit) if n_crit else 0.0, 4),
            "over_triage_rate": round(float((preds < y_true).mean()), 4),
            "under_triage_rate": round(float((preds > y_true).mean()), 4),
            "accuracy": round(float((preds == y_true).mean()), 4),
            "pct_routed_level_1_2": round(float((preds <= 2).mean()), 4),
        })
    return curve


def select_operating_point(curve: List[Dict],
                           max_critical_lane_load: float = 0.35,
                           min_critical_recall: float = 0.65) -> Dict:
    """
    Pick λ under an explicit, operationally meaningful escalation budget.

    The budget is expressed as **emergent-lane load** — the share of arrivals
    the system routes to Level 1–2 — rather than as an abstract over-triage
    rate. That choice is deliberate. "Over-triage" counts a 4→3 move the same
    as a 4→1 move, though one is clinically irrelevant and the other consumes
    a resuscitation bay. Lane load maps directly onto the thing a department
    actually runs out of: staffed capacity in the emergent lane. A charge
    nurse can look at "we will send 30% of arrivals to the emergent lane
    instead of the 19% that truly belong there" and tell you on the spot
    whether that is survivable on a Tuesday night.

    Preference order:
      1. Points meeting both the recall floor and the lane-load ceiling —
         among these, the highest critical recall.
      2. If nothing meets both, honour the safety floor and take the lowest
         lane load that reaches it. Safety binds; throughput gives.
      3. If the safety floor is unreachable at any load, report the maximum
         achievable recall and label it as such, rather than quietly shipping
         a weaker system behind a nicer-looking number.
    """
    feasible = [p for p in curve
                if p["pct_routed_level_1_2"] <= max_critical_lane_load
                and p["critical_recall"] >= min_critical_recall]
    if feasible:
        best = max(feasible,
                   key=lambda p: (p["critical_recall"], -p["pct_routed_level_1_2"]))
        status = "met_both_constraints"
    else:
        meets_safety = [p for p in curve if p["critical_recall"] >= min_critical_recall]
        if meets_safety:
            best = min(meets_safety, key=lambda p: p["pct_routed_level_1_2"])
            status = "safety_floor_met_lane_budget_exceeded"
        else:
            best = max(curve, key=lambda p: p["critical_recall"])
            status = "safety_floor_unreachable"

    return {
        "selected_lambda": best["lambda"],
        "operating_point": best,
        "status": status,
        "budget": {
            "max_critical_lane_load": max_critical_lane_load,
            "min_critical_recall": min_critical_recall,
        },
        "explanation": (
            f"λ = {best['lambda']:g} selected under an escalation budget of "
            f"≤{max_critical_lane_load:.0%} of arrivals routed to the emergent "
            f"lane, with a ≥{min_critical_recall:.0%} critical-recall floor. At "
            f"this point the model alone catches {best['critical_recall']:.1%} "
            f"of Level 1 or 2 patients while sending "
            f"{best['pct_routed_level_1_2']:.1%} of arrivals to that lane."
        ),
    }


def conformal_safety_decision(pred_set: List[int]) -> int:
    """
    Act on the most urgent level the conformal set still admits.

    With a 90% coverage guarantee, a set of {2, 3} means we cannot rule out
    Level 2 at that confidence — and the only defensible action when a Level 2
    cannot be ruled out is to treat the patient as Level 2.
    """
    return min(pred_set) if pred_set else 3


def combine_policies(expected_cost_level: int, conformal_level: int) -> Dict:
    """
    Reconcile the two decision paths.

    They answer different questions — one minimises average harm, the other
    respects a per-patient coverage guarantee — so when they disagree the
    system takes the more urgent of the two. Disagreement is itself recorded,
    because a patient where the two policies diverge is precisely the patient a
    clinician should look at twice.
    """
    final = min(expected_cost_level, conformal_level)
    return {
        "level": final,
        "expected_cost_level": expected_cost_level,
        "conformal_level": conformal_level,
        "policies_agree": expected_cost_level == conformal_level,
        "driver": ("expected_cost" if final == expected_cost_level and
                   final != conformal_level else
                   "conformal_coverage" if final != expected_cost_level else "both"),
    }
