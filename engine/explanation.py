"""
PulseGuard — Decision-Trace Explanation Layer
===================================================

Explains *the decision that was actually made*, not the model in general.

The failure this module exists to prevent is subtle and common. A system
escalates a patient to Level 1 because their oxygen saturation is 66%, and
then displays "top factors: patient age 58, pain score 5" underneath —
because the explanation was built from global feature importance rather than
from the decision path. The numbers are not wrong, they are simply about a
different question, and a clinician who reads them once concludes the system
does not know why it does anything.

So explanations here are constructed in strict causal order:

  1. **What actually decided the level.** If a deterministic safety rule set
     it, that rule is factor #1 with its threshold and citation. Nothing
     outranks the thing that made the decision.
  2. **What moved the model.** Per-patient attribution (SHAP where the model
     supports it, otherwise counterfactual perturbation), never global
     importance.
  3. **What we do not know.** Missing measurements are named as missing.
     An unrecorded blood pressure is never rendered as a value.
  4. **What would change the answer.** The nearest counterfactual, because
     "it would have been Level 3 if the saturation were above 92" tells a nurse
     more about the system's boundaries than any importance ranking.

The same trace object is rendered three ways — for the nurse at the bedside,
for the patient, and for the compliance record — so no two audiences can ever
be shown different accounts of the same decision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from data.features import describe_feature

# Features worth attributing over. Deliberately the clinically meaningful
# ones: attributing over 64 anonymous text-embedding components would be
# accurate and useless.
EXPLAINABLE_FEATURES = [
    "heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp",
    "temperature", "pain_score", "age",
    "heart_rate_z_for_age", "respiratory_rate_z_for_age",
    "systolic_bp_z_for_age", "spo2_z_for_age", "temperature_z_for_age",
    "shock_index", "shock_index_ratio_to_age_normal",
    "hypoxia_burden", "fever_burden", "hypothermia_burden",
    "ews_score", "news2_score", "pews_score", "ews_implied_level",
    "n_vitals_abnormal_for_age", "arrival_by_ambulance",
    "has_high_risk_conditions", "n_chronic_conditions", "history_available",
    "seen_last_72h", "injury_related", "nursing_home_resident", "severe_pain",
]

# Normal reference values used to build counterfactuals
COUNTERFACTUAL_NORMALS = {
    "heart_rate": 80.0, "respiratory_rate": 16.0, "spo2": 98.0,
    "systolic_bp": 120.0, "diastolic_bp": 75.0, "temperature": 36.8,
    "pain_score": 2.0,
}


class ExplanationBuilder:
    """Builds decision traces from a completed triage evaluation."""

    def __init__(self, bundle):
        self.bundle = bundle
        self._explainer = None
        self._explainer_failed = False

    # ── Per-patient model attribution ────────────────────────────────────────

    def _shap_attribution(self, record: Dict) -> Optional[Dict[str, float]]:
        """
        TreeSHAP attribution toward the probability of a critical level.

        Attributing toward P(Level 1–2) rather than toward the predicted class
        keeps the explanation aligned with the decision that matters: this
        system's job is to not miss a sick patient, so the question a factor
        should answer is "did this push the patient toward critical?".
        """
        if self._explainer_failed:
            return None
        try:
            import shap
            if self._explainer is None:
                model = self.bundle.classifier
                inner = getattr(model, "_model", model)
                self._explainer = shap.TreeExplainer(inner)

            X = self.bundle.build_matrix([record])
            values = self._explainer.shap_values(X)

            # Multiclass TreeSHAP returns per-class arrays in varying shapes
            # across versions and model types; normalise to (classes, features).
            arr = np.asarray(values)
            if arr.ndim == 3:
                arr = arr[0].T if arr.shape[0] == X.shape[0] else arr[:, 0, :]
            elif arr.ndim == 2:
                arr = arr[np.newaxis, :, :][0][np.newaxis, :]

            if arr.ndim != 2 or arr.shape[0] < 2:
                return None

            # Push toward critical = contribution to Level 1 + Level 2
            critical_push = arr[0] + arr[1]
            return {
                name: float(critical_push[i])
                for i, name in enumerate(self.bundle.feature_names)
                if i < len(critical_push)
            }
        except Exception:
            self._explainer_failed = True
            return None

    def _perturbation_attribution(self, record: Dict) -> Dict[str, float]:
        """
        Model-agnostic fallback: how much does P(critical) fall if this feature
        is replaced by a normal value?

        All perturbations are scored in a single batched prediction, so the
        whole attribution costs roughly one extra inference.
        """
        base_proba = self.bundle.predict_proba([record])[0]
        base_critical = float(base_proba[0] + base_proba[1])

        candidates = [f for f in EXPLAINABLE_FEATURES
                      if f in COUNTERFACTUAL_NORMALS and record.get(f) is not None]
        if not candidates:
            return {}

        perturbed = []
        for feat in candidates:
            r = dict(record)
            r[feat] = COUNTERFACTUAL_NORMALS[feat]
            perturbed.append(r)

        probas = self.bundle.predict_proba(perturbed)
        return {
            feat: base_critical - float(probas[i][0] + probas[i][1])
            for i, feat in enumerate(candidates)
        }

    # ── Counterfactual ───────────────────────────────────────────────────────

    def nearest_counterfactual(self, record: Dict, current_level: int,
                               safety_engine=None, encounter=None) -> Optional[Dict]:
        """
        Find the single change that would most reduce this patient's urgency.

        Phrased as "the level would be X if Y were Z" — the form a nurse can
        actually test against their own judgement, and the fastest way to learn
        where the system's boundaries sit.
        """
        candidates = [f for f in COUNTERFACTUAL_NORMALS
                      if record.get(f) is not None]
        if not candidates:
            return None

        perturbed = []
        for feat in candidates:
            r = dict(record)
            r[feat] = COUNTERFACTUAL_NORMALS[feat]
            perturbed.append(r)

        results = [self.bundle.predict_one(r) for r in perturbed]

        best = None
        for feat, res in zip(candidates, results):
            new_level = res["model_level"]
            if new_level > current_level:
                improvement = new_level - current_level
                if best is None or improvement > best["levels_changed"]:
                    best = {
                        "feature": feat,
                        "feature_label": describe_feature(feat),
                        "current_value": record.get(feat),
                        "counterfactual_value": COUNTERFACTUAL_NORMALS[feat],
                        "resulting_level": int(new_level),
                        "levels_changed": int(improvement),
                    }

        if best:
            best["statement"] = (
                f"If {best['feature_label'].lower()} were "
                f"{_fmt(best['counterfactual_value'])} instead of "
                f"{_fmt(best['current_value'])}, this patient would be "
                f"Level {best['resulting_level']}."
            )
        return best

    # ── The trace ────────────────────────────────────────────────────────────

    def build(self, record: Dict, model_output: Dict, safety_result: Dict,
              data_quality: Optional[Dict] = None,
              velocity: Optional[Dict] = None) -> Dict:
        """Assemble the full, ordered decision trace for one patient."""
        final_level = safety_result["final_level"]
        fired = safety_result.get("fired_rules", [])

        factors: List[Dict] = []

        # ── 1. Deterministic rules that set the level ──
        deciding = [r for r in fired if r["target_level"] == final_level]
        supporting = [r for r in fired if r["target_level"] != final_level]

        for rule in deciding:
            factors.append({
                "rank": len(factors) + 1,
                "source": "safety_rule",
                "headline": rule["reason"],
                "detail": rule["clinical_rationale"],
                "citation": rule["citation"],
                "rule_id": rule["rule_id"],
                "decisive": True,
            })

        # ── 2. Per-patient model attribution ──
        attribution = self._shap_attribution(record)
        method = "SHAP (per-patient)"
        if not attribution:
            attribution = self._perturbation_attribution(record)
            method = "counterfactual perturbation (per-patient)"

        contributors = [
            (name, value) for name, value in attribution.items()
            if name in EXPLAINABLE_FEATURES and abs(value) > 1e-6
            and _is_recorded(record, name)
        ]
        contributors.sort(key=lambda kv: abs(kv[1]), reverse=True)

        # TreeSHAP attributions are in the model's log-odds margin space, not in
        # probability. Printing them as "changes the probability by 102.2%" was
        # both wrong and visibly wrong — a probability shift cannot exceed 100%,
        # and a clinician who spots one impossible number stops believing the
        # rest of the panel. What a reader can actually use is the direction and
        # the relative weight, so each factor is reported as its share of the
        # total evidence shown. The perturbation fallback *is* a probability
        # delta, and keeps its literal reading.
        total = sum(abs(v) for _, v in contributors[:5]) or 1.0
        probability_space = method.startswith("counterfactual")

        for name, value in contributors[:5]:
            raw = record.get(name)
            direction = ("Pushes this patient toward" if value > 0
                         else "Pulls this patient away from")
            if probability_space:
                detail = (
                    f"{'Increases' if value > 0 else 'Decreases'} the estimated "
                    f"probability of a critical presentation by "
                    f"{min(abs(value), 1.0):.1%} for this patient."
                )
            else:
                detail = (
                    f"{direction} a critical assessment, and accounts for "
                    f"{abs(value) / total:.0%} of the model evidence shown here."
                )
            factors.append({
                "rank": len(factors) + 1,
                "source": "model",
                "headline": _phrase_factor(name, raw, value),
                "detail": detail,
                "attribution": round(float(value), 4),
                "attribution_method": method,
                "feature": name,
                "decisive": False,
            })

        # ── 3. Supporting rules that agreed but did not decide ──
        for rule in supporting[:2]:
            factors.append({
                "rank": len(factors) + 1,
                "source": "safety_rule_supporting",
                "headline": rule["reason"],
                "detail": rule["clinical_rationale"],
                "citation": rule["citation"],
                "rule_id": rule["rule_id"],
                "decisive": False,
            })

        # ── 4. What we do not know ──
        not_recorded = [
            describe_feature(f) for f in
            ["heart_rate", "respiratory_rate", "spo2", "systolic_bp",
             "temperature", "pain_score"]
            if not _is_recorded(record, f)
        ]

        # ── 5. Counterfactual ──
        counterfactual = self.nearest_counterfactual(record, final_level)

        # Describe what actually settled the level. Three distinct situations,
        # and conflating them misleads: a rule that overruled the model is not
        # the same as a rule that happened to agree with it, and a clinician
        # reading the trace needs to know which one they are looking at.
        if safety_result["was_escalated"]:
            decided_by = "deterministic safety rule (overruled the model)"
        elif deciding:
            decided_by = "model and safety rule independently agreed"
        else:
            decided_by = "cost-sensitive model decision (no safety rule fired)"

        return {
            "final_level": final_level,
            "decided_by": decided_by,
            "factors": factors,
            "not_recorded": not_recorded,
            "counterfactual": counterfactual,
            "attribution_method": method,
            "model_evidence": {
                "probabilities": model_output.get("probabilities", {}),
                "most_likely_level": model_output.get("most_likely_level"),
                "critical_probability": round(
                    float(model_output.get("critical_probability", 0.0)), 4),
                "conformal_set": model_output.get("conformal_set"),
                "cost_rationale": model_output.get("cost_decision", {}).get("rationale"),
            },
            "rule_pack": safety_result.get("rule_pack", {}),
        }

    # ── Persona renderings ───────────────────────────────────────────────────

    @staticmethod
    def for_nurse(trace: Dict) -> Dict:
        """Three lines. A triage nurse has seconds, not minutes."""
        top = trace["factors"][:3]
        return {
            "headline": f"Level {trace['final_level']}",
            "because": [f["headline"] for f in top],
            "decided_by": trace["decided_by"],
            "missing": trace["not_recorded"],
            "counterfactual": (trace["counterfactual"] or {}).get("statement"),
        }

    @staticmethod
    def for_patient(trace: Dict) -> Dict:
        """
        Plain language, no jargon, no probabilities.

        Deliberately avoids numbers a patient cannot act on, and never states
        or implies a diagnosis — this is a statement about queue priority.
        """
        level = trace["final_level"]
        wait_message = {
            1: "You are being seen immediately by the emergency team.",
            2: "You are a high priority and will be seen very soon.",
            3: "You will be seen as soon as a clinician is free. Please tell staff if you feel worse.",
            4: "Your condition appears stable. There may be a wait, and patients who arrive more unwell will be seen first.",
            5: "Your condition appears stable and non-urgent. There may be a longer wait.",
        }.get(level, "A clinician will assess you.")

        reasons = []
        for f in trace["factors"][:2]:
            reasons.append(_plain_language(f["headline"]))

        return {
            "priority_message": wait_message,
            "why": reasons,
            "reassurance": (
                "A nurse or doctor reviews every one of these decisions. "
                "If anything changes or you feel worse, tell staff straight away. "
                "Your priority can be raised at any time."
            ),
        }

    @staticmethod
    def for_compliance(trace: Dict, safety_result: Dict, model_output: Dict) -> Dict:
        """The full record: every rule, threshold, citation and version."""
        return {
            "final_level": trace["final_level"],
            "decision_authority": "Deterministic safety engine (models are advisory only)",
            "model_proposed_level": safety_result.get("original_model_level"),
            "escalation_applied": safety_result.get("was_escalated"),
            "levels_escalated": safety_result.get("levels_escalated"),
            "rules_fired": safety_result.get("fired_rules", []),
            "rule_pack": safety_result.get("rule_pack", {}),
            "model_name": model_output.get("model_name"),
            "calibrated_probabilities": model_output.get("probabilities"),
            "conformal_set": model_output.get("conformal_set"),
            "conformal_alpha": 0.10,
            "cost_profile": model_output.get("cost_decision", {}).get("cost_profile"),
            "expected_costs": model_output.get("cost_decision", {}).get("expected_costs"),
            "inference_latency_ms": model_output.get("latency_ms"),
            "attribution_method": trace.get("attribution_method"),
            "unrecorded_observations": trace.get("not_recorded"),
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _is_recorded(record: Dict, feature: str) -> bool:
    """
    True only if the value genuinely exists.

    The guard that prevents an imputed zero from ever being presented to a
    clinician as evidence.
    """
    value = record.get(feature)
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    return True


def _fmt(value) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, float):
        return f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}"
    return str(value)


def _phrase_factor(name: str, value, attribution: float) -> str:
    """Turn a feature and its value into a clinical phrase."""
    label = describe_feature(name)
    direction = "raising urgency" if attribution > 0 else "lowering urgency"

    phrases = {
        "spo2": lambda v: (f"Oxygen saturation {v:.0f}%"
                           + (", below normal" if v < 95 else "")),
        "heart_rate": lambda v: (f"Heart rate {v:.0f} bpm"
                                 + (", elevated" if v > 100 else
                                    ", low" if v < 60 else "")),
        "respiratory_rate": lambda v: (f"Respiratory rate {v:.0f}/min"
                                       + (", elevated" if v > 20 else "")),
        "systolic_bp": lambda v: (f"Systolic blood pressure {v:.0f} mmHg"
                                  + (", low" if v < 100 else
                                     ", high" if v > 180 else "")),
        "temperature": lambda v: (f"Temperature {v:.1f} °C"
                                  + (", febrile" if v > 38 else
                                     ", low" if v < 36 else "")),
        "pain_score": lambda v: f"Pain score {v:.0f}/10",
        "age": lambda v: f"Age {v:.0f}",
        "shock_index": lambda v: (f"Shock index {v:.2f}"
                                  + (", elevated, suggests compensated shock"
                                     if v > 0.9 else "")),
        "ews_score": lambda v: f"Early warning score {v:.0f}",
        "news2_score": lambda v: f"NEWS2 score {v:.0f}",
        "pews_score": lambda v: f"Paediatric early warning score {v:.0f}",
        "n_vitals_abnormal_for_age": lambda v: (
            f"{v:.0f} vital sign{'s' if v != 1 else ''} abnormal for this patient's age"),
        "arrival_by_ambulance": lambda v: ("Arrived by ambulance" if v
                                           else "Did not arrive by ambulance"),
        "has_high_risk_conditions": lambda v: ("Known high-risk medical conditions"
                                               if v else "No high-risk conditions on file"),
        "history_available": lambda v: ("Medical history available" if v
                                        else "No medical history on file"),
        "n_chronic_conditions": lambda v: f"{v:.0f} chronic condition(s) on record",
        "severe_pain": lambda v: "Severe pain reported" if v else "Pain not severe",
        "nursing_home_resident": lambda v: ("Nursing home resident" if v else ""),
        "seen_last_72h": lambda v: ("Seen in this ED within the last 72 hours"
                                    if v else ""),
    }

    if name in phrases:
        try:
            text = phrases[name](float(value))
            if text:
                return text
        except (TypeError, ValueError):
            pass

    if name.endswith("_z_for_age"):
        try:
            z = float(value)
            base = describe_feature(name)
            severity = ("markedly" if abs(z) > 3 else
                        "moderately" if abs(z) > 2 else "mildly")
            return (f"{base}: {severity} "
                    f"{'above' if z > 0 else 'below'} normal ({z:+.1f} SD)")
        except (TypeError, ValueError):
            pass

    return f"{label} ({_fmt(value)}), {direction}"


def _plain_language(text: str) -> str:
    """Strip clinical jargon for the patient-facing view."""
    replacements = {
        "SpO₂": "oxygen level",
        "SpO2": "oxygen level",
        "hypoxaemia": "low oxygen",
        "hypoxemia": "low oxygen",
        "tachycardia": "fast heart rate",
        "bradycardia": "slow heart rate",
        "tachypnoea": "fast breathing",
        "hypotension": "low blood pressure",
        "systolic blood pressure": "blood pressure",
        "respiratory rate": "breathing rate",
        "NEWS2": "standard early-warning",
        "SIRS": "infection-warning",
        "intracranial haemorrhage": "bleeding around the brain",
        "acute coronary event": "a possible heart problem",
        "compensated shock": "early signs of circulation strain",
        "anticoagulated": "taking blood-thinning medication",
    }
    out = text
    for term, plain in replacements.items():
        out = out.replace(term, plain)
    return out.split(".")[0].strip()
