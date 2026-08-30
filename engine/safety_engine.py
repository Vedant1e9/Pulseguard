"""
PulseGuard — Deterministic Safety Engine
==============================================

The final authority on a patient's triage level.

No model, no agent and no LLM sets the level directly. They produce a
*proposal*; this engine decides. The reason is accountability rather than
distrust: when a clinician asks why a patient was made Level 2, the answer has
to be a rule with a citation and a threshold, not a gradient-boosted forest.

Two invariants hold everywhere in this file, and both are unit-tested:

  1. **The engine may only escalate.** Every path takes `min(current, target)`.
     There is no code path that can make a patient less urgent than the model
     proposed. Downgrading is a clinician's decision, recorded as an override.

  2. **Every escalation is traceable.** Each fired rule records its id, the
     values that triggered it, the threshold it compared against, its clinical
     citation, and the rule-pack version in force. That trace is what the
     explanation layer displays and what the audit log stores — the same
     object, so a clinician and an auditor can never be shown different
     accounts of the same decision.

Policy (thresholds, which rules are on, escalation targets) lives in
config/rules_*.yaml. This file holds only the logic that reads it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from engine.rule_pack import RulePack


class FiredRule:
    """One rule that fired, with everything needed to justify it later."""

    def __init__(self, rule_id: str, target_level: int, reason: str,
                 evidence: Dict, pack: RulePack):
        spec = pack.rule(rule_id) or {}
        self.rule_id = rule_id
        self.target_level = target_level
        self.reason = reason
        self.evidence = evidence
        self.category = spec.get("category", "unspecified")
        self.certainty = spec.get("certainty", "precautionary")
        self.citation = spec.get("citation", "not cited")
        self.rationale = " ".join(spec.get("clinical_rationale", "").split())

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "target_level": self.target_level,
            "reason": self.reason,
            "evidence": self.evidence,
            "category": self.category,
            "certainty": self.certainty,
            "citation": self.citation,
            "clinical_rationale": self.rationale,
        }


class SafetyEngine:
    """Deterministic, rule-pack-driven safety layer."""

    def __init__(self, rule_pack: Optional[RulePack] = None):
        self.pack = rule_pack or RulePack.load()

    # ── Public API ───────────────────────────────────────────────────────────

    def evaluate(self, model_level: int, encounter,
                 velocity_result: Optional[Dict] = None,
                 agent_debate: Optional[Dict] = None,
                 ensemble_result: Optional[Dict] = None,
                 model_output: Optional[Dict] = None) -> Dict:
        """
        Apply the rule pack to a proposed level.

        `model_output` is the bundle's `predict_one()` result, used by the
        uncertainty rules. Everything is optional so the engine still works on
        a patient with almost nothing recorded — which is the situation it
        most needs to work in.
        """
        fired: List[FiredRule] = []
        final_level = int(model_level)

        age_group = encounter.age_group.value
        thresholds = self.pack.thresholds_for(age_group)
        vitals = encounter.vitals.to_feature_dict()
        text = self._combined_text(encounter)

        # ── Consciousness (AVPU) ──
        fired += self._check_consciousness(encounter)

        # ── Age-banded critical vital signs ──
        fired += self._check_vitals(vitals, thresholds, age_group)

        # ── Bleeding ──
        fired += self._check_bleeding(encounter)

        # ── Presentations that hide their severity ──
        fired += self._check_pediatric_shock(encounter, vitals, age_group)
        fired += self._check_geriatric_atypical(encounter, text, age_group)
        fired += self._check_anticoagulated_fall(encounter, text, age_group)
        fired += self._check_sepsis(encounter, vitals, thresholds, text)

        # ── Information gaps ──
        fired += self._check_missing_history(encounter, text)

        # ── Statistical uncertainty ──
        fired += self._check_conformal_uncertainty(model_output)

        # ── Deterioration trend ──
        fired += self._check_velocity(velocity_result)

        # ── Apply. Escalation only, always. ──
        applied: List[FiredRule] = []
        for rule in sorted(fired, key=lambda r: r.target_level):
            if rule.target_level < final_level:
                final_level = rule.target_level
                applied.append(rule)
            elif rule.target_level == final_level:
                # Records agreement: the rule independently supports the level
                # the model already chose, which is worth showing a sceptical
                # clinician.
                applied.append(rule)

        was_escalated = final_level < model_level

        return {
            "final_level": int(final_level),
            "original_model_level": int(model_level),
            "was_escalated": was_escalated,
            "levels_escalated": int(model_level - final_level),
            "safety_status": "escalation_applied" if was_escalated else "pass",
            "fired_rules": [r.to_dict() for r in applied],
            "all_triggered_rules": [r.to_dict() for r in fired],
            "reasons": [r.reason for r in applied if r.target_level <= final_level],
            "rules_applied": [r.rule_id for r in applied],
            "age_group": age_group,
            "thresholds_used": thresholds,
            "rule_pack": self.pack.provenance(),
            "evaluated_at": datetime.now().isoformat(),
        }

    # ── Individual rule implementations ──────────────────────────────────────

    def _check_consciousness(self, encounter) -> List[FiredRule]:
        cue = getattr(encounter.staff_cues, "consciousness", None)
        if not cue or cue.value is None:
            return []
        avpu = str(cue.value).lower()

        mapping = [
            ("unresponsive", "UNRESPONSIVE",
             "Patient is unresponsive. Immediate resuscitation required"),
            ("pain", "RESPONDS_TO_PAIN_ONLY",
             "Patient responds only to painful stimulus. Emergent evaluation required"),
            ("verbal", "ALTERED_CONSCIOUSNESS",
             "Altered consciousness, responds to voice only"),
        ]
        for value, rule_id, reason in mapping:
            if avpu == value and self.pack.is_enabled(rule_id):
                target = self.pack.escalation_target(rule_id)
                if target:
                    return [FiredRule(rule_id, target, reason,
                                      {"avpu": avpu}, self.pack)]
        return []

    def _check_vitals(self, vitals: Dict, thresholds: Dict,
                      age_group: str) -> List[FiredRule]:
        if not self.pack.is_enabled("CRITICAL_VITAL_SIGN"):
            return []

        target = self.pack.escalation_target("CRITICAL_VITAL_SIGN", 2) or 2
        extreme_target = (self.pack.rule("CRITICAL_VITAL_SIGN") or {}).get(
            "escalate_to_if_extreme", 1)
        out: List[FiredRule] = []

        def fire(level, reason, evidence):
            out.append(FiredRule("CRITICAL_VITAL_SIGN", level, reason,
                                 evidence, self.pack))

        checks = [
            ("heart_rate", "hr_high", "hr_low", "Heart rate", "bpm", 0),
            ("respiratory_rate", "rr_high", "rr_low", "Respiratory rate", "/min", 0),
            ("systolic_bp", "sbp_high", "sbp_low", "Systolic blood pressure", "mmHg", 0),
            ("temperature", "temp_high", "temp_low", "Temperature", "°C", 1),
        ]

        for field, hi_key, lo_key, label, unit, dp in checks:
            value = vitals.get(field)
            if value is None:
                continue
            hi, lo = thresholds.get(hi_key), thresholds.get(lo_key)
            if hi is not None and value > hi:
                fire(target,
                     f"{label} {value:.{dp}f} {unit} exceeds the critical "
                     f"threshold of {hi} for a {age_group} patient",
                     {"field": field, "value": value, "threshold": hi,
                      "direction": "high", "age_group": age_group})
            elif lo is not None and value < lo:
                fire(target,
                     f"{label} {value:.{dp}f} {unit} is below the critical "
                     f"threshold of {lo} for a {age_group} patient",
                     {"field": field, "value": value, "threshold": lo,
                      "direction": "low", "age_group": age_group})

        # Oxygen saturation has a two-stage threshold: significant, then
        # immediately life-threatening.
        spo2 = vitals.get("spo2")
        if spo2 is not None:
            crit = thresholds.get("spo2_critical")
            low = thresholds.get("spo2_low")
            if crit is not None and spo2 < crit:
                fire(extreme_target,
                     f"Severe hypoxaemia. SpO₂ {spo2:.0f}% is below the "
                     f"critical floor of {crit}%",
                     {"field": "spo2", "value": spo2, "threshold": crit,
                      "direction": "low", "age_group": age_group})
            elif low is not None and spo2 < low:
                fire(target,
                     f"Hypoxaemia. SpO₂ {spo2:.0f}% is below the {low}% "
                     f"threshold for a {age_group} patient",
                     {"field": "spo2", "value": spo2, "threshold": low,
                      "direction": "low", "age_group": age_group})
        return out

    def _check_bleeding(self, encounter) -> List[FiredRule]:
        cue = getattr(encounter.staff_cues, "bleeding", None)
        if not cue or cue.value is None:
            return []
        if str(cue.value).lower() != "uncontrolled":
            return []
        if not self.pack.is_enabled("UNCONTROLLED_BLEEDING"):
            return []
        target = self.pack.escalation_target("UNCONTROLLED_BLEEDING", 2) or 2
        return [FiredRule("UNCONTROLLED_BLEEDING", target,
                          "Uncontrolled bleeding observed. Immediate intervention required",
                          {"bleeding": "uncontrolled"}, self.pack)]

    def _check_pediatric_shock(self, encounter, vitals: Dict,
                               age_group: str) -> List[FiredRule]:
        if age_group != "pediatric" or not self.pack.is_enabled("PEDIATRIC_COMPENSATED_SHOCK"):
            return []

        hr, sbp = vitals.get("heart_rate"), vitals.get("systolic_bp")
        skin_cue = getattr(encounter.staff_cues, "skin_appearance", None)
        skin = str(skin_cue.value).lower() if skin_cue and skin_cue.value else "normal"
        thresholds = self.pack.thresholds_for("pediatric")

        # The dangerous combination: fast heart rate, blood pressure still held
        # up, and skin that says perfusion is already failing.
        if (hr is not None and sbp is not None
                and hr > 140 and sbp >= thresholds.get("sbp_low", 70)
                and skin in ("pale", "mottled", "cyanotic")):
            target = self.pack.escalation_target("PEDIATRIC_COMPENSATED_SHOCK", 2) or 2
            return [FiredRule(
                "PEDIATRIC_COMPENSATED_SHOCK", target,
                f"Paediatric compensated shock suspected. Heart rate {hr:.0f} bpm "
                f"with blood pressure still maintained at {sbp:.0f} mmHg but "
                f"{skin} skin. In children blood pressure is the last thing to "
                f"fall, so a normal reading here is not reassurance.",
                {"heart_rate": hr, "systolic_bp": sbp, "skin": skin},
                self.pack)]
        return []

    def _check_geriatric_atypical(self, encounter, text: str,
                                  age_group: str) -> List[FiredRule]:
        if age_group != "geriatric" or not self.pack.is_enabled("GERIATRIC_ATYPICAL_CARDIAC"):
            return []
        if not encounter.history.has_high_risk_conditions():
            return []

        hits = [kw for kw in self.pack.lexicon("geriatric_atypical_cardiac") if kw in text]
        if not hits:
            return []

        target = self.pack.escalation_target("GERIATRIC_ATYPICAL_CARDIAC", 2) or 2
        return [FiredRule(
            "GERIATRIC_ATYPICAL_CARDIAC", target,
            f"Older patient with cardiac risk history presenting with atypical "
            f"symptoms ({', '.join(hits[:3])}). Possible masked acute coronary "
            f"event. Up to a third of infarctions over 75 present without chest pain.",
            {"matched_symptoms": hits, "age": encounter.age}, self.pack)]

    def _check_anticoagulated_fall(self, encounter, text: str,
                                   age_group: str) -> List[FiredRule]:
        if not self.pack.is_enabled("GERIATRIC_ANTICOAGULATED_FALL"):
            return []
        if age_group != "geriatric":
            return []
        if not any(kw in text for kw in self.pack.lexicon("fall_keywords")):
            return []

        meds = encounter.history.medications
        meds_text = str(meds.value).lower() if meds and meds.value else ""
        anticoags = [a for a in self.pack.lexicon("anticoagulants") if a in meds_text]
        if not anticoags:
            return []

        target = self.pack.escalation_target("GERIATRIC_ANTICOAGULATED_FALL", 2) or 2
        return [FiredRule(
            "GERIATRIC_ANTICOAGULATED_FALL", target,
            f"Older patient with a fall while anticoagulated "
            f"({', '.join(anticoags)}). High risk of intracranial haemorrhage, "
            f"which can present normally and deteriorate hours later.",
            {"anticoagulants": anticoags}, self.pack)]

    def _check_sepsis(self, encounter, vitals: Dict, thresholds: Dict,
                      text: str) -> List[FiredRule]:
        if not self.pack.is_enabled("SEPSIS_PHYSIOLOGY"):
            return []
        if not any(kw in text for kw in self.pack.lexicon("infection_keywords")):
            return []

        criteria = []
        temp, hr, rr = vitals.get("temperature"), vitals.get("heart_rate"), vitals.get("respiratory_rate")
        if temp is not None and (temp > 38.0 or temp < 36.0):
            criteria.append(f"temperature {temp:.1f} °C")
        if hr is not None and hr > 90:
            criteria.append(f"heart rate {hr:.0f} bpm")
        if rr is not None and rr > 20:
            criteria.append(f"respiratory rate {rr:.0f}/min")

        if len(criteria) < 2:
            return []

        target = self.pack.escalation_target("SEPSIS_PHYSIOLOGY", 2) or 2
        return [FiredRule(
            "SEPSIS_PHYSIOLOGY", target,
            f"Possible sepsis. {len(criteria)} SIRS criteria met "
            f"({'; '.join(criteria)}) with a plausible infective source. "
            f"Each hour of delayed antibiotics measurably raises mortality.",
            {"sirs_criteria": criteria, "n_criteria": len(criteria)}, self.pack)]

    def _check_missing_history(self, encounter, text: str) -> List[FiredRule]:
        if encounter.history.history_available:
            return []

        out = []
        if self.pack.is_enabled("ZERO_HISTORY_HIGH_RISK_COMPLAINT"):
            hits = [kw for kw in self.pack.lexicon("high_risk_complaints") if kw in text]
            if hits:
                target = self.pack.escalation_target("ZERO_HISTORY_HIGH_RISK_COMPLAINT", 2) or 2
                out.append(FiredRule(
                    "ZERO_HISTORY_HIGH_RISK_COMPLAINT", target,
                    f"High-risk complaint ('{hits[0]}') with no medical history on "
                    f"file. Absent history is not reassuring history, so the "
                    f"conditions that would make this dangerous cannot be excluded.",
                    {"matched_complaints": hits}, self.pack))

        if self.pack.is_enabled("ZERO_HISTORY_CONSERVATIVE"):
            target = self.pack.escalation_target("ZERO_HISTORY_CONSERVATIVE", 3) or 3
            out.append(FiredRule(
                "ZERO_HISTORY_CONSERVATIVE", target,
                "No medical history available, so a conservative ceiling is applied "
                "until a clinician has assessed the patient.",
                {"history_available": False}, self.pack))
        return out

    def _check_conformal_uncertainty(self, model_output: Optional[Dict]) -> List[FiredRule]:
        if not model_output or not self.pack.is_enabled("CRITICAL_NOT_EXCLUDED"):
            return []

        pred_set = model_output.get("conformal_set") or []
        proposed = model_output.get("model_level")
        if not pred_set or proposed is None:
            return []

        # Only meaningful when the model wanted to send the patient to the back
        # of the queue while a critical level remains inside the guaranteed set.
        if min(pred_set) <= 2 and proposed > 3:
            target = self.pack.escalation_target("CRITICAL_NOT_EXCLUDED", 3) or 3
            crit_p = model_output.get("critical_probability", 0.0)
            return [FiredRule(
                "CRITICAL_NOT_EXCLUDED", target,
                f"A critical level cannot be excluded at 90% coverage "
                f"(prediction set {pred_set}, {crit_p:.1%} probability of Level 1 or 2). "
                f"The patient is held at Level {target} pending clinician review.",
                {"conformal_set": pred_set, "critical_probability": crit_p},
                self.pack)]
        return []

    def _check_velocity(self, velocity_result: Optional[Dict]) -> List[FiredRule]:
        if not velocity_result or not velocity_result.get("should_trigger_reassessment"):
            return []

        risk = velocity_result.get("overall_risk", "low")
        alerts = velocity_result.get("alerts", [])

        if risk == "critical" and self.pack.is_enabled("DETERIORATION_VELOCITY_CRITICAL"):
            target = self.pack.escalation_target("DETERIORATION_VELOCITY_CRITICAL", 2) or 2
            return [FiredRule(
                "DETERIORATION_VELOCITY_CRITICAL", target,
                f"Critical deterioration trend. {'; '.join(alerts[:2])}. Rate of "
                f"change leads absolute thresholds, so this fires before any "
                f"single reading looks abnormal.",
                {"risk": risk, "alerts": alerts}, self.pack)]

        if risk == "high" and self.pack.is_enabled("DETERIORATION_VELOCITY_HIGH"):
            target = self.pack.escalation_target("DETERIORATION_VELOCITY_HIGH", 3) or 3
            return [FiredRule(
                "DETERIORATION_VELOCITY_HIGH", target,
                f"Significant deterioration trend. {'; '.join(alerts[:2])}.",
                {"risk": risk, "alerts": alerts}, self.pack)]
        return []

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _combined_text(encounter) -> str:
        return (f"{encounter.symptoms.get_chief_complaint_text()} "
                f"{encounter.symptoms.get_symptom_text()}").lower()
