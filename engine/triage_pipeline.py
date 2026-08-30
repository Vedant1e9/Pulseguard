"""
PulseGuard — End-to-End Triage Pipeline
=============================================

Orchestrates one patient's journey through the system:

    encounter
      → data quality assessment
      → feature construction (age-normalised physiology, early-warning scores)
      → calibrated risk model  → probability distribution over levels
      → conformal prediction   → guaranteed-coverage set
      → cost-sensitive policy  → proposed level
      → multi-agent review     → structured second opinion (advisory)
      → DETERMINISTIC SAFETY ENGINE → final level  ← sole authority
      → decision-trace explanation
      → audit record

The ordering is the architecture. Models propose; only the rule-driven safety
engine disposes. Every stage's output is retained rather than collapsed into a
number, because the explanation and the audit record are both built from the
same retained trace — which is how a clinician and a regulator end up looking
at identical accounts of the same decision.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.input_schema import (
    AuditLogEntry, DataSource, Measurement, PatientEncounter, TriageResult,
)
from data.data_quality import DataQualityScorer
from data.features import build_features_row
from engine.explanation import ExplanationBuilder
from engine.multi_agent_debate import MultiAgentDebate
from engine.rule_pack import RulePack
from engine.safety_engine import SafetyEngine
from models.clinical_scores import compute_early_warning_score
from models.deterioration_velocity import DeteriorationVelocityModel
from models.triage_model import load_bundle, TriageModelBundle
from models.uncertainty import uncertainty_band_from_set

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), "..",
                           "saved_models", "triage_bundle.joblib")


# ─── Encounter → model record ────────────────────────────────────────────────

def encounter_to_record(encounter: PatientEncounter) -> Dict:
    """
    Flatten a PatientEncounter into the flat record the feature builder wants.

    Unmeasured values stay `None` the whole way through — they are never
    replaced with zero or with a population mean. That single discipline is
    what stops an unrecorded oxygen saturation from being read downstream as a
    saturation of zero.
    """
    vitals = encounter.vitals.to_feature_dict()

    def cue(name: str) -> Optional[str]:
        m = getattr(encounter.staff_cues, name, None)
        return str(m.value).lower() if m is not None and m.value is not None else None

    history = encounter.history
    meds_text = (str(history.medications.value).lower()
                 if history.medications and history.medications.value else "")
    conditions_text = (str(history.known_conditions.value).lower()
                       if history.known_conditions and history.known_conditions.value else "")

    # Map free-text conditions onto the coded flags the model was trained on.
    # Without this a nurse who types "COPD, heart failure" at intake gets a
    # patient scored as having no documented conditions at all — the same
    # feature loss that made the demo board diverge from the validated
    # pipeline, arriving instead through the manual path.
    coded_conditions = _code_conditions(conditions_text)

    record: Dict = {
        "age": encounter.age,
        "sex": encounter.sex,
        "chief_complaint": encounter.symptoms.get_chief_complaint_text(),
        "symptoms_text": encounter.symptoms.get_symptom_text(),
        "history_available": 1.0 if history.history_available else 0.0,
        "has_high_risk_conditions": 1.0 if history.has_high_risk_conditions() else 0.0,
        "n_chronic_conditions": float(sum(coded_conditions.values())),
        "consciousness": cue("consciousness") or "alert",
        "skin_appearance": cue("skin_appearance") or "normal",
        "breathing_difficulty": cue("breathing_difficulty") or "none",
        "arrival_by_ambulance": None,
        "seen_last_72h": None,
        "injury_related": None,
        "nursing_home_resident": None,
        "medications_text": meds_text,
    }
    record.update(vitals)
    record.update(coded_conditions)

    # Structured context (arrival mode, coded conditions, prior-visit flags)
    # takes precedence over the defaults above, since it comes from the record
    # system rather than being inferred from free text.
    record.update({k: v for k, v in (encounter.context or {}).items()
                   if v is not None})
    return record



# Keyword → coded condition flag. Deliberately generous with synonyms and
# abbreviations, because this reads what a nurse actually types under time
# pressure ("CHF", "afib on apixaban", "COPD/emphysema") rather than a
# controlled vocabulary.
CONDITION_KEYWORDS = {
    "cond_asthma": ["asthma"],
    "cond_cancer": ["cancer", "malignancy", "carcinoma", "lymphoma", "leukaemia",
                    "leukemia", "tumour", "tumor", "metasta"],
    "cond_cebvd": ["stroke", "cva", "tia", "cerebrovascular"],
    "cond_ckd": ["chronic kidney", "ckd", "renal impairment", "renal failure"],
    "cond_copd": ["copd", "emphysema", "chronic bronchitis"],
    "cond_chf": ["heart failure", "chf", "cardiac failure", "ccf"],
    "cond_cad": ["coronary", "cad", "ischaemic heart", "ischemic heart", "angina",
                 "myocardial infarction", "mi", "stent", "cabg"],
    "cond_deprn": ["depression", "depressive"],
    "cond_diabtyp1": ["type 1 diabetes", "type i diabetes", "t1dm"],
    "cond_diabtyp2": ["type 2 diabetes", "type ii diabetes", "t2dm"],
    "cond_diabtyp0": ["diabetes", "diabetic", "dm"],
    "cond_esrd": ["esrd", "end-stage renal", "end stage renal", "dialysis"],
    "cond_hpe": ["pulmonary embolism", "pe", "dvt", "deep vein", "vte"],
    "cond_edhiv": ["hiv", "aids"],
    "cond_hyplipid": ["hyperlipid", "high cholesterol", "dyslipid"],
    "cond_htn": ["hypertension", "htn", "high blood pressure"],
    "cond_obesity": ["obesity", "obese"],
    "cond_osa": ["sleep apnoea", "sleep apnea", "osa"],
    "cond_ostprsis": ["osteoporosis"],
    "cond_substab": ["substance", "opioid use", "drug use", "iv drug"],
    "cond_etohab": ["alcohol", "etoh", "alcoholic"],
    "cond_alzhd": ["dementia", "alzheimer", "cognitive impairment"],
}


def _code_conditions(text: str) -> Dict[str, float]:
    """Turn free-text history into the coded condition flags the model uses."""
    text = (text or "").lower()
    coded = {flag: 0.0 for flag in CONDITION_KEYWORDS}
    if not text.strip():
        return coded

    for flag, keywords in CONDITION_KEYWORDS.items():
        for keyword in keywords:
            # Short abbreviations need word boundaries; "mi" must not match
            # "vomiting" and "pe" must not match "pain".
            if len(keyword) <= 3:
                import re
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    coded[flag] = 1.0
                    break
            elif keyword in text:
                coded[flag] = 1.0
                break

    # A specific diabetes type implies the generic flag the survey also codes.
    if coded["cond_diabtyp1"] or coded["cond_diabtyp2"]:
        coded["cond_diabtyp0"] = 1.0
    return coded


# ─── Pipeline ────────────────────────────────────────────────────────────────

class TriagePipeline:
    """The full triage system, ready to score patients."""

    def __init__(self, site: Optional[str] = None,
                 bundle_path: Optional[str] = None):
        self.rule_pack = RulePack.load_site(site) if site else RulePack.load()
        self.safety_engine = SafetyEngine(self.rule_pack)
        self.data_quality_scorer = DataQualityScorer()
        self.velocity_model = DeteriorationVelocityModel()
        self.multi_agent_debate = MultiAgentDebate()

        self.bundle: Optional[TriageModelBundle] = None
        self.explainer: Optional[ExplanationBuilder] = None
        self.bundle_path = bundle_path or BUNDLE_PATH

        self.patients: List[Tuple[PatientEncounter, int, str]] = []
        self.patient_encounters: Dict[str, Tuple] = {}
        self.triage_results: Dict[str, Dict] = {}
        self.audit_log: List[AuditLogEntry] = []
        self.is_ready = False
        self.site = site or "default"

    # ── Setup ────────────────────────────────────────────────────────────────

    def initialize(self, load_demo_patients: bool = True, verbose: bool = True):
        if verbose:
            print("=" * 68)
            print("PulseGuard initialising")
            print("=" * 68)

        if not os.path.exists(self.bundle_path):
            raise FileNotFoundError(
                f"No trained model bundle at {self.bundle_path}.\n"
                f"Train one with:  python -m scripts.train_model"
            )

        self.bundle = load_bundle(self.bundle_path)
        self.explainer = ExplanationBuilder(self.bundle)

        if verbose:
            meta = self.bundle.metadata
            print(f"  Model         : {self.bundle.model_name}")
            print(f"  Trained on    : {meta.get('n_train', '?'):,} real ED visits")
            print(f"  Features      : {meta.get('n_features', '?')}")
            print(f"  Cost profile  : {self.bundle.cost_matrix.name}")
            print(f"  Rule pack     : {self.rule_pack.pack_id} "
                  f"v{self.rule_pack.version} ({self.rule_pack.content_hash()})")

        if load_demo_patients:
            from data.demo_cohort import load_demo_cohort
            self.patients = load_demo_cohort()
            for enc, level, desc in self.patients:
                self.patient_encounters[enc.patient_id] = (enc, level, desc)

            # Serial observations for the deterioration-velocity demonstration.
            # Attached to real board patients so the trend display is anchored
            # to a real presentation rather than an invented one.
            self._attach_vitals_histories()

            if verbose:
                n_real = sum(1 for _, _, d in self.patients if d.startswith("NHAMCS"))
                n_edge = len(self.patients) - n_real
                print(f"  Demo board    : {n_real} real held-out ED visits "
                      f"+ {n_edge} curated edge cases")

        self.is_ready = True
        if verbose:
            print("  Status        : ready\n")
        return self

    def _attach_vitals_histories(self):
        """
        Give a few board patients a series of repeated observations.

        NHAMCS records one set of vitals per visit, so serial observations have
        to be simulated for the deterioration-velocity demonstration. They are
        generated from each patient's own recorded vitals as the starting
        point and labelled as simulated wherever they are displayed — the
        trend is illustrative, the starting physiology is real.
        """
        from datetime import timedelta
        import numpy as np

        rng = np.random.RandomState(11)
        eligible = [(enc, gt) for enc, gt, desc in self.patients
                    if desc.startswith("NHAMCS") and gt in (2, 3)]
        if not eligible:
            return

        for enc, _ in eligible[:3]:
            base = enc.vitals.to_feature_dict()
            hr0 = base.get("heart_rate") or 88
            rr0 = base.get("respiratory_rate") or 18
            spo0 = base.get("spo2") or 97

            history = []
            for step in range(4):
                timestamp = enc.arrival_time + timedelta(minutes=step * 15)
                history.append((timestamp, {
                    "heart_rate": float(hr0 + step * 11 + rng.normal(0, 1.5)),
                    "respiratory_rate": float(rr0 + step * 2.5 + rng.normal(0, 0.5)),
                    "spo2": float(max(85, spo0 - step * 1.8 + rng.normal(0, 0.4))),
                    "systolic_bp": float((base.get("systolic_bp") or 120) - step * 4),
                }))
            self.velocity_model.load_history(enc.patient_id, history)
            self.simulated_trend_patients = getattr(
                self, "simulated_trend_patients", set()) | {enc.patient_id}

    # ── Scoring ──────────────────────────────────────────────────────────────

    def triage_patient(self, encounter: PatientEncounter,
                       ground_truth: Optional[int] = None,
                       store: bool = True) -> TriageResult:
        """Run one patient through the full pipeline."""
        if not self.is_ready:
            raise RuntimeError("Pipeline not initialised. Call initialize() first.")

        t_start = time.perf_counter()
        patient_id = encounter.patient_id

        # ── 1. Data quality ──
        data_quality = self.data_quality_scorer.score_encounter(encounter)

        # ── 2. Features + model ──
        record = encounter_to_record(encounter)
        model_output = self.bundle.predict_one(record)

        # ── 3. Age-appropriate early warning score (shown alongside, as a
        #      familiar reference point for staff who already use it) ──
        ews = compute_early_warning_score(
            {k: record.get(k) for k in
             ["temperature", "heart_rate", "respiratory_rate", "spo2",
              "systolic_bp", "diastolic_bp"]},
            encounter.age,
            record.get("consciousness", "alert"),
            record.get("skin_appearance", "normal"),
            record.get("breathing_difficulty", "none"),
        )

        # ── 4. Deterioration velocity ──
        velocity_result = self.velocity_model.compute_velocity(patient_id)

        # ── 5. Multi-agent review (advisory only) ──
        agent_debate = self.multi_agent_debate.debate(
            encounter,
            {"Risk model": model_output["model_level"],
             "Early warning score": ews["implied_triage_level"]},
            data_quality["overall_score"],
        )

        # ── 6. Safety engine — the only thing that sets the level ──
        safety_result = self.safety_engine.evaluate(
            model_level=model_output["model_level"],
            encounter=encounter,
            velocity_result=velocity_result,
            agent_debate=agent_debate,
            model_output=model_output,
        )
        final_level = safety_result["final_level"]

        # ── 7. Confidence & uncertainty ──
        confidence = self._confidence(model_output, safety_result, data_quality)
        exact_match_probability = float(
            model_output.get("probabilities", {}).get(final_level, 0.0)
        )
        uncertainty_band = self._uncertainty_band(
            model_output, safety_result, data_quality
        )

        # ── 8. Explanation trace ──
        explanation = self.explainer.build(
            record, model_output, safety_result, data_quality, velocity_result
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        result = TriageResult(
            patient_id=patient_id,
            triage_level=final_level,
            confidence_percent=confidence,
            uncertainty_band=uncertainty_band,
            data_quality_percent=data_quality["overall_score"],
            model_agreement=self._agreement(model_output, ews),
            individual_predictions={
                "Risk model (cost-optimal)": model_output["model_level"],
                "Most likely level": model_output["most_likely_level"],
                ews["scale"]: ews["implied_triage_level"],
            },
            safety_status=safety_result["safety_status"],
            safety_reason=("; ".join(safety_result["reasons"])
                           if safety_result["reasons"] else None),
            top_contributing_factors=[f["headline"] for f in explanation["factors"][:5]],
            missing_information=data_quality.get("missing_fields", []),
            recommended_followup_question=data_quality.get("recommendation"),
            recommended_action=self._recommended_action(final_level),
            agent_debate_summary=agent_debate.get("summary"),
            deterioration_velocity=(velocity_result
                                    if velocity_result.get("has_trend_data") else None),
        )

        if store:
            # A patient scored at intake has to become a real member of the
            # board, not just a row in triage_results. Without this the intake
            # page's own closing line — "this patient is now on the board and
            # in the waiting queue" — was untrue: the board, the patient
            # selector and the queue all iterate self.patients, so a newly
            # triaged arrival was scored, logged, counted in the emergent-lane
            # tally, and then invisible everywhere a clinician would look.
            if patient_id not in self.patient_encounters:
                description = "LIVE INTAKE: entered on this device"
                self.patients.append((encounter, ground_truth, description))
                self.patient_encounters[patient_id] = (
                    encounter, ground_truth, description)

            self.triage_results[patient_id] = {
                "result": result,
                "model_output": model_output,
                "safety": safety_result,
                "explanation": explanation,
                "data_quality": data_quality,
                "velocity": velocity_result,
                "agent_debate": agent_debate,
                "early_warning_score": ews,
                "record": record,
                "ground_truth": ground_truth,
                "latency_ms": round(latency_ms, 2),
                "exact_match_probability": round(exact_match_probability, 4),
                "confidence_semantics": "P(true level >= assigned level)",
            }

            self.audit_log.append(AuditLogEntry(
                timestamp=datetime.now(),
                event_type="triage_decision",
                patient_id=patient_id,
                user_id="SYSTEM",
                details={
                    "final_level": final_level,
                    "model_proposed_level": model_output["model_level"],
                    "safety_escalated": safety_result["was_escalated"],
                    "rules_fired": safety_result["rules_applied"],
                    "confidence": confidence,
                    "uncertainty": uncertainty_band,
                    "conformal_set": model_output["conformal_set"],
                    "rule_pack_version": self.rule_pack.version,
                    "rule_pack_hash": self.rule_pack.content_hash(),
                    "model_name": self.bundle.model_name,
                    "latency_ms": round(latency_ms, 2),
                    "ground_truth": ground_truth,
                },
            ))

        return result

    def triage_all_patients(self) -> List[TriageResult]:
        return [self.triage_patient(enc, gt) for enc, gt, _ in self.patients]

    # ── Reassessment ─────────────────────────────────────────────────────────

    def record_observation(self, patient_id: str, vitals: Dict[str, float],
                           recorded_by: str = "TRIAGE_NURSE") -> Optional[Dict]:
        """
        Re-record a waiting patient's vitals and re-score them.

        The brief makes this a requirement rather than a nicety: the system
        "must monitor patients already in the waiting queue and trigger
        re-assessment if wait time exceeds safe thresholds for their severity
        level *or if vitals are re-recorded as worsening*". Wait-time
        monitoring was already live. This is the second half, and it is the
        half that makes deterioration real rather than simulated, because the
        reading comes from a nurse walking the waiting room rather than from a
        fixture generated at boot.

        Re-scoring is **escalate-only**, and deliberately so. A patient whose
        heart rate happens to read lower on one recheck has not necessarily
        improved; they may be tiring. Letting a single favourable observation
        walk a patient back down the queue would turn routine monitoring into
        an unaccountable de-escalation channel, which is precisely the thing
        the override flow exists to keep in a clinician's hands and on the
        record. So the level may rise on new evidence and never fall, and a
        downgrade remains an override with a name attached.

        Returns a before/after summary, or None if the patient is unknown.
        """
        stored = self.triage_results.get(patient_id)
        entry = self.patient_encounters.get(patient_id)
        if not stored or not entry:
            return None

        encounter, ground_truth, _ = entry
        previous_level = stored["result"].triage_level
        previous_vitals = encounter.vitals.to_feature_dict()

        clean = {k: float(v) for k, v in vitals.items() if v is not None}
        if not clean:
            return None

        now = datetime.now()

        # The first re-recording needs the arrival observation behind it, or
        # there is no interval to compute a rate of change over.
        if patient_id not in self.velocity_model.patient_histories:
            baseline = {k: v for k, v in previous_vitals.items() if v is not None}
            if baseline:
                self.velocity_model.add_reading(
                    patient_id, encounter.arrival_time, baseline)
        self.velocity_model.add_reading(patient_id, now, clean)

        # Overwrite only what was actually re-measured. A vital the nurse did
        # not recheck keeps its arrival value rather than becoming unknown.
        for name, value in clean.items():
            existing = getattr(encounter.vitals, name, None)
            unit = existing.unit if existing is not None else ""
            setattr(encounter.vitals, name,
                    Measurement(value=value, unit=unit, timestamp=now,
                                source=DataSource.DEVICE_MEASURED))

        result = self.triage_patient(encounter, ground_truth, store=True)

        proposed_level = result.triage_level
        if proposed_level > previous_level:
            result.triage_level = previous_level
            result.recommended_action = self._recommended_action(previous_level)

        velocity = self.triage_results[patient_id].get("velocity", {})
        changes = [
            f"{name} {previous_vitals[name]:.0f} to {clean[name]:.0f}"
            for name in clean
            if previous_vitals.get(name) is not None
            and abs(previous_vitals[name] - clean[name]) >= 1
        ]

        self.audit_log.append(AuditLogEntry(
            timestamp=now,
            event_type="reassessment",
            patient_id=patient_id,
            user_id=recorded_by,
            details={
                "previous_level": previous_level,
                "proposed_level": proposed_level,
                "final_level": result.triage_level,
                "escalated": result.triage_level < previous_level,
                "held_by_escalate_only_rule": proposed_level > previous_level,
                "vitals_recorded": clean,
                "changes": changes,
                "velocity_risk": velocity.get("overall_risk", "unknown"),
                "velocity_alerts": velocity.get("alerts", [])[:3],
                "rules_fired": self.triage_results[patient_id]["safety"]["rules_applied"],
                "rule_pack_version": self.rule_pack.version,
                "rule_pack_hash": self.rule_pack.content_hash(),
            },
        ))

        return {
            "patient_id": patient_id,
            "previous_level": previous_level,
            "proposed_level": proposed_level,
            "final_level": result.triage_level,
            "escalated": result.triage_level < previous_level,
            "held_by_escalate_only_rule": proposed_level > previous_level,
            "changes": changes,
            "velocity": velocity,
            "result": result,
            "recorded_at": now,
        }

    # ── Override handling ────────────────────────────────────────────────────

    def apply_override(self, patient_id: str, new_level: int) -> bool:
        """
        Make a clinician's override take effect on the live record.

        Recording an override in the audit log while the queue keeps showing
        the system's original level would be worse than not supporting
        overrides at all: the clinician believes they have acted, and the
        department carries on treating the patient as the machine ranked them.
        """
        entry = self.triage_results.get(patient_id)
        if not entry:
            return False

        result = entry["result"]
        entry["system_level_before_override"] = result.triage_level
        result.triage_level = int(new_level)
        result.recommended_action = self._recommended_action(int(new_level))
        result.safety_status = "clinician_override"
        entry["overridden"] = True
        return True

    # ── Internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _confidence(model_output: Dict, safety_result: Dict,
                    data_quality: Dict) -> float:
        """
        Confidence that the patient has NOT been under-triaged.

        The obvious definition — P(true level == assigned level) — is the wrong
        one for a safety system, and visibly so. It reported 5% on a patient
        who was correctly identified as Level 1 by an unresponsiveness rule,
        because the model had spread its probability mass across levels while
        the rule was categorical. A nurse reading "Level 1, 5% confidence"
        learns to distrust the field, which is worse than showing nothing.

        What a triage nurse actually needs to know is whether this patient
        could be sicker than the assigned level. So confidence here is

            P(true level ≥ assigned level)

        — the probability we have not sent someone to the back of a queue they
        do not belong in. It rises when the system escalates (as it should:
        escalation buys safety margin) and falls when a low-urgency assignment
        still carries real probability of a critical presentation, which is
        precisely when a nurse should look twice.

        The exact-match probability is retained separately for the compliance
        record — it is the right number for auditing calibration, just not the
        right number to put in front of someone with nine seconds to decide.
        """
        probs = {int(k): float(v) for k, v in model_output.get("probabilities", {}).items()}
        assigned = safety_result["final_level"]

        # Probability the patient is no MORE urgent than assigned
        not_under_triaged = sum(p for level, p in probs.items() if level >= assigned)

        # An OBSERVED rule is a categorical clinical criterion, not a
        # probabilistic guess: an unresponsive patient is a Level 1 whatever
        # the model's distribution says, so that escalation carries the rule's
        # certainty. A PRECAUTIONARY rule is the opposite — it fires precisely
        # because something is unknown, and inheriting high confidence from it
        # would let the system sound most certain exactly when it knows least.
        observed_rules = [r for r in safety_result.get("fired_rules", [])
                          if r.get("certainty") == "observed"
                          and r.get("target_level") == assigned]
        if observed_rules:
            not_under_triaged = max(not_under_triaged, 0.95)

        dq = data_quality.get("overall_score", 100.0) / 100.0
        confidence = not_under_triaged * 100.0 * (0.85 + 0.15 * dq)

        return round(float(np.clip(confidence, 5.0, 99.0)), 1)

    @staticmethod
    def _uncertainty_band(model_output: Dict, safety_result: Dict,
                          data_quality: Dict) -> str:
        """
        Report how much is genuinely unknown about this patient.

        Not derived from the five-class conformal set. On real triage data
        those sets span nearly the whole scale — a true and useful finding for
        the evaluation report, but as a per-patient signal it labelled every
        single patient "high uncertainty", which conveys nothing.

        The band instead combines the three things that do discriminate:
        whether a critical presentation can be statistically excluded, how
        complete the record is, and whether the escalation rests on an
        observation or on a gap in what we know.
        """
        dq = data_quality.get("overall_score", 100.0) / 100.0
        critical_excluded = model_output.get("critical_excluded_with_confidence", False)
        critical_p = float(model_output.get("critical_probability", 0.0))
        precautionary = [r for r in safety_result.get("fired_rules", [])
                         if r.get("certainty") == "precautionary"
                         and r.get("target_level") == safety_result["final_level"]]

        if dq < 0.55 or (precautionary and critical_p > 0.25):
            return "high"
        if critical_excluded and dq >= 0.75 and not precautionary:
            return "low"
        if critical_p > 0.35 or dq < 0.75 or precautionary:
            return "moderate"
        return "low"

    @staticmethod
    def _agreement(model_output: Dict, ews: Dict) -> float:
        """Agreement between the learned model and the standard clinical score."""
        gap = abs(model_output["model_level"] - ews["implied_triage_level"])
        return round(max(0.0, 1.0 - gap / 4.0), 3)

    @staticmethod
    def _recommended_action(level: int) -> str:
        return {
            1: "IMMEDIATE: Activate resuscitation team. Continuous monitoring. Do not leave unattended.",
            2: "EMERGENT: Physician assessment within 10 minutes. Cardiac monitoring. Priority labs and imaging.",
            3: "URGENT: Physician assessment within 30 minutes. Diagnostics as indicated. Analgesia as needed.",
            4: "LESS URGENT: Physician assessment within 60 minutes. Basic diagnostics if indicated.",
            5: "NON-URGENT: Assess when available. May suit fast-track or an alternative care pathway.",
        }.get(level, "Assess as clinically indicated.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = TriagePipeline().initialize()
    results = pipeline.triage_all_patients()

    print("=" * 78)
    print("TRIAGE RESULTS: curated clinical challenge cases")
    print("=" * 78)

    for result in results:
        stored = pipeline.triage_results[result.patient_id]
        _, gt, desc = pipeline.patient_encounters.get(result.patient_id, (None, "?", ""))
        icon = "⚠" if result.safety_status == "escalation_applied" else "✓"
        print(f"\n{result.patient_id} | Level {result.triage_level} (nurse: {gt}) "
              f"| conf {result.confidence_percent:.0f}% | {result.uncertainty_band} "
              f"uncertainty | set {stored['model_output']['conformal_set']} | {icon}")
        for factor in result.top_contributing_factors[:2]:
            print(f"    • {factor}")

    latencies = [v["latency_ms"] for v in pipeline.triage_results.values()]
    matched = sum(1 for r in results
                  if pipeline.patient_encounters[r.patient_id][1] == r.triage_level)
    escalated = sum(1 for r in results if r.safety_status == "escalation_applied")

    print(f"\n{'=' * 78}")
    print(f"{matched}/{len(results)} match the reference level | "
          f"{escalated} safety escalations | "
          f"median latency {np.median(latencies):.1f} ms")
