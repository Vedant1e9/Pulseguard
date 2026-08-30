"""
PatientTriage.ai — Data Quality Layer
Scores the completeness, freshness, and consistency of patient data.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class DataQualityScorer:
    """
    Computes a composite data quality score (0-100%) for a patient encounter.
    Three sub-scores: completeness, freshness, consistency.
    """

    # Fields considered essential for triage
    ESSENTIAL_VITALS = ["temperature", "heart_rate", "respiratory_rate", "spo2",
                         "systolic_bp", "diastolic_bp"]
    ESSENTIAL_SYMPTOMS = ["chief_complaint"]
    ESSENTIAL_CUES = ["consciousness", "visible_distress", "breathing_difficulty"]

    # Freshness thresholds (minutes)
    FRESH_THRESHOLD = 15       # <15 min = fully fresh
    STALE_THRESHOLD = 60       # >60 min = stale
    VERY_STALE_THRESHOLD = 120  # >120 min = very stale

    def __init__(self):
        self.weights = {
            "completeness": 0.50,
            "freshness": 0.25,
            "consistency": 0.25,
        }

    def score_encounter(self, encounter) -> Dict:
        """
        Score a PatientEncounter and return detailed quality breakdown.
        Returns dict with overall score and sub-scores.
        """
        completeness = self._score_completeness(encounter)
        freshness = self._score_freshness(encounter)
        consistency = self._score_consistency(encounter)

        overall = (
            self.weights["completeness"] * completeness["score"] +
            self.weights["freshness"] * freshness["score"] +
            self.weights["consistency"] * consistency["score"]
        )

        missing_fields = completeness.get("missing_fields", [])
        quality_issues = []
        quality_issues.extend(completeness.get("issues", []))
        quality_issues.extend(freshness.get("issues", []))
        quality_issues.extend(consistency.get("issues", []))

        return {
            "overall_score": round(overall, 1),
            "completeness": completeness,
            "freshness": freshness,
            "consistency": consistency,
            "missing_fields": missing_fields,
            "quality_issues": quality_issues,
            "recommendation": self._get_recommendation(overall, missing_fields),
        }

    def _score_completeness(self, encounter) -> Dict:
        """Score how complete the data is. Missing essentials penalize heavily."""
        total_fields = 0
        present_fields = 0
        missing_fields = []
        issues = []

        # Check vitals
        vitals_dict = encounter.vitals.to_feature_dict()
        for field in self.ESSENTIAL_VITALS:
            total_fields += 1
            if vitals_dict.get(field) is not None:
                present_fields += 1
            else:
                missing_fields.append(f"vitals.{field}")

        # Pain score (important but not essential)
        total_fields += 1
        if vitals_dict.get("pain_score") is not None:
            present_fields += 1
        else:
            missing_fields.append("vitals.pain_score")

        # Check symptoms
        total_fields += 1
        if encounter.symptoms.get_chief_complaint_text():
            present_fields += 1
        else:
            missing_fields.append("symptoms.chief_complaint")
            issues.append("CRITICAL: No chief complaint recorded")

        # Check history availability
        total_fields += 1
        if encounter.history.history_available:
            present_fields += 1
        else:
            missing_fields.append("history (unavailable)")
            issues.append("Medical history unavailable, escalation bias applies")

        # Check staff cues
        for field in self.ESSENTIAL_CUES:
            total_fields += 1
            cue_val = getattr(encounter.staff_cues, field, None)
            if cue_val is not None:
                present_fields += 1
            else:
                missing_fields.append(f"staff_cues.{field}")

        score = (present_fields / total_fields) * 100 if total_fields > 0 else 0

        return {
            "score": score,
            "present": present_fields,
            "total": total_fields,
            "missing_fields": missing_fields,
            "issues": issues,
        }

    def _score_freshness(self, encounter) -> Dict:
        """Score how recent the data is."""
        now = datetime.now()
        issues = []
        freshness_scores = []

        # Check vital sign timestamps
        for field_name in ["temperature", "heart_rate", "respiratory_rate",
                           "spo2", "systolic_bp", "diastolic_bp"]:
            measurement = getattr(encounter.vitals, field_name, None)
            if measurement is not None:
                age_minutes = (now - measurement.timestamp).total_seconds() / 60.0
                if age_minutes <= self.FRESH_THRESHOLD:
                    freshness_scores.append(100)
                elif age_minutes <= self.STALE_THRESHOLD:
                    # Linear decay from 100 to 50
                    score = 100 - 50 * ((age_minutes - self.FRESH_THRESHOLD) /
                                         (self.STALE_THRESHOLD - self.FRESH_THRESHOLD))
                    freshness_scores.append(score)
                    if age_minutes > 30:
                        issues.append(f"{field_name} is {int(age_minutes)} minutes old")
                elif age_minutes <= self.VERY_STALE_THRESHOLD:
                    freshness_scores.append(30)
                    issues.append(f"STALE: {field_name} is {int(age_minutes)} minutes old")
                else:
                    freshness_scores.append(10)
                    issues.append(f"VERY STALE: {field_name} is {int(age_minutes)} minutes old, reassessment recommended")

        score = np.mean(freshness_scores) if freshness_scores else 50
        return {"score": score, "issues": issues}

    def _score_consistency(self, encounter) -> Dict:
        """Check for contradictions between patient report and staff observations."""
        issues = []
        penalty = 0

        # Check: Patient says low severity but staff observes high distress
        if (encounter.symptoms.severity_self_assessed and
            encounter.staff_cues.visible_distress):
            patient_severity = encounter.symptoms.severity_self_assessed.value
            staff_distress = str(encounter.staff_cues.visible_distress.value).lower()

            if patient_severity is not None:
                if patient_severity <= 3 and staff_distress in ["moderate", "severe"]:
                    issues.append("CONTRADICTION: Patient reports low severity but staff observes significant distress")
                    penalty += 20
                elif patient_severity >= 7 and staff_distress == "none":
                    issues.append("NOTE: Patient reports high severity but no visible distress observed by staff")
                    penalty += 10

        # Check: Patient says improving but vitals suggest otherwise
        if encounter.symptoms.progression:
            progression = str(encounter.symptoms.progression.value).lower()
            vitals = encounter.vitals.to_feature_dict()

            if "improving" in progression:
                hr = vitals.get("heart_rate")
                rr = vitals.get("respiratory_rate")
                if hr and hr > 110:
                    issues.append("CONTRADICTION: Patient says improving but heart rate is elevated (>110)")
                    penalty += 15
                if rr and rr > 24:
                    issues.append("CONTRADICTION: Patient says improving but respiratory rate is elevated (>24)")
                    penalty += 15

        # Check: Skin appearance vs. reported status
        if encounter.staff_cues.skin_appearance:
            skin = str(encounter.staff_cues.skin_appearance.value).lower()
            if skin in ["cyanotic", "diaphoretic", "pale"]:
                if (encounter.symptoms.severity_self_assessed and
                    encounter.symptoms.severity_self_assessed.value is not None and
                    encounter.symptoms.severity_self_assessed.value <= 3):
                    issues.append(f"NOTE: Staff observes {skin} skin but patient reports low severity")
                    penalty += 10

        score = max(0, 100 - penalty)
        return {"score": score, "issues": issues}

    def _get_recommendation(self, overall_score, missing_fields) -> Optional[str]:
        """Generate a follow-up question if material information is missing."""
        critical_missing = [f for f in missing_fields if "chief_complaint" in f or
                           "consciousness" in f or "history" in f]

        if "symptoms.chief_complaint" in missing_fields:
            return "What is the patient's primary reason for visiting the emergency department today?"

        if "history (unavailable)" in missing_fields:
            return "Do you have any medical conditions, take any medications, or have any allergies?"

        if any("spo2" in f for f in missing_fields) and any("respiratory_rate" in f for f in missing_fields):
            return "Can the patient's oxygen saturation and respiratory rate be measured?"

        if len(missing_fields) > 4:
            return "Multiple vital signs are missing. Can a complete set of vitals be obtained?"

        if overall_score < 50:
            return "Data quality is low. Consider repeating assessment with more complete information."

        return None
