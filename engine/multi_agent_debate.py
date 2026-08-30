"""
PulseGuard — Multi-Agent Safety Debate (Differentiator B)
A Throughput Agent and a Safety Sentinel Agent produce structured disagreement
that feeds the uncertainty score. The deterministic safety engine still makes
the final call — the agents only INFORM it.

LLM Boundary (Section 5.H): The agents may produce candidate levels and a
structured disagreement summary, but the deterministic safety engine remains
the sole final authority on escalation.
"""

from typing import Dict, Optional
import numpy as np


class ThroughputAgent:
    """
    Optimizes for patient flow and resource efficiency.
    Tends to assign levels that keep the queue moving.
    Not reckless — still respects critical cases — but biases toward
    'just urgent enough' rather than 'maximally cautious.'
    """

    def assess(self, vitals: Dict, age: int, history_available: bool,
               model_predictions: Dict, data_quality_score: float) -> Dict:
        """
        Produce a triage assessment from the throughput perspective.
        """
        # Start with the model consensus
        predictions = [p for p in model_predictions.values()]
        if predictions:
            base_level = int(np.median(predictions))
        else:
            base_level = 3

        reasoning = []

        # Throughput agent looks at objective vital stability
        hr = vitals.get("heart_rate")
        rr = vitals.get("respiratory_rate")
        spo2 = vitals.get("spo2")
        sbp = vitals.get("systolic_bp")

        stable_vitals = 0
        total_checked = 0

        if hr is not None:
            total_checked += 1
            if 60 <= hr <= 100:
                stable_vitals += 1
                reasoning.append("Heart rate within normal range")

        if rr is not None:
            total_checked += 1
            if 12 <= rr <= 20:
                stable_vitals += 1
                reasoning.append("Respiratory rate normal")

        if spo2 is not None:
            total_checked += 1
            if spo2 >= 95:
                stable_vitals += 1
                reasoning.append("Oxygen saturation adequate")

        if sbp is not None:
            total_checked += 1
            if 90 <= sbp <= 160:
                stable_vitals += 1
                reasoning.append("Blood pressure within acceptable range")

        # If most vitals are stable, throughput agent may suggest one level lower
        stability_ratio = stable_vitals / total_checked if total_checked > 0 else 0.5

        adjusted_level = base_level
        if stability_ratio >= 0.75 and base_level > 1:
            adjusted_level = min(base_level + 1, 5)  # Suggest one level less urgent
            reasoning.append(f"Vitals predominantly stable ({stable_vitals}/{total_checked}), "
                           "patient may be suitable for less urgent track")
        elif stability_ratio < 0.5:
            reasoning.append("Multiple abnormal vitals, maintaining urgency assessment")

        # High data quality supports the throughput position
        if data_quality_score >= 80:
            reasoning.append("Good data quality supports assessment confidence")

        return {
            "agent": "Throughput Agent",
            "recommended_level": adjusted_level,
            "reasoning": reasoning,
            "stability_ratio": round(stability_ratio, 2),
            "base_model_level": base_level,
            "priority": "flow_efficiency",
        }


class SafetySentinelAgent:
    """
    Maximizes caution and patient safety.
    Looks for red flags, worst-case scenarios, and edge cases
    that might be missed by purely data-driven models.
    Biases toward escalation under any uncertainty.
    """

    RED_FLAG_KEYWORDS = [
        "worst", "severe", "sudden", "acute", "unresponsive",
        "can't breathe", "crushing", "tearing", "thunderclap",
        "worsening", "deteriorating", "collapse", "seizure",
        "blood", "bleeding", "unconscious", "confused",
    ]

    def assess(self, vitals: Dict, age: int, sex: str,
               history_available: bool, has_high_risk_conditions: bool,
               chief_complaint: str, symptoms_text: str,
               model_predictions: Dict, consciousness: str = "alert",
               progression: str = "stable") -> Dict:
        """
        Produce a triage assessment from the safety perspective.
        """
        predictions = [p for p in model_predictions.values()]
        base_level = min(predictions) if predictions else 3  # Start with most urgent model

        reasoning = []
        escalation_flags = 0

        # Check for red flag keywords in complaints
        combined_text = f"{chief_complaint} {symptoms_text}".lower()
        found_flags = [kw for kw in self.RED_FLAG_KEYWORDS if kw in combined_text]
        if found_flags:
            escalation_flags += len(found_flags)
            reasoning.append(f"Red flag keywords detected: {', '.join(found_flags[:3])}")

        # Age-related risk
        if age >= 65:
            escalation_flags += 1
            reasoning.append("Geriatric patient, higher risk of atypical presentation and rapid decompensation")
        elif age <= 5:
            escalation_flags += 1
            reasoning.append("Young pediatric patient, higher risk of compensated shock and rapid deterioration")

        # Unknown history
        if not history_available:
            escalation_flags += 2
            reasoning.append("CAUTION: No medical history available, cannot rule out high-risk conditions")

        # Known high-risk conditions
        if has_high_risk_conditions:
            escalation_flags += 1
            reasoning.append("Patient has known high-risk medical conditions")

        # Consciousness concern
        if consciousness and consciousness.lower() != "alert":
            escalation_flags += 2
            reasoning.append(f"Altered consciousness: {consciousness}")

        # Worsening progression
        if progression and "worsening" in str(progression).lower():
            escalation_flags += 1
            reasoning.append("Symptoms reported as worsening")

        # Vital sign concerns (any abnormality is a flag)
        hr = vitals.get("heart_rate")
        spo2 = vitals.get("spo2")
        sbp = vitals.get("systolic_bp")

        if hr is not None and (hr > 120 or hr < 50):
            escalation_flags += 1
            reasoning.append(f"Concerning heart rate: {hr} bpm")

        if spo2 is not None and spo2 < 94:
            escalation_flags += 2
            reasoning.append(f"Low oxygen saturation: {spo2}%")

        if sbp is not None and (sbp < 90 or sbp > 180):
            escalation_flags += 1
            reasoning.append(f"Concerning blood pressure: {sbp} mmHg")

        # Adjust level based on flags
        adjusted_level = base_level
        if escalation_flags >= 4:
            adjusted_level = max(1, base_level - 2)
        elif escalation_flags >= 2:
            adjusted_level = max(1, base_level - 1)

        adjusted_level = max(1, min(adjusted_level, 5))

        reasoning.append(f"Safety assessment: {escalation_flags} risk flags identified")

        return {
            "agent": "Safety Sentinel Agent",
            "recommended_level": adjusted_level,
            "reasoning": reasoning,
            "escalation_flags": escalation_flags,
            "base_model_level": min(predictions) if predictions else 3,
            "priority": "patient_safety",
        }


class MultiAgentDebate:
    """
    Orchestrates the debate between Throughput Agent and Safety Sentinel.
    Produces a structured disagreement summary that feeds the safety engine.
    The agents INFORM the safety engine — they do NOT set the final level.
    """

    def __init__(self):
        self.throughput_agent = ThroughputAgent()
        self.safety_agent = SafetySentinelAgent()

    def debate(self, encounter, model_predictions: Dict,
               data_quality_score: float) -> Dict:
        """
        Run the debate and produce a structured disagreement summary.
        """
        vitals = encounter.vitals.to_feature_dict()
        consciousness = ""
        if encounter.staff_cues.consciousness:
            consciousness = str(encounter.staff_cues.consciousness.value)

        progression = ""
        if encounter.symptoms.progression:
            progression = str(encounter.symptoms.progression.value)

        # Get each agent's assessment
        throughput_result = self.throughput_agent.assess(
            vitals=vitals,
            age=encounter.age,
            history_available=encounter.history.history_available,
            model_predictions=model_predictions,
            data_quality_score=data_quality_score,
        )

        safety_result = self.safety_agent.assess(
            vitals=vitals,
            age=encounter.age,
            sex=encounter.sex,
            history_available=encounter.history.history_available,
            has_high_risk_conditions=encounter.history.has_high_risk_conditions(),
            chief_complaint=encounter.symptoms.get_chief_complaint_text(),
            symptoms_text=encounter.symptoms.get_symptom_text(),
            model_predictions=model_predictions,
            consciousness=consciousness,
            progression=progression,
        )

        # Compute disagreement
        throughput_level = throughput_result["recommended_level"]
        safety_level = safety_result["recommended_level"]
        level_difference = abs(throughput_level - safety_level)

        if level_difference == 0:
            disagreement_level = "none"
        elif level_difference == 1:
            disagreement_level = "low"
        elif level_difference == 2:
            disagreement_level = "moderate"
        else:
            disagreement_level = "high"

        # Build structured disagreement summary
        summary_parts = []
        if disagreement_level == "none":
            summary_parts.append(
                f"Both agents agree on Level {throughput_level}."
            )
        else:
            summary_parts.append(
                f"Throughput Agent recommends Level {throughput_level} "
                f"(priority: flow efficiency). "
                f"Safety Sentinel recommends Level {safety_level} "
                f"(priority: patient safety). "
                f"Disagreement: {level_difference} level(s)."
            )

        # Key points of contention
        if disagreement_level in ["moderate", "high"]:
            summary_parts.append("\nKey points of contention:")
            # Throughput reasoning
            summary_parts.append(f"  Throughput: {'; '.join(throughput_result['reasoning'][:2])}")
            # Safety reasoning
            summary_parts.append(f"  Safety: {'; '.join(safety_result['reasoning'][:2])}")

        summary = "\n".join(summary_parts)

        return {
            "throughput_agent": throughput_result,
            "safety_agent": safety_result,
            "throughput_level": throughput_level,
            "safety_agent_level": safety_level,
            "level_difference": level_difference,
            "disagreement_level": disagreement_level,
            "summary": summary,
            "recommendation": safety_level,  # Default to safety agent's recommendation
        }
