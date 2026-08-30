"""
PatientTriage.ai — Reassessment Engine
Triggers reassessment when wait time exceeds safe thresholds, vitals worsen,
or deterioration velocity warrants re-evaluation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional


class ReassessmentEngine:
    """
    Monitors waiting patients and triggers reassessment events.
    Per Section 5.D: trigger if wait time exceeds safe threshold or
    a newly recorded vital is worse than the prior one.
    With Differentiator A: additionally trigger on rising rate of change
    even before an absolute threshold is crossed.
    """

    # Reassessment intervals by triage level
    REASSESSMENT_INTERVALS = {
        1: 0,     # Level 1: immediate — should not be waiting
        2: 10,    # Every 10 minutes
        3: 30,    # Every 30 minutes
        4: 60,    # Every 60 minutes
        5: 120,   # Every 120 minutes
    }

    def __init__(self):
        self.reassessment_log = []

    def check_reassessment_needed(self, patient_id: str, triage_level: int,
                                    arrival_time: datetime,
                                    last_reassessment: Optional[datetime],
                                    current_vitals: Dict,
                                    previous_vitals: Optional[Dict],
                                    velocity_result: Optional[Dict]) -> Dict:
        """
        Check if a patient needs reassessment.
        Returns dict with should_reassess, reasons, and urgency.
        """
        now = datetime.now()
        reasons = []
        urgency = "routine"

        # ── Trigger 1: Wait time exceeds threshold ──
        wait_minutes = (now - arrival_time).total_seconds() / 60.0
        threshold = self.REASSESSMENT_INTERVALS.get(triage_level, 60)

        if threshold > 0:
            time_since_last = wait_minutes
            if last_reassessment:
                time_since_last = (now - last_reassessment).total_seconds() / 60.0

            if time_since_last >= threshold:
                reasons.append(
                    f"Time since last assessment ({time_since_last:.0f} min) "
                    f"exceeds interval ({threshold} min) for Level {triage_level}"
                )
                urgency = "standard"

        # ── Trigger 2: Worsening vitals ──
        if previous_vitals and current_vitals:
            worsening = self._compare_vitals(previous_vitals, current_vitals)
            if worsening:
                reasons.extend(worsening)
                urgency = "elevated"

        # ── Trigger 3: Deterioration velocity (Differentiator A) ──
        if velocity_result and velocity_result.get("has_trend_data"):
            if velocity_result.get("should_trigger_reassessment"):
                risk = velocity_result.get("overall_risk", "low")
                alerts = velocity_result.get("alerts", [])
                reasons.append(
                    f"Deterioration velocity alert ({risk} risk): "
                    f"{'; '.join(alerts[:2])}"
                )
                if risk == "critical":
                    urgency = "immediate"
                else:
                    urgency = "elevated"

        should_reassess = len(reasons) > 0

        result = {
            "patient_id": patient_id,
            "should_reassess": should_reassess,
            "reasons": reasons,
            "urgency": urgency,
            "wait_minutes": round(wait_minutes, 1),
            "triage_level": triage_level,
            "timestamp": now.isoformat(),
        }

        if should_reassess:
            self.reassessment_log.append(result)

        return result

    def _compare_vitals(self, previous: Dict, current: Dict) -> List[str]:
        """Compare current vitals against previous and flag worsening."""
        worsening = []

        # Heart rate: rising is worsening (usually)
        prev_hr = previous.get("heart_rate") or previous.get("hr")
        curr_hr = current.get("heart_rate") or current.get("hr")
        if prev_hr and curr_hr and curr_hr > prev_hr + 15:
            worsening.append(f"Heart rate increased: {prev_hr:.0f} → {curr_hr:.0f} bpm")

        # Respiratory rate: rising is worsening
        prev_rr = previous.get("respiratory_rate") or previous.get("rr")
        curr_rr = current.get("respiratory_rate") or current.get("rr")
        if prev_rr and curr_rr and curr_rr > prev_rr + 4:
            worsening.append(f"Respiratory rate increased: {prev_rr:.0f} → {curr_rr:.0f}/min")

        # SpO2: dropping is worsening
        prev_spo2 = previous.get("spo2")
        curr_spo2 = current.get("spo2")
        if prev_spo2 and curr_spo2 and curr_spo2 < prev_spo2 - 2:
            worsening.append(f"SpO2 decreased: {prev_spo2:.0f}% → {curr_spo2:.0f}%")

        # Systolic BP: dropping is worsening
        prev_sbp = previous.get("systolic_bp") or previous.get("sbp")
        curr_sbp = current.get("systolic_bp") or current.get("sbp")
        if prev_sbp and curr_sbp and curr_sbp < prev_sbp - 20:
            worsening.append(f"Systolic BP dropped: {prev_sbp:.0f} → {curr_sbp:.0f} mmHg")

        # Temperature: rising above 38.5 is worsening
        prev_temp = previous.get("temperature") or previous.get("temp")
        curr_temp = current.get("temperature") or current.get("temp")
        if prev_temp and curr_temp and curr_temp > prev_temp + 0.5 and curr_temp > 38.0:
            worsening.append(f"Temperature rising: {prev_temp:.1f} → {curr_temp:.1f}°C")

        # Pain: increasing is worsening
        prev_pain = previous.get("pain_score") or previous.get("pain")
        curr_pain = current.get("pain_score") or current.get("pain")
        if prev_pain is not None and curr_pain is not None and curr_pain > prev_pain + 2:
            worsening.append(f"Pain increased: {prev_pain:.0f} → {curr_pain:.0f}/10")

        return worsening

    def get_reassessment_schedule(self) -> Dict[int, int]:
        """Return the reassessment schedule."""
        return dict(self.REASSESSMENT_INTERVALS)
