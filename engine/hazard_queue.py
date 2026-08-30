"""
PatientTriage.ai — Time-Decay Hazard Queue (Differentiator E)
Orders the waiting queue by a live hazard score that shifts with
wait time and latest vitals, instead of a static triage level.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class HazardQueueManager:
    """
    Manages a waiting queue ordered by a dynamic hazard score.
    The hazard score combines triage level, wait time, age risk,
    and deterioration velocity into a single sortable priority.
    """

    # Wait time risk multipliers by triage level (minutes before concern)
    SAFE_WAIT_THRESHOLDS = {
        1: 0,     # Level 1: immediate, should never wait
        2: 10,    # Level 2: 10 minutes max
        3: 30,    # Level 3: 30 minutes
        4: 60,    # Level 4: 60 minutes
        5: 120,   # Level 5: 120 minutes
    }

    # Age-group risk modifiers
    AGE_RISK = {
        "pediatric": 1.3,    # Kids decompensate faster
        "adult": 1.0,
        "geriatric": 1.25,   # Elderly decompensate faster
    }

    def __init__(self):
        self.queue = {}  # patient_id -> queue entry dict

    def add_patient(self, patient_id: str, triage_level: int,
                    age_group: str, arrival_time: datetime,
                    confidence: float = 80.0,
                    uncertainty: str = "low",
                    velocity_risk: str = "low",
                    vitals_summary: Dict = None):
        """Add a patient to the hazard queue."""
        self.queue[patient_id] = {
            "patient_id": patient_id,
            "triage_level": triage_level,
            "age_group": age_group,
            "arrival_time": arrival_time,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "velocity_risk": velocity_risk,
            "vitals_summary": vitals_summary or {},
            "last_updated": datetime.now(),
            "reassessment_count": 0,
            "status": "waiting",
        }

    def remove_patient(self, patient_id: str):
        """Remove a patient from the queue (called for treatment)."""
        if patient_id in self.queue:
            self.queue[patient_id]["status"] = "called"

    def compute_hazard_score(self, patient_id: str) -> Dict:
        """
        Compute the live hazard score for a patient.
        Higher score = more urgent = higher priority.

        Score = base_urgency × wait_time_factor × age_modifier × uncertainty_modifier × velocity_modifier
        """
        entry = self.queue.get(patient_id)
        if not entry or entry["status"] != "waiting":
            return {"score": 0, "components": {}}

        now = datetime.now()

        # ── Component 1: Base urgency (inverted triage level) ──
        # Level 1 → 100, Level 2 → 80, Level 3 → 60, Level 4 → 40, Level 5 → 20
        base_urgency = (6 - entry["triage_level"]) * 20

        # ── Component 2: Wait time factor ──
        wait_minutes = (now - entry["arrival_time"]).total_seconds() / 60.0
        safe_wait = self.SAFE_WAIT_THRESHOLDS.get(entry["triage_level"], 60)

        if safe_wait == 0:
            # Level 1: should be seen immediately
            wait_factor = 2.0 + (wait_minutes / 5.0)  # Escalates rapidly
        elif wait_minutes <= safe_wait:
            # Within safe window: modest linear increase
            wait_factor = 1.0 + 0.3 * (wait_minutes / safe_wait)
        elif wait_minutes <= safe_wait * 2:
            # Past safe threshold: accelerating increase
            overage = (wait_minutes - safe_wait) / safe_wait
            wait_factor = 1.3 + 0.7 * overage
        else:
            # Way past safe threshold: high urgency
            overage = (wait_minutes - safe_wait) / safe_wait
            wait_factor = 2.0 + 0.5 * overage

        # ── Component 3: Age modifier ──
        age_modifier = self.AGE_RISK.get(entry["age_group"], 1.0)

        # ── Component 4: Uncertainty modifier ──
        # Higher uncertainty = higher hazard (precautionary principle)
        uncertainty_modifier = {
            "low": 1.0,
            "moderate": 1.15,
            "high": 1.3,
        }.get(entry["uncertainty"], 1.0)

        # ── Component 5: Velocity modifier ──
        velocity_modifier = {
            "low": 1.0,
            "moderate": 1.2,
            "high": 1.5,
            "critical": 2.0,
            "insufficient_data": 1.1,  # Slight caution when no trend data
        }.get(entry["velocity_risk"], 1.0)

        # ── Compute final score ──
        hazard_score = (base_urgency * wait_factor * age_modifier *
                       uncertainty_modifier * velocity_modifier)

        return {
            "score": round(hazard_score, 1),
            "patient_id": patient_id,
            "triage_level": entry["triage_level"],
            "wait_minutes": round(wait_minutes, 1),
            "safe_wait_threshold": safe_wait,
            "wait_exceeded": wait_minutes > safe_wait,
            "components": {
                "base_urgency": base_urgency,
                "wait_factor": round(wait_factor, 2),
                "age_modifier": age_modifier,
                "uncertainty_modifier": uncertainty_modifier,
                "velocity_modifier": velocity_modifier,
            },
        }

    def get_ordered_queue(self) -> List[Dict]:
        """
        Get the full queue ordered by hazard score (highest first).
        This is the live, dynamic queue that replaces static ordering.
        """
        scored_patients = []
        for patient_id, entry in self.queue.items():
            if entry["status"] == "waiting":
                hazard = self.compute_hazard_score(patient_id)
                scored_patients.append({
                    **entry,
                    "hazard_score": hazard["score"],
                    "wait_minutes": hazard["wait_minutes"],
                    "wait_exceeded": hazard["wait_exceeded"],
                    "hazard_components": hazard["components"],
                })

        # Sort by hazard score, highest first
        scored_patients.sort(key=lambda x: x["hazard_score"], reverse=True)

        # Add queue position
        for i, p in enumerate(scored_patients):
            p["queue_position"] = i + 1

        return scored_patients

    def get_patients_needing_reassessment(self) -> List[Dict]:
        """
        Identify patients whose wait time has exceeded safe thresholds.
        These patients need vitals rechecked and potentially re-triaged.
        """
        needs_reassessment = []
        for patient_id, entry in self.queue.items():
            if entry["status"] != "waiting":
                continue

            hazard = self.compute_hazard_score(patient_id)

            should_reassess = False
            reason = ""

            if hazard["wait_exceeded"]:
                should_reassess = True
                reason = (f"Wait time ({hazard['wait_minutes']:.0f} min) exceeds "
                         f"safe threshold ({hazard['safe_wait_threshold']} min) "
                         f"for Level {entry['triage_level']}")

            if entry["velocity_risk"] in ["high", "critical"]:
                should_reassess = True
                reason = f"Active deterioration trend detected (velocity risk: {entry['velocity_risk']})"

            if should_reassess:
                needs_reassessment.append({
                    "patient_id": patient_id,
                    "reason": reason,
                    "hazard_score": hazard["score"],
                    "wait_minutes": hazard["wait_minutes"],
                    "triage_level": entry["triage_level"],
                    "age_group": entry["age_group"],
                })

        return sorted(needs_reassessment, key=lambda x: x["hazard_score"], reverse=True)

    def update_patient(self, patient_id: str, **kwargs):
        """Update a patient's queue entry (e.g., after reassessment)."""
        if patient_id in self.queue:
            self.queue[patient_id].update(kwargs)
            self.queue[patient_id]["last_updated"] = datetime.now()

    def get_queue_stats(self) -> Dict:
        """Get queue statistics."""
        waiting = [e for e in self.queue.values() if e["status"] == "waiting"]
        if not waiting:
            return {"count": 0}

        levels = [e["triage_level"] for e in waiting]
        now = datetime.now()
        waits = [(now - e["arrival_time"]).total_seconds() / 60.0 for e in waiting]

        return {
            "count": len(waiting),
            "by_level": {i: levels.count(i) for i in range(1, 6)},
            "avg_wait_minutes": round(np.mean(waits), 1),
            "max_wait_minutes": round(max(waits), 1),
            "patients_past_threshold": len(self.get_patients_needing_reassessment()),
        }
