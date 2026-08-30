"""
PulseGuard — Deterioration Velocity Model (Differentiator A)
Models the rate of change on repeated vitals so a patient trending worse
gets caught before crossing an absolute threshold.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class DeteriorationVelocityModel:
    """
    Analyzes time-series vital signs to detect deterioration trends.
    Computes velocity (rate of change) and acceleration for each vital.
    Triggers alerts when velocity exceeds safe thresholds even if
    absolute values are still within normal range.
    """

    # Velocity thresholds: max acceptable change per hour
    # Positive = worsening direction for each vital
    VELOCITY_THRESHOLDS = {
        "hr": {"warning": 10, "critical": 20, "direction": 1},     # Rising HR is bad
        "rr": {"warning": 4, "critical": 8, "direction": 1},       # Rising RR is bad
        "spo2": {"warning": -2, "critical": -4, "direction": -1},   # Falling SpO2 is bad
        "sbp": {"warning": -15, "critical": -30, "direction": -1},  # Falling SBP is bad
        "temp": {"warning": 0.5, "critical": 1.0, "direction": 1},  # Rising temp is bad
        "pain": {"warning": 2, "critical": 4, "direction": 1},      # Rising pain is bad
    }

    # Absolute critical thresholds (any single reading)
    ABSOLUTE_CRITICAL = {
        "hr": {"low": 40, "high": 150},
        "rr": {"low": 8, "high": 30},
        "spo2": {"low": 90, "high": 101},
        "sbp": {"low": 80, "high": 200},
        "temp": {"low": 35.0, "high": 39.5},
    }

    def __init__(self):
        self.patient_histories = {}  # patient_id -> list of (timestamp, vitals_dict)

    def add_reading(self, patient_id: str, timestamp: datetime, vitals: Dict[str, float]):
        """Add a vital signs reading to a patient's history."""
        if patient_id not in self.patient_histories:
            self.patient_histories[patient_id] = []
        self.patient_histories[patient_id].append((timestamp, vitals))
        # Sort by timestamp
        self.patient_histories[patient_id].sort(key=lambda x: x[0])

    def load_history(self, patient_id: str, history: List[Tuple[datetime, Dict]]):
        """Load a complete history for a patient."""
        self.patient_histories[patient_id] = sorted(history, key=lambda x: x[0])

    def compute_velocity(self, patient_id: str) -> Dict:
        """
        Compute velocity (rate of change per hour) for each vital.
        Returns detailed analysis including velocity, acceleration,
        and risk assessment.
        """
        history = self.patient_histories.get(patient_id, [])

        if len(history) < 2:
            return {
                "has_trend_data": False,
                "vitals_velocity": {},
                "overall_risk": "insufficient_data",
                "alert": None,
                "details": "Need at least 2 readings for velocity calculation."
            }

        velocities = {}
        alerts = []
        risk_score = 0

        # Get the last two readings for velocity
        t1, v1 = history[-2]
        t2, v2 = history[-1]
        hours_elapsed = max((t2 - t1).total_seconds() / 3600, 0.01)

        # Also compute over full window if >2 readings
        t_first, v_first = history[0]
        full_hours = max((t2 - t_first).total_seconds() / 3600, 0.01)

        for vital_name, thresholds in self.VELOCITY_THRESHOLDS.items():
            if vital_name not in v1 or vital_name not in v2:
                continue

            val1 = v1[vital_name]
            val2 = v2[vital_name]
            current_val = val2

            # Short-term velocity (last interval)
            velocity = (val2 - val1) / hours_elapsed

            # Long-term velocity (full window)
            if vital_name in v_first:
                long_velocity = (val2 - v_first[vital_name]) / full_hours
            else:
                long_velocity = velocity

            # Acceleration (if >2 readings)
            acceleration = 0
            if len(history) >= 3:
                t0, v0 = history[-3]
                if vital_name in v0:
                    hours_01 = max((t1 - t0).total_seconds() / 3600, 0.01)
                    prev_velocity = (val1 - v0[vital_name]) / hours_01
                    hours_12 = max((t2 - t1).total_seconds() / 3600, 0.01)
                    acceleration = (velocity - prev_velocity) / hours_12

            # Determine direction (worsening or improving)
            direction = thresholds["direction"]
            worsening_velocity = velocity * direction  # positive = worsening

            # Risk assessment
            vital_risk = "stable"
            if abs(worsening_velocity) >= abs(thresholds["critical"]):
                vital_risk = "critical"
                risk_score += 3
                alerts.append(f"CRITICAL VELOCITY: {vital_name} changing at "
                             f"{velocity:+.1f}/hr (threshold: {thresholds['critical']})")
            elif abs(worsening_velocity) >= abs(thresholds["warning"]):
                vital_risk = "warning"
                risk_score += 1
                alerts.append(f"WARNING: {vital_name} trending "
                             f"{'up' if velocity > 0 else 'down'} at {velocity:+.1f}/hr")

            # Check if acceleration is increasing (getting worse faster)
            accelerating = acceleration * direction > 0
            if accelerating and vital_risk != "stable":
                risk_score += 1
                alerts.append(f"ACCELERATING: {vital_name} deterioration is speeding up")

            velocities[vital_name] = {
                "current_value": current_val,
                "velocity_per_hour": round(velocity, 2),
                "long_term_velocity": round(long_velocity, 2),
                "acceleration": round(acceleration, 2),
                "is_worsening": worsening_velocity > 0,
                "is_accelerating": accelerating,
                "risk_level": vital_risk,
                "readings_count": len(history),
            }

        # Overall risk assessment
        if risk_score >= 5:
            overall_risk = "critical"
        elif risk_score >= 2:
            overall_risk = "high"
        elif risk_score >= 1:
            overall_risk = "moderate"
        else:
            overall_risk = "low"

        return {
            "has_trend_data": True,
            "vitals_velocity": velocities,
            "overall_risk": overall_risk,
            "risk_score": risk_score,
            "alerts": alerts,
            "readings_count": len(history),
            "time_window_hours": round(full_hours, 2),
            "should_trigger_reassessment": risk_score >= 2,
        }

    def should_escalate(self, patient_id: str) -> Tuple[bool, int, str]:
        """
        Determine if velocity data warrants escalation.
        Returns (should_escalate, levels_to_escalate, reason).
        """
        analysis = self.compute_velocity(patient_id)

        if not analysis["has_trend_data"]:
            return False, 0, "No trend data available"

        risk = analysis["overall_risk"]
        alerts = analysis.get("alerts", [])

        if risk == "critical":
            return True, 2, f"Critical deterioration velocity detected: {'; '.join(alerts[:2])}"
        elif risk == "high":
            return True, 1, f"Significant deterioration trend: {'; '.join(alerts[:2])}"
        elif risk == "moderate":
            return False, 0, f"Moderate trend detected (monitoring): {'; '.join(alerts[:1])}"
        else:
            return False, 0, "Vitals trending stable"

    def predict_time_to_critical(self, patient_id: str, vital_name: str) -> Optional[Dict]:
        """
        Predict when a vital will cross a critical threshold based on current velocity.
        Returns estimated time to critical (hours) or None if stable/improving.
        """
        analysis = self.compute_velocity(patient_id)
        if not analysis["has_trend_data"] or vital_name not in analysis["vitals_velocity"]:
            return None

        vital_data = analysis["vitals_velocity"][vital_name]
        current = vital_data["current_value"]
        velocity = vital_data["velocity_per_hour"]

        if vital_name not in self.ABSOLUTE_CRITICAL or velocity == 0:
            return None

        thresholds = self.ABSOLUTE_CRITICAL[vital_name]

        # Check which threshold we're approaching
        if velocity > 0:
            target = thresholds["high"]
            if current < target:
                hours = (target - current) / velocity
                return {"hours_to_critical": round(hours, 1),
                        "threshold": target, "direction": "rising"}
        else:
            target = thresholds["low"]
            if current > target:
                hours = (current - target) / abs(velocity)
                return {"hours_to_critical": round(hours, 1),
                        "threshold": target, "direction": "falling"}

        return None
