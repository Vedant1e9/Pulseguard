"""Shared test helpers for constructing PatientEncounter fixtures."""
from data.input_schema import (
    PatientEncounter, VitalSigns, SelfReportedSymptoms, MedicalHistory,
    StaffObservedCues, Measurement,
)


def m(value):
    """Wrap a raw value in a Measurement, or return None if value is None/empty."""
    if value is None or value == "":
        return None
    return Measurement(value=value)


def make_encounter(
    patient_id="TEST-001", age=40, sex="M",
    temp=37.0, hr=80, rr=16, spo2=98, sbp=120, dbp=80, pain=2,
    chief_complaint="", symptoms_text="", progression=None,
    history_available=True, conditions="", medications="",
    consciousness="alert", bleeding="none", skin="normal",
    breathing_difficulty="none", visible_distress="none", mobility="ambulatory",
):
    """Build a fully-populated PatientEncounter with sane defaults, so each
    test only has to specify the fields relevant to the rule it's exercising.
    """
    return PatientEncounter(
        patient_id=patient_id,
        age=age,
        sex=sex,
        vitals=VitalSigns(
            temperature=m(temp), heart_rate=m(hr), respiratory_rate=m(rr),
            spo2=m(spo2), systolic_bp=m(sbp), diastolic_bp=m(dbp), pain_score=m(pain),
        ),
        symptoms=SelfReportedSymptoms(
            chief_complaint=m(chief_complaint), symptoms=m(symptoms_text),
            progression=m(progression),
        ),
        history=MedicalHistory(
            history_available=history_available,
            known_conditions=m(conditions), medications=m(medications),
        ),
        staff_cues=StaffObservedCues(
            consciousness=m(consciousness), bleeding=m(bleeding), skin_appearance=m(skin),
            breathing_difficulty=m(breathing_difficulty), visible_distress=m(visible_distress),
            mobility=m(mobility),
        ),
    )
