"""
PatientTriage.ai — Demo Cohort Builder
======================================

Builds the patient board the prototype demonstrates on, from two sources that
do different jobs:

**Real held-out ED visits (the majority).** Drawn from the NHAMCS test fold —
hospitals the model never trained on. These are actual patients with actual
nurse-assigned triage levels and actual recorded outcomes, so every number on
the demo board is a real comparison rather than a rehearsal against data we
invented. Sampling is stratified so the board contains critical patients at a
visible rate rather than the 1.8% a random draw would give.

**Curated clinical edge cases (a small, clearly-labelled set).** Real survey
data records vital signs and complaint codes but not the bedside observations
a triage nurse makes — whether a patient is unresponsive, whether their skin
is mottled, whether bleeding is controlled. Those observations drive several
of the safety rules, so a handful of synthetic cases exist to exercise them.
They are labelled SYNTHETIC everywhere they appear, and they are never counted
in any accuracy figure. Their only job is to demonstrate that a rule fires.

Keeping the two visibly separate is the point. A demo that quietly mixes
invented patients into a real cohort is how a prototype ends up reporting
numbers nobody can reproduce.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.input_schema import (
    DataSource, Measurement, MedicalHistory, PatientEncounter,
    SelfReportedSymptoms, StaffObservedCues, VitalSigns,
)

REAL_SOURCE_TAG = "NHAMCS (real, held-out hospital)"
SYNTHETIC_SOURCE_TAG = "SYNTHETIC: curated edge case"


def _m(value, unit: str = "", source: DataSource = DataSource.DEVICE_MEASURED,
       when: Optional[datetime] = None) -> Optional[Measurement]:
    """Wrap a value, preserving missingness as None rather than inventing one."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return Measurement(value=float(value) if isinstance(value, (int, float, np.number))
                       else value,
                       unit=unit, source=source,
                       timestamp=when or datetime.now())


def nhamcs_row_to_encounter(row: pd.Series, arrival_offset_minutes: float = 0.0
                            ) -> PatientEncounter:
    """
    Convert one real NHAMCS visit into a PatientEncounter.

    Bedside observations (AVPU, skin, bleeding) are left as None, not guessed.
    The survey never recorded them, and fabricating a plausible-looking value
    would let rules fire on evidence that does not exist — the fastest way to
    make a demo look better than the system is.
    """
    now = datetime.now()
    arrival = now - timedelta(minutes=arrival_offset_minutes)

    vitals = VitalSigns(
        temperature=_m(row.get("temperature"), "°C", when=arrival),
        heart_rate=_m(row.get("heart_rate"), "bpm", when=arrival),
        respiratory_rate=_m(row.get("respiratory_rate"), "breaths/min", when=arrival),
        spo2=_m(row.get("spo2"), "%", when=arrival),
        systolic_bp=_m(row.get("systolic_bp"), "mmHg", when=arrival),
        diastolic_bp=_m(row.get("diastolic_bp"), "mmHg", when=arrival),
        pain_score=_m(row.get("pain_score"), "/10",
                      DataSource.PATIENT_REPORTED, when=arrival),
    )

    complaint = str(row.get("chief_complaint") or "").strip()
    symptoms = str(row.get("symptoms_text") or "").strip()

    symptom_obj = SelfReportedSymptoms(
        chief_complaint=_m(complaint or None, source=DataSource.PATIENT_REPORTED,
                           when=arrival),
        symptoms=_m(symptoms or None, source=DataSource.PATIENT_REPORTED, when=arrival),
        severity_self_assessed=_m(row.get("pain_score"), "/10",
                                  DataSource.PATIENT_REPORTED, when=arrival),
    )

    history_available = bool(row.get("history_available", 0))
    conditions = _condition_text(row)
    history = MedicalHistory(
        history_available=history_available,
        known_conditions=_m(conditions or None, source=DataSource.EHR_IMPORTED,
                            when=arrival) if history_available else None,
    )

    # Carry the structured context the model was trained on. Dropping it here
    # would score the patient with 25 fewer features than the evaluation used.
    context = {
        key: (None if pd.isna(row.get(key)) else float(row.get(key)))
        for key in ["arrival_by_ambulance", "seen_last_72h", "injury_related",
                    "nursing_home_resident", "n_chronic_conditions",
                    "has_high_risk_conditions"]
        if key in row
    }
    for key in row.index:
        if key.startswith("cond_"):
            value = row.get(key)
            context[key] = None if pd.isna(value) else float(value)

    encounter = PatientEncounter(
        patient_id=str(row.get("record_id", "REAL")),
        age=int(round(float(row.get("age", 40)))),
        sex=str(row.get("sex") or "F"),
        arrival_time=arrival,
        vitals=vitals,
        symptoms=symptom_obj,
        history=history,
        staff_cues=StaffObservedCues(),   # deliberately empty — see docstring
        context=context,
    )
    return encounter


CONDITION_LABELS = {
    "cond_asthma": "Asthma", "cond_cancer": "Cancer",
    "cond_cebvd": "Cerebrovascular disease / prior stroke",
    "cond_ckd": "Chronic kidney disease", "cond_copd": "COPD",
    "cond_chf": "Congestive heart failure",
    "cond_cad": "Coronary artery disease", "cond_deprn": "Depression",
    "cond_diabtyp1": "Diabetes type 1", "cond_diabtyp2": "Diabetes type 2",
    "cond_esrd": "End-stage renal disease", "cond_hpe": "Prior PE/DVT",
    "cond_edhiv": "HIV", "cond_htn": "Hypertension",
    "cond_hyplipid": "Hyperlipidaemia", "cond_obesity": "Obesity",
    "cond_alzhd": "Dementia", "cond_substab": "Substance use disorder",
    "cond_etohab": "Alcohol use disorder", "cond_osa": "Obstructive sleep apnoea",
    "cond_ostprsis": "Osteoporosis", "cond_diabtyp0": "Diabetes",
}


def _condition_text(row: pd.Series) -> str:
    present = [label for col, label in CONDITION_LABELS.items()
               if float(row.get(col, 0) or 0) == 1.0]
    return ", ".join(present)


# ─── Building the board ──────────────────────────────────────────────────────

def build_demo_board(df_test: pd.DataFrame, n: int = 25,
                     seed: int = 7) -> List[Tuple[PatientEncounter, int, str]]:
    """
    Sample a stratified board of real held-out patients.

    Stratification matters for a demo: a random 25 from a real ED yields at
    most one critical patient, and a triage system demonstrated on nobody who
    is sick demonstrates nothing. The sample deliberately over-represents
    Levels 1–2 relative to the true 18%, and every reported metric is computed
    on the full unstratified test fold instead, never on this board.
    """
    rng = np.random.RandomState(seed)
    quotas = {1: 3, 2: 6, 3: 8, 4: 6, 5: 2}

    picked = []
    for level, quota in quotas.items():
        pool = df_test[df_test["triage_level"] == level]
        # Prefer records with enough recorded vitals to make a legible demo
        pool = pool.assign(_completeness=pool[
            ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "temperature"]
        ].notna().sum(axis=1))
        pool = pool[pool["_completeness"] >= 4]
        if len(pool) == 0:
            continue
        take = min(quota, len(pool))
        picked.append(pool.sample(take, random_state=rng.randint(1 << 30)))

    board_df = pd.concat(picked).sample(frac=1.0, random_state=seed)
    board_df = board_df.head(n)

    encounters = []
    for i, (_, row) in enumerate(board_df.iterrows()):
        # Spread arrivals across the last two hours so the waiting queue and
        # its time-decay hazard scoring have something real to order.
        offset = float(rng.randint(0, 125))
        enc = nhamcs_row_to_encounter(row, arrival_offset_minutes=offset)
        enc.patient_id = f"ED-{i + 1:03d}"

        outcome = _describe_outcome(row)
        desc = (f"{REAL_SOURCE_TAG}. "
                f"{int(row['age'])}y {row.get('sex', '?')}. "
                f"{row.get('chief_complaint') or 'Complaint not coded'}. "
                f"Outcome: {outcome}.")
        encounters.append((enc, int(row["triage_level"]), desc))

    return encounters


def _describe_outcome(row: pd.Series) -> str:
    """What actually happened to this patient — the independent ground truth."""
    if float(row.get("out_critical_care_unit", 0) or 0) == 1:
        return "admitted to critical care"
    if float(row.get("out_dieded", 0) or 0) == 1:
        return "died in the ED"
    if float(row.get("outcome_admitted", 0) or 0) == 1:
        return "admitted to hospital"
    if float(row.get("out_lwbs", 0) or 0) == 1:
        return "left without being seen"
    return "discharged from the ED"


# ─── Curated edge cases ──────────────────────────────────────────────────────

def build_edge_cases() -> List[Tuple[PatientEncounter, int, str]]:
    """
    Synthetic cases that exercise bedside-observation safety rules.

    Each one exists to fire a specific rule that real survey data cannot
    trigger, because NHAMCS never recorded whether a patient was unresponsive
    or whether their skin was mottled. Labelled SYNTHETIC, excluded from every
    accuracy metric.
    """
    now = datetime.now()

    def cue(value, source=DataSource.NURSE_OBSERVED):
        return Measurement(value=value, source=source, timestamp=now)

    cases = []

    # ── Unresponsive patient — fires UNRESPONSIVE (observed → Level 1) ──
    cases.append((
        PatientEncounter(
            patient_id="EDGE-001", age=58, sex="M", arrival_time=now - timedelta(minutes=2),
            vitals=VitalSigns(
                temperature=_m(35.8, "°C"), heart_rate=_m(38, "bpm"),
                respiratory_rate=_m(6, "breaths/min"), spo2=_m(78, "%"),
                systolic_bp=_m(72, "mmHg"), diastolic_bp=_m(40, "mmHg"),
            ),
            symptoms=SelfReportedSymptoms(
                chief_complaint=_m("Found unresponsive at home",
                                   source=DataSource.NURSE_OBSERVED),
            ),
            history=MedicalHistory(history_available=False),
            staff_cues=StaffObservedCues(
                consciousness=cue("unresponsive"), breathing_difficulty=cue("severe"),
                mobility=cue("immobile"), skin_appearance=cue("cyanotic"),
                visible_distress=cue("severe"),
            ),
        ), 1,
        f"{SYNTHETIC_SOURCE_TAG}. Exercises UNRESPONSIVE + CRITICAL_VITAL_SIGN. "
        f"No history on file, so the zero-history escalation also applies."))

    # ── Paediatric compensated shock — normal BP is the trap ──
    cases.append((
        PatientEncounter(
            patient_id="EDGE-002", age=4, sex="F", arrival_time=now - timedelta(minutes=8),
            vitals=VitalSigns(
                temperature=_m(38.9, "°C"), heart_rate=_m(168, "bpm"),
                respiratory_rate=_m(38, "breaths/min"), spo2=_m(95, "%"),
                systolic_bp=_m(88, "mmHg"), diastolic_bp=_m(52, "mmHg"),
                pain_score=_m(4, "/10", DataSource.PATIENT_REPORTED),
            ),
            symptoms=SelfReportedSymptoms(
                chief_complaint=_m("Fever, not drinking, very sleepy",
                                   source=DataSource.PATIENT_REPORTED),
                symptoms=_m("Parents report reduced wet nappies since yesterday",
                            source=DataSource.PATIENT_REPORTED),
            ),
            history=MedicalHistory(history_available=True),
            staff_cues=StaffObservedCues(
                consciousness=cue("alert"), skin_appearance=cue("mottled"),
                visible_distress=cue("moderate"), mobility=cue("assisted"),
            ),
        ), 2,
        f"{SYNTHETIC_SOURCE_TAG}. Exercises PEDIATRIC_COMPENSATED_SHOCK. Blood "
        f"pressure is still normal for age, and in a child that is the last thing "
        f"to fail, so an adult-calibrated reading would call this stable."))

    # ── Geriatric fall on an anticoagulant — looks well, is not ──
    cases.append((
        PatientEncounter(
            patient_id="EDGE-003", age=82, sex="F", arrival_time=now - timedelta(minutes=25),
            vitals=VitalSigns(
                temperature=_m(36.6, "°C"), heart_rate=_m(78, "bpm"),
                respiratory_rate=_m(16, "breaths/min"), spo2=_m(97, "%"),
                systolic_bp=_m(142, "mmHg"), diastolic_bp=_m(78, "mmHg"),
                pain_score=_m(3, "/10", DataSource.PATIENT_REPORTED),
            ),
            symptoms=SelfReportedSymptoms(
                chief_complaint=_m("Fell at home and hit her head, small forehead cut",
                                   source=DataSource.PATIENT_REPORTED),
                symptoms=_m("Did not lose consciousness. Slight headache.",
                            source=DataSource.PATIENT_REPORTED),
            ),
            history=MedicalHistory(
                history_available=True,
                known_conditions=_m("Atrial fibrillation, hypertension",
                                    source=DataSource.EHR_IMPORTED),
                medications=_m("Apixaban, bisoprolol", source=DataSource.EHR_IMPORTED),
            ),
            staff_cues=StaffObservedCues(
                consciousness=cue("alert"), visible_distress=cue("mild"),
                mobility=cue("ambulatory"), skin_appearance=cue("normal"),
            ),
        ), 2,
        f"{SYNTHETIC_SOURCE_TAG}. Exercises GERIATRIC_ANTICOAGULATED_FALL. Every "
        f"vital sign is normal and she walked in. An intracranial bleed on "
        f"apixaban can look exactly like this for hours."))

    # ── Uncontrolled bleeding ──
    cases.append((
        PatientEncounter(
            patient_id="EDGE-004", age=29, sex="M", arrival_time=now - timedelta(minutes=4),
            vitals=VitalSigns(
                temperature=_m(36.4, "°C"), heart_rate=_m(118, "bpm"),
                respiratory_rate=_m(22, "breaths/min"), spo2=_m(97, "%"),
                systolic_bp=_m(108, "mmHg"), diastolic_bp=_m(62, "mmHg"),
                pain_score=_m(8, "/10", DataSource.PATIENT_REPORTED),
            ),
            symptoms=SelfReportedSymptoms(
                chief_complaint=_m("Deep forearm laceration from broken glass",
                                   source=DataSource.PATIENT_REPORTED),
            ),
            history=MedicalHistory(history_available=False),
            staff_cues=StaffObservedCues(
                consciousness=cue("alert"), bleeding=cue("uncontrolled"),
                visible_distress=cue("moderate"), skin_appearance=cue("pale"),
            ),
        ), 2,
        f"{SYNTHETIC_SOURCE_TAG}. Exercises UNCONTROLLED_BLEEDING. Blood pressure "
        f"is still held up. In haemorrhage, it falls last."))

    # ── Zero-history high-risk complaint ──
    cases.append((
        PatientEncounter(
            patient_id="EDGE-005", age=44, sex="F", arrival_time=now - timedelta(minutes=12),
            vitals=VitalSigns(
                temperature=_m(36.9, "°C"), heart_rate=_m(96, "bpm"),
                respiratory_rate=_m(20, "breaths/min"), spo2=_m(96, "%"),
                systolic_bp=_m(138, "mmHg"), diastolic_bp=_m(84, "mmHg"),
                pain_score=_m(9, "/10", DataSource.PATIENT_REPORTED),
            ),
            symptoms=SelfReportedSymptoms(
                chief_complaint=_m("Worst headache of my life, came on suddenly",
                                   source=DataSource.PATIENT_REPORTED),
                symptoms=_m("Started 40 minutes ago while sitting. Neck feels stiff.",
                            source=DataSource.PATIENT_REPORTED),
            ),
            history=MedicalHistory(history_available=False),
            staff_cues=StaffObservedCues(
                consciousness=cue("alert"), visible_distress=cue("severe"),
                mobility=cue("ambulatory"),
            ),
        ), 1,
        f"{SYNTHETIC_SOURCE_TAG}. Exercises ZERO_HISTORY_HIGH_RISK_COMPLAINT. "
        f"First-time patient, nothing on file, and a thunderclap headache. The "
        f"vitals are unremarkable and the presentation is not."))

    # ── Sepsis hiding behind an ordinary complaint ──
    cases.append((
        PatientEncounter(
            patient_id="EDGE-006", age=71, sex="M", arrival_time=now - timedelta(minutes=35),
            vitals=VitalSigns(
                temperature=_m(38.4, "°C"), heart_rate=_m(104, "bpm"),
                respiratory_rate=_m(24, "breaths/min"), spo2=_m(94, "%"),
                systolic_bp=_m(104, "mmHg"), diastolic_bp=_m(60, "mmHg"),
                pain_score=_m(4, "/10", DataSource.PATIENT_REPORTED),
            ),
            symptoms=SelfReportedSymptoms(
                chief_complaint=_m("Burning urination and feeling generally unwell",
                                   source=DataSource.PATIENT_REPORTED),
                symptoms=_m("Off his food for two days, a bit confused this morning "
                            "according to his daughter",
                            source=DataSource.PATIENT_REPORTED),
            ),
            history=MedicalHistory(
                history_available=True,
                known_conditions=_m("Type 2 diabetes, chronic kidney disease",
                                    source=DataSource.EHR_IMPORTED),
            ),
            staff_cues=StaffObservedCues(
                consciousness=cue("alert"), visible_distress=cue("mild"),
                mobility=cue("assisted"), skin_appearance=cue("flushed"),
            ),
        ), 2,
        f"{SYNTHETIC_SOURCE_TAG}. Exercises SEPSIS_PHYSIOLOGY. A urinary "
        f"complaint that would ordinarily be Level 4, with three SIRS criteria "
        f"and a diabetic, renally-impaired host."))

    return cases


def load_demo_cohort(n_real: int = 25, include_edge_cases: bool = True,
                     years: Tuple[int, ...] = (2021, 2022),
                     seed: int = 7) -> List[Tuple[PatientEncounter, int, str]]:
    """
    Assemble the full demo board: real held-out patients + edge cases.

    Uses the same hospital-grouped split as training, so the demo board is
    drawn only from departments the model has never seen.
    """
    from data.real.nhamcs_loader import load_clean
    from models.triage_model import grouped_split

    df = load_clean(years=years, verbose=False).reset_index(drop=True)
    idx = grouped_split(df)
    df_test = df.loc[idx["test"]]

    board = build_demo_board(df_test, n=n_real, seed=seed)
    if include_edge_cases:
        board += build_edge_cases()
    return board


def load_test_fold(years: Tuple[int, ...] = (2021, 2022)) -> pd.DataFrame:
    """The full held-out test fold — what every reported metric is computed on."""
    from data.real.nhamcs_loader import load_clean
    from models.triage_model import grouped_split

    df = load_clean(years=years, verbose=False).reset_index(drop=True)
    idx = grouped_split(df)
    return df.loc[idx["test"]].reset_index(drop=True)
