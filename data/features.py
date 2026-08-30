"""
PatientTriage.ai — Clinical Feature Engineering
===============================================

Turns a patient record into the feature vector the models consume.

Three ideas do most of the work here:

1. **Age-normalised physiology.** A heart rate of 150 is an emergency in a
   70-year-old and unremarkable in a 2-year-old. Rather than hoping a tree
   model rediscovers that interaction from age × heart_rate splits, every
   vital is additionally expressed as a z-score against the published normal
   range *for that patient's age band*. This is the single most important
   feature-level answer to the brief's warning that an adult-calibrated model
   carries silent risk for children and the elderly.

2. **Established clinical scores as features.** NEWS2 and PEWS encode decades
   of validation work. Handing the model the score directly means its capacity
   goes to the residual signal instead of re-deriving known medicine.

3. **Missingness as signal, never as zero.** Every vital carries a companion
   `*_missing` indicator and the value itself stays NaN. In an ED, *which*
   vitals went unrecorded is itself informative — a patient too agitated for a
   blood pressure cuff is not a patient with a blood pressure of zero.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models.clinical_scores import compute_news2, compute_pews

# ─── Age-banded reference physiology ─────────────────────────────────────────
# (low, high) of the published normal range per age band. The midpoint is
# treated as the reference mean and (high - low) / 4 as the reference SD, so a
# value at the edge of normal sits at roughly ±2 z.
# Paediatric bands follow PALS reference ranges; adult and geriatric bands
# follow standard adult physiology, with the geriatric systolic band shifted
# upward to reflect age-related arterial stiffening.

VITAL_REFERENCE = {
    "heart_rate": [
        (0, 1, 100, 160), (1, 3, 90, 150), (3, 6, 80, 140),
        (6, 12, 70, 120), (12, 18, 60, 100), (18, 65, 60, 100),
        (65, 200, 60, 95),
    ],
    "respiratory_rate": [
        (0, 1, 30, 60), (1, 3, 24, 40), (3, 6, 22, 34),
        (6, 12, 18, 30), (12, 18, 12, 20), (18, 65, 12, 20),
        (65, 200, 12, 22),
    ],
    "systolic_bp": [
        (0, 1, 70, 100), (1, 3, 72, 104), (3, 6, 75, 110),
        (6, 12, 80, 120), (12, 18, 90, 130), (18, 65, 100, 140),
        (65, 200, 110, 150),
    ],
    "diastolic_bp": [
        (0, 1, 40, 60), (1, 3, 42, 64), (3, 6, 45, 70),
        (6, 12, 50, 78), (12, 18, 55, 84), (18, 65, 60, 90),
        (65, 200, 60, 90),
    ],
    "spo2": [(0, 200, 95, 100)],
    "temperature": [(0, 200, 36.5, 37.5)],
}

VITALS = ["temperature", "heart_rate", "respiratory_rate", "spo2",
          "systolic_bp", "diastolic_bp", "pain_score"]

CHRONIC_FLAGS = [
    "cond_etohab", "cond_alzhd", "cond_asthma", "cond_cancer", "cond_cebvd",
    "cond_ckd", "cond_copd", "cond_chf", "cond_cad", "cond_deprn",
    "cond_diabtyp1", "cond_diabtyp2", "cond_diabtyp0", "cond_esrd",
    "cond_hpe", "cond_edhiv", "cond_hyplipid", "cond_htn", "cond_obesity",
    "cond_osa", "cond_ostprsis", "cond_substab",
]


def _reference_range(vital: str, age: float):
    for lo, hi, ref_lo, ref_hi in VITAL_REFERENCE.get(vital, []):
        if lo <= age < hi:
            return ref_lo, ref_hi
    bands = VITAL_REFERENCE.get(vital)
    return (bands[-1][2], bands[-1][3]) if bands else (None, None)


def age_normalised_z(vital: str, value, age: float):
    """
    Express a vital as a z-score against the normal range for this age.

    Returns NaN for a missing value rather than 0 — a missing measurement is
    not a normal measurement, and collapsing the two is precisely how an
    unmeasured saturation becomes an invisible respiratory failure.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    ref_lo, ref_hi = _reference_range(vital, age)
    if ref_lo is None:
        return np.nan
    mean = (ref_lo + ref_hi) / 2.0
    sd = max((ref_hi - ref_lo) / 4.0, 1e-6)
    return (float(value) - mean) / sd


def is_abnormal_for_age(vital: str, value, age: float) -> bool:
    """True when a vital falls outside the normal range for this patient's age."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    ref_lo, ref_hi = _reference_range(vital, age)
    if ref_lo is None:
        return False
    return not (ref_lo <= float(value) <= ref_hi)


def age_group_of(age: float) -> str:
    if age < 18:
        return "pediatric"
    if age >= 65:
        return "geriatric"
    return "adult"


# ─── Row-level feature construction ──────────────────────────────────────────

def build_features_row(record: Dict) -> Dict[str, float]:
    """
    Build the full feature dict for one patient record.

    `record` needs: age, sex, the seven vitals (None/NaN where unmeasured),
    and optionally history/condition flags and staff-observed consciousness.
    Anything absent degrades to NaN rather than to a fabricated value.
    """
    age = float(record.get("age") or 0.0)
    f: Dict[str, float] = {}

    # ── Demographics & context ──
    f["age"] = age
    f["age_group_ordinal"] = {"pediatric": 0, "adult": 1, "geriatric": 2}[age_group_of(age)]
    f["is_pediatric"] = 1.0 if age < 18 else 0.0
    f["is_geriatric"] = 1.0 if age >= 65 else 0.0
    f["is_infant"] = 1.0 if age < 1 else 0.0
    sex = record.get("sex")
    f["sex_male"] = 1.0 if sex == "M" else (0.0 if sex == "F" else np.nan)

    for key in ["arrival_by_ambulance", "seen_last_72h", "injury_related",
                "nursing_home_resident", "history_available",
                "has_high_risk_conditions"]:
        val = record.get(key)
        f[key] = np.nan if val is None else float(val)

    f["n_chronic_conditions"] = float(record.get("n_chronic_conditions") or 0.0)
    for flag in CHRONIC_FLAGS:
        f[flag] = float(record.get(flag) or 0.0)

    # ── Raw vitals (NaN preserved) ──
    vitals: Dict[str, Optional[float]] = {}
    for v in VITALS:
        raw = record.get(v)
        val = np.nan if raw is None else float(raw)
        if isinstance(val, float) and np.isnan(val):
            val = np.nan
        vitals[v] = val
        f[v] = val
        f[f"{v}_missing"] = 1.0 if (val is None or np.isnan(val)) else 0.0

    f["n_vitals_missing"] = float(sum(f[f"{v}_missing"] for v in VITALS))

    # ── Age-normalised physiology ──
    for v in ["heart_rate", "respiratory_rate", "systolic_bp",
              "diastolic_bp", "spo2", "temperature"]:
        f[f"{v}_z_for_age"] = age_normalised_z(v, vitals[v], age)

    f["n_vitals_abnormal_for_age"] = float(sum(
        is_abnormal_for_age(v, vitals[v], age)
        for v in ["heart_rate", "respiratory_rate", "systolic_bp", "spo2", "temperature"]
    ))

    # ── Haemodynamic derivations ──
    hr, sbp, dbp = vitals["heart_rate"], vitals["systolic_bp"], vitals["diastolic_bp"]

    # Shock index (HR/SBP) rises before either vital alone looks alarming —
    # the classic signature of compensated shock, and the reason a patient can
    # look "stable on paper" while decompensating.
    f["shock_index"] = (hr / sbp) if _ok(hr) and _ok(sbp) and sbp > 0 else np.nan

    # Age-adjusted shock index matters because the healthy ratio itself moves
    # with age: a normal toddler sits near 1.4, a normal adult near 0.6.
    if _ok(f["shock_index"]):
        ref_hr_lo, ref_hr_hi = _reference_range("heart_rate", age)
        ref_sbp_lo, ref_sbp_hi = _reference_range("systolic_bp", age)
        expected = ((ref_hr_lo + ref_hr_hi) / 2) / ((ref_sbp_lo + ref_sbp_hi) / 2)
        f["shock_index_ratio_to_age_normal"] = f["shock_index"] / expected
    else:
        f["shock_index_ratio_to_age_normal"] = np.nan

    f["pulse_pressure"] = (sbp - dbp) if _ok(sbp) and _ok(dbp) else np.nan
    f["mean_arterial_pressure"] = (
        (sbp + 2 * dbp) / 3.0 if _ok(sbp) and _ok(dbp) else np.nan
    )

    # Hypoxia burden: distance below the 94% threshold, floored at zero.
    spo2 = vitals["spo2"]
    f["hypoxia_burden"] = max(0.0, 94.0 - spo2) if _ok(spo2) else np.nan

    temp = vitals["temperature"]
    f["fever_burden"] = max(0.0, temp - 38.0) if _ok(temp) else np.nan
    f["hypothermia_burden"] = max(0.0, 36.0 - temp) if _ok(temp) else np.nan

    # ── Validated early-warning scores ──
    consciousness = record.get("consciousness") or "alert"
    vitals_for_score = {k: (None if not _ok(v) else v) for k, v in vitals.items()}

    if age < 16:
        pews = compute_pews(vitals_for_score, age, consciousness,
                            record.get("skin_appearance"),
                            record.get("breathing_difficulty"))
        f["ews_score"] = float(pews["score"])
        f["ews_implied_level"] = float(pews["implied_triage_level"])
        f["ews_is_pediatric_scale"] = 1.0
        f["news2_score"] = np.nan
        f["pews_score"] = float(pews["score"])
    else:
        news = compute_news2(vitals_for_score, consciousness)
        f["ews_score"] = float(news["score"])
        f["ews_implied_level"] = float(news["implied_triage_level"])
        f["ews_is_pediatric_scale"] = 0.0
        f["news2_score"] = float(news["score"])
        f["news2_max_single"] = float(news["max_single_parameter"])
        f["pews_score"] = np.nan

    f.setdefault("news2_max_single", np.nan)

    # ── Pain ──
    pain = vitals["pain_score"]
    f["severe_pain"] = 1.0 if _ok(pain) and pain >= 7 else (0.0 if _ok(pain) else np.nan)

    return f


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def build_feature_frame(df: pd.DataFrame, show_progress: bool = False) -> pd.DataFrame:
    """Vectorised-enough feature build over a cohort DataFrame."""
    records = df.to_dict("records")
    rows = [build_features_row(r) for r in records]
    out = pd.DataFrame(rows, index=df.index)
    return out


def feature_names(df_features: pd.DataFrame) -> List[str]:
    return list(df_features.columns)


# ─── Human-readable descriptions, used by the explanation layer ──────────────

FEATURE_DESCRIPTIONS = {
    "age": "Patient age",
    "sex_male": "Sex",
    "temperature": "Body temperature",
    "heart_rate": "Heart rate",
    "respiratory_rate": "Respiratory rate",
    "spo2": "Oxygen saturation",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "pain_score": "Pain score",
    "heart_rate_z_for_age": "Heart rate relative to normal for age",
    "respiratory_rate_z_for_age": "Respiratory rate relative to normal for age",
    "systolic_bp_z_for_age": "Blood pressure relative to normal for age",
    "spo2_z_for_age": "Oxygen saturation relative to normal",
    "temperature_z_for_age": "Temperature relative to normal",
    "shock_index": "Shock index (heart rate ÷ systolic BP)",
    "shock_index_ratio_to_age_normal": "Shock index relative to normal for age",
    "pulse_pressure": "Pulse pressure",
    "mean_arterial_pressure": "Mean arterial pressure",
    "hypoxia_burden": "Degree of hypoxia below 94%",
    "fever_burden": "Fever above 38 °C",
    "hypothermia_burden": "Temperature below 36 °C",
    "ews_score": "Early warning score (NEWS2/PEWS)",
    "news2_score": "NEWS2 score",
    "pews_score": "Paediatric early warning score",
    "ews_implied_level": "Triage level implied by early warning score",
    "n_vitals_abnormal_for_age": "Number of vitals abnormal for age",
    "n_vitals_missing": "Number of vitals not recorded",
    "arrival_by_ambulance": "Arrived by ambulance",
    "seen_last_72h": "Seen in this ED within 72 hours",
    "injury_related": "Injury-related visit",
    "nursing_home_resident": "Nursing home resident",
    "history_available": "Medical history availability",
    "has_high_risk_conditions": "Known high-risk conditions",
    "n_chronic_conditions": "Number of chronic conditions",
    "severe_pain": "Severe pain (≥7/10)",
    "is_pediatric": "Paediatric patient",
    "is_geriatric": "Geriatric patient",
    "is_infant": "Infant (under 1 year)",
}


def describe_feature(name: str) -> str:
    if name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[name]
    if name.endswith("_missing"):
        base = name[: -len("_missing")]
        return f"{FEATURE_DESCRIPTIONS.get(base, base)} not recorded"
    if name.startswith("cond_"):
        return f"History of {name[5:].upper()}"
    return name.replace("_", " ").capitalize()
