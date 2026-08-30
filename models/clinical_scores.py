"""
PulseGuard — Standard Clinical Early-Warning Scores
=========================================================

Implements the published early-warning scores that emergency departments
already use, for two purposes:

1. **As engineered features** for the ML models — a NEWS2 score is a far
   stronger single predictor than any raw vital, and giving the model the
   clinically-validated aggregation for free means it spends its capacity
   on the residual signal instead of re-deriving 20 years of clinical
   research from a few thousand synthetic rows.

2. **As a benchmark baseline** — the honest question a clinical reviewer
   asks is not "is your model accurate?" but "is it better than the score
   my department already uses?". Every evaluation in this project reports
   PulseGuard against NEWS2/PEWS on the identical cohort.

References (implemented from published definitions):
  - NEWS2: Royal College of Physicians, National Early Warning Score 2 (2017)
  - PEWS:  Brighton Paediatric Early Warning Score (Monaghan, 2005) —
           implemented here in its vitals-derived approximation, since the
           behavioural sub-score requires bedside observation we encode
           separately as staff cues.

NOTE: these are decision-SUPPORT scores. They are reproduced here for
research/prototype benchmarking on synthetic data only.
"""

from typing import Dict, Optional


# ─── NEWS2 (adults and adolescents ≥16) ──────────────────────────────────────

def news2_respiratory_rate(rr: Optional[float]) -> int:
    if rr is None:
        return 0
    if rr <= 8:
        return 3
    if rr <= 11:
        return 1
    if rr <= 20:
        return 0
    if rr <= 24:
        return 2
    return 3


def news2_spo2_scale1(spo2: Optional[float]) -> int:
    if spo2 is None:
        return 0
    if spo2 <= 91:
        return 3
    if spo2 <= 93:
        return 2
    if spo2 <= 95:
        return 1
    return 0


def news2_systolic_bp(sbp: Optional[float]) -> int:
    if sbp is None:
        return 0
    if sbp <= 90:
        return 3
    if sbp <= 100:
        return 2
    if sbp <= 110:
        return 1
    if sbp <= 219:
        return 0
    return 3


def news2_pulse(hr: Optional[float]) -> int:
    if hr is None:
        return 0
    if hr <= 40:
        return 3
    if hr <= 50:
        return 1
    if hr <= 90:
        return 0
    if hr <= 110:
        return 1
    if hr <= 130:
        return 2
    return 3


def news2_temperature(temp: Optional[float]) -> int:
    if temp is None:
        return 0
    if temp <= 35.0:
        return 3
    if temp <= 36.0:
        return 1
    if temp <= 38.0:
        return 0
    if temp <= 39.0:
        return 1
    return 2


def news2_consciousness(avpu: Optional[str]) -> int:
    """ACVPU: Alert = 0, anything else (Confusion/Voice/Pain/Unresponsive) = 3."""
    if avpu is None:
        return 0
    return 0 if str(avpu).lower() == "alert" else 3


def compute_news2(vitals: Dict, consciousness: Optional[str] = "alert",
                  on_oxygen: bool = False) -> Dict:
    """
    Compute the aggregate NEWS2 score and its clinical risk band.

    Returns the aggregate, the per-parameter breakdown, the risk band, and
    the RCP-recommended monitoring response — all of which are surfaced in
    the UI beside our own recommendation so a clinician can see whether the
    two agree.
    """
    components = {
        "respiratory_rate": news2_respiratory_rate(vitals.get("respiratory_rate")),
        "spo2": news2_spo2_scale1(vitals.get("spo2")),
        "air_or_oxygen": 2 if on_oxygen else 0,
        "systolic_bp": news2_systolic_bp(vitals.get("systolic_bp")),
        "pulse": news2_pulse(vitals.get("heart_rate")),
        "consciousness": news2_consciousness(consciousness),
        "temperature": news2_temperature(vitals.get("temperature")),
    }
    aggregate = sum(components.values())
    max_single = max(components.values()) if components else 0

    # RCP risk banding
    if aggregate >= 7:
        band, response = "high", "Emergency assessment by critical-care-competent team; continuous monitoring."
    elif aggregate >= 5 or max_single == 3:
        band, response = "medium", "Urgent review by a clinician; monitoring at least hourly."
    elif aggregate >= 1:
        band, response = "low", "Ward-based response; monitoring 4 to 6 hourly."
    else:
        band, response = "none", "Routine monitoring, minimum 12 hourly."

    # Map the NEWS2 band onto our 5-level scale so the two can be compared
    # head-to-head. This mapping is the conventional ED interpretation, not
    # part of the published score.
    if aggregate >= 9:
        implied_level = 1
    elif aggregate >= 7:
        implied_level = 2
    elif aggregate >= 5 or max_single == 3:
        implied_level = 3
    elif aggregate >= 1:
        implied_level = 4
    else:
        implied_level = 5

    return {
        "score": aggregate,
        "components": components,
        "max_single_parameter": max_single,
        "risk_band": band,
        "recommended_response": response,
        "implied_triage_level": implied_level,
        "scale": "NEWS2 (RCP 2017)",
    }


# ─── PEWS (paediatric) ───────────────────────────────────────────────────────

# Age-banded normal ranges used by the paediatric score (PALS reference ranges)
PEDS_HR_NORMAL = [
    (0, 1, 100, 160),
    (1, 3, 90, 150),
    (3, 6, 80, 140),
    (6, 12, 70, 120),
    (12, 18, 60, 100),
]

PEDS_RR_NORMAL = [
    (0, 1, 30, 60),
    (1, 3, 24, 40),
    (3, 6, 22, 34),
    (6, 12, 18, 30),
    (12, 18, 12, 20),
]

PEDS_SBP_LOW = [
    (0, 1, 70),
    (1, 3, 72),
    (3, 6, 75),
    (6, 12, 80),
    (12, 18, 90),
]


def _band_lookup(table, age: float, default):
    for lo, hi, *vals in table:
        if lo <= age < hi:
            return vals if len(vals) > 1 else vals[0]
    return default


def compute_pews(vitals: Dict, age: float, consciousness: Optional[str] = "alert",
                 skin: Optional[str] = "normal",
                 respiratory_effort: Optional[str] = "none") -> Dict:
    """
    Brighton-style Paediatric Early Warning Score (0–3 per domain, max 9).

    Deliberately separate from NEWS2: applying an adult-calibrated score to a
    child is precisely the silent safety risk this project exists to remove.
    A 4-year-old with a heart rate of 150 scores 0 on the paediatric scale and
    2 on the adult scale — the adult score would generate a false alarm, while
    an adult score applied to a child in compensated shock does the opposite
    and stays silent until decompensation.
    """
    # ── Behaviour / neuro domain ──
    c = str(consciousness or "alert").lower()
    if c == "unresponsive":
        behaviour = 3
    elif c == "pain":
        behaviour = 2
    elif c == "verbal":
        behaviour = 1
    else:
        behaviour = 0

    # ── Cardiovascular domain ──
    hr = vitals.get("heart_rate")
    sbp = vitals.get("systolic_bp")
    skin_val = str(skin or "normal").lower()

    cardio = 0
    hr_lo, hr_hi = _band_lookup(PEDS_HR_NORMAL, age, (60, 100))
    if hr is not None:
        if hr > hr_hi + 30 or hr < hr_lo - 20:
            cardio = max(cardio, 3)
        elif hr > hr_hi + 20:
            cardio = max(cardio, 2)
        elif hr > hr_hi:
            cardio = max(cardio, 1)

    if skin_val in ("mottled", "cyanotic"):
        cardio = max(cardio, 3)
    elif skin_val == "pale":
        cardio = max(cardio, 1)

    sbp_low = _band_lookup(PEDS_SBP_LOW, age, 90)
    if sbp is not None and sbp < sbp_low:
        # Hypotension in a child is a LATE and ominous sign — it means
        # compensation has already failed.
        cardio = 3

    # ── Respiratory domain ──
    rr = vitals.get("respiratory_rate")
    spo2 = vitals.get("spo2")
    effort = str(respiratory_effort or "none").lower()

    resp = 0
    rr_lo, rr_hi = _band_lookup(PEDS_RR_NORMAL, age, (12, 20))
    if rr is not None:
        if rr > rr_hi + 20 or rr < rr_lo - 5:
            resp = max(resp, 3)
        elif rr > rr_hi + 10:
            resp = max(resp, 2)
        elif rr > rr_hi:
            resp = max(resp, 1)

    if spo2 is not None:
        if spo2 < 90:
            resp = 3
        elif spo2 < 94:
            resp = max(resp, 2)

    if effort == "severe":
        resp = 3
    elif effort == "moderate":
        resp = max(resp, 2)
    elif effort == "mild":
        resp = max(resp, 1)

    total = behaviour + cardio + resp

    if total >= 7:
        band, implied_level = "critical", 1
    elif total >= 5:
        band, implied_level = "high", 2
    elif total >= 3:
        band, implied_level = "medium", 3
    elif total >= 1:
        band, implied_level = "low", 4
    else:
        band, implied_level = "none", 5

    return {
        "score": total,
        "components": {"behaviour": behaviour, "cardiovascular": cardio, "respiratory": resp},
        "risk_band": band,
        "implied_triage_level": implied_level,
        "scale": "PEWS (Brighton, vitals-derived)",
    }


def compute_early_warning_score(vitals: Dict, age: float,
                                consciousness: Optional[str] = "alert",
                                skin: Optional[str] = "normal",
                                respiratory_effort: Optional[str] = "none",
                                on_oxygen: bool = False) -> Dict:
    """
    Age-appropriate early-warning score dispatcher.

    This one function is the single-line answer to the brief's central
    clinical complaint: a fever of 38.5 °C does not mean the same thing in a
    3-year-old and a 75-year-old, so we never run one score across all ages.
    """
    if age < 16:
        result = compute_pews(vitals, age, consciousness, skin, respiratory_effort)
    else:
        result = compute_news2(vitals, consciousness, on_oxygen)
    result["age_appropriate_scale"] = result["scale"]
    return result
