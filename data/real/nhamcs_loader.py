"""
PulseGuard — Real-World Data Loader (NHAMCS)
==================================================

Loads the **National Hospital Ambulatory Medical Care Survey — Emergency
Department** public-use file from the CDC/NCHS.

Why this dataset
----------------
Every triage prototype can claim accuracy on data it invented. NHAMCS is the
CDC's nationally representative probability sample of US emergency department
visits, and for each visit it records:

  * the triage level an actual triage nurse assigned at the bedside (IMMEDR),
  * the initial vital signs actually measured at that visit,
  * the patient's reason for visit in the NCHS classification,
  * documented chronic conditions, and
  * **what actually happened to the patient** — admitted, admitted to a
    critical-care unit, or died in the ED.

That last group is what makes this dataset worth the parsing effort. It gives
us a ground truth that is independent of the nurse's own label, so we can
measure something far more meaningful than "does the model agree with the
nurse?": we can measure how often a patient who genuinely needed critical care
would have been sent to the waiting room.

Sample: 16,025 ED visits across 188 hospitals (2022 public-use file).

Data use
--------
NHAMCS public-use files are released by NCHS for statistical reporting and
analysis. All direct identifiers are removed by NCHS before release, and the
readme's terms prohibit any attempt at re-identification or linkage to other
individually identifiable data. This project uses the file for statistical
analysis only, makes no re-identification attempt, and performs no linkage —
consistent with those terms. Records are patient-level survey records, not
patients of any identifiable institution.

Because NHAMCS is a *survey*, PATWT (visit weight) is required to produce
national estimates. Model training uses unweighted records (each record is one
real visit, which is what a model should learn from); population-level
statistics reported in the evaluation are weighted where noted.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RFV_CODES_PATH = os.path.join(HERE, "rfv_codes.json")

# ─── Fixed-width record layout ───────────────────────────────────────────────
# Positions transcribed from the NHAMCS micro-data file documentation
# (doc22-ed-508.pdf, Section II.A "Codebook of Emergency Department
# Micro-Data File"). Documented as 1-indexed inclusive ranges; converted to
# 0-indexed half-open slices at parse time.

LAYOUT_2022: Dict[str, tuple] = {
    # ── Visit ──
    "VMONTH": (1, 2),
    "VDAYR": (3, 3),
    "ARRTIME": (4, 7),
    "WAITTIME": (8, 11),
    # ── Demographics ──
    "AGE": (16, 18),
    "AGER": (19, 19),
    "AGEDAYS": (20, 22),
    "RESIDNCE": (23, 24),
    "SEX": (25, 25),
    "RACERETH": (32, 32),
    "ARREMS": (33, 34),
    # ── Initial vital signs ──
    "TEMPF": (48, 51),
    "PULSE": (52, 54),
    "RESPR": (55, 57),
    "BPSYS": (58, 60),
    "BPDIAS": (61, 63),
    "POPCT": (64, 66),
    # ── Triage ──
    "IMMEDR": (67, 68),
    "PAINSCALE": (69, 70),
    "SEEN72": (71, 72),
    # ── Reason for visit ──
    "RFV1": (73, 77),
    "RFV2": (78, 82),
    "RFV3": (83, 87),
    "EPISODE": (98, 99),
    "INJURY": (100, 101),
    # ── Chronic conditions (single char each, 0/1) ──
    "ETOHAB": (152, 152),
    "ALZHD": (153, 153),
    "ASTHMA": (154, 154),
    "CANCER": (155, 155),
    "CEBVD": (156, 156),
    "CKD": (157, 157),
    "COPD": (158, 158),
    "CHF": (159, 159),
    "CAD": (160, 160),
    "DEPRN": (161, 161),
    "DIABTYP1": (162, 162),
    "DIABTYP2": (163, 163),
    "DIABTYP0": (164, 164),
    "ESRD": (165, 165),
    "HPE": (166, 166),
    "EDHIV": (167, 167),
    "HYPLIPID": (168, 168),
    "HTN": (169, 169),
    "OBESITY": (170, 170),
    "OSA": (171, 171),
    "OSTPRSIS": (172, 172),
    "SUBSTAB": (173, 173),
    "NOCHRON": (174, 174),
    "TOTCHRON": (175, 176),
    # ── Visit disposition / OUTCOMES ──
    "LWBS": (491, 491),
    "LBTC": (492, 492),
    "LEFTAMA": (493, 493),
    "DOA": (494, 494),
    "DIEDED": (495, 495),
    "TRANPSYC": (497, 497),
    "TRANOTH": (498, 498),
    "ADMITHOS": (499, 499),
    "OBSHOS": (500, 500),
    "ADMIT": (503, 504),
    "LOS": (507, 508),
    # ── Identifiers / survey design ──
    "HOSPCODE": (544, 546),
    "PATCODE": (547, 549),
    "CSTRATM": (2345, 2352),
    "CPSUM": (2353, 2358),
    "PATWT": (2359, 2369),
    "BOARDED": (2379, 2382),
}

# The 2021 file is byte-identical to 2022 through position 176 (demographics,
# vital signs, triage level, reason-for-visit, chronic conditions) but the
# visit-disposition block onward is shifted two characters earlier, because
# 2022 added two items upstream. Reusing the 2022 layout on 2021 silently
# produced a 9.6% critical-care rate — nearly four times the true figure —
# which is exactly the kind of error a fixed-width parse hides in plain sight,
# since every value it yields is still a well-formed number.
_SHIFTED_IN_2021 = {
    "LWBS": (489, 489),
    "LBTC": (490, 490),
    "LEFTAMA": (491, 491),
    "DOA": (492, 492),
    "DIEDED": (493, 493),
    "TRANPSYC": (495, 495),
    "TRANOTH": (496, 496),
    "ADMITHOS": (497, 497),
    "OBSHOS": (498, 498),
    "ADMIT": (501, 502),
    "LOS": (505, 506),
    "HOSPCODE": (542, 544),
    "PATCODE": (545, 547),
    "CSTRATM": (2343, 2350),
    "CPSUM": (2351, 2356),
    "PATWT": (2357, 2367),
    "BOARDED": (2377, 2380),
}

LAYOUT_2021 = {**LAYOUT_2022, **_SHIFTED_IN_2021}

LAYOUTS = {
    2022: LAYOUT_2022,
    2021: LAYOUT_2021,
}

# NHAMCS missing-data conventions
MISSING_CODES = {-9, -8, -7}

CHRONIC_CONDITIONS = [
    "ETOHAB", "ALZHD", "ASTHMA", "CANCER", "CEBVD", "CKD", "COPD", "CHF",
    "CAD", "DEPRN", "DIABTYP1", "DIABTYP2", "DIABTYP0", "ESRD", "HPE",
    "EDHIV", "HYPLIPID", "HTN", "OBESITY", "OSA", "OSTPRSIS", "SUBSTAB",
]

# Conditions that materially raise the risk of rapid deterioration and so
# feed the safety engine's "high-risk history" flag.
HIGH_RISK_CONDITIONS = [
    "CANCER", "CEBVD", "CKD", "COPD", "CHF", "CAD", "ESRD", "HPE",
    "EDHIV", "DIABTYP1", "ALZHD",
]


# ─── Parsing ─────────────────────────────────────────────────────────────────

def _read_fixed_width(path: str, layout: Dict[str, tuple]) -> pd.DataFrame:
    """Read the fixed-width ASCII record file using an explicit layout."""
    names = list(layout.keys())
    colspecs = [(layout[n][0] - 1, layout[n][1]) for n in names]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, dtype=str,
                     header=None, encoding="latin-1")
    return df


def _to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _blank_missing(series: pd.Series, extra_missing: Optional[set] = None) -> pd.Series:
    """
    Convert NHAMCS missing sentinels to NaN.

    This matters more than it looks. NHAMCS encodes 'blank' as -9 and 'not
    applicable' as -7. Feeding those into a model as numbers would teach it
    that an unrecorded blood pressure is a blood pressure of negative nine —
    the exact class of silent error this project is built to prevent.
    """
    missing = set(MISSING_CODES)
    if extra_missing:
        missing |= extra_missing
    return series.where(~series.isin(missing), np.nan)


def load_raw(year: int = 2022, data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load and decode one NHAMCS ED public-use year into a tidy DataFrame."""
    data_dir = data_dir or HERE
    layout = LAYOUTS.get(year)
    if layout is None:
        raise ValueError(f"No verified record layout for NHAMCS year {year}.")

    # Accept either the extracted file or the original zip
    candidates = [
        os.path.join(data_dir, f"ed{year}", f"ed{year}"),
        os.path.join(data_dir, f"ed{year}", f"ED{year}"),
        os.path.join(data_dir, f"ed{year}"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)

    if path is None:
        zip_path = os.path.join(data_dir, f"ed{year}.zip")
        if not os.path.isfile(zip_path):
            raise FileNotFoundError(
                f"NHAMCS {year} data not found. Download it with:\n"
                f"  python -m data.real.nhamcs_loader --download {year}"
            )
        with zipfile.ZipFile(zip_path) as zf:
            extract_dir = os.path.join(data_dir, f"ed{year}")
            zf.extractall(extract_dir)
        path = next((p for p in candidates if os.path.isfile(p)), None)

    df = _read_fixed_width(path, layout)
    df = _to_numeric(df, [c for c in df.columns if c not in ("RFV1", "RFV2", "RFV3")])
    df["survey_year"] = year
    return df


# ─── Cleaning into clinical units ────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw survey codes into clinical values our pipeline speaks.

    Every conversion here is a documented NHAMCS decode, not an assumption:
    temperature carries an implied decimal and is Fahrenheit; a pulse of 998
    means 'Doppler', not 998 bpm; sex is coded 1=Female, 2=Male.
    """
    out = pd.DataFrame(index=df.index)

    # ── Identifiers (survey record ids, not patient identifiers) ──
    out["record_id"] = (
        df["survey_year"].astype(str) + "-"
        + df["HOSPCODE"].astype("Int64").astype(str) + "-"
        + df["PATCODE"].astype("Int64").astype(str)
    )
    out["hospital_id"] = df["HOSPCODE"].astype("Int64")
    out["survey_year"] = df["survey_year"]
    out["patient_weight"] = _blank_missing(df["PATWT"])

    # ── Demographics ──
    age = _blank_missing(df["AGE"])
    # AGEDAYS gives sub-1-year infants their true age; a 3-day-old and an
    # 11-month-old are clinically nothing alike and both code as AGE=0.
    age_days = _blank_missing(df["AGEDAYS"])
    age = np.where(age_days.notna(), age_days / 365.25, age)
    out["age"] = pd.Series(age, index=df.index).astype(float)

    sex_raw = _blank_missing(df["SEX"])
    out["sex"] = sex_raw.map({1: "F", 2: "M"})

    # Race/ethnicity, retained solely to audit the system for disparate impact.
    # It is never a model feature — a triage model must not learn to treat
    # patients differently by race — but a system nobody checks for disparity
    # is a system that hides it, so the evaluation reports under-triage rates
    # broken out by this field.
    out["race_ethnicity"] = _blank_missing(df["RACERETH"]).map({
        1: "Non-Hispanic White", 2: "Non-Hispanic Black",
        3: "Hispanic", 4: "Non-Hispanic Other",
    })

    out["arrival_by_ambulance"] = _blank_missing(df["ARREMS"]).map({1: 1, 2: 0})
    out["seen_last_72h"] = _blank_missing(df["SEEN72"]).map({1: 1, 2: 0})
    out["injury_related"] = _blank_missing(df["INJURY"]).map({1: 1, 2: 0})
    out["nursing_home_resident"] = (_blank_missing(df["RESIDNCE"]) == 2).astype(float)

    # ── Vital signs ──
    # TEMPF: implied decimal between 3rd and 4th digit, Fahrenheit.
    tempf = _blank_missing(df["TEMPF"]) / 10.0
    tempf = tempf.where((tempf >= 89.6) & (tempf <= 105.6))
    out["temperature"] = (tempf - 32.0) * 5.0 / 9.0  # → Celsius

    # PULSE: 998 = "Doppler" (a measurement method note, not a rate).
    pulse = _blank_missing(df["PULSE"], extra_missing={998})
    out["heart_rate"] = pulse.where((pulse > 0) & (pulse <= 240))

    respr = _blank_missing(df["RESPR"])
    out["respiratory_rate"] = respr.where((respr > 0) & (respr <= 150))

    bpsys = _blank_missing(df["BPSYS"])
    # A recorded systolic of 0 is a documentation artifact, not asystole:
    # a real cardiac arrest in this survey shows up in the outcome fields.
    out["systolic_bp"] = bpsys.where((bpsys >= 43) & (bpsys <= 289))

    bpdias = _blank_missing(df["BPDIAS"], extra_missing={998})
    out["diastolic_bp"] = bpdias.where((bpdias >= 22) & (bpdias <= 190))

    popct = _blank_missing(df["POPCT"])
    out["spo2"] = popct.where((popct > 0) & (popct <= 100))

    pain = _blank_missing(df["PAINSCALE"])
    out["pain_score"] = pain.where((pain >= 0) & (pain <= 10))

    # ── Triage label (the nurse's actual assignment) ──
    # 0 = "no triage performed", 7 = "ED does not triage", -8/-9 missing.
    immed = _blank_missing(df["IMMEDR"], extra_missing={0, 7})
    out["triage_level"] = immed.where(immed.between(1, 5))

    # ── Chronic conditions ──
    for cond in CHRONIC_CONDITIONS:
        out[f"cond_{cond.lower()}"] = (_blank_missing(df[cond]) == 1).astype(float)

    total_chron = _blank_missing(df["TOTCHRON"])
    out["n_chronic_conditions"] = total_chron.clip(lower=0)
    out["has_high_risk_conditions"] = (
        df[HIGH_RISK_CONDITIONS].apply(lambda c: _blank_missing(c) == 1).any(axis=1)
    ).astype(float)

    # `history_available` models what the ED actually knows at triage. NHAMCS
    # records the chronic-condition item as entirely blank when the facility
    # had no record to draw on — which is exactly the "first-time patient with
    # nothing on file" case the brief asks us to handle.
    nochron = _blank_missing(df["NOCHRON"])
    out["history_available"] = (nochron != 2).astype(float)

    # ── Reason for visit → chief complaint text ──
    rfv_map = load_rfv_codes()
    for i in (1, 2, 3):
        col = f"RFV{i}"
        out[f"rfv{i}_code"] = df[col].astype(str).str.strip()
    out["chief_complaint"] = out.apply(
        lambda r: _rfv_text(r, rfv_map, first_only=True), axis=1
    )
    out["symptoms_text"] = out.apply(
        lambda r: _rfv_text(r, rfv_map, first_only=False), axis=1
    )

    # ── OUTCOMES (independent of the triage label) ──
    for col in ["ADMITHOS", "OBSHOS", "DIEDED", "DOA", "TRANOTH", "LWBS", "LBTC"]:
        out[f"out_{col.lower()}"] = (_blank_missing(df[col]) == 1).astype(float)

    admit_unit = _blank_missing(df["ADMIT"])
    out["out_critical_care_unit"] = (admit_unit == 1).astype(float)
    out["out_hospital_los_days"] = _blank_missing(df["LOS"])

    # A composite, outcome-based definition of "this patient was genuinely
    # sick". Deliberately conservative: it counts only outcomes that are hard
    # to argue with — critical care, death, or an inpatient admission.
    out["outcome_critical"] = (
        (out["out_critical_care_unit"] == 1)
        | (out["out_dieded"] == 1)
        | (out["out_doa"] == 1)
    ).astype(int)

    out["outcome_admitted"] = (
        (out["out_admithos"] == 1)
        | (out["out_obshos"] == 1)
        | (out["out_critical_care_unit"] == 1)
        | (out["out_tranoth"] == 1)
    ).astype(int)

    out["waiting_time_minutes"] = _blank_missing(df["WAITTIME"])
    out["boarded_minutes"] = _blank_missing(df["BOARDED"])
    out["arrival_time_raw"] = _blank_missing(df["ARRTIME"])

    # Survey design variables, for weighted national estimates
    out["stratum"] = _blank_missing(df["CSTRATM"])
    out["psu"] = _blank_missing(df["CPSUM"])

    return out


def _rfv_text(row, rfv_map: Dict[str, str], first_only: bool = False) -> str:
    """Turn NCHS reason-for-visit codes into readable complaint text."""
    codes = [row.get("rfv1_code")]
    if not first_only:
        codes += [row.get("rfv2_code"), row.get("rfv3_code")]
    parts = []
    for c in codes:
        if not c or c in ("-9", "nan", "None", ""):
            continue
        label = rfv_map.get(str(c).strip())
        if label:
            parts.append(label.strip().rstrip(","))
    return "; ".join(parts)


def load_rfv_codes() -> Dict[str, str]:
    """Load the NCHS Reason-for-Visit classification code → label map."""
    if os.path.isfile(RFV_CODES_PATH):
        with open(RFV_CODES_PATH) as f:
            return json.load(f)
    return {}


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_parse(df: pd.DataFrame) -> Dict:
    """
    Sanity-check a parsed year against known clinical and survey facts.

    A fixed-width parse that is off by even one column produces numbers that
    still *look* like data, so this runs before any parsed year is trusted.
    """
    checks = {}

    tri = df["triage_level"].dropna()
    checks["n_records"] = int(len(df))
    checks["triage_coverage"] = round(float(len(tri) / max(len(df), 1)), 4)
    checks["triage_values_valid"] = bool(tri.isin([1, 2, 3, 4, 5]).all())
    dist = (tri.value_counts(normalize=True).sort_index() * 100).round(2).to_dict()
    checks["triage_distribution_pct"] = {int(k): v for k, v in dist.items()}

    # Clinically plausible central tendencies
    for col, (lo, hi) in {
        "heart_rate": (60, 110),
        "respiratory_rate": (14, 24),
        "systolic_bp": (110, 145),
        "spo2": (94, 100),
        "temperature": (36.0, 37.6),
        "age": (25, 55),
    }.items():
        med = df[col].median()
        checks[f"{col}_median"] = None if pd.isna(med) else round(float(med), 2)
        checks[f"{col}_plausible"] = bool(lo <= med <= hi) if not pd.isna(med) else False

    checks["sex_values_valid"] = bool(set(df["sex"].dropna().unique()) <= {"M", "F"})
    checks["pct_admitted"] = round(float(df["outcome_admitted"].mean() * 100), 2)
    checks["pct_critical_outcome"] = round(float(df["outcome_critical"].mean() * 100), 2)
    checks["complaint_text_coverage"] = round(
        float((df["chief_complaint"].str.len() > 0).mean()), 4
    )
    checks["n_hospitals"] = int(df["hospital_id"].nunique())

    # Cross-field consistency. Critical-care admissions are a subset of all
    # admissions and are always a minority of them; if the parse says
    # otherwise, a column offset has moved the disposition block. An earlier
    # version of this loader passed every single-field range check on a
    # misaligned 2021 file and was only caught by this ratio.
    ratio = (checks["pct_critical_outcome"] / checks["pct_admitted"]
             if checks["pct_admitted"] > 0 else 1.0)
    checks["critical_to_admitted_ratio"] = round(ratio, 3)
    checks["disposition_block_consistent"] = bool(ratio <= 0.5)

    plausible_flags = [v for k, v in checks.items() if k.endswith("_plausible")]
    checks["all_checks_passed"] = bool(
        checks["triage_values_valid"]
        and checks["sex_values_valid"]
        and all(plausible_flags)
        and checks["triage_coverage"] > 0.5
        and 5 <= checks["pct_admitted"] <= 30
        and checks["disposition_block_consistent"]
    )
    return checks


def load_clean(years=(2022,), data_dir: Optional[str] = None,
               require_triage: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Load, clean, validate and concatenate one or more NHAMCS ED years."""
    frames = []
    for year in years:
        raw = load_raw(year, data_dir)
        cleaned = clean(raw)
        checks = validate_parse(cleaned)
        if verbose:
            print(f"  NHAMCS {year}: {checks['n_records']:,} visits, "
                  f"{checks['n_hospitals']} hospitals, "
                  f"triage coverage {checks['triage_coverage']:.1%}, "
                  f"validation {'PASSED' if checks['all_checks_passed'] else 'FAILED'}")
        if not checks["all_checks_passed"]:
            raise ValueError(
                f"NHAMCS {year} failed parse validation. Refusing to train on it. "
                f"Checks: {checks}"
            )
        frames.append(cleaned)

    df = pd.concat(frames, ignore_index=True)
    if require_triage:
        df = df[df["triage_level"].notna()].reset_index(drop=True)
    df["triage_level"] = df["triage_level"].astype(int)
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

NHAMCS_BASE = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS"


def download(year: int, data_dir: Optional[str] = None):
    """Download one NHAMCS ED year from the CDC public FTP mirror."""
    import urllib.request
    data_dir = data_dir or HERE
    os.makedirs(data_dir, exist_ok=True)
    for name in (f"ed{year}.zip", f"ED{year}.zip"):
        url = f"{NHAMCS_BASE}/{name}"
        dest = os.path.join(data_dir, f"ed{year}.zip")
        try:
            print(f"  Downloading {url} …")
            urllib.request.urlretrieve(url, dest)
            print(f"  ✓ Saved to {dest}")
            return dest
        except Exception:
            continue
    raise RuntimeError(f"Could not download NHAMCS ED {year}.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="NHAMCS ED data loader")
    ap.add_argument("--download", type=int, help="Download a survey year")
    ap.add_argument("--years", type=int, nargs="+", default=[2022])
    args = ap.parse_args()

    if args.download:
        download(args.download)

    df = load_clean(tuple(args.years))
    print(f"\nLoaded {len(df):,} real ED visits with a nurse-assigned triage level.")
    print("\nTriage level distribution (unweighted):")
    print((df["triage_level"].value_counts(normalize=True).sort_index() * 100).round(1))
    print("\nOutcome rates:")
    print(f"  Admitted:            {df['outcome_admitted'].mean():.1%}")
    print(f"  Critical outcome:    {df['outcome_critical'].mean():.2%}")
    print("\nVital sign availability:")
    for c in ["heart_rate", "respiratory_rate", "systolic_bp", "spo2",
              "temperature", "pain_score"]:
        print(f"  {c:20s} {df[c].notna().mean():.1%}")
