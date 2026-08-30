"""
PatientTriage.ai — Input Schema & Data Models
Dataclass-based models for all four core input categories.
Each field carries: value, unit, timestamp, source, confidence, quality
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import json


# ─── Enumerations ───────────────────────────────────────────────────────────

class TriageLevel(Enum):
    """Five-level triage scale. Level 1 = highest urgency."""
    RESUSCITATION = 1
    EMERGENT = 2
    URGENT = 3
    LESS_URGENT = 4
    NON_URGENT = 5


class UncertaintyBand(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class AgeGroup(Enum):
    PEDIATRIC = "pediatric"       # 0-17
    ADULT = "adult"               # 18-64
    GERIATRIC = "geriatric"       # 65+


class DataSource(Enum):
    PATIENT_REPORTED = "patient_reported"
    NURSE_OBSERVED = "nurse_observed"
    DEVICE_MEASURED = "device_measured"
    EHR_IMPORTED = "ehr_imported"
    MANUAL_ENTRY = "manual_entry"
    # A value that reached the record through speech: dictated at handover,
    # transcribed, extracted, and then confirmed on screen by the nurse who
    # said it. Tracked separately from MANUAL_ENTRY because the failure modes
    # differ. A typo is idiosyncratic; a mis-transcription is systematic, and
    # an auditor reviewing a decision needs to be able to find every value
    # that passed through a speech model.
    VOICE_TRANSCRIBED = "voice_transcribed"
    SYSTEM_GENERATED = "system_generated"


class SafetyStatus(Enum):
    PASS = "pass"
    ESCALATION_APPLIED = "escalation_applied"


class OverrideDirection(Enum):
    UPGRADE = "upgrade"       # System under-triaged
    DOWNGRADE = "downgrade"   # System over-triaged


# ─── Base Measurement ───────────────────────────────────────────────────────

@dataclass
class Measurement:
    """A single measured or reported value with metadata."""
    value: Any
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: DataSource = DataSource.MANUAL_ENTRY
    confidence: float = 1.0    # 0.0 to 1.0
    quality: float = 1.0       # 0.0 to 1.0 (data quality score)


# ─── Category 1: Vitals ─────────────────────────────────────────────────────

@dataclass
class VitalSigns:
    """Vital signs measurements."""
    temperature: Optional[Measurement] = None          # °C
    heart_rate: Optional[Measurement] = None            # bpm
    respiratory_rate: Optional[Measurement] = None      # breaths/min
    spo2: Optional[Measurement] = None                  # %
    systolic_bp: Optional[Measurement] = None           # mmHg
    diastolic_bp: Optional[Measurement] = None          # mmHg
    pain_score: Optional[Measurement] = None            # 0-10

    def to_feature_dict(self) -> Dict[str, Optional[float]]:
        """Extract numeric values for ML models."""
        return {
            "temperature": self.temperature.value if self.temperature else None,
            "heart_rate": self.heart_rate.value if self.heart_rate else None,
            "respiratory_rate": self.respiratory_rate.value if self.respiratory_rate else None,
            "spo2": self.spo2.value if self.spo2 else None,
            "systolic_bp": self.systolic_bp.value if self.systolic_bp else None,
            "diastolic_bp": self.diastolic_bp.value if self.diastolic_bp else None,
            "pain_score": self.pain_score.value if self.pain_score else None,
        }

    def get_quality_scores(self) -> Dict[str, float]:
        """Return quality scores for each available vital."""
        scores = {}
        for fname in ["temperature", "heart_rate", "respiratory_rate", "spo2",
                       "systolic_bp", "diastolic_bp", "pain_score"]:
            m = getattr(self, fname)
            if m is not None:
                scores[fname] = m.quality
        return scores


# ─── Category 2: Self-Reported Symptoms ─────────────────────────────────────

@dataclass
class SelfReportedSymptoms:
    """Patient-reported symptoms. Uncertain statements stay uncertain."""
    chief_complaint: Optional[Measurement] = None       # free text
    symptoms: Optional[Measurement] = None              # free text list
    severity_self_assessed: Optional[Measurement] = None  # 1-10
    onset: Optional[Measurement] = None                 # free text (e.g., "2 hours ago")
    duration: Optional[Measurement] = None              # free text
    progression: Optional[Measurement] = None           # "improving" / "stable" / "worsening" / "uncertain"

    def get_chief_complaint_text(self) -> str:
        if self.chief_complaint and self.chief_complaint.value:
            return str(self.chief_complaint.value)
        return ""

    def get_symptom_text(self) -> str:
        if self.symptoms and self.symptoms.value:
            return str(self.symptoms.value)
        return ""


# ─── Category 3: Medical History ─────────────────────────────────────────────

@dataclass
class MedicalHistory:
    """
    Medical history with explicit history_available flag.
    Empty history ≠ "no medical history" — it means "medical history unknown."
    """
    history_available: bool = False
    known_conditions: Optional[Measurement] = None      # list of conditions
    medications: Optional[Measurement] = None           # list of medications
    allergies: Optional[Measurement] = None             # list of allergies
    prior_visits: Optional[Measurement] = None          # count or list

    def has_high_risk_conditions(self) -> bool:
        """Check for known high-risk conditions."""
        if not self.history_available or not self.known_conditions:
            return False
        high_risk = ["cardiac", "heart", "diabetes", "copd", "asthma", "cancer",
                     "immunocompromised", "transplant", "renal", "liver", "stroke",
                     "seizure", "bleeding disorder", "anticoagulant"]
        conditions_text = str(self.known_conditions.value).lower()
        return any(term in conditions_text for term in high_risk)


# ─── Category 4: Nurse / Staff Observed Cues ─────────────────────────────────

@dataclass
class StaffObservedCues:
    """
    Observations by clinical staff — kept clearly separate from patient-reported info.
    """
    visible_distress: Optional[Measurement] = None        # none / mild / moderate / severe
    breathing_difficulty: Optional[Measurement] = None     # none / mild / moderate / severe
    consciousness: Optional[Measurement] = None            # alert / verbal / pain / unresponsive (AVPU)
    mobility: Optional[Measurement] = None                 # ambulatory / assisted / immobile
    bleeding: Optional[Measurement] = None                 # none / controlled / uncontrolled
    skin_appearance: Optional[Measurement] = None          # normal / pale / flushed / cyanotic / diaphoretic


# ─── Complete Patient Encounter ──────────────────────────────────────────────

@dataclass
class PatientEncounter:
    """A complete patient encounter combining all four input categories."""
    patient_id: str
    age: int
    sex: str                           # "M" / "F" / "Other"
    arrival_time: datetime = field(default_factory=datetime.now)
    vitals: VitalSigns = field(default_factory=VitalSigns)
    symptoms: SelfReportedSymptoms = field(default_factory=SelfReportedSymptoms)
    history: MedicalHistory = field(default_factory=MedicalHistory)
    staff_cues: StaffObservedCues = field(default_factory=StaffObservedCues)

    # Structured context the model was trained on that has no natural home in
    # the four clinical categories above — arrival mode, coded chronic
    # conditions, prior-visit flags.
    #
    # This exists because of a measured bug rather than a hunch: without it,
    # building an encounter from a record and scoring it dropped 25 features
    # the model uses, and changed 15% of triage decisions relative to scoring
    # the same patient directly. The demo was quietly running a weaker system
    # than the one the evaluation validated.
    context: Dict[str, Any] = field(default_factory=dict)

    # Triage output (filled by the pipeline)
    triage_level: Optional[TriageLevel] = None
    confidence: Optional[float] = None
    uncertainty_band: Optional[UncertaintyBand] = None
    data_quality_score: Optional[float] = None
    safety_status: Optional[SafetyStatus] = None
    safety_reason: Optional[str] = None

    @property
    def age_group(self) -> AgeGroup:
        if self.age < 18:
            return AgeGroup.PEDIATRIC
        elif self.age >= 65:
            return AgeGroup.GERIATRIC
        else:
            return AgeGroup.ADULT

    def get_wait_time_minutes(self) -> float:
        """Minutes since arrival."""
        return (datetime.now() - self.arrival_time).total_seconds() / 60.0


# ─── Triage Result ───────────────────────────────────────────────────────────

@dataclass
class TriageResult:
    """Complete triage output as required by Section 5.B."""
    patient_id: str
    triage_level: int                           # 1-5
    confidence_percent: float                    # 0-100
    uncertainty_band: str                        # low / moderate / high
    data_quality_percent: float                  # 0-100
    model_agreement: float                       # 0-1 (proportion of models agreeing)
    individual_predictions: Dict[str, int]       # model_name -> predicted_level
    safety_status: str                          # "pass" or "escalation_applied"
    safety_reason: Optional[str]                 # why escalation was applied
    top_contributing_factors: List[str]           # pulled from real features
    missing_information: List[str]               # what's missing
    recommended_followup_question: Optional[str]  # if something material is missing
    recommended_action: str                      # e.g., "Immediate physician assessment"
    agent_debate_summary: Optional[str] = None   # from multi-agent debate
    deterioration_velocity: Optional[Dict] = None  # from velocity model
    timestamp: datetime = field(default_factory=datetime.now)


# ─── Override Record ─────────────────────────────────────────────────────────

@dataclass
class OverrideRecord:
    """Clinician override record per Section 5 and regulatory requirements."""
    timestamp: datetime
    clinician_id: str
    clinician_role: str
    patient_id: str
    system_recommendation: int          # triage level 1-5
    system_confidence: float
    system_uncertainty: str
    override_level: int                 # clinician's triage level
    override_direction: str             # "upgrade" or "downgrade"
    justification_code: str             # from controlled vocabulary
    justification_text: str             # free text
    second_clinician_concurred: Optional[bool] = None  # required for L1-L2 downgrades


# ─── Audit Log Entry ─────────────────────────────────────────────────────────

@dataclass
class AuditLogEntry:
    """Immutable audit log entry."""
    timestamp: datetime
    event_type: str                    # "triage_decision", "override", "reassessment", "data_access"
    patient_id: str
    user_id: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "patient_id": self.patient_id,
            "user_id": self.user_id,
            "details": self.details
        }
