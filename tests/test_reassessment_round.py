"""
Re-recording a waiting patient's vitals.

The brief makes this a requirement: the system "must monitor patients already
in the waiting queue and trigger re-assessment if wait time exceeds safe
thresholds for their severity level or if vitals are re-recorded as worsening".

The dangerous half is not the escalation. It is the possibility that routine
monitoring quietly becomes a de-escalation channel that nobody signed for.
"""

import pytest


def _a_waiting_patient(pipeline, level=3):
    """
    A patient at the requested level, or the least urgent one available.

    The fallback matters: the demo board is drawn from a real cohort and does
    not guarantee a patient at every level, so pinning a test to Level 4 makes
    it skip silently on some boards. A skipped safety test is a test nobody
    notices has stopped running.
    """
    candidates = [(e.patient_id, pipeline.triage_results[e.patient_id]["result"].triage_level)
                  for e, _, _ in pipeline.patients
                  if e.patient_id in pipeline.triage_results]
    assert candidates, "no scored patients on the board"
    exact = [pid for pid, lvl in candidates if lvl == level]
    if exact:
        return exact[0]
    return max(candidates, key=lambda kv: kv[1])[0]


def _worse_than(vitals):
    return {"heart_rate": (vitals.get("heart_rate") or 90) + 45,
            "respiratory_rate": (vitals.get("respiratory_rate") or 18) + 14,
            "spo2": max(85, (vitals.get("spo2") or 98) - 12)}


# ── It does the thing the brief asks for ─────────────────────────────────────

def test_worsening_vitals_escalate_a_waiting_patient(pipeline):
    pid = _a_waiting_patient(pipeline)
    before = pipeline.triage_results[pid]["result"].triage_level
    arrival = pipeline.patient_encounters[pid][0].vitals.to_feature_dict()

    outcome = pipeline.record_observation(pid, _worse_than(arrival))

    assert outcome is not None
    assert outcome["final_level"] < before, "worsening vitals did not escalate"
    assert outcome["escalated"] is True


def test_a_recheck_produces_trend_data_where_there_was_none(pipeline):
    """One recheck must be enough to compute a rate of change."""
    pid = _a_waiting_patient(pipeline, level=4)
    arrival = pipeline.patient_encounters[pid][0].vitals.to_feature_dict()

    pipeline.record_observation(pid, _worse_than(arrival))

    velocity = pipeline.triage_results[pid]["velocity"]
    assert velocity["has_trend_data"], "no trend computed after a recheck"
    assert velocity["readings_count"] >= 2


# ── The half that carries the risk ───────────────────────────────────────────

def test_an_improved_reading_never_de_escalates(pipeline):
    """
    A heart rate that falls may mean a patient is tiring, not improving.
    Re-scoring is escalate-only; a downgrade stays a clinician override.
    """
    pid = _a_waiting_patient(pipeline)
    arrival = pipeline.patient_encounters[pid][0].vitals.to_feature_dict()

    pipeline.record_observation(pid, _worse_than(arrival))
    escalated = pipeline.triage_results[pid]["result"].triage_level

    outcome = pipeline.record_observation(
        pid, {"heart_rate": 72, "respiratory_rate": 14, "spo2": 99})

    assert outcome["final_level"] == escalated, "a recheck walked a patient down"
    assert outcome["final_level"] <= escalated
    assert outcome["held_by_escalate_only_rule"] is True
    assert pipeline.triage_results[pid]["result"].triage_level == escalated


def test_repeated_reassessment_never_reduces_urgency(pipeline):
    """Across a whole round of rechecks, urgency is monotone."""
    pid = _a_waiting_patient(pipeline)
    levels = [pipeline.triage_results[pid]["result"].triage_level]

    for vitals in ({"heart_rate": 130, "spo2": 90},
                   {"heart_rate": 70, "spo2": 99},
                   {"heart_rate": 145, "spo2": 86},
                   {"heart_rate": 68, "spo2": 100}):
        pipeline.record_observation(pid, vitals)
        levels.append(pipeline.triage_results[pid]["result"].triage_level)

    assert levels == sorted(levels, reverse=True), \
        f"urgency decreased during a reassessment round: {levels}"


# ── It is recorded ───────────────────────────────────────────────────────────

def test_every_reassessment_is_audited(pipeline):
    pid = _a_waiting_patient(pipeline)
    before = sum(1 for e in pipeline.audit_log if e.event_type == "reassessment")

    pipeline.record_observation(pid, {"heart_rate": 132, "spo2": 89},
                                recorded_by="NURSE_07")

    events = [e for e in pipeline.audit_log if e.event_type == "reassessment"]
    assert len(events) == before + 1

    entry = events[-1]
    assert entry.patient_id == pid
    assert entry.user_id == "NURSE_07"
    for key in ("previous_level", "proposed_level", "final_level",
                "vitals_recorded", "rules_fired", "rule_pack_hash"):
        assert key in entry.details, f"audit entry is missing {key}"


def test_unrechecked_vitals_keep_their_arrival_value(pipeline):
    """A vital the nurse did not re-measure must not become unknown."""
    pid = _a_waiting_patient(pipeline)
    encounter = pipeline.patient_encounters[pid][0]
    arrival = encounter.vitals.to_feature_dict()
    if arrival.get("temperature") is None:
        pytest.skip("Patient has no arrival temperature to preserve.")
    original_temp = arrival["temperature"]

    pipeline.record_observation(pid, {"heart_rate": 128})

    after = encounter.vitals.to_feature_dict()
    assert after["temperature"] == original_temp
    assert after["heart_rate"] == 128


# ── Guards ───────────────────────────────────────────────────────────────────

def test_unknown_patient_returns_none(pipeline):
    assert pipeline.record_observation("NO-SUCH-PATIENT", {"heart_rate": 90}) is None


def test_empty_observation_returns_none(pipeline):
    pid = _a_waiting_patient(pipeline)
    assert pipeline.record_observation(pid, {}) is None
    assert pipeline.record_observation(pid, {"heart_rate": None}) is None


def test_reassessment_still_returns_a_confidence(pipeline):
    """The no-score-without-confidence invariant survives re-scoring."""
    pid = _a_waiting_patient(pipeline)
    outcome = pipeline.record_observation(pid, {"heart_rate": 135, "spo2": 88})
    assert outcome["result"].confidence_percent is not None
    assert outcome["result"].uncertainty_band in ("low", "moderate", "high")
