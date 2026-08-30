"""
Tests for engine/override_audit.py — clinician override capture and the
hash-chained, append-only audit trail. The problem statement requires that
overrides "remain reviewable... with a clear audit trail" and that the
prototype "capture at least one clinician override and show what the
system logs." These tests verify the audit trail is actually tamper-evident,
not just labeled that way.
"""
import json
import os
import tempfile

import pytest

from engine.override_audit import OverrideAuditManager


@pytest.fixture
def manager():
    tmpdir = tempfile.mkdtemp()
    log_path = os.path.join(tmpdir, "audit_log.jsonl")
    return OverrideAuditManager(log_file_path=log_path)


def test_override_is_recorded(manager):
    result = manager.record_override(
        clinician_id="RN-100", clinician_role="Triage Nurse", patient_id="PT-001",
        system_level=3, system_confidence=82.0, system_uncertainty="low",
        override_level=2, justification_code="CLINICAL_JUDGMENT",
        justification_text="Patient looks worse than vitals suggest.",
    )
    assert result["recorded"] is True
    assert result["record"]["override_direction"] == "upgrade"
    assert len(manager.get_overrides()) == 1


def test_downgrade_of_critical_level_requires_second_clinician(manager):
    """Downgrading a Level 1-2 recommendation must be blocked without
    documented second-clinician concurrence — this is the specific
    liability control the business proposal commits to."""
    blocked = manager.record_override(
        clinician_id="RN-100", clinician_role="Triage Nurse", patient_id="PT-002",
        system_level=1, system_confidence=90.0, system_uncertainty="low",
        override_level=3, justification_code="CLINICAL_JUDGMENT",
        justification_text="Patient looks stable.",
    )
    assert blocked["recorded"] is False

    allowed = manager.record_override(
        clinician_id="RN-100", clinician_role="Triage Nurse", patient_id="PT-002",
        system_level=1, system_confidence=90.0, system_uncertainty="low",
        override_level=3, justification_code="CLINICAL_JUDGMENT",
        justification_text="Patient looks stable.",
        second_clinician_concurred=True,
    )
    assert allowed["recorded"] is True


def test_audit_log_is_hash_chained_and_intact_by_default(manager):
    manager.record_override(
        clinician_id="RN-1", clinician_role="RN", patient_id="PT-A",
        system_level=3, system_confidence=70.0, system_uncertainty="moderate",
        override_level=2, justification_code="CLINICAL_JUDGMENT", justification_text="x",
    )
    manager.record_acceptance("RN-2", "RN", "PT-B", system_level=4)

    result = manager.verify_integrity()
    assert result["intact"] is True
    assert result["total_entries"] == 2


def test_tampering_with_an_entry_is_detected(manager):
    manager.record_override(
        clinician_id="RN-1", clinician_role="RN", patient_id="PT-A",
        system_level=3, system_confidence=70.0, system_uncertainty="moderate",
        override_level=2, justification_code="CLINICAL_JUDGMENT", justification_text="x",
    )
    manager.record_acceptance("RN-2", "RN", "PT-B", system_level=4)

    # Simulate someone editing a past entry in place
    manager.audit_log[0]["details"]["patient_id"] = "TAMPERED"

    result = manager.verify_integrity()
    assert result["intact"] is False
    assert result["tampered_at_index"] == 0


def test_deleting_an_entry_is_detected(manager):
    manager.record_override(
        clinician_id="RN-1", clinician_role="RN", patient_id="PT-A",
        system_level=3, system_confidence=70.0, system_uncertainty="moderate",
        override_level=2, justification_code="CLINICAL_JUDGMENT", justification_text="x",
    )
    manager.record_acceptance("RN-2", "RN", "PT-B", system_level=4)
    manager.record_acceptance("RN-3", "RN", "PT-C", system_level=5)

    del manager.audit_log[1]  # remove the middle entry

    result = manager.verify_integrity()
    assert result["intact"] is False


def test_entries_are_persisted_append_only_to_disk(manager):
    manager.record_override(
        clinician_id="RN-1", clinician_role="RN", patient_id="PT-A",
        system_level=3, system_confidence=70.0, system_uncertainty="moderate",
        override_level=2, justification_code="CLINICAL_JUDGMENT", justification_text="x",
    )
    manager.record_acceptance("RN-2", "RN", "PT-B", system_level=4)

    with open(manager.log_file_path) as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 2
    assert all("entry_hash" in line and "prev_hash" in line for line in lines)


def test_get_audit_log_filters_by_patient_and_event_type(manager):
    manager.record_override(
        clinician_id="RN-1", clinician_role="RN", patient_id="PT-A",
        system_level=3, system_confidence=70.0, system_uncertainty="moderate",
        override_level=2, justification_code="CLINICAL_JUDGMENT", justification_text="x",
    )
    manager.record_acceptance("RN-2", "RN", "PT-B", system_level=4)

    only_a = manager.get_audit_log(patient_id="PT-A")
    assert len(only_a) == 1
    assert only_a[0]["patient_id"] == "PT-A"

    only_overrides = manager.get_audit_log(event_type="clinician_override")
    assert len(only_overrides) == 1


# ─── Regression: the second-clinician gate must fail closed ──────────────────

def test_downgrading_a_critical_patient_requires_explicit_concurrence():
    """
    The highest-consequence control in the system.

    An earlier build tested `second_clinician_concurred is None`, so an
    unticked checkbox arrived as False, passed the guard, and recorded the
    downgrade. The control appeared to work while permitting exactly the action
    it existed to prevent.
    """
    from engine.override_audit import OverrideAuditManager

    manager = OverrideAuditManager()

    for value in (None, False):
        result = manager.record_override(
            clinician_id="DR-1", clinician_role="Emergency physician",
            patient_id="P-1", system_level=1, system_confidence=90.0,
            system_uncertainty="low", override_level=4,
            justification_code="CLINICAL_JUDGMENT",
            justification_text="Patient looks well",
            second_clinician_concurred=value,
        )
        assert result["recorded"] is False, (
            f"Downgrade of a Level 1 patient was recorded with "
            f"second_clinician_concurred={value!r}")
        assert "second" in result["error"].lower()

    approved = manager.record_override(
        clinician_id="DR-1", clinician_role="Emergency physician",
        patient_id="P-1", system_level=1, system_confidence=90.0,
        system_uncertainty="low", override_level=4,
        justification_code="CLINICAL_JUDGMENT",
        justification_text="Second clinician reviewed and concurs",
        second_clinician_concurred=True,
    )
    assert approved["recorded"] is True


def test_raising_urgency_never_requires_a_second_clinician():
    """Escalation is always allowed — friction belongs only on the risky path."""
    from engine.override_audit import OverrideAuditManager

    manager = OverrideAuditManager()
    result = manager.record_override(
        clinician_id="DR-2", clinician_role="Charge nurse",
        patient_id="P-2", system_level=4, system_confidence=70.0,
        system_uncertainty="moderate", override_level=2,
        justification_code="PHYSICAL_EXAM",
        justification_text="Looks far worse than the vitals suggest",
    )
    assert result["recorded"] is True
    assert result["record"]["override_direction"] == "upgrade"
