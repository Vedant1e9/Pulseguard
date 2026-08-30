"""
PatientTriage.ai — Clinician Override & Audit Trail
Captures clinician overrides with structured justification
and maintains a tamper-evident audit log.

Tamper-evidence mechanism: each audit entry is chained to the previous
entry via SHA-256 (entry_hash = sha256(prev_hash + canonical_json(entry))),
the same pattern used by append-only ledgers. Editing or deleting any past
entry breaks every hash after it, which verify_integrity() detects. Entries
are also written to an append-only JSONL file as they're recorded — the file
is opened in append mode only, never rewritten, and never used to load state
back in. This is prototype-scoped tamper-evidence (the hash chain resets
each process run); a production deployment would persist the chain head
across restarts and write to a WORM store or database instead of a local file.
"""

from datetime import datetime
from typing import Dict, List, Optional
import hashlib
import json
import os


class OverrideAuditManager:
    """
    Manages clinician overrides and audit trail.
    Audit log entries are hash-chained and append-only once written.
    """

    GENESIS_HASH = "0" * 64
    DEFAULT_LOG_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "audit_log.jsonl",
    )

    # Controlled vocabulary for override justifications
    JUSTIFICATION_CODES = {
        "CLINICAL_INCONSISTENT": "Clinical presentation inconsistent with model assessment",
        "ADDITIONAL_HISTORY": "Additional history obtained after initial assessment",
        "PHYSICAL_EXAM": "Physical examination findings not captured by system",
        "CLINICAL_JUDGMENT": "Clinical judgment based on experience with similar presentations",
        "PATIENT_PREFERENCE": "Patient preference or request",
        "RESOURCE_CONSIDERATION": "Resource availability consideration",
        "FAMILY_INPUT": "Additional information from family member or caregiver",
        "LAB_RESULTS": "Laboratory or imaging results obtained",
        "SPECIALIST_CONSULT": "Specialist consultation recommendation",
        "KNOWN_PATIENT": "Known patient with established baseline",
        "OTHER": "Other (see free text)",
    }

    def __init__(self, log_file_path: Optional[str] = None):
        self.overrides = []  # List of override records
        self.audit_log = []  # Full audit trail (hash-chained)
        self._chain_head = self.GENESIS_HASH
        self.log_file_path = log_file_path or self.DEFAULT_LOG_FILE
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

    def _append_chained(self, entry: Dict) -> Dict:
        """
        Append an entry to the tamper-evident hash chain and the
        append-only log file. Returns the chained entry (with
        prev_hash / entry_hash attached) that was actually stored.
        """
        prev_hash = self._chain_head
        payload = json.dumps(entry, sort_keys=True, default=str)
        entry_hash = hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()
        chained_entry = {**entry, "prev_hash": prev_hash, "entry_hash": entry_hash}

        self._chain_head = entry_hash
        self.audit_log.append(chained_entry)

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(chained_entry, default=str) + "\n")

        return chained_entry

    def verify_integrity(self) -> Dict:
        """
        Recompute the hash chain over the in-memory audit log and report
        whether it is intact. Returns tampered_at_index if any entry's
        stored hash doesn't match what its content + prev_hash produce.
        """
        expected_prev = self.GENESIS_HASH
        for i, entry in enumerate(self.audit_log):
            stored_prev = entry.get("prev_hash")
            stored_hash = entry.get("entry_hash")
            body = {k: v for k, v in entry.items() if k not in ("prev_hash", "entry_hash")}
            payload = json.dumps(body, sort_keys=True, default=str)
            recomputed = hashlib.sha256((expected_prev + payload).encode("utf-8")).hexdigest()

            if stored_prev != expected_prev or stored_hash != recomputed:
                return {
                    "intact": False,
                    "tampered_at_index": i,
                    "total_entries": len(self.audit_log),
                }
            expected_prev = stored_hash

        return {
            "intact": True,
            "total_entries": len(self.audit_log),
            "chain_head": expected_prev,
        }

    def record_override(self, clinician_id: str, clinician_role: str,
                        patient_id: str,
                        system_level: int, system_confidence: float,
                        system_uncertainty: str,
                        override_level: int,
                        justification_code: str,
                        justification_text: str,
                        second_clinician_concurred: Optional[bool] = None) -> Dict:
        """
        Record a clinician override.
        Returns the immutable override record.
        """
        # Determine direction
        if override_level < system_level:
            direction = "upgrade"  # System under-triaged
        elif override_level > system_level:
            direction = "downgrade"  # System over-triaged
        else:
            direction = "confirmed"  # Clinician agrees

        # Validate: downgrades of L1–L2 require a second clinician's concurrence.
        #
        # The check is on truthiness, not on `is None`. An earlier version
        # tested `is None`, which meant an unticked "second clinician concurs"
        # checkbox arrived as False, passed the guard, and recorded the
        # downgrade — the control appeared to work while permitting exactly the
        # action it existed to prevent. This is the highest-consequence gate in
        # the system, so it fails closed.
        if direction == "downgrade" and system_level <= 2:
            if not second_clinician_concurred:
                return {
                    "error": ("Downgrading a Level 1 or 2 patient requires a second "
                              "clinician to concur. Nothing has been recorded."),
                    "recorded": False,
                }

        record = {
            "timestamp": datetime.now().isoformat(),
            "clinician_id": clinician_id,
            "clinician_role": clinician_role,
            "patient_id": patient_id,
            "system_recommendation": system_level,
            "system_confidence": system_confidence,
            "system_uncertainty": system_uncertainty,
            "override_level": override_level,
            "override_direction": direction,
            "justification_code": justification_code,
            "justification_description": self.JUSTIFICATION_CODES.get(justification_code, justification_code),
            "justification_text": justification_text,
            "second_clinician_concurred": second_clinician_concurred,
        }

        self.overrides.append(record)

        # Add to the hash-chained, append-only audit log. `details` is a
        # copy of `record`, never the same object — record must stay
        # untouched after chaining, or a later mutation would silently
        # change what the stored hash was computed over.
        self._append_chained({
            "timestamp": datetime.now().isoformat(),
            "event_type": "clinician_override",
            "patient_id": patient_id,
            "user_id": clinician_id,
            "details": dict(record),
        })

        return {"recorded": True, "record": record}

    def record_acceptance(self, clinician_id: str, clinician_role: str,
                          patient_id: str, system_level: int) -> Dict:
        """Record that a clinician accepted the system recommendation."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "triage_accepted",
            "clinician_id": clinician_id,
            "clinician_role": clinician_role,
            "patient_id": patient_id,
            "accepted_level": system_level,
        }

        self._append_chained({
            "timestamp": datetime.now().isoformat(),
            "event_type": "triage_accepted",
            "patient_id": patient_id,
            "user_id": clinician_id,
            "details": dict(record),
        })

        return {"recorded": True, "record": record}

    def get_overrides(self, patient_id: Optional[str] = None) -> List[Dict]:
        """Get override records, optionally filtered by patient."""
        if patient_id:
            return [o for o in self.overrides if o["patient_id"] == patient_id]
        return self.overrides

    def get_override_stats(self) -> Dict:
        """Get aggregate override statistics."""
        if not self.overrides:
            return {
                "total_overrides": 0,
                "upgrade_count": 0,
                "downgrade_count": 0,
                "confirmed_count": 0,
            }

        directions = [o["override_direction"] for o in self.overrides]
        codes = [o["justification_code"] for o in self.overrides]

        return {
            "total_overrides": len(self.overrides),
            "upgrade_count": directions.count("upgrade"),
            "downgrade_count": directions.count("downgrade"),
            "confirmed_count": directions.count("confirmed"),
            "most_common_justification": max(set(codes), key=codes.count) if codes else None,
            "justification_distribution": {c: codes.count(c) for c in set(codes)},
        }

    def get_audit_log(self, patient_id: Optional[str] = None,
                      event_type: Optional[str] = None) -> List[Dict]:
        """Get audit log entries with optional filters."""
        entries = self.audit_log
        if patient_id:
            entries = [e for e in entries if e.get("patient_id") == patient_id]
        if event_type:
            entries = [e for e in entries if e.get("event_type") == event_type]
        return entries

    def add_audit_entry(self, event_type: str, patient_id: str,
                        user_id: str, details: Dict):
        """Add a generic audit log entry."""
        self._append_chained({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "patient_id": patient_id,
            "user_id": user_id,
            "details": dict(details),
        })
