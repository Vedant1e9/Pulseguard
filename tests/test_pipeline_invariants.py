"""
End-to-end invariants for the triage pipeline.

These are the properties that must hold for every patient, not the ones that
happen to hold for the demo board.
"""
import numpy as np
import pytest

from tests.helpers import make_encounter


def test_every_patient_gets_a_valid_level_and_confidence(pipeline):
    """No score is ever returned without an uncertainty indicator."""
    for encounter, _, _ in pipeline.patients:
        result = pipeline.triage_results[encounter.patient_id]["result"]
        assert result.triage_level in (1, 2, 3, 4, 5)
        assert 0 <= result.confidence_percent <= 100
        assert result.uncertainty_band in ("low", "moderate", "high")
        assert result.recommended_action


def test_missing_vitals_never_become_zero(pipeline):
    """
    A missing measurement must stay missing all the way to the model.

    Imputing zero would make an unrecorded oxygen saturation indistinguishable
    from a saturation of 0% — the exact silent failure this pipeline is built
    to prevent.
    """
    from engine.triage_pipeline import encounter_to_record

    encounter = make_encounter(spo2=None, hr=None, sbp=None)
    record = encounter_to_record(encounter)
    assert record["spo2"] is None
    assert record["heart_rate"] is None
    assert record["systolic_bp"] is None


def test_explanation_names_the_rule_that_decided_the_level(pipeline):
    """
    The regression this guards: an earlier build escalated a patient to Level 1
    for severe hypoxaemia and then listed "patient age" as the top factor,
    because explanations came from global feature importance rather than from
    the decision path.
    """
    encounter = make_encounter(patient_id="TEST-EXPLAIN", spo2=72,
                               consciousness="unresponsive")
    pipeline.triage_patient(encounter)
    trace = pipeline.triage_results["TEST-EXPLAIN"]["explanation"]

    assert trace["factors"], "Explanation produced no factors"
    top = trace["factors"][0]
    assert top["source"] == "safety_rule", (
        "The rule that set the level must be the first factor shown")
    assert top["decisive"] is True


def test_explanation_never_presents_an_unrecorded_value_as_evidence(pipeline):
    """Unmeasured vitals belong in 'not recorded', never in the factor list."""
    encounter = make_encounter(patient_id="TEST-MISSING", spo2=None,
                               sbp=None, temp=None)
    pipeline.triage_patient(encounter)
    stored = pipeline.triage_results["TEST-MISSING"]
    trace = stored["explanation"]

    assert trace["not_recorded"], "Missing observations should be listed"
    for factor in trace["factors"]:
        if factor["source"] == "model":
            feature = factor.get("feature")
            assert stored["record"].get(feature) is not None, (
                f"Explanation cited '{feature}', which was never recorded")


def test_confidence_is_high_when_an_observed_rule_decides(pipeline):
    """
    Regression: confidence once read 5% on a correctly-identified Level 1
    patient, because it reported P(exact level match) while a categorical rule
    had set the level. An unresponsive patient is a Level 1 with certainty.
    """
    encounter = make_encounter(patient_id="TEST-CONF", consciousness="unresponsive",
                               spo2=70, hr=35)
    result = pipeline.triage_patient(encounter)
    assert result.triage_level == 1
    assert result.confidence_percent >= 80


def test_conformal_set_is_a_contiguous_severity_range(pipeline):
    """
    Triage levels are ordinal. A set like {1, 2, 4, 5} — "critical or nearly
    non-urgent, but definitely not in between" — is not something a clinician
    can act on.
    """
    for encounter, _, _ in pipeline.patients:
        pred_set = pipeline.triage_results[encounter.patient_id]["model_output"]["conformal_set"]
        assert pred_set
        assert pred_set == list(range(min(pred_set), max(pred_set) + 1))


def test_override_changes_the_patient_level(pipeline):
    """
    An override that is logged but does not change the patient's actual
    priority is worse than no override at all: the clinician believes they have
    acted while the department carries on with the machine's ranking.
    """
    encounter = make_encounter(patient_id="TEST-OVERRIDE")
    result = pipeline.triage_patient(encounter)
    original = result.triage_level
    target = 1 if original != 1 else 3

    assert pipeline.apply_override("TEST-OVERRIDE", target) is True
    assert pipeline.triage_results["TEST-OVERRIDE"]["result"].triage_level == target
    assert pipeline.triage_results["TEST-OVERRIDE"]["system_level_before_override"] == original


def test_audit_entry_written_for_every_decision(pipeline):
    encounter = make_encounter(patient_id="TEST-AUDIT")
    pipeline.triage_patient(encounter)
    entries = [e for e in pipeline.audit_log if e.patient_id == "TEST-AUDIT"]
    assert entries
    details = entries[-1].details
    for key in ("final_level", "model_proposed_level", "rules_fired",
                "rule_pack_version", "rule_pack_hash", "model_name", "latency_ms"):
        assert key in details, f"Audit record missing '{key}'"


def test_latency_is_within_a_clinically_usable_budget(pipeline):
    """Triage happens under time pressure; the assistant cannot be the delay."""
    latencies = [v["latency_ms"] for v in pipeline.triage_results.values()]
    assert np.median(latencies) < 250, (
        f"Median latency {np.median(latencies):.0f} ms is too slow for triage")


def test_pipeline_survives_a_nearly_empty_record(pipeline):
    encounter = make_encounter(
        patient_id="TEST-SPARSE", temp=None, hr=None, rr=None, spo2=None,
        sbp=None, dbp=None, pain=None, chief_complaint="", history_available=False)
    result = pipeline.triage_patient(encounter)
    assert result.triage_level in (1, 2, 3, 4, 5)
    # With nothing known, the system must not route the patient to the back
    assert result.triage_level <= 3


# ─── Regression: the demo must run the system the evaluation validated ──────

def test_encounter_round_trip_preserves_every_model_feature():
    """
    Scoring a patient via a PatientEncounter must match scoring the source
    record directly.

    This regression cost 15% of decisions. Building an encounter from a record
    silently dropped 25 features — arrival mode and every coded chronic
    condition — so the demo board was running a measurably weaker system than
    the one the evaluation reported on, with nothing to indicate it.
    """
    import numpy as np

    from data.demo_cohort import load_test_fold, nhamcs_row_to_encounter
    from engine.triage_pipeline import encounter_to_record
    from models.decision_policy import expected_cost_decision
    from models.triage_model import _align_proba, load_bundle

    df = load_test_fold().sample(200, random_state=11)
    bundle = load_bundle("saved_models/triage_bundle.joblib")
    model = bundle.calibrated_classifier or bundle.classifier

    def decide(records):
        proba = _align_proba(model.predict_proba(bundle.build_matrix(records)), model)
        return np.array([expected_cost_decision(p, bundle.cost_matrix)["decision"]
                         for p in proba])

    direct = decide(df.to_dict("records"))
    round_tripped = decide([encounter_to_record(nhamcs_row_to_encounter(row))
                            for _, row in df.iterrows()])

    agreement = float((direct == round_tripped).mean())
    assert agreement == 1.0, (
        f"Encounter round-trip changed {(1 - agreement):.1%} of decisions, "
        f"the demo is not running the validated pipeline")


def test_free_text_history_maps_onto_coded_conditions():
    """A nurse typing 'COPD, heart failure' must not score as no history."""
    from engine.triage_pipeline import _code_conditions

    coded = _code_conditions("COPD, heart failure and type 2 diabetes")
    assert coded["cond_copd"] == 1.0
    assert coded["cond_chf"] == 1.0
    assert coded["cond_diabtyp2"] == 1.0
    assert coded["cond_diabtyp0"] == 1.0, "A specific diabetes type implies the generic flag"

    # Short abbreviations must not match inside unrelated words
    noise = _code_conditions("vomiting, generalised pain, feels unwell")
    assert noise["cond_cad"] == 0.0, "'mi' matched inside 'vomiting'"
    assert noise["cond_hpe"] == 0.0, "'pe' matched inside 'pain'"
