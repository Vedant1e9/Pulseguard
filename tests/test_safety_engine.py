"""
Tests for the deterministic safety engine — the sole authority on triage level.

The first test in this file is the one that matters most: the engine must be
structurally incapable of making a patient less urgent. Everything else in the
system is advisory, so if that invariant breaks, no other guarantee holds.
"""
import pytest

from engine.safety_engine import SafetyEngine
from tests.helpers import make_encounter


# ─── The core invariant ──────────────────────────────────────────────────────

def test_engine_can_never_reduce_urgency(engine):
    """
    Exhaustive check across representative presentations and every possible
    model proposal. A higher level number means less urgent, so the final
    level must never exceed what the model proposed.
    """
    scenarios = [
        make_encounter(age=4, hr=190, spo2=75),
        make_encounter(age=80, hr=140, skin="pale"),
        make_encounter(consciousness="unresponsive"),
        make_encounter(bleeding="uncontrolled"),
        make_encounter(history_available=False, chief_complaint="chest pain"),
        make_encounter(),
        make_encounter(temp=None, hr=None, rr=None, spo2=None, sbp=None, dbp=None),
    ]
    for encounter in scenarios:
        for model_level in (1, 2, 3, 4, 5):
            result = engine.evaluate(model_level, encounter)
            assert result["final_level"] <= model_level, (
                f"Engine reduced urgency from {model_level} to "
                f"{result['final_level']} — violates escalate-only design")
            assert 1 <= result["final_level"] <= 5


# ─── Individual rules ────────────────────────────────────────────────────────

def test_unresponsive_patient_forces_level_1(engine):
    result = engine.evaluate(4, make_encounter(consciousness="unresponsive"))
    assert result["final_level"] == 1
    assert "UNRESPONSIVE" in result["rules_applied"]


def test_severe_hypoxaemia_escalates_to_resuscitation(engine):
    result = engine.evaluate(4, make_encounter(spo2=78))
    assert result["final_level"] == 1
    assert result["safety_status"] == "escalation_applied"


def test_uncontrolled_bleeding_escalates(engine):
    result = engine.evaluate(4, make_encounter(bleeding="uncontrolled"))
    assert result["final_level"] <= 2
    assert "UNCONTROLLED_BLEEDING" in result["rules_applied"]


def test_zero_history_high_risk_complaint_escalates(engine):
    result = engine.evaluate(
        4, make_encounter(history_available=False,
                          chief_complaint="worst headache of my life"))
    assert result["final_level"] <= 2


def test_zero_history_alone_applies_conservative_ceiling(engine):
    result = engine.evaluate(
        5, make_encounter(history_available=False, chief_complaint="sore thumb"))
    assert result["final_level"] <= 3


# ─── Age-banded thresholds ───────────────────────────────────────────────────

def test_same_heart_rate_judged_differently_by_age(engine):
    """
    A heart rate of 135 is unremarkable in a toddler and alarming in an
    85-year-old. This is the silent safety risk the brief calls out, so it is
    tested directly rather than assumed from the config.
    """
    child = engine.evaluate(3, make_encounter(age=3, hr=135, sbp=95))
    elder = engine.evaluate(3, make_encounter(age=85, hr=135, sbp=130))

    assert child["final_level"] == 3, "Normal paediatric tachycardia should not escalate"
    assert elder["final_level"] <= 2, "Geriatric tachycardia at 135 should escalate"


def test_paediatric_compensated_shock_detected_despite_normal_bp(engine):
    """The trap: blood pressure still normal, child already in shock."""
    encounter = make_encounter(age=4, hr=165, sbp=92, skin="mottled")
    result = engine.evaluate(4, encounter)
    assert result["final_level"] <= 2
    assert "PEDIATRIC_COMPENSATED_SHOCK" in result["rules_applied"]


def test_geriatric_anticoagulated_fall_escalates_despite_normal_vitals(engine):
    encounter = make_encounter(
        age=82, hr=76, rr=16, spo2=97, sbp=140,
        chief_complaint="Fell at home and hit her head",
        medications="Apixaban, bisoprolol", history_available=True)
    result = engine.evaluate(4, encounter)
    assert result["final_level"] <= 2
    assert "GERIATRIC_ANTICOAGULATED_FALL" in result["rules_applied"]


# ─── Traceability ────────────────────────────────────────────────────────────

def test_every_escalation_carries_a_citation_and_rationale(engine):
    """An escalation a clinician cannot interrogate is not acceptable output."""
    result = engine.evaluate(4, make_encounter(consciousness="unresponsive"))
    assert result["fired_rules"]
    for rule in result["fired_rules"]:
        assert rule["reason"]
        assert rule["citation"] and rule["citation"] != "not cited"
        assert rule["clinical_rationale"]
        assert rule["certainty"] in ("observed", "precautionary")


def test_decision_records_the_rule_pack_version_and_hash(engine):
    result = engine.evaluate(3, make_encounter())
    pack = result["rule_pack"]
    assert pack["pack_id"] and pack["version"]
    assert len(pack["content_hash"]) == 16


def test_engine_handles_a_patient_with_no_recorded_vitals(engine):
    """The situation the engine most needs to survive: almost nothing recorded."""
    encounter = make_encounter(temp=None, hr=None, rr=None, spo2=None,
                               sbp=None, dbp=None, pain=None)
    result = engine.evaluate(3, encounter)
    assert 1 <= result["final_level"] <= 3


# ─── Site rule packs ─────────────────────────────────────────────────────────

def test_rural_pack_escalates_earlier_than_urban_default():
    """
    The rural site tightens adult thresholds because definitive care is 90
    minutes away. A heart rate that passes at the urban site must escalate here.
    """
    from engine.rule_pack import RulePack

    urban = SafetyEngine(RulePack.load())
    rural = SafetyEngine(RulePack.load_site("rural_community"))

    encounter = make_encounter(age=45, hr=140, sbp=120, spo2=96)
    assert urban.evaluate(3, encounter)["final_level"] == 3
    assert rural.evaluate(3, encounter)["final_level"] <= 2
