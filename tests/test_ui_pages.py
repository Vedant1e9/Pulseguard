"""
Smoke tests for every UI page.

These exist because of a specific failure: an evaluation output key was
renamed, three pages kept referencing the old name, and the unit tests all
passed while the application raised a KeyError in the browser. Streamlit
renderers are ordinary functions, so calling them headlessly against the real
pipeline catches exactly that class of bug — a page that is broken for a judge
clicking through it, but invisible to logic tests.
"""
import json
import os

import pytest

from engine.hazard_queue import HazardQueueManager
from engine.override_audit import OverrideAuditManager

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "evaluation", "saved_results", "evaluation_full.json")


@pytest.fixture(scope="module")
def evaluation_results():
    if not os.path.exists(RESULTS):
        pytest.skip("Evaluation results not generated yet.")
    with open(RESULTS) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def queue(pipeline):
    manager = HazardQueueManager()
    for encounter, _, _ in pipeline.patients:
        stored = pipeline.triage_results[encounter.patient_id]
        velocity = stored.get("velocity", {})
        manager.add_patient(
            patient_id=encounter.patient_id,
            triage_level=stored["result"].triage_level,
            age_group=encounter.age_group.value,
            arrival_time=encounter.arrival_time,
            confidence=stored["result"].confidence_percent,
            uncertainty=stored["result"].uncertainty_band,
            velocity_risk=(velocity.get("overall_risk", "low")
                           if velocity.get("has_trend_data") else "insufficient_data"),
        )
    return manager


def test_every_page_renders(pipeline, queue, evaluation_results):
    from ui.ai_boundary import render_ai_boundary
    from ui.audit_log import render_audit_log
    from ui.clinician_review import render_clinician_review
    from ui.dashboard import render_dashboard
    from ui.governance import render_governance
    from ui.model_evaluation_ui import render_model_evaluation
    from ui.reassessment_round import render_reassessment_round
    from ui.robustness_ui import render_robustness
    from ui.safety_frontier import render_safety_frontier
    from ui.triage_result import render_triage_result
    from ui.voice_intake import render_voice_intake
    from ui.waiting_queue import render_waiting_queue
    from ui.what_if_explorer import render_what_if

    override = OverrideAuditManager()

    pages = {
        "dashboard": lambda: render_dashboard(pipeline, queue, evaluation_results,
                                              "Triage nurse"),
        "patient_detail": lambda: render_triage_result(pipeline, "Triage nurse"),
        "waiting_queue": lambda: render_waiting_queue(queue, pipeline),
        "review_override": lambda: render_clinician_review(pipeline, override, queue),
        "what_if": lambda: render_what_if(pipeline),
        "model_performance": lambda: render_model_evaluation(evaluation_results, pipeline),
        "safety_frontier": lambda: render_safety_frontier(evaluation_results, pipeline),
        "robustness_surge": lambda: render_robustness(evaluation_results, pipeline),
        "governance": lambda: render_governance(pipeline),
        "audit_log": lambda: render_audit_log(pipeline, override),
        "spoken_handover": lambda: render_voice_intake(pipeline, queue),
        "ai_boundary": lambda: render_ai_boundary(pipeline),
        "reassessment_round": lambda: render_reassessment_round(pipeline, queue),
    }

    failures = {}
    for name, render in pages.items():
        try:
            render()
        except Exception as exc:      # noqa: BLE001 — we want every failure, not the first
            failures[name] = f"{type(exc).__name__}: {exc}"

    assert not failures, f"Pages raised while rendering: {failures}"


def test_pages_degrade_gracefully_without_evaluation_results(pipeline, queue):
    """A judge cloning the repo before running the evaluation must not see a crash."""
    from ui.model_evaluation_ui import render_model_evaluation
    from ui.robustness_ui import render_robustness
    from ui.safety_frontier import render_safety_frontier

    for render in (render_model_evaluation, render_safety_frontier, render_robustness):
        render(None, pipeline)
