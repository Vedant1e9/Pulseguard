"""PatientTriage.ai — Clinician review and override."""

import pandas as pd
import streamlit as st

from engine.override_audit import OverrideAuditManager
from ui.components import (
    LEVEL_NAMES, confidence_row, factor_list, level_header, safety_banner,
)


def render_clinician_review(pipeline, override_manager, queue_manager=None):
    st.title("Clinician review & override")
    safety_banner()

    st.markdown(
        "The system's recommendation is advisory. A licensed clinician can "
        "accept it or replace it, and either way the decision is recorded in a "
        "tamper-evident log. **An override changes the patient's actual "
        "priority and re-orders the queue**. It is a clinical action, not an "
        "annotation."
    )

    ids = [enc.patient_id for enc, _, _ in pipeline.patients
           if enc.patient_id in pipeline.triage_results]
    selected = st.selectbox("Select patient", ids)
    stored = pipeline.triage_results[selected]
    result = stored["result"]
    encounter = pipeline.patient_encounters[selected][0]

    level_header(result.triage_level, result.recommended_action,
                 escalated=(result.safety_status == "escalation_applied"),
                 overridden=stored.get("overridden", False))

    st.markdown(f"**{encounter.age}y {encounter.sex}** · "
                f"{encounter.symptoms.get_chief_complaint_text() or 'complaint not coded'}")
    confidence_row(result, stored["model_output"])

    st.markdown("**System reasoning**")
    factor_list(stored["explanation"]["factors"], limit=3)

    st.markdown("---")

    # ── Clinician identity ──
    id_col, role_col = st.columns(2)
    clinician_id = id_col.text_input("Clinician ID", value="DR-4471",
                                     help="Recorded against the decision. In "
                                          "deployment this comes from the "
                                          "authenticated session, not a text box.")
    clinician_role = role_col.selectbox(
        "Role", ["Emergency physician", "Senior registrar", "Nurse practitioner",
                 "Charge nurse"])

    accept_col, override_col = st.columns(2)

    with accept_col:
        st.markdown("#### Accept")
        st.caption("Confirms the recommendation and records your agreement.")
        if st.button("✅ Accept recommendation", type="primary",
                     use_container_width=True):
            override_manager.record_acceptance(
                clinician_id=clinician_id, clinician_role=clinician_role,
                patient_id=selected, system_level=result.triage_level)
            st.success(f"Level {result.triage_level} accepted and logged.")

    with override_col:
        st.markdown("#### Override")
        new_level = st.selectbox(
            "Set triage level", [1, 2, 3, 4, 5],
            index=result.triage_level - 1,
            format_func=lambda lvl: f"Level {lvl}: {LEVEL_NAMES[lvl]}")

        justification_code = st.selectbox(
            "Reason", list(OverrideAuditManager.JUSTIFICATION_CODES.keys()),
            format_func=lambda c: OverrideAuditManager.JUSTIFICATION_CODES[c])
        justification_text = st.text_area(
            "Clinical note",
            placeholder="What did you see that the system did not?",
            height=80)

        is_downgrade = new_level > result.triage_level
        needs_second = is_downgrade and result.triage_level <= 2
        second_clinician = None

        if needs_second:
            st.warning(
                "⚠️ Reducing the urgency of a Level 1 or 2 patient requires a "
                "second clinician to concur. This is the single highest-risk "
                "action in the system.")
            second_clinician = st.checkbox(
                "A second clinician has reviewed this patient and concurs")

        if st.button("⚠️ Submit override", use_container_width=True):
            if new_level == result.triage_level:
                st.info("That matches the current level. Use Accept instead.")
            elif not justification_text.strip():
                st.error("A clinical note is required for every override.")
            else:
                record = override_manager.record_override(
                    clinician_id=clinician_id,
                    clinician_role=clinician_role,
                    patient_id=selected,
                    system_level=result.triage_level,
                    system_confidence=result.confidence_percent,
                    system_uncertainty=result.uncertainty_band,
                    override_level=new_level,
                    justification_code=justification_code,
                    justification_text=justification_text,
                    second_clinician_concurred=second_clinician,
                )
                if record.get("recorded"):
                    previous = result.triage_level
                    pipeline.apply_override(selected, new_level)
                    if queue_manager is not None:
                        queue_manager.update_patient(selected, triage_level=new_level)
                    st.success(
                        f"Override recorded and applied: Level {previous} → "
                        f"**{new_level}**. The queue has been re-ordered.")
                    st.rerun()
                else:
                    st.error(record.get("error", "Override could not be recorded."))

    # ── Override history and what it is for ──
    st.markdown("---")
    st.subheader("Override history")

    overrides = override_manager.get_overrides()
    if not overrides:
        st.info("No overrides recorded in this session.")
        return

    st.dataframe(pd.DataFrame([{
        "Time": o["timestamp"][11:19],
        "Patient": o["patient_id"],
        "System": o["system_recommendation"],
        "Clinician": o["override_level"],
        "Direction": o["override_direction"],
        "Reason": o["justification_code"],
        "2nd clinician": o.get("second_clinician_concurred", "n/a"),
    } for o in overrides]), use_container_width=True, hide_index=True)

    stats = override_manager.get_override_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Overrides", stats["total_overrides"])
    c2.metric("Raised urgency", stats["upgrade_count"])
    c3.metric("Lowered urgency", stats["downgrade_count"])

    st.info(
        "**These records are the system's learning signal.** Overrides are "
        "reviewed in aggregate at the weekly triage huddle: a reason code that "
        "keeps recurring points at a rule that needs retuning, and a cluster of "
        "upgrades on one presentation type points at a gap in the model. The "
        "controlled vocabulary exists so that analysis is possible at all. "
        "Free text alone cannot be counted.",
        icon="🔁")
