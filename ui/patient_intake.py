"""PatientTriage.ai — New patient intake."""

from datetime import datetime
from typing import Optional

import streamlit as st

from data.input_schema import (
    DataSource, Measurement, MedicalHistory, PatientEncounter,
    SelfReportedSymptoms, StaffObservedCues, VitalSigns,
)
from engine.explanation import ExplanationBuilder
from ui.components import (
    confidence_row, factor_list, level_header,
    missing_info_panel, rule_pack_footer, safety_banner,
)


def _optional_number(label: str, minimum: float, maximum: float,
                     unit: str, help_text: str = "",
                     step: float = 1.0,
                     decimals: int = 0, run: int = 0) -> Optional[float]:
    """
    A vital sign field that starts EMPTY, where empty means "not measured".

    Pre-filling these with normal values is a genuine safety hazard: a nurse
    who skips a field silently submits "normal", and the system scores a
    patient on a measurement nobody took. Empty means unmeasured, and the
    system handles unmeasured — it widens the uncertainty band rather than
    inventing a number.

    An earlier version paired each field with a separate "n/a" checkbox. That
    was worse, not better: leaving a field blank and ticking n/a expressed the
    same thing two different ways, so a nurse had to stop and work out whether
    the distinction mattered. It doesn't. One empty field, one meaning.
    """
    # Explicit format: floats default to two decimal places, so a heart rate
    # entered as 128 was displayed back to the nurse as "128.00 bpm". Counts
    # get no decimals; temperature gets one.
    return st.number_input(f"{label} ({unit})", min_value=float(minimum),
                           max_value=float(maximum), value=None, step=step,
                           format=f"%.{decimals}f",
                           placeholder="not measured",
                           help=help_text, key=f"intake_{run}_val_{label}")


def render_patient_intake(pipeline, queue_manager=None):
    st.title("New patient intake")
    safety_banner()

    # The result of the last run is rendered here, at the top, before the blank
    # form below it. Streamlit reruns top-to-bottom on submit, so stashing the
    # patient id in session state is what lets the decision appear above the
    # form rather than two screens below the button that produced it.
    last_id = st.session_state.get("intake_last_patient")
    if last_id and last_id in pipeline.triage_results:
        _render_result(pipeline, last_id)
        st.markdown("---")
        st.markdown("#### Triage another patient")

    st.info(
        "Vital sign fields start **empty**, not at a normal value. A field left "
        "blank is recorded as *not measured* and widens the uncertainty band. "
        "It is never silently treated as normal.", icon="ℹ️")

    # Streamlit keeps widget state across a rerun, and deleting the keys after
    # a submit is not enough to clear a form. Versioning every key instead
    # means each submitted assessment retires its widgets and the next patient
    # starts on genuinely blank ones — without this the form reopened holding
    # the previous patient's age, vitals and complaint, which is the same
    # "silently inherited value" hazard the empty-by-default vitals exist to
    # prevent, and far worse when it is a whole assessment rather than a field.
    run = st.session_state.get("intake_run", 0)

    with st.form(f"intake_{run}"):
        st.markdown("#### Patient")
        p1, p2, p3, p4 = st.columns(4)
        patient_id = p1.text_input("Patient ID", key=f"intake_{run}_patient_id",
                                   value=f"NEW-{datetime.now():%H%M%S}")
        age = p2.number_input("Age (years)", 0, 120, value=None,
                              placeholder="required", key=f"intake_{run}_age")
        sex = p3.selectbox("Sex", ["F", "M", "Other"], key=f"intake_{run}_sex")
        arrival_mode = p4.selectbox("Arrival", ["Walk-in", "Ambulance"],
                                    key=f"intake_{run}_arrival")

        st.markdown("---")
        st.markdown("#### Vital signs")
        v1, v2, v3 = st.columns(3)
        with v1:
            temperature = _optional_number("Temperature", 30.0, 43.0, "°C",
                                           "Age-banded thresholds apply.", 0.1,
                                           decimals=1, run=run)
            heart_rate = _optional_number("Heart rate", 0, 300, "bpm",
                                          "Compared against normal for this age.", run=run)
        with v2:
            respiratory_rate = _optional_number("Respiratory rate", 0, 80, "/min", run=run)
            spo2 = _optional_number("Oxygen saturation", 50, 100, "%",
                                    "Below the age-banded floor triggers escalation.", run=run)
        with v3:
            systolic = _optional_number("Systolic BP", 0, 300, "mmHg", run=run)
            diastolic = _optional_number("Diastolic BP", 0, 200, "mmHg", run=run)

        pain_col, pain_na_col = st.columns([4, 1])
        with pain_col:
            pain = st.slider("Pain score (patient-reported)", 0, 10, 0,
                             key=f"intake_{run}_pain")
        with pain_na_col:
            st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
            # A slider has no empty state, so pain is the one vital that still
            # needs an explicit "not asked" — otherwise 0 would be recorded as
            # "no pain" for every patient nobody got round to asking.
            pain_not_asked = st.checkbox("Not asked", key=f"intake_{run}_pain_na",
                                         help="A slider cannot be left blank. "
                                              "Tick this if pain was not assessed, "
                                              "so it is not recorded as zero.")

        st.markdown("---")
        st.markdown("#### Presenting complaint")
        complaint = st.text_input(
            "Chief complaint", key=f"intake_{run}_complaint",
            placeholder="e.g. Chest pain and shortness of breath for two hours")
        symptoms = st.text_area(
            "Additional detail / nursing note", height=80, key=f"intake_{run}_note",
            placeholder="Onset, duration, associated symptoms, what the patient "
                        "or family reports…")

        st.markdown("---")
        st.markdown("#### Bedside observations")
        st.caption(
            "Your direct assessment. These drive several deterministic safety "
            "rules that vital signs alone cannot trigger.")
        o1, o2, o3 = st.columns(3)
        with o1:
            consciousness = st.selectbox(
                "Consciousness (AVPU)",
                ["alert", "verbal", "pain", "unresponsive"],
                key=f"intake_{run}_consciousness",
                help="Alert / responds to Voice / responds to Pain / Unresponsive")
            breathing = st.selectbox("Breathing difficulty",
                                     ["none", "mild", "moderate", "severe"],
                                     key=f"intake_{run}_breathing")
        with o2:
            distress = st.selectbox("Visible distress",
                                    ["none", "mild", "moderate", "severe"],
                                    key=f"intake_{run}_distress")
            mobility = st.selectbox("Mobility",
                                    ["ambulatory", "assisted", "immobile"],
                                    key=f"intake_{run}_mobility")
        with o3:
            bleeding = st.selectbox("Bleeding",
                                    ["none", "controlled", "uncontrolled"],
                                    key=f"intake_{run}_bleeding")
            skin = st.selectbox("Skin appearance",
                                ["normal", "pale", "flushed", "cyanotic",
                                 "mottled", "diaphoretic"],
                                key=f"intake_{run}_skin")

        st.markdown("---")
        st.markdown("#### Medical history")
        history_available = st.toggle("A medical record is available", value=False,
                                      key=f"intake_{run}_history_available")
        if history_available:
            h1, h2, h3 = st.columns(3)
            conditions = h1.text_input("Known conditions",
                                       key=f"intake_{run}_conditions",
                                       placeholder="e.g. Atrial fibrillation, COPD")
            medications = h2.text_input("Current medications",
                                        key=f"intake_{run}_medications",
                                        placeholder="e.g. Apixaban, metformin")
            allergies = h3.text_input("Allergies", key=f"intake_{run}_allergies",
                                      placeholder="e.g. Penicillin")
        else:
            st.warning(
                "⚠️ No record available. The system will apply its conservative "
                "ceiling. A first-time patient with a high-risk complaint is "
                "escalated rather than assumed well.")
            conditions = medications = allergies = ""

        submitted = st.form_submit_button("🩺 Run triage", type="primary",
                                          use_container_width=True)

    if not submitted:
        return

    if age is None:
        st.error("Age is required. Every threshold in this system is age-banded.")
        return

    def m(value, unit="", source=DataSource.DEVICE_MEASURED):
        if value is None:
            return None
        return Measurement(value=value, unit=unit, source=source,
                           timestamp=datetime.now())

    encounter = PatientEncounter(
        patient_id=patient_id, age=int(age), sex=sex, arrival_time=datetime.now(),
        vitals=VitalSigns(
            temperature=m(temperature, "°C"),
            heart_rate=m(heart_rate, "bpm"),
            respiratory_rate=m(respiratory_rate, "breaths/min"),
            spo2=m(spo2, "%"),
            systolic_bp=m(systolic, "mmHg"),
            diastolic_bp=m(diastolic, "mmHg"),
            pain_score=(None if pain_not_asked
                        else m(pain, "/10", DataSource.PATIENT_REPORTED)),
        ),
        symptoms=SelfReportedSymptoms(
            chief_complaint=m(complaint or None, source=DataSource.PATIENT_REPORTED),
            symptoms=m(symptoms or None, source=DataSource.PATIENT_REPORTED),
        ),
        history=MedicalHistory(
            history_available=history_available,
            known_conditions=m(conditions or None, source=DataSource.EHR_IMPORTED),
            medications=m(medications or None, source=DataSource.EHR_IMPORTED),
            allergies=m(allergies or None, source=DataSource.EHR_IMPORTED),
        ),
        # Arrival mode is a real model feature — patients brought in by
        # ambulance are a materially sicker population. It was being collected
        # on the form and then discarded, which is worse than not asking: it
        # costs the nurse a click and buys nothing.
        context={"arrival_by_ambulance": 1.0 if arrival_mode == "Ambulance" else 0.0},
        staff_cues=StaffObservedCues(
            visible_distress=m(distress, source=DataSource.NURSE_OBSERVED),
            breathing_difficulty=m(breathing, source=DataSource.NURSE_OBSERVED),
            consciousness=m(consciousness, source=DataSource.NURSE_OBSERVED),
            mobility=m(mobility, source=DataSource.NURSE_OBSERVED),
            bleeding=m(bleeding, source=DataSource.NURSE_OBSERVED),
            skin_appearance=m(skin, source=DataSource.NURSE_OBSERVED),
        ),
    )

    with st.spinner("Running triage …"):
        result = pipeline.triage_patient(encounter)

    # Join the live waiting queue. The pipeline puts the patient on the board;
    # the queue is a separate structure and has to be told as well, or the
    # closing line below would be half true.
    if queue_manager is not None:
        stored = pipeline.triage_results[patient_id]
        velocity = stored.get("velocity", {})
        queue_manager.add_patient(
            patient_id=patient_id,
            triage_level=result.triage_level,
            age_group=encounter.age_group.value,
            arrival_time=encounter.arrival_time,
            confidence=result.confidence_percent,
            uncertainty=result.uncertainty_band,
            velocity_risk=(velocity.get("overall_risk", "low")
                           if velocity.get("has_trend_data")
                           else "insufficient_data"),
        )

    st.session_state["intake_run"] = run + 1

    st.session_state["intake_last_patient"] = patient_id
    st.rerun()


def _render_result(pipeline, patient_id: str):
    """The decision for the patient just entered, rendered above the form."""
    stored = pipeline.triage_results[patient_id]
    result = stored["result"]
    trace = stored["explanation"]

    st.success(f"Triage complete for **{patient_id}**.", icon="✅")
    level_header(result.triage_level, result.recommended_action,
                 escalated=(result.safety_status == "escalation_applied"))
    confidence_row(result, stored["model_output"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Why this level")
        st.caption(f"Decided by: **{trace['decided_by']}**")
        factor_list(trace["factors"], limit=5)
        if trace.get("counterfactual"):
            st.info(f"🔄 {trace['counterfactual']['statement']}")
    with right:
        st.markdown("#### What we don't know")
        missing_info_panel(trace["not_recorded"],
                           result.recommended_followup_question)

    nurse_view = ExplanationBuilder.for_nurse(trace)
    if nurse_view.get("counterfactual"):
        st.caption(nurse_view["counterfactual"])

    st.caption(f"Scored in {stored['latency_ms']:.0f} ms. "
               f"This patient is now on the board and in the waiting queue.")
    rule_pack_footer(stored["safety"].get("rule_pack", {}))
