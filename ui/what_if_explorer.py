"""PulseGuard — What-if explorer: test the system's boundaries."""

from copy import deepcopy

import pandas as pd
import streamlit as st

from data.input_schema import DataSource, Measurement
from ui.components import LEVEL_NAMES, factor_list, level_badge, safety_banner


def render_what_if(pipeline):
    st.title("What-if explorer")
    safety_banner()

    st.markdown(
        "Change one value and watch the recommendation move. This exists to "
        "build **earned** trust rather than blind trust: a nurse who has "
        "probed where the system is sensitive and where it is stable knows "
        "when to lean on it and when to overrule it, and that judgement is "
        "worth far more than any accuracy figure on a slide."
    )

    ids = [enc.patient_id for enc, _, _ in pipeline.patients
           if enc.patient_id in pipeline.triage_results]
    selected = st.selectbox("Select patient", ids)

    encounter, reference, description = pipeline.patient_encounters[selected]
    stored = pipeline.triage_results[selected]
    original = stored["result"]
    vitals = encounter.vitals.to_feature_dict()

    st.markdown(f"**Current:** {level_badge(original.triage_level)} · "
                f"{original.confidence_percent:.0f}% confident not under-triaged · "
                f"{original.uncertainty_band} uncertainty")

    st.markdown("---")
    st.subheader("Adjust")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # One decimal: a float slider defaults to two, and "38.40 °C" is not
        # how any thermometer or clinician writes a temperature.
        temperature = st.slider("Temperature (°C)", 33.0, 43.0,
                                float(vitals.get("temperature") or 37.0), 0.1,
                                format="%.1f")
        heart_rate = st.slider("Heart rate (bpm)", 20, 220,
                               int(vitals.get("heart_rate") or 80))
    with c2:
        respiratory_rate = st.slider("Respiratory rate (/min)", 4, 60,
                                     int(vitals.get("respiratory_rate") or 16))
        spo2 = st.slider("Oxygen saturation (%)", 50, 100,
                         int(vitals.get("spo2") or 98))
    with c3:
        systolic = st.slider("Systolic BP (mmHg)", 40, 260,
                             int(vitals.get("systolic_bp") or 120))
        diastolic = st.slider("Diastolic BP (mmHg)", 20, 160,
                              int(vitals.get("diastolic_bp") or 76))
    with c4:
        pain = st.slider("Pain score", 0, 10, int(vitals.get("pain_score") or 0))
        consciousness = st.selectbox(
            "Consciousness (AVPU)", ["alert", "verbal", "pain", "unresponsive"],
            index=["alert", "verbal", "pain", "unresponsive"].index(
                str(encounter.staff_cues.consciousness.value).lower()
                if encounter.staff_cues.consciousness else "alert"))
        history_available = st.checkbox("Medical history available",
                                        value=encounter.history.history_available)

    modified = deepcopy(encounter)
    modified.patient_id = f"{selected}-whatif"

    def m(value, unit="", source=DataSource.DEVICE_MEASURED):
        return Measurement(value=value, unit=unit, source=source)

    modified.vitals.temperature = m(temperature, "°C")
    modified.vitals.heart_rate = m(heart_rate, "bpm")
    modified.vitals.respiratory_rate = m(respiratory_rate, "breaths/min")
    modified.vitals.spo2 = m(spo2, "%")
    modified.vitals.systolic_bp = m(systolic, "mmHg")
    modified.vitals.diastolic_bp = m(diastolic, "mmHg")
    modified.vitals.pain_score = m(pain, "/10", DataSource.PATIENT_REPORTED)
    modified.staff_cues.consciousness = m(consciousness, source=DataSource.NURSE_OBSERVED)
    modified.history.history_available = history_available

    changes = []
    for label, before, after, fmt in [
        ("Temperature", vitals.get("temperature"), temperature, "{:.1f}"),
        ("Heart rate", vitals.get("heart_rate"), heart_rate, "{:.0f}"),
        ("Respiratory rate", vitals.get("respiratory_rate"), respiratory_rate, "{:.0f}"),
        ("Oxygen saturation", vitals.get("spo2"), spo2, "{:.0f}"),
        ("Systolic BP", vitals.get("systolic_bp"), systolic, "{:.0f}"),
        ("Diastolic BP", vitals.get("diastolic_bp"), diastolic, "{:.0f}"),
        ("Pain score", vitals.get("pain_score"), pain, "{:.0f}"),
    ]:
        if before is None:
            if after is not None:
                changes.append(f"{label}: not measured → {fmt.format(after)}")
        elif abs(float(before) - float(after)) > 1e-6:
            changes.append(f"{label}: {fmt.format(float(before))} → {fmt.format(after)}")

    original_avpu = (str(encounter.staff_cues.consciousness.value).lower()
                     if encounter.staff_cues.consciousness else "alert")
    if consciousness != original_avpu:
        changes.append(f"Consciousness: {original_avpu} → {consciousness}")
    if history_available != encounter.history.history_available:
        changes.append(
            f"History: {'available' if encounter.history.history_available else 'none'}"
            f" → {'available' if history_available else 'none'}")

    if not changes:
        st.info("👆 Move any control above to see how the recommendation responds.")
        return

    with st.spinner("Re-triaging …"):
        new_result = pipeline.triage_patient(modified, store=True)
    new_stored = pipeline.triage_results[modified.patient_id]

    st.markdown("---")
    st.subheader("Result")

    col_before, col_arrow, col_after = st.columns([2, 1, 2])
    with col_before:
        st.markdown("**Original**")
        st.markdown(level_badge(original.triage_level))
        st.caption(f"{original.confidence_percent:.0f}% not under-triaged · "
                   f"{original.uncertainty_band} uncertainty")
    with col_arrow:
        delta = original.triage_level - new_result.triage_level
        st.markdown("<div style='text-align:center;font-size:34px;margin-top:22px;'>"
                    + ("⬆️" if delta > 0 else "⬇️" if delta < 0 else "➡️")
                    + "</div>", unsafe_allow_html=True)
    with col_after:
        st.markdown("**With your changes**")
        st.markdown(level_badge(new_result.triage_level))
        st.caption(f"{new_result.confidence_percent:.0f}% not under-triaged · "
                   f"{new_result.uncertainty_band} uncertainty")

    st.markdown("**What you changed**")
    for change in changes:
        st.markdown(f"- 🔄 {change}")

    if new_result.triage_level != original.triage_level:
        direction = ("more urgent" if new_result.triage_level < original.triage_level
                     else "less urgent")
        st.success(f"The recommendation moved **{direction}**: Level "
                   f"{original.triage_level} → **{new_result.triage_level}**.")
    else:
        st.info(
            f"The recommendation held at Level {original.triage_level}. Stability "
            f"is information too. It tells you this decision does not hinge on "
            f"the value you just moved.")

    if new_result.safety_status == "escalation_applied":
        st.warning(f"⚖️ A safety rule fired: {new_result.safety_reason}")

    st.markdown("**Why, now**")
    factor_list(new_stored["explanation"]["factors"], limit=4)

    # Keep the scratch patient out of the real board and queue
    pipeline.triage_results.pop(modified.patient_id, None)
