"""PatientTriage.ai — Patient detail view with the full decision trace."""

import pandas as pd
import streamlit as st

from engine.explanation import ExplanationBuilder
from ui.components import (
    LEVEL_NAMES, confidence_row, factor_list, level_header,
    missing_info_panel, rule_pack_footer, safety_banner, source_tag,
)


def render_triage_result(pipeline, role: str = "Triage nurse"):
    st.title("Patient detail")
    safety_banner()

    ids = [enc.patient_id for enc, _, _ in pipeline.patients
           if enc.patient_id in pipeline.triage_results]
    if not ids:
        st.warning("No patients on the board.")
        return

    selected = st.selectbox("Select patient", ids)
    stored = pipeline.triage_results[selected]
    result = stored["result"]
    encounter, reference_level, description = pipeline.patient_encounters[selected]
    model_output = stored["model_output"]
    safety = stored["safety"]
    trace = stored["explanation"]

    source_tag(description)

    level_header(
        result.triage_level, result.recommended_action,
        escalated=(result.safety_status == "escalation_applied"),
        overridden=stored.get("overridden", False),
    )

    # ── Patient summary strip ──
    vitals = encounter.vitals.to_feature_dict()
    st.markdown(
        f"**{encounter.age}y {encounter.sex}** · "
        f"{encounter.symptoms.get_chief_complaint_text() or 'complaint not coded'} · "
        f"arrived {encounter.arrival_time.strftime('%H:%M')}"
    )

    vital_cols = st.columns(7)
    labels = [("heart_rate", "HR", "bpm"), ("respiratory_rate", "RR", "/min"),
              ("spo2", "SpO₂", "%"), ("systolic_bp", "SBP", "mmHg"),
              ("diastolic_bp", "DBP", "mmHg"), ("temperature", "Temp", "°C"),
              ("pain_score", "Pain", "/10")]
    for col, (key, label, unit) in zip(vital_cols, labels):
        value = vitals.get(key)
        if value is None:
            col.metric(label, "n/a", help="Not recorded. Shown as unknown, never "
                                        "substituted with a normal value.")
        else:
            fmt = f"{value:.1f}" if key == "temperature" else f"{value:.0f}"
            col.metric(label, fmt)

    st.markdown("---")
    confidence_row(result, model_output)
    st.markdown("---")

    # ── The decision trace ──
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Why this level")
        st.caption(
            f"Decided by: **{trace['decided_by']}**. Ordered by what actually "
            f"drove the decision. A rule that set the level appears first, "
            f"model evidence after it."
        )
        factor_list(trace["factors"], limit=6)

        counterfactual = trace.get("counterfactual")
        if counterfactual:
            st.info(f"🔄 **{counterfactual['statement']}**")

    with right:
        st.subheader("What we don't know")
        missing_info_panel(trace["not_recorded"], result.recommended_followup_question)

        st.markdown("---")
        st.markdown("**Model view**")
        probs = model_output["probabilities"]
        prob_df = pd.DataFrame({
            "Level": [f"{lvl}: {LEVEL_NAMES[lvl]}" for lvl in [1, 2, 3, 4, 5]],
            "Probability": [probs.get(lvl, probs.get(str(lvl), 0.0)) for lvl in [1, 2, 3, 4, 5]],
        }).set_index("Level")
        st.bar_chart(prob_df, height=190)

        st.caption(
            f"Most likely level on its own: **{model_output['most_likely_level']}**. "
            f"Probability this patient is Level 1 or 2: "
            f"**{model_output['critical_probability']:.1%}**."
        )

    # ── Why the system didn't just pick the most likely level ──
    cost = model_output.get("cost_decision", {})
    if cost.get("differs_from_argmax"):
        st.warning(
            f"⚖️ **The most likely level was not chosen.** {cost.get('rationale')}",
            icon="⚖️")

    # ── Deterioration trend ──
    velocity = stored.get("velocity") or {}
    if velocity.get("has_trend_data"):
        st.markdown("---")
        st.subheader("Deterioration trend")
        simulated = selected in getattr(pipeline, "simulated_trend_patients", set())
        if simulated:
            st.caption(
                "⚠️ Serial observations are **simulated** from this patient's real "
                "starting vitals, because the survey records one observation per visit. "
                "The trend illustrates the mechanism; the baseline physiology is real."
            )
        for alert in velocity.get("alerts", [])[:4]:
            st.markdown(f"- 📈 {alert}")

    # ── Persona views ──
    st.markdown("---")
    st.subheader("Explanation views")
    tab_nurse, tab_patient, tab_compliance = st.tabs(
        ["👩‍⚕️ Nurse", "🧑 Patient", "📋 Compliance"])

    with tab_nurse:
        nurse_view = ExplanationBuilder.for_nurse(trace)
        st.markdown(f"### {nurse_view['headline']}: {LEVEL_NAMES[result.triage_level]}")
        for line in nurse_view["because"]:
            st.markdown(f"- {line}")
        if nurse_view.get("counterfactual"):
            st.caption(f"🔄 {nurse_view['counterfactual']}")
        if nurse_view["missing"]:
            st.caption(f"Not recorded: {', '.join(nurse_view['missing'])}")

    with tab_patient:
        patient_view = ExplanationBuilder.for_patient(trace)
        st.markdown(f"**{patient_view['priority_message']}**")
        if patient_view["why"]:
            st.markdown("What we noticed:")
            for line in patient_view["why"]:
                st.markdown(f"- {line}")
        st.info(patient_view["reassurance"])
        st.caption(
            "Deliberately free of probabilities and clinical terms, and never "
            "states or implies a diagnosis. It describes queue priority only."
        )

    with tab_compliance:
        compliance = ExplanationBuilder.for_compliance(trace, safety, model_output)
        st.json(compliance, expanded=False)

        if safety.get("fired_rules"):
            st.markdown("**Rules that fired**")
            st.dataframe(pd.DataFrame([{
                "Rule": r["rule_id"],
                "Certainty": r.get("certainty", "n/a"),
                "Target level": r["target_level"],
                "Category": r["category"],
                "Citation": r["citation"],
            } for r in safety["fired_rules"]]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        f"Inference latency **{model_output['latency_ms']:.1f} ms** · "
        f"end-to-end **{stored['latency_ms']:.1f} ms** · "
        f"model **{model_output['model_name']}** · "
        f"attribution via {trace.get('attribution_method')}"
    )
    rule_pack_footer(safety.get("rule_pack", {}))

    # Only NHAMCS records carry a nurse's own assignment and a recorded
    # outcome. A live intake has neither, and printing "Level None" beside it
    # reads as a system fault rather than an absent comparator.
    if description.startswith("NHAMCS") and reference_level is not None:
        st.caption(f"Reference (triage nurse's own assignment): **Level {reference_level}** · "
                   f"{description.split('Outcome:')[-1].strip()}")
