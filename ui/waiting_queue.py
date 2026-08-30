"""PatientTriage.ai — Waiting queue, ordered by live clinical hazard."""

import pandas as pd
import streamlit as st

from ui.components import LEVEL_EMOJIS, safety_banner


def render_waiting_queue(queue_manager, pipeline):
    st.title("Waiting queue")
    safety_banner()

    st.markdown(
        "Ordered by a **live hazard score**, not by triage level alone. A "
        "Level 3 patient who has waited 50 minutes and whose vitals are "
        "drifting outranks a Level 3 who arrived five minutes ago. That is "
        "what a charge nurse does in their head, made explicit and consistent."
    )

    ordered = queue_manager.get_ordered_queue()
    if not ordered:
        st.info("No patients currently waiting.")
        return

    stats = queue_manager.get_queue_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Waiting", stats["count"])
    c2.metric("Average wait", f"{stats['avg_wait_minutes']:.0f} min")
    c3.metric("Longest wait", f"{stats['max_wait_minutes']:.0f} min")
    c4.metric("Past safe interval", stats["patients_past_threshold"])

    st.markdown("---")

    rows = []
    for entry in ordered:
        pid = entry["patient_id"]
        stored = pipeline.triage_results.get(pid, {})
        encounter = pipeline.patient_encounters.get(pid, (None,))[0]

        rows.append({
            "#": entry["queue_position"],
            "Patient": pid,
            "Level": f"{LEVEL_EMOJIS.get(entry['triage_level'], '⚪')} {entry['triage_level']}",
            "Complaint": ((encounter.symptoms.get_chief_complaint_text() or "not recorded")[:40]
                          if encounter else "not recorded"),
            "Waited": f"{entry['wait_minutes']:.0f} min",
            "Why here": _plain_reason(entry),
            "Uncertainty": entry["uncertainty"],
        })

    # "Why here" is the column this page exists for, so it gets the width. Left
    # to auto-sizing it was clipped mid-sentence at the right-hand edge, which
    # turns a stated reason back into the opaque number it replaced.
    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True,
        height=420,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Patient": st.column_config.TextColumn(width="small"),
            "Level": st.column_config.TextColumn(width="small"),
            "Complaint": st.column_config.TextColumn(width="medium"),
            "Waited": st.column_config.TextColumn(width="small"),
            "Why here": st.column_config.TextColumn(width="large"),
            "Uncertainty": st.column_config.TextColumn(width="small"),
        })

    st.caption(
        "**Why here** replaces the raw hazard score. A number like 1386 is not "
        "something a nurse can act on or challenge; a sentence is."
    )

    # ── Reassessment alerts, grouped so they don't become wallpaper ──
    st.markdown("---")
    st.subheader("Due for reassessment")

    alerts = queue_manager.get_patients_needing_reassessment()
    if not alerts:
        st.success("✓ Nobody is past their safe reassessment interval.")
        return

    deteriorating = [a for a in alerts if "deterioration" in a["reason"].lower()]
    overdue = [a for a in alerts if a not in deteriorating]

    if deteriorating:
        st.error(f"**{len(deteriorating)} showing a deterioration trend. Review now.**")
        for a in deteriorating:
            st.markdown(f"- **{a['patient_id']}** (Level {a['triage_level']}, "
                        f"{a['age_group']}): {a['reason']}")

    if overdue:
        with st.expander(f"⏱️ {len(overdue)} past their safe wait interval",
                         expanded=len(overdue) <= 4):
            for a in overdue:
                st.markdown(f"- **{a['patient_id']}** (Level {a['triage_level']}): "
                            f"{a['reason']}")

    st.caption(
        "Deterioration alerts are separated from wait-time alerts on purpose. "
        "Mixing them produces a single undifferentiated list that staff learn "
        "to dismiss, the alert-fatigue failure the brief warns about. Only the "
        "first group asks for immediate action."
    )


def _plain_reason(entry) -> str:
    """
    Say why a patient is where they are, in words.

    The hazard score is a product of five multipliers; naming the dominant one
    is more useful than showing the product, because it tells a nurse what
    would have to change for the ordering to change.
    """
    components = entry.get("hazard_components", {})
    reasons = []

    if entry["triage_level"] <= 2:
        reasons.append(f"Level {entry['triage_level']}")
    if entry.get("wait_exceeded"):
        reasons.append(f"waited past the {entry['triage_level']}-level safe interval")
    if components.get("velocity_modifier", 1.0) >= 1.5:
        reasons.append("vitals trending worse")
    if components.get("age_modifier", 1.0) > 1.0:
        reasons.append(f"{entry['age_group']} risk adjustment")
    if components.get("uncertainty_modifier", 1.0) >= 1.15:
        reasons.append("unresolved uncertainty")

    if not reasons:
        return f"Level {entry['triage_level']}, within safe wait window"
    return "; ".join(reasons).capitalize()
