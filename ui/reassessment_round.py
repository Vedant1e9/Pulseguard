"""PulseGuard: Reassessment round, for patients already waiting."""

from datetime import datetime

import pandas as pd
import streamlit as st

from ui.components import (
    LEVEL_EMOJIS, LEVEL_NAMES, factor_list, level_header, safety_banner,
)

# The vitals a nurse actually rechecks on a waiting-room round. Blood pressure
# is omitted on purpose: a cuff reading is not something anyone repeats every
# fifteen minutes on a corridor trolley, and offering a field nobody fills
# teaches staff to ignore the form.
RECHECK = [
    ("heart_rate", "Heart rate", "bpm", 0, 300),
    ("respiratory_rate", "Respiratory rate", "/min", 0, 80),
    ("spo2", "Oxygen saturation", "%", 50, 100),
    ("temperature", "Temperature", "°C", 30.0, 43.0),
]


def render_reassessment_round(pipeline, queue_manager):
    st.title("Reassessment round")
    safety_banner()

    st.markdown(
        "A triage level is a decision made at one moment about a patient who "
        "then sits in a waiting room for two hours. This page is the round a "
        "nurse walks to check whether that decision is still true."
    )

    st.info(
        "The brief requires the system to trigger re-assessment **if wait time "
        "exceeds safe thresholds for a patient's severity level, or if vitals "
        "are re-recorded as worsening**. Wait-time monitoring runs "
        "continuously and drives the list below. This page is the second "
        "half: re-recording a vital, and letting the system act on it.",
        icon="🔁")

    _render_outcome(pipeline)

    # ── Who to see, and why ──
    due = queue_manager.get_patients_needing_reassessment()
    ordered = queue_manager.get_ordered_queue()
    stats = queue_manager.get_queue_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Waiting", stats.get("count", 0))
    c2.metric("Due for reassessment", len(due))
    c3.metric("Past safe interval", stats.get("patients_past_threshold", 0))
    c4.metric("Longest wait", f"{stats.get('max_wait_minutes', 0):.0f} min")

    st.markdown("---")
    st.markdown("### Who to see next")

    if not ordered:
        st.info("No patients currently waiting.")
        return

    due_ids = {a["patient_id"] for a in due}
    rows = []
    for entry in ordered:
        pid = entry["patient_id"]
        stored = pipeline.triage_results.get(pid)
        if not stored:
            continue
        rows.append({
            "Patient": pid,
            "Level": f"{LEVEL_EMOJIS.get(entry['triage_level'], '⚪')} "
                     f"{entry['triage_level']}",
            "Waited": f"{entry['wait_minutes']:.0f} min",
            "Due": "yes" if pid in due_ids else "",
            "Uncertainty": entry["uncertainty"],
        })

    st.dataframe(
        pd.DataFrame(rows), use_container_width=True, hide_index=True, height=260,
        column_config={
            "Patient": st.column_config.TextColumn(width="small"),
            "Level": st.column_config.TextColumn(width="small"),
            "Waited": st.column_config.TextColumn(width="small"),
            "Due": st.column_config.TextColumn(
                width="small",
                help="Past the safe reassessment interval for this level, or "
                     "already showing a deterioration trend."),
            "Uncertainty": st.column_config.TextColumn(width="small"),
        })

    # ── Re-record ──
    st.markdown("---")
    st.markdown("### Re-record a patient's vitals")

    ids = [r["Patient"] for r in rows]
    default = next((i for i, r in enumerate(rows) if r["Due"] == "yes"), 0)
    selected = st.selectbox("Patient", ids, index=default)

    encounter = pipeline.patient_encounters[selected][0]
    stored = pipeline.triage_results[selected]
    current = stored["result"]
    at_arrival = encounter.vitals.to_feature_dict()

    st.markdown(
        f"**{encounter.age}y {encounter.sex}** · "
        f"{encounter.symptoms.get_chief_complaint_text() or 'complaint not coded'} · "
        f"currently {LEVEL_EMOJIS.get(current.triage_level, '⚪')} "
        f"**Level {current.triage_level}, {LEVEL_NAMES.get(current.triage_level, '?')}**")

    history = pipeline.velocity_model.patient_histories.get(selected, [])
    if history:
        st.caption(f"{len(history)} observation(s) already on file, the most "
                   f"recent at {history[-1][0]:%H:%M}.")
    else:
        st.caption("Only the arrival observation is on file. One recheck gives "
                   "the system an interval to compute a rate of change over.")

    with st.form("recheck"):
        st.caption(
            "Leave a field blank if you did not recheck it. A vital you do not "
            "re-measure keeps its arrival value rather than becoming unknown, "
            "and only what you enter here is treated as a new observation.")
        cols = st.columns(4)
        entered = {}
        for i, (key, label, unit, lo, hi) in enumerate(RECHECK):
            with cols[i]:
                # Streamlit matches a number input's numeric type to its format
                # string. Passing float bounds alongside "%d" renders a type
                # warning directly above the field a nurse is reading, so the
                # bounds are cast to match the format.
                if key == "temperature":
                    entered[key] = st.number_input(
                        f"{label} ({unit})", min_value=float(lo),
                        max_value=float(hi), value=None,
                        placeholder="not rechecked", step=0.1, format="%.1f",
                        key=f"recheck_{key}")
                else:
                    entered[key] = st.number_input(
                        f"{label} ({unit})", min_value=int(lo),
                        max_value=int(hi), value=None,
                        placeholder="not rechecked", step=1, format="%d",
                        key=f"recheck_{key}")
                previous = at_arrival.get(key)
                st.caption(f"at arrival: {previous:.0f}" if previous is not None
                           else "not recorded at arrival")

        submitted = st.form_submit_button(
            "🔁 Record observation and re-score", type="primary",
            use_container_width=True)

    if not submitted:
        return

    measured = {k: v for k, v in entered.items() if v is not None}
    if not measured:
        st.error("Enter at least one vital sign to record an observation.")
        return

    with st.spinner("Re-scoring …"):
        outcome = pipeline.record_observation(selected, measured,
                                              recorded_by="TRIAGE_NURSE")

    if outcome is None:
        st.error("Could not record that observation.")
        return

    # `update_patient` merges rather than increments, and stamps its own
    # last_updated, so the running count is read and advanced here. That count
    # is what resets a patient's reassessment clock: without it, a patient who
    # was just checked stays on the "due" list and the round never converges.
    seen = queue_manager.queue.get(selected, {}).get("reassessment_count", 0)
    queue_manager.update_patient(
        selected,
        triage_level=outcome["final_level"],
        confidence=outcome["result"].confidence_percent,
        uncertainty=outcome["result"].uncertainty_band,
        velocity_risk=(outcome["velocity"].get("overall_risk", "low")
                       if outcome["velocity"].get("has_trend_data")
                       else "insufficient_data"),
        reassessment_count=seen + 1,
    )

    st.session_state["reassess_last"] = selected
    st.rerun()


def _render_outcome(pipeline):
    """What the last recorded observation did, shown at the top of the page."""
    pid = st.session_state.get("reassess_last")
    if not pid:
        return

    event = next((e for e in reversed(pipeline.audit_log)
                  if e.event_type == "reassessment" and e.patient_id == pid), None)
    if not event:
        return

    d = event.details
    stored = pipeline.triage_results[pid]
    result = stored["result"]
    before, after = d["previous_level"], d["final_level"]

    if d["escalated"]:
        st.error(
            f"**{pid} escalated from Level {before} to Level {after} on "
            f"re-recorded vitals.** The patient was already waiting; nothing "
            f"about their presentation changed except the numbers.", icon="📈")
    elif d["held_by_escalate_only_rule"]:
        st.warning(
            f"**{pid} held at Level {after}.** The new reading scored as Level "
            f"{d['proposed_level']}, which is less urgent. Re-scoring is "
            f"escalate-only: one favourable observation does not walk a "
            f"patient back down the queue, because a heart rate that falls may "
            f"mean a patient is tiring rather than improving. A downgrade "
            f"stays a clinician override, with a name attached to it.",
            icon="🛡️")
    else:
        st.info(f"**{pid} remains at Level {after}.** The new observation did "
                f"not change the assessment.", icon="✅")

    if d["changes"]:
        st.caption("Recorded: " + " · ".join(d["changes"]))

    velocity = stored.get("velocity", {})
    if velocity.get("has_trend_data"):
        v1, v2, v3 = st.columns(3)
        v1.metric("Deterioration risk", velocity.get("overall_risk", "unknown").title())
        v2.metric("Observations on file", velocity.get("readings_count", 0))
        v3.metric("Window", f"{velocity.get('time_window_hours', 0):.1f} h")
        for alert in velocity.get("alerts", [])[:3]:
            st.markdown(f"- 📈 {alert}")

    level_header(result.triage_level, result.recommended_action,
                 escalated=(result.safety_status == "escalation_applied"))

    with st.expander("Why this level now", expanded=d["escalated"]):
        factor_list(stored["explanation"]["factors"], limit=4)

    st.caption(
        f"Logged to the audit trail as a `reassessment` event at "
        f"{event.timestamp:%H:%M:%S}, with the vitals recorded, the level "
        f"before and after, the rules that fired, and the rule-pack hash in "
        f"force. Visible on the **Audit log** page.")
    st.markdown("---")
