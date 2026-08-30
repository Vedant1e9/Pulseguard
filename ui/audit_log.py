"""PulseGuard — Audit log and tamper-evidence."""

import os

import pandas as pd
import streamlit as st

from ui.components import safety_banner


def render_audit_log(pipeline, override_manager):
    st.title("Audit log")
    safety_banner()

    st.markdown(
        "Every triage decision, acceptance and override is written to a "
        "hash-chained, append-only log. Each entry is sealed with "
        "`SHA-256(previous_hash + entry)`, so altering or deleting any past "
        "entry breaks every hash after it and the chain check finds it. Under "
        "HIPAA these records are retained for six years."
    )

    # ── Integrity ──
    with st.expander("🔒 Verify chain integrity", expanded=True):
        col_check, col_info = st.columns([1, 3])
        with col_check:
            run = st.button("Run integrity check", type="primary")
        with col_info:
            # Shown relative to the deployment root. The absolute path
            # exposes the operator's home directory on a screen a compliance
            # officer may well be sharing or projecting.
            log_path = str(override_manager.log_file_path)
            try:
                log_path = os.path.relpath(
                    log_path, os.path.dirname(os.path.dirname(__file__)))
            except ValueError:
                pass
            st.caption(f"Append-only log file: `{log_path}`")

        if run:
            outcome = override_manager.verify_integrity()
            if outcome["intact"]:
                st.success(
                    f"✅ Chain intact. {outcome['total_entries']} entries verified. "
                    f"Head hash `{outcome['chain_head'][:24]}…`")
            else:
                st.error(
                    f"⚠️ Tampering detected at entry #{outcome['tampered_at_index']} "
                    f"of {outcome['total_entries']}. The stored hash does not match "
                    f"the recomputed hash for that entry's content.")

        st.caption(
            "**Prototype scope:** the chain head resets each time the process "
            "restarts. A production deployment would persist the head across "
            "restarts and write to WORM storage, so that the guarantee survives "
            "the machine as well as the session."
        )

    # ── Decision records ──
    st.markdown("---")
    st.subheader("Triage decisions")

    entries = []
    for entry in pipeline.audit_log:
        details = entry.details
        entries.append({
            "Time": entry.timestamp.strftime("%H:%M:%S"),
            "Patient": entry.patient_id,
            "Level": details.get("final_level"),
            "Model proposed": details.get("model_proposed_level"),
            "Escalated": "yes" if details.get("safety_escalated") else "no",
            "Rules fired": ", ".join(details.get("rules_fired", [])) or "none",
            "Confidence": f"{details.get('confidence', 0):.0f}%",
            "Rule pack": f"v{details.get('rule_pack_version')}",
            "Model": details.get("model_name", "n/a"),
            "Latency": f"{details.get('latency_ms', 0):.0f} ms",
        })

    if entries:
        df = pd.DataFrame(entries)
        f1, f2 = st.columns(2)
        only_escalated = f1.checkbox("Show only safety escalations")
        patient_filter = f2.multiselect("Filter by patient",
                                        sorted(df["Patient"].unique()))
        view = df
        if only_escalated:
            view = view[view["Escalated"] == "yes"]
        if patient_filter:
            view = view[view["Patient"].isin(patient_filter)]
        st.dataframe(view, use_container_width=True, hide_index=True, height=380)
        st.caption(f"Showing {len(view)} of {len(df)} decision records.")
    else:
        st.info("No triage decisions recorded yet.")

    st.info(
        "Each record stores the model version, the **rule pack version and "
        "content hash**, the level the model proposed, the level actually "
        "assigned, and every rule that fired. That is enough to reconstruct any "
        "past decision exactly, including which clinical policy text was in "
        "force at the time, which a version string alone would not prove.",
        icon="🧾")

    # ── Clinician actions ──
    st.markdown("---")
    st.subheader("Clinician actions")

    clinician_entries = override_manager.get_audit_log()
    if not clinician_entries:
        st.info("No clinician acceptances or overrides recorded in this session.")
        return

    rows = []
    for entry in clinician_entries:
        details = entry.get("details", {})
        rows.append({
            "Time": entry.get("timestamp", "")[11:19],
            "Event": entry.get("event_type"),
            "Patient": entry.get("patient_id"),
            "Clinician": entry.get("user_id"),
            "Role": details.get("clinician_role", "n/a"),
            "System": details.get("system_recommendation",
                                  details.get("accepted_level", "n/a")),
            "Clinician level": details.get("override_level", "n/a"),
            "Reason": details.get("justification_code", "n/a"),
            "2nd clinician": details.get("second_clinician_concurred", "n/a"),
            "Entry hash": entry.get("entry_hash", "")[:12],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    stats = override_manager.get_override_stats()
    if stats["total_overrides"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total overrides", stats["total_overrides"])
        c2.metric("Urgency raised", stats["upgrade_count"])
        c3.metric("Urgency lowered", stats["downgrade_count"])
        if stats.get("justification_distribution"):
            st.markdown("**Reasons given**")
            st.dataframe(pd.DataFrame([
                {"Reason": k, "Count": v}
                for k, v in stats["justification_distribution"].items()
            ]), use_container_width=True, hide_index=True)
            st.caption(
                "This distribution is the feedback loop's input. A reason code "
                "that keeps recurring is evidence that a rule threshold needs "
                "revisiting, which is why overrides use a controlled "
                "vocabulary and not free text alone."
            )
