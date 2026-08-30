"""PatientTriage.ai — Patient board (the ED's live view)."""

from datetime import datetime

import pandas as pd
import streamlit as st

from ui.components import LEVEL_EMOJIS, safety_banner


def render_dashboard(pipeline, queue_manager, eval_results, role: str):
    st.title("Emergency department board")
    safety_banner()

    stats = queue_manager.get_queue_stats()
    needs_reassessment = queue_manager.get_patients_needing_reassessment()

    n_real = sum(1 for _, _, d in pipeline.patients if d.startswith("NHAMCS"))
    n_edge = sum(1 for _, _, d in pipeline.patients if d.startswith("SYNTHETIC"))
    n_live = len(pipeline.patients) - n_real - n_edge

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients waiting", stats.get("count", 0))
    c2.metric("Longest wait", f"{stats.get('max_wait_minutes', 0):.0f} min")

    emergent = sum(1 for v in pipeline.triage_results.values()
                   if v["result"].triage_level <= 2)
    c3.metric("In emergent lane", emergent,
              help="Patients the system routed to Level 1 or 2.")

    if needs_reassessment:
        c4.metric("Due for reassessment", len(needs_reassessment),
                  help="Waited past the safe interval for their level, or "
                       "showing a deterioration trend.")
    else:
        c4.metric("Reassessment", "✓ none due")

    composition = (
        f"Board composition: **{n_real} real held-out ED visits** from the CDC "
        f"NHAMCS survey, plus **{n_edge} synthetic edge cases** that exercise "
        f"bedside-observation safety rules"
    )
    if n_live:
        composition += (
            f", plus **{n_live} live intake{'s' if n_live != 1 else ''}** "
            f"entered on this device")
    composition += "."
    st.caption(
        composition + " Synthetic cases are labelled wherever they appear and "
        "excluded from all accuracy figures."
    )

    st.markdown("---")

    # ── The board ──
    rows = []
    for enc, reference_level, desc in pipeline.patients:
        stored = pipeline.triage_results.get(enc.patient_id)
        if not stored:
            continue
        result = stored["result"]
        waited = (datetime.now() - enc.arrival_time).total_seconds() / 60.0
        escalated = result.safety_status == "escalation_applied"
        overridden = stored.get("overridden", False)

        rows.append({
            "Patient": enc.patient_id,
            # Age and sex share a cell. Eleven columns did not fit the content
            # width, so the two right-hand columns — the escalation flag and the
            # provenance label, both of which change how a row should be read —
            # were being clipped off the edge of the table.
            "Age/Sex": f"{enc.age} {enc.sex}",
            "Presenting complaint": (enc.symptoms.get_chief_complaint_text()
                                     or "not coded")[:52],
            "Level": f"{LEVEL_EMOJIS.get(result.triage_level, '⚪')} {result.triage_level}",
            "Reference": ("n/a" if reference_level is None
                          else str(reference_level)),
            "Confidence": f"{result.confidence_percent:.0f}%",
            "Uncertainty": result.uncertainty_band,
            "Waited": f"{waited:.0f} min",
            "Flag": ("escalated" if escalated else
                     "overridden" if overridden else ""),
            "Source": ("synthetic" if desc.startswith("SYNTHETIC")
                       else "real" if desc.startswith("NHAMCS") else "live"),
        })

    df = pd.DataFrame(rows)

    f1, f2 = st.columns([2, 1])
    with f1:
        levels = st.multiselect("Filter by level", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    with f2:
        source_filter = st.selectbox("Source", ["All", "Real patients only",
                                                "Synthetic edge cases only",
                                                "Live intakes only"])

    view = df[df["Level"].str[-1].astype(int).isin(levels)]
    if source_filter == "Real patients only":
        view = view[view["Source"] == "real"]
    elif source_filter == "Synthetic edge cases only":
        view = view[view["Source"] == "synthetic"]
    elif source_filter == "Live intakes only":
        view = view[view["Source"] == "live"]

    view = view.sort_values(["Level", "Waited"], ascending=[True, False])
    st.dataframe(
        view, use_container_width=True, hide_index=True, height=460,
        column_config={
            "Patient": st.column_config.TextColumn(width="small"),
            "Age/Sex": st.column_config.TextColumn(width="small"),
            "Presenting complaint": st.column_config.TextColumn(width="medium"),
            "Level": st.column_config.TextColumn(width="small"),
            "Reference": st.column_config.TextColumn(
                "Nurse", width="small",
                help="The level an actual triage nurse assigned. Shown as n/a "
                     "for synthetic cases and live intakes, which have none."),
            "Confidence": st.column_config.TextColumn(
                width="small",
                help="Probability this patient is no more urgent than the "
                     "assigned level, i.e. not under-triaged."),
            "Uncertainty": st.column_config.TextColumn(width="small"),
            "Waited": st.column_config.TextColumn(width="small"),
            "Flag": st.column_config.TextColumn(width="small"),
            "Source": st.column_config.TextColumn(width="small"),
        })

    st.caption(
        "**Nurse** is the level an actual triage nurse assigned (for real "
        "patients). The system is a second opinion, not a grader. Disagreement "
        "is expected and is where the clinical value lies."
    )

    # ── Where the system disagreed with the nurse, and why it matters ──
    st.markdown("---")
    st.subheader("Where this system disagreed with the triage nurse")

    escalations, de_escalations = [], []
    for enc, reference_level, desc in pipeline.patients:
        stored = pipeline.triage_results.get(enc.patient_id)
        if not stored or desc.startswith("SYNTHETIC") or reference_level is None:
            continue
        level = stored["result"].triage_level
        if level < reference_level:
            escalations.append((enc, reference_level, level, stored, desc))
        elif level > reference_level:
            de_escalations.append((enc, reference_level, level, stored, desc))

    col_up, col_down = st.columns(2)

    with col_up:
        st.markdown(f"**Prioritised higher than the nurse: {len(escalations)}**")
        st.caption("The cases worth reviewing: the system saw something that "
                   "warranted a faster look.")
        for enc, ref, level, stored, desc in escalations[:6]:
            outcome = desc.split("Outcome:")[-1].strip().rstrip(".")
            hit = "✅" if outcome in ("admitted to critical care", "died in the ED",
                                     "admitted to hospital") else ""
            with st.expander(f"{enc.patient_id}: Level {ref} → **{level}** {hit}"):
                st.markdown(f"**Complaint:** {enc.symptoms.get_chief_complaint_text() or 'not recorded'}")
                st.markdown(f"**What actually happened:** {outcome}")
                for factor in stored["result"].top_contributing_factors[:2]:
                    st.markdown(f"- {factor}")

    with col_down:
        st.markdown(f"**Prioritised lower than the nurse: {len(de_escalations)}**")
        st.caption("Shown with equal prominence. A tool that only displays its "
                   "wins is a tool nobody should trust.")
        for enc, ref, level, stored, desc in de_escalations[:6]:
            outcome = desc.split("Outcome:")[-1].strip().rstrip(".")
            miss = "⚠️" if outcome in ("admitted to critical care", "died in the ED") else ""
            with st.expander(f"{enc.patient_id}: Level {ref} → **{level}** {miss}"):
                st.markdown(f"**Complaint:** {enc.symptoms.get_chief_complaint_text() or 'not recorded'}")
                st.markdown(f"**What actually happened:** {outcome}")
                for factor in stored["result"].top_contributing_factors[:2]:
                    st.markdown(f"- {factor}")

    # ── Headline validated performance ──
    if eval_results:
        st.markdown("---")
        st.subheader("Validated performance")
        pm = eval_results["primary_metrics"]["with_confidence_intervals"]
        outcome = eval_results["outcome_validation"]
        cohort = eval_results["cohort"]

        m1, m2, m3, m4 = st.columns(4)
        cr = pm["critical_recall"]
        m1.metric("Critical recall", f"{cr['point']:.1%}",
                  help=f"95% CI {cr['ci_low']:.1%} to {cr['ci_high']:.1%}, "
                       f"bootstrapped clustered by hospital.")
        ll = pm["critical_lane_load"]
        m2.metric("Emergent-lane load", f"{ll['point']:.1%}",
                  help=f"True critical prevalence is "
                       f"{cohort['critical_prevalence_pct']}%. This is the "
                       f"operational cost of the recall above.")
        m3.metric("AUROC (critical)",
                  f"{eval_results['discrimination']['auroc_critical_level_1_2']:.3f}")

        ours = outcome["PatientTriage.ai (cost-sensitive policy)"]["critical_outcome_capture_rate"]
        nurses = outcome["Triage nurses (the reference standard)"]["critical_outcome_capture_rate"]
        m4.metric("ICU/death cases caught", f"{ours:.1%}",
                  delta=f"{(ours - nurses) * 100:+.1f} pts vs triage nurses",
                  help="Of patients who actually went to critical care or died "
                       "in the ED, the share this system would have routed to "
                       "the emergent lane.")

        st.caption(
            f"Measured on {cohort['test_fold_visits']:,} visits from "
            f"{cohort['test_fold_hospitals']} hospitals held out of training, "
            f"calibration and threshold selection."
        )
