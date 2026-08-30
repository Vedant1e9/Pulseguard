"""PatientTriage.ai — The safety–throughput frontier and site profiles."""

import pandas as pd
import streamlit as st

from ui.components import safety_banner


def render_safety_frontier(eval_results, pipeline):
    st.title("Safety and throughput frontier")
    safety_banner()

    if not eval_results:
        st.error("No evaluation results found. Run `python -m evaluation.full_evaluation`.")
        return

    st.markdown(
        "The brief asks for a system **tuned** to the asymmetry between "
        "under-triage and over-triage rather than one that pretends the "
        "tension away. This page is that tuning, made visible. There is no "
        "setting that catches every critical patient without flooding the "
        "emergent lane, so the question is not *whether* to trade, but "
        "**where to stand on the curve, and who decides**."
    )

    curve = eval_results.get("operating_curve", [])
    if not curve:
        st.warning("No operating curve in the evaluation results.")
        return

    df = pd.DataFrame(curve)
    cohort = eval_results["cohort"]
    prevalence = cohort["critical_prevalence_pct"] / 100.0

    chart_df = pd.DataFrame({
        "Emergent-lane load": df["pct_routed_level_1_2"],
        "Critical recall": df["critical_recall"],
    }).set_index("Emergent-lane load")
    # Axis titles: without them the chart is two unlabelled numeric axes, and
    # the whole point of the page is which quantity is being traded for which.
    st.line_chart(chart_df, height=340,
                  x_label="Share of patients routed to the emergent lane",
                  y_label="Share of critical patients caught")

    st.caption(
        f"Each point is one setting of λ, the single parameter scaling the cost "
        f"of under-triage. True critical prevalence in this cohort is "
        f"**{prevalence:.1%}**. An ideal system would sit at that lane load "
        f"with 100% recall, and the gap between that point and this curve is "
        f"the honest measure of how hard triage actually is."
    )

    # ── Where we chose to stand ──
    meta = pipeline.bundle.metadata
    operating = meta.get("operating_point", {})
    budget = meta.get("escalation_budget", {})

    st.markdown("---")
    st.subheader("Where this deployment stands, and why")

    if operating:
        point = operating.get("operating_point", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("λ (under-triage cost scale)", f"{operating.get('selected_lambda')}")
        c2.metric("Critical recall", f"{point.get('critical_recall', 0):.1%}")
        c3.metric("Emergent-lane load", f"{point.get('pct_routed_level_1_2', 0):.1%}")
        c4.metric("Exact agreement", f"{point.get('accuracy', 0):.1%}")
        st.info(operating.get("explanation", ""), icon="🎯")
        st.caption(
            f"Budget in force: no more than "
            f"**{budget.get('max_critical_lane_load', 0):.0%}** of arrivals routed "
            f"to the emergent lane, with a critical-recall floor of "
            f"**{budget.get('min_critical_recall', 0):.0%}**. The budget is stated "
            f"in lane load rather than over-triage rate because lane load is what "
            f"a department actually runs out of. A charge nurse can tell you on "
            f"the spot whether it is staffable."
        )

    with st.expander("The full curve"):
        st.dataframe(pd.DataFrame({
            "λ": df["lambda"],
            "Critical recall": df["critical_recall"].map("{:.1%}".format),
            "Critical under-triage": df["critical_under_triage_rate"].map("{:.1%}".format),
            "Emergent-lane load": df["pct_routed_level_1_2"].map("{:.1%}".format),
            "Under-triage (any)": df["under_triage_rate"].map("{:.1%}".format),
            "Exact agreement": df["accuracy"].map("{:.1%}".format),
        }), use_container_width=True, hide_index=True, height=380)
        st.caption(
            "λ = 0 is the accuracy-maximising system: it agrees with the nurse "
            "most often and misses most critical patients. Large λ escalates "
            "everyone: perfect recall, unusable department. Neither end is a "
            "product."
        )

    # ── Site profiles ──
    st.markdown("---")
    st.subheader("The same assistant across different hospitals")
    st.markdown(
        "Hospitals differ enormously in what they can do and how fast help "
        "arrives. Rather than shipping one risk appetite, the cost matrix is a "
        "versioned, inspectable object that a medical director signs off. Same "
        "model, same code, different stance, and every difference is a diff in "
        "a file, not a hidden constant."
    )

    profiles = eval_results.get("site_profile_comparison", {})
    if profiles:
        rows = []
        for name, m in profiles.items():
            rows.append({
                "Site profile": name.replace("_", " ").title(),
                "Critical recall": f"{m['critical_recall']:.1%}",
                "Critical under-triage": f"{m['critical_under_triage_rate']:.1%}",
                "Emergent-lane load": f"{m['critical_lane_load']:.1%}",
                "Exact agreement": f"{m['accuracy']:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for name, m in profiles.items():
            with st.expander(f"Why {name.replace('_', ' ')} is tuned this way"):
                st.markdown(m["rationale"])

    # ── The cost matrix itself ──
    st.markdown("---")
    with st.expander("The cost matrix in force"):
        cm = pipeline.bundle.cost_matrix
        st.markdown(f"**{cm.name}** (v{cm.version})")
        st.caption(cm.rationale)
        st.dataframe(pd.DataFrame({
            "True level": [1, 2, 3, 4, 5],
            "Cost per level of UNDER-triage": [cm.under_cost.get(l) for l in [1, 2, 3, 4, 5]],
            "Cost per level of OVER-triage": [cm.over_cost.get(l) for l in [1, 2, 3, 4, 5]],
        }), use_container_width=True, hide_index=True)
        st.caption(
            "Read the first row as: sending a Level 1 patient to the Level 3 "
            "queue is priced at two levels of under-triage. Everything "
            "contestable about this system's risk appetite lives in this table, "
            "in one place, where a clinician can argue with it."
        )
