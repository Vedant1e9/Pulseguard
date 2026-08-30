"""PatientTriage.ai — Model performance, validated on held-out hospitals."""

import pandas as pd
import streamlit as st

from ui.components import safety_banner


def render_model_evaluation(eval_results, pipeline):
    st.title("Model performance")
    safety_banner()

    if not eval_results:
        st.error("No evaluation results found. Run `python -m evaluation.full_evaluation`.")
        return

    cohort = eval_results["cohort"]
    st.markdown(
        f"Every figure below is measured on **{cohort['test_fold_visits']:,} real "
        f"emergency department visits** from **{cohort['test_fold_hospitals']} "
        f"hospitals that were held out** of training, calibration, conformal "
        f"fitting and threshold selection. Confidence intervals are bootstrapped "
        f"clustered by hospital."
    )

    with st.expander("The cohort"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total visits", f"{cohort['total_visits_with_triage_label']:,}")
        c2.metric("Hospitals", cohort["total_hospitals"])
        c3.metric("Critical prevalence", f"{cohort['critical_prevalence_pct']}%")
        st.caption(
            f"Source: CDC/NCHS National Hospital Ambulatory Medical Care Survey, "
            f"{', '.join(str(y) for y in cohort['survey_years'])}. Each record is "
            f"a real ED visit with a triage level assigned by an actual triage "
            f"nurse, real recorded vitals, and the patient's actual outcome. "
            f"Age mix: {cohort['age_bands']['pediatric']:,} paediatric, "
            f"{cohort['age_bands']['adult']:,} adult, "
            f"{cohort['age_bands']['geriatric']:,} geriatric."
        )

    # ── Primary safety metrics ──
    st.markdown("---")
    st.subheader("Primary safety metrics")
    st.caption("Critical recall and critical under-triage come first. Accuracy is "
               "reported but never used to select a model.")

    ci = eval_results["primary_metrics"]["with_confidence_intervals"]

    def metric_with_ci(col, label, key, help_text=None, invert=False):
        d = ci.get(key, {})
        if d.get("point") is None:
            return
        col.metric(label, f"{d['point']:.1%}",
                   help=(help_text or "") +
                        f" 95% CI {d['ci_low']:.1%} to {d['ci_high']:.1%}.")

    c1, c2, c3, c4 = st.columns(4)
    metric_with_ci(c1, "Critical recall", "critical_recall",
                   "Level 1 and 2 patients correctly routed to the emergent lane.")
    metric_with_ci(c2, "Critical under-triage", "critical_under_triage_rate",
                   "Critical patients sent to the waiting room. The number that "
                   "matters most.")
    metric_with_ci(c3, "Emergent-lane load", "critical_lane_load",
                   f"True critical prevalence is {cohort['critical_prevalence_pct']}%.")
    metric_with_ci(c4, "Within one level", "within_one_level",
                   "Agreement with the triage nurse to within one level.")

    disc = eval_results["discrimination"]
    d1, d2, d3 = st.columns(3)
    d1.metric("AUROC, critical (L1 or 2)", f"{disc['auroc_critical_level_1_2']:.3f}")
    d2.metric("AUROC, hospital admission", f"{disc['auroc_vs_hospital_admission']:.3f}",
              help="Predicting an outcome that was unknown at triage time.")
    if disc.get("auroc_vs_critical_outcome"):
        d3.metric("AUROC, ICU or death", f"{disc['auroc_vs_critical_outcome']:.3f}")
    st.caption(disc["note"])

    # ── Baselines ──
    st.markdown("---")
    st.subheader("Against the alternatives")
    st.caption(eval_results.get("baseline_comparison_note", ""))

    baseline_rows = []
    for name, m in eval_results["baseline_comparison"].items():
        baseline_rows.append({
            "Approach": name,
            "Critical recall": f"{m['critical_recall']:.1%}",
            "Critical under-triage": f"{m['critical_under_triage_rate']:.1%}",
            "Emergent-lane load": f"{m['critical_lane_load']:.1%}",
            "Exact agreement": f"{m['accuracy']:.1%}",
        })
    st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)

    argmax = eval_results["baseline_comparison"]["Same model, accuracy-maximising argmax"]
    ours = eval_results["baseline_comparison"]["PatientTriage.ai (cost-sensitive policy)"]
    st.success(
        f"**The decision rule matters more than the model.** The same trained "
        f"model on the same patients catches {argmax['critical_recall']:.1%} of "
        f"critical cases when it picks the most likely level, and "
        f"{ours['critical_recall']:.1%} when it minimises expected clinical harm. "
        f"Accuracy falls from {argmax['accuracy']:.1%} to {ours['accuracy']:.1%}. "
        f"That is the trade being made, deliberately, and it is the right way "
        f"round for triage.", icon="⚖️")

    # ── Outcome validation ──
    st.markdown("---")
    st.subheader("Validated against what happened to the patient")
    outcome = eval_results["outcome_validation"]
    st.caption(
        f"Agreeing with the triage nurse is the easy question. This is the hard "
        f"one: of the **{outcome['n_critical_outcome']} patients in the test fold "
        f"who were admitted to critical care or died in the emergency "
        f"department**, how many would each approach have sent to the waiting "
        f"room? This ground truth was unknowable at triage time."
    )

    outcome_rows = []
    for name, m in outcome.items():
        if not isinstance(m, dict) or "critical_outcome_capture_rate" not in m:
            continue
        outcome_rows.append({
            "Approach": name,
            "ICU/death cases caught": f"{m['critical_outcome_capture_rate']:.1%}",
            "Sent to waiting room": m["critical_outcome_patients_routed_to_waiting_room"],
        })
    st.dataframe(pd.DataFrame(outcome_rows), use_container_width=True, hide_index=True)
    st.caption(outcome["Triage nurses (the reference standard)"]["note"])

    # ── Calibration ──
    st.markdown("---")
    st.subheader("Is the confidence figure trustworthy?")
    cal = eval_results["calibration"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Expected calibration error", f"{cal['expected_calibration_error']:.3f}",
              help="Average gap between stated confidence and observed accuracy. "
                   "Lower is better.")
    k2.metric("Brier score", f"{cal['brier_score']:.3f}")
    k3.metric("Mean confidence", f"{cal['mean_confidence']:.1%}")
    st.caption(cal["interpretation"])

    curve = cal.get("reliability_curve", [])
    if curve:
        rel = pd.DataFrame(curve)
        chart = pd.DataFrame({
            "Stated confidence": rel["mean_confidence"],
            "Observed accuracy": rel["accuracy"],
            "Perfect calibration": rel["mean_confidence"],
        }).set_index("Stated confidence")
        st.line_chart(chart, height=280)
        st.caption("A perfectly calibrated system's two lines lie on top of each "
                   "other. Above the diagonal means underconfident; below means "
                   "overconfident, which in a triage system is the dangerous side.")

    # ── Conformal ──
    crit_conf = eval_results.get("conformal_critical_exclusion")
    if crit_conf:
        st.markdown("---")
        st.subheader("Statistical guarantee")
        g1, g2, g3 = st.columns(3)
        g1.metric("Patients cleared", f"{crit_conf['pct_patients_cleared']:.1%}",
                  help="Share of patients for whom a critical presentation can "
                       "be excluded at the stated confidence.")
        g2.metric("Critical patients missed", f"{crit_conf['empirical_miss_rate']:.1%}",
                  help=f"Target ≤ {crit_conf['target_max_miss_rate']:.0%}.")
        g3.metric("Guarantee holds", "✅ yes" if crit_conf["guarantee_holds"] else "⚠️ no")
        st.caption(crit_conf["interpretation"])

        five_class = eval_results.get("conformal_5class", {})
        if five_class:
            st.caption(
                f"The five-class conformal set is reported for completeness and "
                f"is deliberately not used as a decision rule: at "
                f"{five_class['empirical_coverage']:.1%} empirical coverage its "
                f"mean width is {five_class['mean_set_size']:.1f} of 5 levels. "
                f"That width is a real finding, because a triage level is not "
                f"identifiable to a single value from triage-time data. But it "
                f"is not something a nurse can act on, so the actionable "
                f"guarantee is the binary one above."
            )

    # ── Fairness ──
    st.markdown("---")
    st.subheader("Fairness audit")
    fairness = eval_results["fairness"]
    st.caption(fairness["note"])

    for dimension, label in [("by_age_band", "Age band"),
                             ("by_sex", "Sex"),
                             ("by_race_ethnicity", "Race / ethnicity")]:
        rows = []
        for group, m in fairness.get(dimension, {}).items():
            if not isinstance(m, dict):
                continue
            if m.get("estimate_suppressed"):
                rows.append({label: group, "n": m["n"],
                             "Critical recall": "suppressed",
                             "Critical under-triage": "suppressed",
                             "Note": m["reason"]})
            else:
                rows.append({
                    label: group,
                    "n": m["n"],
                    "Critical recall": (f"{m['critical_recall']:.1%}"
                                        if m.get("critical_recall") is not None else "n/a"),
                    "Critical under-triage": (f"{m['critical_under_triage_rate']:.1%}"
                                              if m.get("critical_under_triage_rate") is not None
                                              else "n/a"),
                    "Note": "",
                })
        if rows:
            st.markdown(f"**{label}**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            gap = fairness.get(f"{dimension}_max_under_triage_gap")
            if gap is not None:
                st.caption(f"Largest critical under-triage gap across groups: **{gap:.1%}**")

    # ── Cross-site generalisation ──
    cross = eval_results.get("cross_site", {})
    if cross.get("n_hospitals_evaluated"):
        st.markdown("---")
        st.subheader("Does it work at the next hospital?")
        s1, s2, s3 = st.columns(3)
        s1.metric("Hospitals evaluated", cross["n_hospitals_evaluated"])
        s2.metric("Critical recall (mean)", f"{cross['critical_recall_mean']:.1%}")
        s3.metric("Range across sites",
                  f"{cross['critical_recall_min']:.0%} to {cross['critical_recall_max']:.0%}")
        st.caption(cross["interpretation"])

        st.dataframe(pd.DataFrame(cross["per_hospital"]).rename(columns={
            "hospital": "Hospital", "n_visits": "Visits", "n_critical": "Critical",
            "critical_recall": "Critical recall",
            "critical_lane_load": "Lane load", "accuracy": "Exact agreement",
        }), use_container_width=True, hide_index=True, height=260)

    temporal = eval_results.get("temporal_validation", {})
    if temporal.get("metrics"):
        st.markdown("---")
        st.subheader("Does it still work next year?")
        t1, t2, t3 = st.columns(3)
        t1.metric("Critical recall", f"{temporal['metrics']['critical_recall']:.1%}")
        t2.metric("AUROC", f"{temporal['auroc_critical']:.3f}")
        t3.metric("Tested on", f"{temporal['n_test']:,} visits")
        st.caption(temporal["interpretation"])

    # ── Leaderboard ──
    st.markdown("---")
    with st.expander("Model selection: how the winner was chosen"):
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "evaluation", "saved_results", "training_report.json")
        if os.path.exists(path):
            with open(path) as f:
                training = json.load(f)
            st.caption(training.get("selection_criterion", ""))
            st.dataframe(pd.DataFrame([{
                "Model": m["model"],
                "AUROC (critical)": f"{m['auroc_critical']:.3f}",
                "Critical recall": f"{m['critical_recall']:.1%}",
                "Lane load": f"{m['over_triage_rate']:.1%}",
                "λ": m["selected_lambda"],
                "Train time": f"{m['train_seconds']}s",
            } for m in training.get("leaderboard", [])]),
                use_container_width=True, hide_index=True)
            st.caption(
                f"Selected: **{training.get('selected_model')}**. Calibration "
                f"method chosen on a third disjoint fold: "
                f"**{training.get('calibration', {}).get('method_selected')}**."
            )
