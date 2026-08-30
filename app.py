"""
PulseGuard — Emergency Department Triage Assistant
=======================================================

Streamlit front end. Four roles see four different applications, because a
triage nurse with nine seconds and a compliance officer preparing for an audit
need almost nothing in common.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="PulseGuard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

    /* 1180px was comfortable for prose but too narrow for the board, which
       carries ten columns; the two rightmost — the escalation flag and the
       provenance label, both of which change how a row should be read — were
       clipped off the edge. Prose keeps its own measure below. */
    .block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1360px; }
    .block-container [data-testid="stMarkdownContainer"] > p,
    .block-container [data-testid="stCaptionContainer"] > p { max-width: 78ch; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #16233c 100%);
    }
    /* The blanket light colour is what makes the sidebar legible on its dark
       gradient — but it was also inherited by the two dropdowns, which keep a
       light background. That rendered "Triage nurse" and "Urban ED (default)"
       as near-white text on near-white, i.e. the role and rule pack the user
       is signed in under were effectively invisible. Exclude the input
       surfaces and give them explicit dark text instead. */
    [data-testid="stSidebar"] *:not([data-baseweb="select"] *):not([data-baseweb="popover"] *) {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] input {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #475569 !important; }
    [data-testid="stSidebar"] hr { border-color: #26334d !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 5px 9px; border-radius: 6px; transition: background .15s ease;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(96,165,250,.16) !important;
    }

    /* Metric cards.
       These previously hard-coded a light background while inheriting the
       theme's text colour — which rendered white text on a near-white card and
       made the four headline numbers on the board completely invisible under a
       dark theme. Both surface and text are now set together, and the dark
       variant is defined explicitly rather than left to inherit. */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 10px;
    }
    [data-testid="stMetric"] * { color: #0f172a !important; }
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 1.45rem !important;
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] { color: #475569 !important; }

    @media (prefers-color-scheme: dark) {
        [data-testid="stMetric"] {
            background: #16233c;
            border-color: #2a3b57;
        }
        [data-testid="stMetric"] * { color: #e2e8f0 !important; }
        [data-testid="stMetricValue"] { color: #f8fafc !important; }
        [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        border: none; color: #fff; font-weight: 600; font-size: 15px;
        padding: 12px 18px; border-radius: 9px; transition: all .2s ease;
        box-shadow: 0 4px 14px rgba(29,78,216,.3);
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-1px); box-shadow: 0 6px 18px rgba(29,78,216,.42);
    }

    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        border-radius: 8px !important; border: 1.5px solid #e2e8f0 !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,.12) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* Decision banner (ui/components.py: level_header). The band colour and
       tint are set inline per triage level; only the two text rows are
       themed here. Previously both were hardcoded slate grays, which read
       fine on the light tint in light mode but went near-invisible once the
       page background (and so the tint) turned dark — the same failure
       mode already fixed once above for st.metric. */
    .pt-level-eyebrow { font-size:12px; letter-spacing:1.2px; text-transform:uppercase;
                        color:#64748b; font-weight:700; }
    .pt-level-action { font-size:15px; color:#334155; }

    /* Decision-trace factor cards (ui/components.py: factor_list). Fixed
       light backgrounds with dark text — readable in light mode, but an
       opaque light card stranded on a dark page in dark mode. */
    .pt-factor { padding:10px 14px; border-radius:6px; margin-bottom:8px;
                border-left-width:4px; border-left-style:solid; }
    .pt-factor-rule { border-left-color:#DC2626; background:#fef2f2; }
    .pt-factor-rule .pt-factor-head { font-weight:600; color:#7f1d1d; }
    .pt-factor-rule .pt-factor-detail { font-size:13px; color:#7f1d1d; opacity:.85; margin-top:4px; }
    .pt-factor-rule .pt-factor-meta { font-size:11px; color:#991b1b; opacity:.7; margin-top:6px; }

    .pt-factor-support { border-left-color:#d97706; background:#fffbeb; }
    .pt-factor-support .pt-factor-head { font-weight:600; color:#78350f; }
    .pt-factor-support .pt-factor-meta { font-size:11px; color:#92400e; opacity:.75; margin-top:6px; }

    .pt-factor-evidence { border-left-color:#cbd5e1; background:#f8fafc; }
    .pt-factor-evidence .pt-factor-head { font-weight:600; color:#0f172a; }
    .pt-factor-evidence .pt-factor-detail { font-size:13px; color:#475569; margin-top:4px; }
    .pt-factor-evidence .pt-arrow-up { color:#b45309; }
    .pt-factor-evidence .pt-arrow-down { color:#0369a1; }

    @media (prefers-color-scheme: dark) {
        .pt-level-eyebrow { color:#94a3b8; }
        .pt-level-action { color:#e2e8f0; }

        .pt-factor-rule { background:#3f1d1d; border-left-color:#ef4444; }
        .pt-factor-rule .pt-factor-head,
        .pt-factor-rule .pt-factor-detail { color:#fecaca; }
        .pt-factor-rule .pt-factor-meta { color:#fca5a5; }

        .pt-factor-support { background:#3f2d0f; border-left-color:#f59e0b; }
        .pt-factor-support .pt-factor-head { color:#fde68a; }
        .pt-factor-support .pt-factor-meta { color:#fcd34d; }

        .pt-factor-evidence { background:#1e293b; border-left-color:#475569; }
        .pt-factor-evidence .pt-factor-head { color:#e2e8f0; }
        .pt-factor-evidence .pt-factor-detail { color:#94a3b8; }
        .pt-factor-evidence .pt-arrow-up { color:#fbbf24; }
        .pt-factor-evidence .pt-arrow-down { color:#38bdf8; }
    }
</style>
""", unsafe_allow_html=True)


# ─── Boot ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model and patient board …")
def boot(site: str):
    from engine.triage_pipeline import TriagePipeline
    from engine.hazard_queue import HazardQueueManager
    from engine.override_audit import OverrideAuditManager
    from engine.reassessment import ReassessmentEngine

    pipeline = TriagePipeline(site=None if site == "default" else site)
    pipeline.initialize(verbose=False)
    pipeline.triage_all_patients()

    queue = HazardQueueManager()
    for enc, _, _ in pipeline.patients:
        stored = pipeline.triage_results.get(enc.patient_id)
        if not stored:
            continue
        result = stored["result"]
        velocity = stored.get("velocity", {})
        queue.add_patient(
            patient_id=enc.patient_id,
            triage_level=result.triage_level,
            age_group=enc.age_group.value,
            arrival_time=enc.arrival_time,
            confidence=result.confidence_percent,
            uncertainty=result.uncertainty_band,
            velocity_risk=(velocity.get("overall_risk", "low")
                           if velocity.get("has_trend_data") else "insufficient_data"),
        )

    return pipeline, queue, OverrideAuditManager(), ReassessmentEngine()


@st.cache_data(show_spinner=False)
def load_evaluation():
    import json
    path = os.path.join(os.path.dirname(__file__),
                        "evaluation", "saved_results", "evaluation_full.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🏥 PulseGuard")
st.sidebar.caption("Emergency department triage assistant")
st.sidebar.markdown("---")

st.sidebar.markdown("**Signed in as**")
role = st.sidebar.selectbox(
    "Role",
    ["Triage nurse", "Emergency physician", "Clinical analyst", "Compliance officer"],
    label_visibility="collapsed",
    help="Role determines what this application shows and what it lets you do. "
         "Access is scoped to the minimum necessary for each role.",
)

from engine.rule_pack import available_site_packs

# `available_site_packs()` scans config/ and already returns "default", so
# prepending it listed "Urban ED (default)" twice in the dropdown. De-duplicate
# while keeping the default first, which is the order a user expects.
site_options = ["default"] + [p for p in available_site_packs() if p != "default"]
site = st.sidebar.selectbox(
    "Site rule pack", site_options,
    format_func=lambda s: {"default": "Urban ED (default)",
                           "rural_community": "Rural community ED"}.get(s, s),
    help="The same assistant, flexed to a department's capability and risk "
         "appetite. Switching this reloads the clinical rules and thresholds.",
)

pipeline, queue_manager, override_manager, reassessment_engine = boot(site)
eval_results = load_evaluation()

st.sidebar.markdown("---")

PAGES_BY_ROLE = {
    "Triage nurse": ["Patient board", "New patient intake", "Spoken handover",
                     "Patient detail", "Waiting queue", "Reassessment round"],
    "Emergency physician": ["Patient board", "Patient detail", "Waiting queue",
                            "Reassessment round", "Review & override",
                            "What-if explorer"],
    "Clinical analyst": ["Patient board", "Model performance", "Safety frontier",
                         "Robustness & surge", "AI boundary", "Patient detail"],
    "Compliance officer": ["Audit log", "Clinical rule governance",
                           "AI boundary", "Model performance", "Patient detail"],
}

page = st.sidebar.radio("Navigate", PAGES_BY_ROLE[role], label_visibility="collapsed")

st.sidebar.markdown("---")
meta = pipeline.bundle.metadata
st.sidebar.caption(
    f"**Model** {pipeline.bundle.model_name}  \n"
    f"Trained on {meta.get('n_train', 0):,} real ED visits  \n"
    f"**Rules** {pipeline.rule_pack.pack_id} v{pipeline.rule_pack.version}  \n"
    f"`{pipeline.rule_pack.content_hash()}`"
)
st.sidebar.caption(
    "⚕️ Advisory only. Does not replace clinical judgment."
)


# ─── Routing ─────────────────────────────────────────────────────────────────

if page == "Patient board":
    from ui.dashboard import render_dashboard
    render_dashboard(pipeline, queue_manager, eval_results, role)

elif page == "New patient intake":
    from ui.patient_intake import render_patient_intake
    render_patient_intake(pipeline, queue_manager)

elif page == "Spoken handover":
    from ui.voice_intake import render_voice_intake
    render_voice_intake(pipeline, queue_manager)

elif page == "AI boundary":
    from ui.ai_boundary import render_ai_boundary
    render_ai_boundary(pipeline)

elif page == "Patient detail":
    from ui.triage_result import render_triage_result
    render_triage_result(pipeline, role)

elif page == "Waiting queue":
    from ui.waiting_queue import render_waiting_queue
    render_waiting_queue(queue_manager, pipeline)

elif page == "Reassessment round":
    from ui.reassessment_round import render_reassessment_round
    render_reassessment_round(pipeline, queue_manager)

elif page == "Review & override":
    from ui.clinician_review import render_clinician_review
    render_clinician_review(pipeline, override_manager, queue_manager)

elif page == "What-if explorer":
    from ui.what_if_explorer import render_what_if
    render_what_if(pipeline)

elif page == "Model performance":
    from ui.model_evaluation_ui import render_model_evaluation
    render_model_evaluation(eval_results, pipeline)

elif page == "Safety frontier":
    from ui.safety_frontier import render_safety_frontier
    render_safety_frontier(eval_results, pipeline)

elif page == "Robustness & surge":
    from ui.robustness_ui import render_robustness
    render_robustness(eval_results, pipeline)

elif page == "Clinical rule governance":
    from ui.governance import render_governance
    render_governance(pipeline)

elif page == "Audit log":
    from ui.audit_log import render_audit_log
    render_audit_log(pipeline, override_manager)
