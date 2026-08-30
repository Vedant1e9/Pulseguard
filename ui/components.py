"""PulseGuard — Shared UI components."""

from typing import Dict, List, Optional

import streamlit as st

LEVEL_COLORS = {1: "#DC2626", 2: "#EA580C", 3: "#CA8A04", 4: "#16A34A", 5: "#0284C7"}
LEVEL_NAMES = {1: "Resuscitation", 2: "Emergent", 3: "Urgent",
               4: "Less urgent", 5: "Non-urgent"}
LEVEL_EMOJIS = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}
LEVEL_TARGETS = {1: "immediately", "2": "10 min", 2: "10 min",
                 3: "30 min", 4: "60 min", 5: "when available"}

UNCERTAINTY_STYLE = {
    "low": ("🟢", "Low", "The record is complete and a critical presentation is "
                         "statistically excluded."),
    "moderate": ("🟡", "Moderate", "Some ambiguity remains. Worth a second look "
                                   "if anything changes."),
    "high": ("🔴", "High", "Key information is missing or a critical presentation "
                           "cannot be ruled out."),
}


def level_badge(level: int) -> str:
    return f"{LEVEL_EMOJIS.get(level, '⚪')} **Level {level}**: {LEVEL_NAMES.get(level, '?')}"


def safety_banner():
    st.caption(
        "⚕️ Clinical decision support prototype. Advisory only. Every "
        "recommendation is reviewable and overridable by a licensed clinician. "
        "Not validated for clinical use."
    )


def level_header(level: int, action: str, escalated: bool = False,
                 overridden: bool = False):
    """The big coloured decision banner at the top of a patient view."""
    color = LEVEL_COLORS.get(level, "#666")
    name = LEVEL_NAMES.get(level, "Unknown")
    emoji = LEVEL_EMOJIS.get(level, "⚪")

    tag = ""
    if overridden:
        tag = ('<span style="background:#1e293b;color:#fff;padding:3px 10px;'
               'border-radius:99px;font-size:12px;font-weight:600;margin-left:10px;">'
               'CLINICIAN OVERRIDE</span>')
    elif escalated:
        tag = ('<span style="background:#7c2d12;color:#fff;padding:3px 10px;'
               'border-radius:99px;font-size:12px;font-weight:600;margin-left:10px;">'
               'SAFETY ESCALATION</span>')

    st.markdown(f"""
    <div style="background:{color}18; border-left:8px solid {color};
                padding:18px 22px; border-radius:10px; margin:8px 0 18px 0;">
      <div class="pt-level-eyebrow">Triage decision</div>
      <div style="font-size:34px; font-weight:800; color:{color}; margin:2px 0 6px 0;">
        {emoji} Level {level}: {name}{tag}
      </div>
      <div class="pt-level-action">{action}</div>
    </div>
    """, unsafe_allow_html=True)


def confidence_row(result, model_output: Optional[Dict] = None):
    """
    The three numbers a nurse reads first.

    Confidence is labelled with what it actually measures. "Confidence: 91%"
    is meaningless on its own; "91% confident this patient is not sicker than
    Level 3" is a statement someone can act on.
    """
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Not under-triaged", f"{result.confidence_percent:.0f}%",
                  help="Probability this patient is no more urgent than the "
                       "assigned level. This is the safety-relevant question, "
                       "not the probability of an exact level match.")
    with c2:
        icon, label, explain = UNCERTAINTY_STYLE.get(
            result.uncertainty_band, ("⚪", "Unknown", ""))
        st.metric("Uncertainty", f"{icon} {label}", help=explain)
    with c3:
        st.metric("Record completeness", f"{result.data_quality_percent:.0f}%",
                  help="How much of the expected triage record was actually "
                       "captured. Low completeness widens uncertainty.")


def factor_list(factors: List[Dict], limit: int = 5):
    """
    Render the decision trace.

    Rules that decided the level are visually distinct from model evidence,
    because a clinician needs to know instantly whether a deterministic
    criterion fired or whether a statistical model leaned a certain way.
    """
    for f in factors[:limit]:
        source = f.get("source")
        if source == "safety_rule":
            st.markdown(
                f"""<div class="pt-factor pt-factor-rule">
                  <div class="pt-factor-head">⚖️ {f['headline']}</div>
                  <div class="pt-factor-detail">{f.get('detail', '')}</div>
                  <div class="pt-factor-meta">
                    Rule {f.get('rule_id', '')} · {f.get('citation', '')}</div>
                </div>""", unsafe_allow_html=True)
        elif source == "safety_rule_supporting":
            st.markdown(
                f"""<div class="pt-factor pt-factor-support">
                  <div class="pt-factor-head">⚑ {f['headline']}</div>
                  <div class="pt-factor-meta">
                    Supporting rule {f.get('rule_id', '')} · {f.get('citation', '')}</div>
                </div>""", unsafe_allow_html=True)
        else:
            attribution = f.get("attribution", 0.0)
            arrow = "▲" if attribution > 0 else "▼"
            arrow_class = "pt-arrow-up" if attribution > 0 else "pt-arrow-down"
            st.markdown(
                f"""<div class="pt-factor pt-factor-evidence">
                  <div class="pt-factor-head">
                    <span class="{arrow_class}">{arrow}</span> {f['headline']}</div>
                  <div class="pt-factor-detail">{f.get('detail', '')}</div>
                </div>""", unsafe_allow_html=True)


def missing_info_panel(not_recorded: List[str], followup: Optional[str] = None):
    if not_recorded:
        st.markdown("**Not recorded**")
        for item in not_recorded:
            st.markdown(f"- ⚪ {item}")
        st.caption(
            "These are shown as unknown, never as normal. A missing measurement "
            "widens the uncertainty band rather than being filled with a "
            "plausible-looking value."
        )
        if followup:
            st.info(f"💬 **Ask next:** {followup}")
    else:
        st.success("✓ All expected observations were recorded.")


def rule_pack_footer(rule_pack: Dict):
    if not rule_pack:
        return
    st.caption(
        f"Decision governed by rule pack **{rule_pack.get('pack_id')}** "
        f"v{rule_pack.get('version')} (`{rule_pack.get('content_hash')}`) · "
        f"{rule_pack.get('n_rules_enabled')}/{rule_pack.get('n_rules_total')} rules "
        f"enabled · jurisdiction: {rule_pack.get('jurisdiction')}"
    )


def source_tag(description: str):
    """Make the provenance of every demo patient unmissable."""
    if description.startswith("SYNTHETIC"):
        st.warning(
            "**Synthetic edge case.** Constructed to exercise a specific "
            "safety rule that real survey data cannot trigger, because NHAMCS "
            "does not record bedside observations. Excluded from every "
            "accuracy metric.", icon="🧪")
    elif description.startswith("NHAMCS"):
        st.info(
            "**Real patient record** from the CDC's National Hospital "
            "Ambulatory Medical Care Survey, drawn from a hospital held out of "
            "training. The reference level is the assignment an actual triage "
            "nurse made.", icon="🏥")
    elif description.startswith("LIVE INTAKE"):
        st.info(
            "**Live intake.** Entered through this application on this device. "
            "There is no nurse reference level to compare against, so this "
            "patient appears in no accuracy figure.", icon="🩺")
