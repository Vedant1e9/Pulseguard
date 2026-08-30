"""PatientTriage.ai — Clinical rule governance."""

import pandas as pd
import streamlit as st

from engine.rule_pack import RulePack, available_site_packs
from ui.components import safety_banner


def render_governance(pipeline):
    st.title("Clinical rule governance")
    safety_banner()

    pack = pipeline.rule_pack
    provenance = pack.provenance()

    st.markdown(
        "Every escalation this system makes comes from a rule in a versioned, "
        "human-readable file, not from model internals. A clinician can read "
        "the whole policy, argue with any line of it, and change it without a "
        "code deployment. This page is that file, rendered."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rule pack", provenance["pack_id"])
    c2.metric("Version", provenance["version"])
    c3.metric("Rules enabled",
              f"{provenance['n_rules_enabled']}/{provenance['n_rules_total']}")
    c4.metric("Content hash", provenance["content_hash"])

    st.caption(
        f"Jurisdiction: **{provenance['jurisdiction']}** · effective "
        f"{provenance['effective_date']} · source `{provenance['source']}`. "
        f"The content hash is recorded against every triage decision, so an "
        f"audit years later can reconstruct exactly which policy text was in "
        f"force. A version string can be forgotten on edit, a content hash "
        f"cannot."
    )

    st.markdown("---")
    st.subheader("The rules")

    rules = pack.summary_table()
    for rule in rules:
        spec = pack.rule(rule["id"]) or {}
        certainty = spec.get("certainty", "n/a")
        status = "🟢 enabled" if rule["enabled"] else "⚪ disabled"
        target = (f"→ Level {rule['escalates_to']}" if rule["escalates_to"]
                  else "→ triggers reassessment")

        with st.expander(f"**{rule['id']}** · {status} · {target}"):
            st.markdown(f"**What it does:** {rule['description']}")
            st.markdown(f"**Why:** {rule['rationale']}")
            meta = st.columns(3)
            meta[0].caption(f"**Category**  \n{rule['category']}")
            meta[1].caption(f"**Certainty**  \n{certainty}")
            meta[2].caption(f"**Citation**  \n{rule['citation']}")

    st.info(
        "**`certainty` is not decoration.** An *observed* rule fires on a "
        "measured fact. An unresponsive patient is a Level 1 whatever the "
        "model thinks, so its escalation carries high confidence. A "
        "*precautionary* rule fires because something is unknown, and inheriting "
        "high confidence from it would make the system sound most certain "
        "exactly when it knows least. The two produce different confidence "
        "figures on the patient screen.", icon="🎚️")

    # ── Age-banded thresholds ──
    st.markdown("---")
    st.subheader("Age-banded vital sign thresholds")
    st.markdown(
        "A single adult-calibrated threshold set is the silent safety risk the "
        "brief names explicitly: it cries wolf on children, whose normal heart "
        "rate is adult tachycardia, and stays silent on the elderly, who "
        "decompensate at numbers a younger adult tolerates."
    )

    threshold_rows = []
    for band in ["pediatric", "adult", "geriatric"]:
        t = pack.thresholds_for(band)
        threshold_rows.append({
            "Age band": band.capitalize(),
            "HR high": t.get("hr_high"), "HR low": t.get("hr_low"),
            "RR high": t.get("rr_high"), "RR low": t.get("rr_low"),
            "SpO₂ low": t.get("spo2_low"), "SpO₂ critical": t.get("spo2_critical"),
            "SBP low": t.get("sbp_low"), "SBP high": t.get("sbp_high"),
            "Temp high": t.get("temp_high"), "Temp low": t.get("temp_low"),
        })
    st.dataframe(pd.DataFrame(threshold_rows), use_container_width=True, hide_index=True)
    st.caption("Source for each band: " + " · ".join(
        f"**{b}**: {pack.thresholds_for(b).get('citation', 'not cited')}"
        for b in ["pediatric", "adult", "geriatric"]))

    # ── Site comparison ──
    st.markdown("---")
    st.subheader("How another site differs")
    st.markdown(
        "Site packs are **deltas**, not copies. A rural department's file "
        "contains only what it changes, so reviewing its policy means reading "
        "a dozen lines rather than diffing three hundred, and an improvement "
        "to a shared clinical rule reaches every site without being re-applied "
        "by hand."
    )

    other_sites = [s for s in available_site_packs()]
    if not other_sites:
        st.caption("No site overlay packs configured.")
        return

    site = st.selectbox("Compare against", other_sites,
                        format_func=lambda s: s.replace("_", " ").title())
    try:
        other = RulePack.load_site(site)
    except FileNotFoundError:
        st.warning("Site pack not found.")
        return

    st.markdown(f"**{other.pack_id}** v{other.version}")
    st.caption(other.data.get("description", ""))

    diff_rows = []
    for band in ["pediatric", "adult", "geriatric"]:
        base_t, other_t = pack.thresholds_for(band), other.thresholds_for(band)
        for key in sorted(set(base_t) | set(other_t)):
            if key == "citation":
                continue
            if base_t.get(key) != other_t.get(key):
                diff_rows.append({
                    "Age band": band, "Threshold": key,
                    "This site": base_t.get(key), site: other_t.get(key),
                })

    for rule in other.data["rules"]:
        base_rule = pack.rule(rule["id"]) or {}
        if base_rule.get("escalate_to") != rule.get("escalate_to"):
            diff_rows.append({
                "Age band": "n/a", "Threshold": f"{rule['id']} escalates to",
                "This site": base_rule.get("escalate_to"),
                site: rule.get("escalate_to"),
            })

    if diff_rows:
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Every one of these differences is a deliberate clinical decision "
            "with a written rationale in the site's pack file, reviewable in "
            "version control."
        )
    else:
        st.info("No threshold differences between these packs.")
