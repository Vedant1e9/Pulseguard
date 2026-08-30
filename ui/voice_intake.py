"""PulseGuard: Spoken handover intake."""

import hashlib
from datetime import datetime

import streamlit as st

from data.input_schema import (
    DataSource, Measurement, MedicalHistory, PatientEncounter,
    SelfReportedSymptoms, StaffObservedCues, VitalSigns,
)
from engine.voice_intake import (
    CATEGORICAL_VALUES, CONFIDENCE_FLOOR, SAMPLE_HANDOVERS,
    extract, extraction_backend_status, transcribe,
)
from engine.explanation import ExplanationBuilder
from ui.components import (
    confidence_row, factor_list, level_header,
    missing_info_panel, rule_pack_footer, safety_banner,
)

VITAL_LABELS = {
    "temperature": ("Temperature", "°C"), "heart_rate": ("Heart rate", "bpm"),
    "respiratory_rate": ("Respiratory rate", "/min"), "spo2": ("Oxygen saturation", "%"),
    "systolic_bp": ("Systolic BP", "mmHg"), "diastolic_bp": ("Diastolic BP", "mmHg"),
    "pain_score": ("Pain score", "/10"),
}
CUE_FIELDS = ["consciousness", "breathing_difficulty", "visible_distress",
              "mobility", "bleeding", "skin_appearance"]


def render_voice_intake(pipeline, queue_manager=None):
    st.title("Spoken handover intake")
    safety_banner()

    st.markdown(
        "A triage decision has to be made in seconds by a nurse who is already "
        "holding a blood pressure cuff. The bottleneck in that moment is the "
        "keyboard, not the model. This page takes the sentence a nurse says "
        "out loud anyway during handover and drafts the same structured record "
        "the typed form produces."
    )

    st.warning(
        "**Speech drafts the record. It never sets the level.** Everything "
        "below is a *suggestion* that a nurse confirms before anything is "
        "scored. The extractor cannot write a triage level, a confidence or a "
        "safety decision, because those fields are not in its output schema at "
        "all. The deterministic safety engine remains the only thing that sets "
        "an urgency, exactly as it does for the typed form.", icon="🛡️")

    status = extraction_backend_status()
    _render_result_if_any(pipeline)

    # ── 1. Capture ──
    st.markdown("### 1. Capture the handover")

    tab_speak, tab_type, tab_sample = st.tabs(
        ["🎙️ Record", "⌨️ Type or paste", "📋 Sample handovers"])

    transcript = st.session_state.get("voice_transcript", "")
    backend = st.session_state.get("voice_asr_backend", "none")
    asr_ms = st.session_state.get("voice_asr_ms", 0.0)

    with tab_speak:
        st.markdown("**Press the microphone, say the handover, press stop.**")
        audio = st.audio_input(
            "Record the handover", key="voice_mic",
            help="Records straight from your microphone. No file to choose, "
                 "no upload step.")

        if status["asr_backends"]:
            first = status["asr_backends"][0]
            if "on device" in first:
                st.caption(f"Transcribed by **{first}**. The recording never "
                           f"leaves this machine.")
            else:
                st.caption(f"Transcribed by **{first}**. This backend sends "
                           f"audio off the machine; install `faster-whisper` "
                           f"to keep it on device.")
        else:
            st.info(
                "No speech-to-text backend is available, so audio is captured "
                "but not transcribed. Install one with `pip install "
                "faster-whisper` to transcribe on device, or set "
                "`OPENAI_API_KEY`. Until then, **Type or paste** and **Sample "
                "handovers** exercise every stage after transcription.",
                icon="ℹ️")

        if audio is not None:
            raw = audio.getvalue()
            # Transcribe as soon as a recording exists, rather than behind a
            # second button. The whole point of the page is to remove steps
            # from a nurse's hands, and "record, then also press transcribe"
            # puts one back. The digest guards against re-transcribing the
            # same clip on every rerun of the script.
            digest = hashlib.sha1(raw).hexdigest()
            st.caption(f"Captured {len(raw) / 1024:.0f} KB of audio.")

            if not status["asr_backends"]:
                pass
            elif st.session_state.get("voice_audio_digest") != digest:
                with st.spinner("Transcribing …"):
                    text, used, ms = transcribe(raw)
                st.session_state["voice_audio_digest"] = digest
                if text:
                    st.session_state.update(voice_transcript=text,
                                            voice_asr_backend=used,
                                            voice_asr_ms=ms)
                    st.rerun()
                else:
                    st.error(
                        f"Transcription produced no text ({used}). Re-record, "
                        f"or use **Type or paste**.")
            elif st.button("Transcribe again", key="voice_retranscribe"):
                st.session_state.pop("voice_audio_digest", None)
                st.rerun()

    with tab_type:
        typed = st.text_area(
            "Handover transcript", value=transcript, height=130,
            placeholder="Eighty one year old female brought in by ambulance, "
                        "heart rate one eighteen, sats ninety on air …",
            key="voice_typed")
        if st.button("Extract from this text", type="primary",
                     use_container_width=True):
            st.session_state.update(voice_transcript=typed,
                                    voice_asr_backend="typed directly",
                                    voice_asr_ms=0.0)
            st.rerun()

    with tab_sample:
        st.caption(
            "Four recorded-cadence handovers, including numbers spoken as "
            "words and a deliberately noisy one.")
        for label, text in SAMPLE_HANDOVERS.items():
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{label}**  \n<span style='font-size:13px;opacity:.8'>"
                        f"{text}</span>", unsafe_allow_html=True)
            if c2.button("Use", key=f"sample_{label}"):
                st.session_state.update(voice_transcript=text,
                                        voice_asr_backend="sample transcript",
                                        voice_asr_ms=0.0)
                st.rerun()

    if not transcript:
        return

    # ── 2. Extract ──
    st.markdown("---")
    st.markdown("### 2. What the system heard")
    st.info(f"“{transcript}”")

    result = extract(transcript)
    for warning in result.warnings:
        st.caption(f"⚠️ {warning}")

    t1, t2, t3, t4 = st.columns(4)
    # Short labels: four metric tiles in the content column truncate their
    # headings at anything longer, and a truncated label on a provenance tile
    # defeats the point of showing provenance.
    t1.metric("Source", backend if backend != "none" else "n/a",
              help=f"How this transcript reached the system. "
                   f"{asr_ms:.0f} ms" if asr_ms else
                   "How this transcript reached the system.")
    t2.metric("Extractor", "Claude" if "Claude" in result.extraction_backend
              else "Rules",
              help=result.extraction_backend)
    t3.metric("Latency", f"{result.extraction_ms:.0f} ms",
              help="Time to turn the transcript into candidate fields.")
    t4.metric("Drafted", len(result.fields),
              delta=f"{len(result.rejected)} rejected" if result.rejected else None,
              delta_color="off")

    if result.rejected:
        with st.expander(f"🚫 {len(result.rejected)} candidate(s) rejected at the "
                         f"boundary", expanded=True):
            st.caption(
                "Kept visible on purpose. A nurse who said a number and cannot "
                "see it needs to know the system heard something and discarded "
                "it, rather than assuming it was never listening.")
            for f in result.rejected:
                st.markdown(f"- **{f.name}** heard as `{f.value}`. {f.note}")

    # ── 3. Confirm ──
    st.markdown("---")
    st.markdown("### 3. Confirm before scoring")
    st.caption(
        "Nothing below has been scored yet. Every value is editable, and any "
        "field the extractor was unsure about was left empty rather than "
        "guessed, which the pipeline handles by widening the uncertainty band.")

    drafted = result.values()

    with st.form("voice_confirm"):
        st.markdown("#### Patient")
        p1, p2, p3, p4 = st.columns(4)
        patient_id = p1.text_input("Patient ID", value=f"VOX-{datetime.now():%H%M%S}")
        age = p2.number_input(
            "Age (years)", 0, 120,
            value=int(drafted["age"]) if "age" in drafted else None,
            placeholder="required", format="%d")
        sex_options = CATEGORICAL_VALUES["sex"]
        sex = p3.selectbox("Sex", sex_options,
                           index=sex_options.index(drafted["sex"])
                           if drafted.get("sex") in sex_options else 0)
        arrival_default = 1 if drafted.get("arrival_by_ambulance") else 0
        arrival_mode = p4.selectbox("Arrival", ["Walk-in", "Ambulance"],
                                    index=arrival_default)

        st.markdown("#### Vital signs")
        vitals_in = {}
        cols = st.columns(4)
        for i, (key, (label, unit)) in enumerate(VITAL_LABELS.items()):
            with cols[i % 4]:
                heard = drafted.get(key)
                # Temperature is the only vital carrying a decimal. Streamlit
                # matches the widget's numeric type to its format string, so
                # the pre-filled value has to be cast to match or the field
                # renders a type warning above the number a nurse is checking.
                if key == "temperature":
                    vitals_in[key] = st.number_input(
                        f"{label} ({unit})",
                        value=float(heard) if heard is not None else None,
                        placeholder="not measured", format="%.1f", step=0.1,
                        key=f"vox_{key}")
                else:
                    vitals_in[key] = st.number_input(
                        f"{label} ({unit})",
                        value=int(heard) if heard is not None else None,
                        placeholder="not measured", format="%d", step=1,
                        key=f"vox_{key}")
                st.caption(_provenance(result, key))

        st.markdown("#### Presenting complaint")
        complaint = st.text_input("Chief complaint",
                                  value=str(drafted.get("chief_complaint", "")))

        st.markdown("#### Bedside observations")
        cue_in = {}
        cue_cols = st.columns(3)
        for i, key in enumerate(CUE_FIELDS):
            options = CATEGORICAL_VALUES[key]
            heard = drafted.get(key)
            with cue_cols[i % 3]:
                cue_in[key] = st.selectbox(
                    key.replace("_", " ").capitalize(), options,
                    index=options.index(heard) if heard in options else 0,
                    key=f"vox_cue_{key}")
                st.caption(_provenance(result, key))

        history_available = st.toggle(
            "A medical record is available",
            value=bool(drafted.get("history_available", False)))

        submitted = st.form_submit_button(
            "🩺 Confirm and run triage", type="primary", use_container_width=True)

    if not submitted:
        return

    if age is None:
        st.error("Age is required. Every threshold in this system is age-banded.")
        return

    def m(value, unit="", source=DataSource.VOICE_TRANSCRIBED):
        if value is None or value == "":
            return None
        return Measurement(value=value, unit=unit, source=source,
                           timestamp=datetime.now())

    encounter = PatientEncounter(
        patient_id=patient_id, age=int(age), sex=sex, arrival_time=datetime.now(),
        vitals=VitalSigns(**{
            k: m(v, VITAL_LABELS[k][1]) for k, v in vitals_in.items()
        }),
        symptoms=SelfReportedSymptoms(chief_complaint=m(complaint or None)),
        history=MedicalHistory(history_available=history_available),
        context={"arrival_by_ambulance": 1.0 if arrival_mode == "Ambulance" else 0.0},
        staff_cues=StaffObservedCues(**{k: m(v) for k, v in cue_in.items()}),
    )

    with st.spinner("Running triage …"):
        triage = pipeline.triage_patient(encounter)

    if queue_manager is not None:
        stored = pipeline.triage_results[patient_id]
        velocity = stored.get("velocity", {})
        queue_manager.add_patient(
            patient_id=patient_id, triage_level=triage.triage_level,
            age_group=encounter.age_group.value,
            arrival_time=encounter.arrival_time,
            confidence=triage.confidence_percent,
            uncertainty=triage.uncertainty_band,
            velocity_risk=(velocity.get("overall_risk", "low")
                           if velocity.get("has_trend_data") else "insufficient_data"))

    st.session_state["voice_last_patient"] = patient_id
    st.session_state["voice_transcript"] = ""
    st.rerun()


def _provenance(result, key: str) -> str:
    """Say where a pre-filled value came from, or that nothing was heard."""
    f = result.fields.get(key)
    if not f:
        return "not heard, left unmeasured"
    return f"heard at {f.confidence:.0%}: “{f.evidence[:44]}”"


def _render_result_if_any(pipeline):
    """The decision from the last confirmed handover, rendered above the page."""
    last = st.session_state.get("voice_last_patient")
    if not last or last not in pipeline.triage_results:
        return

    stored = pipeline.triage_results[last]
    result, trace = stored["result"], stored["explanation"]

    st.success(f"Triage complete for **{last}**, from a spoken handover.",
               icon="✅")
    level_header(result.triage_level, result.recommended_action,
                 escalated=(result.safety_status == "escalation_applied"))
    confidence_row(result, stored["model_output"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Why this level")
        st.caption(f"Decided by: **{trace['decided_by']}**")
        factor_list(trace["factors"], limit=5)
    with right:
        st.markdown("#### What we don't know")
        missing_info_panel(trace["not_recorded"],
                           result.recommended_followup_question)

    st.caption(
        f"Scored in {stored['latency_ms']:.0f} ms. Values that reached this "
        f"record through speech are stamped `voice_transcribed` in the audit "
        f"log, so an auditor can find every value that passed through a speech "
        f"model.")
    rule_pack_footer(stored["safety"].get("rule_pack", {}))
    st.markdown("---")
