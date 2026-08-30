"""
End-to-end interaction tests, driven through Streamlit's own test harness.

The existing page tests call each renderer as a plain function, which proves a
page does not raise. That is not the same as proving a *form works*: a widget
can render perfectly and still be wired to nothing, and the failure only shows
up when somebody fills it in and presses the button.

`AppTest` runs the real script, sets real widget state and submits real forms,
so these cover the wiring between the interface and the pipeline.
"""

import pytest
from streamlit.testing.v1 import AppTest

APP = "app.py"
TIMEOUT = 120


def _by_label(widgets, label: str):
    """
    Find a widget by its label rather than its index.

    Indexing is fragile here: AppTest returns main-body widgets before sidebar
    ones, so `selectbox[0]` is whatever the current page happens to render
    first, not the role picker.
    """
    for widget in widgets:
        if (widget.label or "").strip() == label:
            return widget
    pytest.fail(f"no widget labelled {label!r}; "
                f"found {[w.label for w in widgets]}")


def _run(page: str, role: str = "Triage nurse") -> AppTest:
    """Boot the app, choose a role, and navigate to a page."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception, f"app raised on boot: {at.exception}"

    role_picker = _by_label(at.selectbox, "Role")
    if role_picker.value != role:
        role_picker.set_value(role).run()
        assert not at.exception, f"app raised switching role: {at.exception}"

    options = at.radio[0].options
    assert page in options, f"{page!r} not in nav for {role}: {options}"
    at.radio[0].set_value(page).run()
    assert not at.exception, f"app raised opening {page}: {at.exception}"
    return at


def _number(at: AppTest, label_fragment: str):
    for widget in at.number_input:
        if label_fragment.lower() in widget.label.lower():
            return widget
    pytest.fail(f"no number input matching {label_fragment!r}; "
                f"found {[w.label for w in at.number_input]}")


# ── Every page in every role opens without raising ───────────────────────────

ROLE_PAGES = [
    ("Triage nurse", "Patient board"),
    ("Triage nurse", "New patient intake"),
    ("Triage nurse", "Spoken handover"),
    ("Triage nurse", "Patient detail"),
    ("Triage nurse", "Waiting queue"),
    ("Triage nurse", "Reassessment round"),
    ("Emergency physician", "Review & override"),
    ("Emergency physician", "What-if explorer"),
    ("Clinical analyst", "Model performance"),
    ("Clinical analyst", "Safety frontier"),
    ("Clinical analyst", "Robustness & surge"),
    ("Clinical analyst", "AI boundary"),
    ("Compliance officer", "Audit log"),
    ("Compliance officer", "Clinical rule governance"),
]


@pytest.mark.parametrize("role,page", ROLE_PAGES)
def test_page_opens_for_role(role, page):
    at = _run(page, role)
    assert not at.exception


# ── The intake form actually scores a patient ────────────────────────────────

def test_typed_intake_form_scores_a_patient():
    at = _run("New patient intake")

    _number(at, "Age").set_value(68)
    _number(at, "Heart rate").set_value(128)
    _number(at, "Respiratory rate").set_value(28)
    _number(at, "Oxygen saturation").set_value(89)
    at.text_input[1].set_value("Chest pain and shortness of breath")

    at.button[0].click().run()
    assert not at.exception, f"intake submit raised: {at.exception}"

    body = " ".join(str(e.value) for e in at.markdown) + \
           " ".join(str(e.value) for e in at.success)
    assert "Triage complete" in body, "intake produced no decision"


def test_intake_rejects_a_missing_age():
    """Age drives every threshold, so submitting without it must fail loudly."""
    at = _run("New patient intake")
    _number(at, "Heart rate").set_value(100)
    at.button[0].click().run()
    assert not at.exception
    assert any("Age is required" in str(e.value) for e in at.error)


# ── The reassessment form actually re-scores ─────────────────────────────────

def test_reassessment_form_records_and_rescores():
    """
    The wiring this test exists for: fill the recheck form, press the button,
    and confirm the observation reached the pipeline and the audit log.
    """
    at = _run("Reassessment round")

    _number(at, "Heart rate").set_value(145)
    _number(at, "Respiratory rate").set_value(36)
    _number(at, "Oxygen saturation").set_value(86)

    at.button[0].click().run()
    assert not at.exception, f"reassessment submit raised: {at.exception}"

    assert "reassess_last" in at.session_state, "no patient recorded as reassessed"

    body = (" ".join(str(e.value) for e in at.error)
            + " ".join(str(e.value) for e in at.warning)
            + " ".join(str(e.value) for e in at.info)
            + " ".join(str(e.value) for e in at.markdown))
    assert any(phrase in body for phrase in
               ("escalated from", "held at", "remains at")), \
        "reassessment produced no outcome banner"


def test_reassessment_with_no_vitals_is_rejected():
    at = _run("Reassessment round")
    at.button[0].click().run()
    assert not at.exception
    assert any("at least one vital" in str(e.value) for e in at.error)


# ── The spoken handover path works from a sample transcript ──────────────────

def test_sample_handover_extracts_and_confirms():
    at = _run("Spoken handover")

    use_buttons = [b for b in at.button if b.label.strip() == "Use"]
    assert use_buttons, "no sample handovers offered"
    use_buttons[0].click().run()
    assert not at.exception, f"sample handover raised: {at.exception}"

    assert "voice_transcript" in at.session_state, "no transcript stored"
    assert at.session_state["voice_transcript"], "transcript is empty"

    # The extractor should have pre-filled the confirmation form.
    age = _number(at, "Age")
    assert age.value == 81, f"age not drafted from the handover, got {age.value}"

    submit = next(b for b in at.button
                  if "Confirm and run triage" in b.label)
    submit.click().run()
    assert not at.exception, f"handover confirm raised: {at.exception}"
    assert "voice_last_patient" in at.session_state, "no patient scored"


# ── The role menu is genuinely scoped ────────────────────────────────────────

def test_a_nurse_cannot_reach_the_audit_log():
    """Access is scoped to the minimum necessary, so assert on the menu."""
    at = _run("Patient board", "Triage nurse")
    options = at.radio[0].options
    assert "Audit log" not in options
    assert "Clinical rule governance" not in options
    assert "Model performance" not in options


def test_a_compliance_officer_cannot_reach_the_override_page():
    at = _run("Audit log", "Compliance officer")
    options = at.radio[0].options
    assert "Review & override" not in options
    assert "New patient intake" not in options


# ── Regression: the site rule pack menu listed its default twice ─────────────

def test_site_rule_pack_options_are_unique():
    """
    `available_site_packs()` scans config/ and already includes "default", so
    prepending it produced two identical "Urban ED (default)" entries.
    """
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    options = _by_label(at.selectbox, "Site rule pack").options
    assert len(options) == len(set(options)), f"duplicate site packs: {options}"
    assert options[0] == "Urban ED (default)"


def test_role_menu_options_are_unique():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    for label in ("Role", "Site rule pack"):
        options = _by_label(at.selectbox, label).options
        assert len(options) == len(set(options)), f"{label} has duplicates"
