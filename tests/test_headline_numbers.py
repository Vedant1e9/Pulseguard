"""
The published headline numbers.

These are the figures that end up on a slide, in a proposal and on a CV, which
makes them the numbers with the shortest path to being wrong in public. Two
things are pinned here:

  * the classification metrics agree with scikit-learn, so the hand-rolled
    arithmetic in the generator cannot quietly drift; and
  * the generated document says what it must say, including the parts that are
    unflattering.
"""

import json
import os

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from scripts.generate_metrics import classification_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "evaluation", "saved_results")
EVAL_JSON = os.path.join(RESULTS, "evaluation_full.json")
NUMBERS_JSON = os.path.join(RESULTS, "headline_numbers.json")
NUMBERS_MD = os.path.join(RESULTS, "HEADLINE_NUMBERS.md")


@pytest.fixture(scope="module")
def evaluation():
    if not os.path.exists(EVAL_JSON):
        pytest.skip("Evaluation results not generated yet.")
    with open(EVAL_JSON, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def labelled(evaluation):
    """Reconstruct y_true / y_pred from the stored confusion matrix."""
    cm = evaluation["confusion_matrix"]
    labels, matrix = cm["labels"], cm["matrix"]
    y_true, y_pred = [], []
    for i, truth in enumerate(labels):
        for j, pred in enumerate(labels):
            y_true += [truth] * matrix[i][j]
            y_pred += [pred] * matrix[i][j]
    return np.array(y_true), np.array(y_pred), labels


# ── The arithmetic must match a reference implementation ─────────────────────

def test_macro_and_weighted_f1_match_sklearn(evaluation, labelled):
    y_true, y_pred, _ = labelled
    ours = classification_metrics(evaluation["confusion_matrix"])["five_class"]
    assert ours["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0), abs=1e-4)
    assert ours["weighted_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="weighted", zero_division=0), abs=1e-4)
    assert ours["accuracy"] == pytest.approx(
        accuracy_score(y_true, y_pred), abs=1e-4)


def test_per_class_metrics_match_sklearn(evaluation, labelled):
    y_true, y_pred, labels = labelled
    ours = classification_metrics(evaluation["confusion_matrix"])["five_class"]["per_class"]
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    for i, label in enumerate(labels):
        got = ours[f"level_{label}"]
        assert got["precision"] == pytest.approx(p[i], abs=1e-4)
        assert got["recall"] == pytest.approx(r[i], abs=1e-4)
        assert got["f1"] == pytest.approx(f[i], abs=1e-4)
        assert got["support"] == s[i]


def test_binary_critical_metrics_match_sklearn(evaluation, labelled):
    y_true, y_pred, _ = labelled
    ours = classification_metrics(evaluation["confusion_matrix"])["critical_binary"]
    p, r, f, _ = precision_recall_fscore_support(
        (y_true <= 2).astype(int), (y_pred <= 2).astype(int),
        average="binary", zero_division=0)
    assert ours["precision"] == pytest.approx(p, abs=1e-4)
    assert ours["recall"] == pytest.approx(r, abs=1e-4)
    assert ours["f1"] == pytest.approx(f, abs=1e-4)


def test_binary_recall_agrees_with_the_headline_critical_recall(evaluation):
    """
    The F1 block and the safety block must describe the same system.

    They are computed by different code paths, so a disagreement between them
    would mean one of the two published numbers is stale.
    """
    derived = classification_metrics(evaluation["confusion_matrix"])["critical_binary"]
    reported = evaluation["primary_metrics"]["with_confidence_intervals"]["critical_recall"]["point"]
    assert derived["recall"] == pytest.approx(reported, abs=5e-3)


# ── The published artifact must exist and stay honest ────────────────────────

@pytest.fixture(scope="module")
def published():
    if not os.path.exists(NUMBERS_JSON):
        pytest.skip("Run `python -m scripts.generate_metrics` first.")
    with open(NUMBERS_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_challenge_givens_are_kept_separate_from_results(published):
    """
    Scope must never be presented as achievement.

    "100 to 500+ visits per day" is the brief describing a target environment.
    If it migrates into the measured section it becomes a claim about
    throughput this system delivered, which would be false.
    """
    given = published["given_by_the_challenge"]
    assert given, "the challenge-supplied numbers must be listed"
    measured = json.dumps(published["measured_from_this_prototype"])
    assert "100 to 500" not in measured
    assert any("not achieved throughput" in row["kind"] for row in given)


def test_the_five_level_scale_is_reported_as_five(published):
    """
    A regression with real consequences: an earlier version counted the levels
    that happened to appear on the 31-patient demo board and published
    "3 levels implemented", which reads as though the scale was never built.
    """
    scale = published["measured_from_this_prototype"]["prototype"]["triage_scale"]
    assert scale["n_levels_implemented"] == 5
    assert scale["levels_implemented"] == [1, 2, 3, 4, 5]


def test_two_input_modalities_are_reported(published):
    modalities = published["measured_from_this_prototype"]["prototype"]["input_modalities"]
    assert modalities["count"] == 2
    assert any("voice" in m or "spoken" in m for m in modalities["modalities"])


def test_the_extractor_can_write_no_decision_field(published):
    handover = published["measured_from_this_prototype"]["prototype"]["spoken_handover"]
    assert handover["decision_fields_extractable"] == 0
    assert handover["extractable_input_fields"] > 0


def test_every_scored_patient_carries_a_confidence(published):
    invariants = published["measured_from_this_prototype"]["prototype"]["invariants"]
    assert invariants["every_score_has_a_confidence"] is True
    assert invariants["patients_scored"] == invariants["patients_with_a_confidence"]


def test_the_document_warns_against_quoting_macro_f1(published):
    """
    Macro F1 is 0.24 here, by design. The document must say so rather than
    leaving the number to be quoted bare into a CV.
    """
    if not os.path.exists(NUMBERS_MD):
        pytest.skip("Markdown artifact not generated.")
    text = open(NUMBERS_MD, encoding="utf-8").read()
    assert "Do not put that on" in text
    assert "cannot hurt anyone" in text
    macro = published["measured_from_this_prototype"]["classification"]["five_class"]["macro_f1"]
    assert f"{macro:.3f}" in text, "the actual macro F1 must appear, not be hidden"


def test_the_document_keeps_the_outcome_caveat_attached(published):
    if not os.path.exists(NUMBERS_MD):
        pytest.skip("Markdown artifact not generated.")
    text = open(NUMBERS_MD, encoding="utf-8").read()
    assert "suggestive" in text
    assert "No prospective or clinical validation" in text


def test_no_em_dashes_in_the_published_numbers():
    for path in (NUMBERS_MD, NUMBERS_JSON):
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        assert "—" not in text and "–" not in text, f"{path} contains a dash"


# ── Clinical predictive values ───────────────────────────────────────────────

def test_predictive_values_match_a_direct_computation(evaluation, labelled):
    """
    NPV and PPV are the two numbers a clinician weighs against each other, so
    they are checked against confusion counts computed independently here
    rather than trusted from the generator.
    """
    y_true, y_pred, _ = labelled
    truth, pred = (y_true <= 2), (y_pred <= 2)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())

    cpv = classification_metrics(evaluation["confusion_matrix"])["clinical_predictive_values"]
    assert cpv["positive_predictive_value"] == pytest.approx(tp / (tp + fp), abs=1e-4)
    assert cpv["negative_predictive_value"] == pytest.approx(tn / (tn + fn), abs=1e-4)
    assert cpv["sensitivity_recall"] == pytest.approx(tp / (tp + fn), abs=1e-4)
    assert cpv["specificity"] == pytest.approx(tn / (tn + fp), abs=1e-4)


def test_npv_is_higher_than_ppv(evaluation):
    """
    A system deliberately tuned to accept false alarms rather than misses must
    show exactly this asymmetry. If PPV ever exceeded NPV the cost policy would
    have stopped doing its job, and the headline claim would be wrong.
    """
    cpv = classification_metrics(evaluation["confusion_matrix"])["clinical_predictive_values"]
    assert cpv["negative_predictive_value"] > cpv["positive_predictive_value"]


# ── The exchange rate behind the headline gain ───────────────────────────────

def test_marginal_cost_is_derived_from_the_baseline_table(evaluation):
    from scripts.generate_metrics import marginal_cost_of_safety
    mc = marginal_cost_of_safety(evaluation)
    ours = evaluation["baseline_comparison"]["PulseGuard (cost-sensitive policy)"]
    argmax = evaluation["baseline_comparison"]["Same model, accuracy-maximising argmax"]

    assert mc["n_test_fold"] == ours["n"]
    assert mc["extra_critical_patients_caught"] == round(
        (ours["critical_recall"] - argmax["critical_recall"]) * ours["n_critical"])
    assert mc["patients_over_triaged_per_extra_critical_caught"] > 0


def test_the_exchange_rate_is_reported_with_the_recall_gain(published):
    """
    A recall improvement quoted without its cost is not a claim, it is an
    advertisement. The document must carry both.
    """
    if not os.path.exists(NUMBERS_MD):
        pytest.skip("Markdown artifact not generated.")
    text = open(NUMBERS_MD, encoding="utf-8").read()
    assert "over-triaged per extra critical patient caught" in text
    mc = published["measured_from_this_prototype"]["marginal_cost_of_safety"]
    assert str(mc["patients_over_triaged_per_extra_critical_caught"]) in text


# ── Engineering metrics ──────────────────────────────────────────────────────

def test_coverage_is_measured_not_asserted(published):
    cov = published["measured_from_this_prototype"]["test_coverage"]
    if not cov.get("available"):
        pytest.skip("pytest-cov not installed in this environment.")
    assert 0 < cov["runtime_code_pct"] <= 100
    assert cov["safety_engine_pct"] is not None, \
        "coverage of the component that sets a triage level must be reported"
    assert cov["safety_engine_pct"] >= cov["runtime_code_pct"], \
        "the safety engine should be better covered than the codebase average"


def test_codebase_counts_are_plausible(published):
    cb = published["measured_from_this_prototype"]["codebase"]
    assert cb["source_files"] > 20 and cb["source_lines"] > 5000
    assert cb["test_files"] >= 10 and cb["test_lines"] > 1000
