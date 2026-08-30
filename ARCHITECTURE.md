# Architecture: PulseGuard

## The one-sentence version

Statistical models estimate risk; a deterministic, versioned rule engine makes
the decision; every layer's output is retained so the explanation a clinician
reads and the record an auditor reads are built from the same object.

---

## Why this shape

The obvious architecture for a triage assistant is a classifier that outputs a
level. It fails for three reasons that only appear once you take the clinical
problem seriously.

**A classifier optimises the wrong thing.** Trained to maximise accuracy, it
learns that predicting Level 3 is usually right, and on our data, argmax
catches only 20.1% of critical patients while agreeing with the nurse 58.2% of
the time. Accuracy and safety point in opposite directions here, so the
decision rule has to be a separate, explicit layer.

**A classifier cannot be held accountable.** "The model said Level 2" is not an
answer a clinician can give a family, a coroner, or a regulator. The thing that
sets the level has to be a rule with a threshold and a citation.

**A classifier has no way to say "I don't know".** Triage data is genuinely
ambiguous; a system that cannot express that will express false confidence
instead.

So the architecture separates estimation, decision and authority.

---

## Layers

```
┌──────────────────────────────────────────────────────────────────────┐
│  INTAKE                                                              │
│  Vitals · self-reported symptoms · history · bedside observations    │
│  Each value carries source, timestamp and quality. Missing stays     │
│  missing, nothing is imputed anywhere in this pipeline.             │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FEATURE CONSTRUCTION           data/features.py                     │
│  • Age-normalised z-scores against published ranges per age band     │
│  • NEWS2 (adults) / PEWS (children) as engineered features           │
│  • Shock index, pulse pressure, hypoxia and fever burden             │
│  • Missing-indicator per vital, missingness is signal               │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  RISK ESTIMATION                models/triage_model.py               │
│  XGBoost over 135 features, selected against 4 alternatives at a     │
│  matched escalation budget. Probabilities calibrated on a fold       │
│  disjoint from training.                        →  P(level | patient)│
└───────────────┬──────────────────────────────────┬───────────────────┘
                ▼                                  ▼
┌──────────────────────────────┐  ┌────────────────────────────────────┐
│  CONFORMAL UNCERTAINTY       │  │  COST-SENSITIVE DECISION           │
│  models/uncertainty.py       │  │  models/decision_policy.py         │
│  Finite-sample coverage      │  │  argmin_p Σ P(t)·Cost(p,t)         │
│  guarantee. Binary           │  │  Versioned cost matrix per site.   │
│  critical-exclusion at 95%.  │  │        → proposed level            │
└───────────────┬──────────────┘  └────────────────┬───────────────────┘
                └────────────────┬─────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  MULTI-AGENT REVIEW             engine/multi_agent_debate.py         │
│  Throughput agent vs safety sentinel. Structured disagreement feeds  │
│  the record. ADVISORY ONLY, sets nothing.                           │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
╔══════════════════════════════════════════════════════════════════════╗
║  ★ DETERMINISTIC SAFETY ENGINE ★    engine/safety_engine.py          ║
║                                                                      ║
║  THE SOLE AUTHORITY ON THE TRIAGE LEVEL.                             ║
║  15 versioned clinical rules from config/rules_*.yaml.               ║
║  Two invariants, both unit-tested:                                   ║
║    1. May only escalate. No code path reduces urgency.               ║
║    2. Every escalation records rule id, threshold, evidence,          ║
║       citation, certainty class and rule-pack content hash.          ║
╚════════════════════════════════┬═════════════════════════════════════╝
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  EXPLANATION                    engine/explanation.py                │
│  Decision trace in causal order: the rule that decided → per-patient │
│  SHAP → supporting rules → what was never recorded → counterfactual. │
│  Rendered for nurse, patient and compliance from ONE object.         │
└────────────────────────────────┬─────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│  QUEUE & AUDIT                                                       │
│  Hazard-ordered waiting queue · hash-chained append-only audit trail │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Where each kind of logic lives, and why

The brief asks teams to be explicit about when they use deterministic logic,
rules, statistics, ML or an LLM. Our allocation:

| Concern | Mechanism | Why not something else |
|---|---|---|
| Risk estimation from vitals and complaint | Gradient-boosted trees | Tabular clinical data with ~30 informative features; trees beat deep learning here and stay auditable |
| Turning risk into a level | Explicit cost matrix, expected-harm minimisation | The asymmetry is a *values* question, not a modelling one. It belongs in a signed-off object, not in weights |
| Hard clinical criteria | Deterministic YAML rules | An unresponsive patient is Level 1 by definition. Learning that from data would be absurd and unreliable |
| Uncertainty | Split-conformal prediction | Gives a distribution-free finite-sample guarantee. Softmax confidence gives none |
| Confidence calibration | Isotonic/Platt on a disjoint fold | Raw model scores are not probabilities |
| Age adaptation | Feature engineering **and** rule thresholds | Belt and braces on the failure mode the brief calls out explicitly |
| Explanation | Decision trace + per-patient SHAP | Global feature importance answers a different question than "why this patient" |
| Narrative for patients | Templated text | An LLM here would risk fabricating clinical claims for zero benefit |
| **Anywhere in the decision path** | **No LLM** | Determinism, reproducibility, offline operation and explainability are all requirements. An LLM compromises all four |

---

## The data path

```
CDC NHAMCS fixed-width public-use file
        │
        ├─ data/real/nhamcs_loader.py    explicit per-year record layout
        │                                 + parse validation that refuses to
        │                                   train on a misaligned file
        ▼
   clean cohort (20,702 visits, 176 hospitals)
        │
        ├─ grouped_split()   by HOSPITAL, into four disjoint folds
        │       train ─────▶ fit classifiers
        │       calibrate ─▶ fit probability calibration
        │       conformal ─▶ fit conformal thresholds + select operating point
        │       test ──────▶ reported results, touched once
        ▼
   data/demo_cohort.py   demo board = real held-out patients
                         + labelled synthetic edge cases
```

The parse validation is not ceremonial. Reusing the 2022 record layout on the
2021 file produced a 9.6% critical-care rate, four times the truth, while
passing every field-level range check, because the disposition block is shifted
two characters. The loader now cross-checks that critical-care admissions are a
minority of all admissions, which is what caught it.

---

## Design decisions worth defending

**Split by hospital, not by row.** Patients from one department share
documentation habits and case mix. A random split lets a model memorise a
hospital's coding style and report it as skill. Grouped splitting also answers
the commercially relevant question: does this work at the *next* hospital?

**Missing values are never imputed.** The primary models are NaN-native and
every vital carries a missing-indicator. Imputing zero makes an unrecorded
saturation indistinguishable from a saturation of 0%; imputing the mean makes
an unmeasured patient look average, which is exactly the wrong prior for
someone nobody has had time to assess.

**Policy is configuration, logic is code.** Rule *logic* is Python, testable in
a pull request. Rule *policy*, thresholds, which rules are active, escalation
targets, is YAML that a medical director can read and sign. Site packs are
deltas, so a rural ED's policy is a dozen reviewable lines rather than a fork.

**Confidence measures the safety-relevant question.** The displayed figure is
P(patient is not sicker than the assigned level), not P(exact match). The
latter read 5% on a correctly-identified Level 1 patient, because a categorical
rule had set the level while the model spread its probability mass, a number
that teaches staff to ignore the field.

**Overrides change the patient's actual priority.** Recording an override while
the queue keeps the system's ranking would be worse than not supporting
overrides: the clinician believes they have acted, and the department carries
on regardless.

---

## Scaling across hospitals

Three things flex per site, and nothing else has to:

1. **The rule pack** (`config/rules_*.yaml`), thresholds, which rules are on.
2. **The cost matrix** (`models/decision_policy.py`), the department's risk
   appetite, expressed as prices.
3. **The escalation budget**, how much of the board the site can staff.

A rural ED with a 90-minute transfer time tightens thresholds (act earlier,
because rescue is slower), disables rules it cannot action locally, and shortens
reassessment intervals. Same model, same code, and every difference is a diff in
a version-controlled file with a written clinical rationale.

Integration with hospital systems is deliberately out of scope for the
prototype, but the schema is built for it: every value carries a `DataSource`
(`EHR_IMPORTED`, `DEVICE_MEASURED`, `NURSE_OBSERVED`, `PATIENT_REPORTED`), so
an integration populates fields without changing any downstream logic, and a
site with no EHR integration simply produces more `history_available = False`
patients, which the safety engine already handles as a first-class case.
