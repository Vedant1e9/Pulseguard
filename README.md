# PatientTriage.ai

**Accenture Innovation Challenge 2026, Round 2, Problem Track 2**

> ⚕️ A prototype clinical decision support system for research and demonstration.
> It does not diagnose, does not treat, and must not replace assessment by a
> qualified healthcare professional. No prospective clinical validation has
> been performed.

---

## What this is

An AI-assisted triage assistant for emergency departments that acts as a
second opinion for the triage nurse. It is built around one asymmetry: **missing
a critical patient is categorically worse than over-prioritising a minor one**,
and a system that optimises for average accuracy gets that exactly backwards.

The clearest evidence for the whole design sits in a single comparison. The
same trained model, on the same held-out patients:

| Decision rule | Critical patients caught |
|---|---|
| Pick the most likely level (accuracy-maximising) | **20.1%** |
| Minimise expected clinical harm (this system) | **68.2%** |

Identical model. Identical data. The difference is entirely in how the
probability is turned into an action, which is the part most triage
prototypes never build.

---

## What this repository contains

**This repository is the code and the working prototype.** The business proposal
and the pitch presentation are part of the AIC submission but are delivered
through the challenge portal, not published here.

**Start here: [`SUBMISSION.md`](SUBMISSION.md)** — the technical submission
document: the data, the results, how a decision is made, where the AI boundary
sits, and where a judge could push back.

**Supporting evidence, in the order a sceptical reader would want it**

| Document | Answers |
|---|---|
| [`evaluation/saved_results/HEADLINE_NUMBERS.md`](evaluation/saved_results/HEADLINE_NUMBERS.md) | Every quotable figure, generated rather than hand-entered |
| [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) | The full evaluation protocol and narrative |
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | The complete picture in one document |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the pieces fit together |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, training data, limitations |
| [`DATA_PROTECTION.md`](DATA_PROTECTION.md) | Jurisdiction, retention, consent, audio as PHI |

---

## Validated on real emergency department data

Not on data we invented. The system is trained and evaluated on the CDC's
**National Hospital Ambulatory Medical Care Survey (NHAMCS)**: a nationally
representative probability sample of US emergency department visits.

- **20,702 real ED visits** with a triage level assigned by an actual triage nurse
- **176 hospitals**, survey years 2021 to 2022
- Real recorded vital signs, coded reasons for visit, documented chronic conditions
- And critically: **what actually happened to each patient**, admitted,
  admitted to critical care, or died in the department

That last field allows the evaluation that matters. Agreeing with the triage
nurse is the easy question; whether a patient who went on to need critical care
would have been sent to the waiting room is the real one.

> Of the **66 patients** in the held-out test fold who were admitted to critical
> care or died in the ED, this system would have routed **80.3%** to the
> emergent lane. **The triage nurses who actually saw them routed 75.8%.**

Every metric is computed on hospitals held out from training, calibration,
conformal fitting and threshold selection, because a random row split lets a
model memorise a department's documentation habits and report it as skill.

Headline numbers, with the challenge-supplied scope kept separate from what this prototype measured: [`evaluation/saved_results/HEADLINE_NUMBERS.md`](evaluation/saved_results/HEADLINE_NUMBERS.md)  
Regenerate them at any time with `python -m scripts.generate_metrics`.  
Full evaluation narrative: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md)

---

## How it works

```
  Patient encounter
        │
        ├─▶ Data quality assessment          missing stays missing, never zero
        │
        ├─▶ Feature construction              age-normalised physiology,
        │                                     NEWS2 / PEWS, shock index
        │
        ├─▶ Calibrated risk model             XGBoost, isotonic/Platt calibrated
        │        │                            on a disjoint fold
        │        ├─▶ Conformal prediction     finite-sample coverage guarantee
        │        └─▶ Cost-sensitive policy    argmin expected clinical harm
        │
        ├─▶ Multi-agent review                structured second opinion (advisory)
        │
        ├─▶ ★ DETERMINISTIC SAFETY ENGINE ★   the ONLY thing that sets the level
        │                                     15 versioned clinical rules, YAML
        │
        ├─▶ Decision-trace explanation        the rule that decided, then SHAP
        │
        └─▶ Hash-chained audit record         model + rule pack version + hash
```

**Models propose. Only the rule engine disposes.** When a clinician asks why a
patient was made Level 2, the answer is a rule with a threshold and a citation,
never a gradient-boosted forest.

### The five ideas that carry the system

**1. Cost-sensitive decisions, not argmax.** The level chosen minimises expected
clinical harm under an explicit, versioned cost matrix. Everything contestable
about the system's risk appetite lives in one auditable object a medical
director can read and sign off.

**2. An explicit escalation budget.** There is no setting that catches every
critical patient without flooding the emergent lane. Rather than hide that, the
operating point is selected against a stated budget, *≤35% of arrivals to the
emergent lane, ≥65% critical recall*, and the whole safety and throughput frontier
is published.

**3. Age-normalised physiology.** Every vital is additionally expressed as a
z-score against the published normal range *for that patient's age band*. A
heart rate of 150 is unremarkable in a toddler and an emergency in a
70-year-old; a single adult-calibrated model cannot express that, and the brief
names it as a silent safety risk.

**4. Guaranteed uncertainty, not a vibe.** Split-conformal prediction provides a
finite-sample coverage guarantee on whether a critical presentation can be
*excluded*. The confidence figure shown to a nurse is P(patient is not sicker
than the assigned level), the safety-relevant question, not the probability of
an exact match.

**5. Explanations that explain the actual decision.** The rule that set the
level appears first, with its citation. Model evidence comes from per-patient
SHAP, never global feature importance. Unmeasured vitals are named as
unrecorded and are never presented as evidence.

---

## Quick start

Two commands. The trained model and the compressed survey data are both in the
repository, so there is no download step and no training step before the
application runs.

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch takes about 2.5 seconds: the NHAMCS archives extract themselves,
the model bundle loads, and the 31-patient board is scored.

Run the test suite (135 tests, about 8 seconds):

```bash
python -m pytest -q
```

### Optional: spoken handover intake

The **Spoken handover** page records from the microphone and drafts a
structured record from what a nurse says. It works without any of this, using a
deterministic clinical parser on typed, pasted or sample transcripts. To enable
real transcription and language-model extraction:

```bash
pip install faster-whisper          # on-device, no audio leaves the machine
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY, for field extraction
export PT_ALLOW_CLOUD_AUDIO=0       # forbid audio leaving the machine
```

### Rebuilding from source

Neither is needed to run the application. Both are here because every number
this repository publishes should be reproducible.

```bash
python -m data.real.nhamcs_loader --download 2022   # refresh the survey data
python -m scripts.train_model                       # retrain the bundle
python -m evaluation.full_evaluation                # reproduce every metric
```

---

## What the four roles see

The application changes shape by role, because a triage nurse with nine seconds
and a compliance officer preparing for an audit need almost nothing in common.

| Role | Sees |
|---|---|
| **Triage nurse** | Patient board, typed intake, **spoken handover**, patient detail, waiting queue, **reassessment round** |
| **Emergency physician** | + reassessment round, review & override, what-if explorer |
| **Clinical analyst** | + model performance, safety frontier, robustness & surge, **AI boundary** |
| **Compliance officer** | Audit log, clinical rule governance, **AI boundary**, model performance |

---

## Assumptions

| | |
|---|---|
| **Jurisdiction** | United States, HIPAA + California CMIA |
| **Severity scale** | Five-level (1 = highest urgency) |
| **Reference department** | Mid-size urban ED, ~300 visits/day, on-site critical care |
| **Alternative profile** | Rural community ED, 12 beds, 90-minute transfer time |
| **Data availability** | Mixed, roughly half of arrivals have prior records |

Both site profiles ship as versioned rule packs in [`config/`](config/). The
rural pack is a delta over the default, so reviewing its policy means reading a
dozen lines rather than diffing three hundred.

---

## Repository layout

```
app.py                          Streamlit application, role-aware entry point
requirements.txt                Pinned dependencies

config/
  rules_default.yaml            Versioned clinical rule pack, 15 rules
  rules_rural_community.yaml    Rural profile, a delta over the default

data/
  real/nhamcs_loader.py         CDC NHAMCS parser, with parse validation
  real/ed2021.zip ed2022.zip    Compressed survey data, extracted on first run
  features.py                   Age-normalised physiology, clinical scores
  data_quality.py               Missingness assessment, provenance tagging
  demo_cohort.py                Demo board: real held-out patients + edge cases
  input_schema.py               Provenance-tagged encounter schema

engine/
  safety_engine.py              Deterministic rules, sole authority on the level
  rule_pack.py                  Versioned policy loader with content hashing
  triage_pipeline.py            End-to-end orchestration
  explanation.py                Decision-trace explanations, three personas
  voice_intake.py               Spoken handover: transcription, extraction, boundary
  multi_agent_debate.py         Structured second opinion, advisory only
  reassessment.py               Escalate-only re-scoring of waiting patients
  hazard_queue.py               Time-decay waiting queue
  override_audit.py             Hash-chained, append-only audit trail

models/
  triage_model.py               Training, hospital-grouped splits, calibration
  decision_policy.py            Cost matrices, site profiles, operating curve
  uncertainty.py                Calibration metrics, conformal prediction
  clinical_scores.py            NEWS2 and PEWS
  deterioration_velocity.py     Trend scoring for the waiting queue

ui/                             14 interface pages across 4 clinical roles
evaluation/
  full_evaluation.py            Full harness, writes JSON + headline metrics
  saved_results/                Generated evaluation artefacts, incl. HEADLINE_NUMBERS.md
scripts/
  train_model.py                Retrain and rewrite the bundle
  generate_metrics.py           Regenerate every published figure
saved_models/triage_bundle.joblib   Trained bundle, tracked so a clone runs immediately
tests/                          135 tests, including regressions for fixed bugs
```

## Known limitations

Stated plainly, because a prototype that hides these is worth less than one
that names them.

- **NHAMCS is a survey, not a live feed.** It records vitals and coded
  complaints but not free-text nursing notes or bedside observations, so the
  model sees less than a nurse does. The safety rules consume those
  observations; the learned model cannot.
- **Exact agreement with the nurse is 46.8%.** This is deliberate, the system
  escalates on purpose, but it also reflects that triage labels are genuinely
  noisy. Within-one-level agreement is 93.1%.
- **The five-class conformal set is very wide** (mean width **4.81 of 5**;
  empirical coverage 98.6% against a 90% target, and only 0.7% of patients get a
  single-value set). That is a real finding rather than a tuning failure: a
  triage level is not identifiable to a single value from triage-time data. The
  actionable guarantee is the binary critical-exclusion one instead, which
  clears 26.6% of patients of a critical presentation at 95% confidence.
- **Serial vital observations are simulated.** NHAMCS records one set per visit,
  so the deterioration-velocity demonstration builds a trend from each patient's
  real starting physiology. Labelled as such wherever it appears.
- **Cross-site variance is real.** Critical recall ranges 47% to 100% across the 26
  evaluated hospitals. The spread, not the mean, is what a new deployment should
  be planned against.
- **No prospective validation.** These are retrospective results on historical
  survey data. They do not establish clinical safety or efficacy.
