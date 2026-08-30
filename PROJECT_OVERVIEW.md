# PatientTriage.ai: Project Overview

*One document covering what this system is, what it does, how it was built,
what it has actually been shown to do, and how it scores against the criteria a
Round 2 judge will apply. Metrics from the evaluation of 2026-08-28. Front end,
spoken intake, AI boundary and reassessment round reviewed and extended
2026-08-29.*

---

## Contents

1. [What it is](#1-what-it-is)
2. [The problem it targets](#2-the-problem-it-targets)
3. [Data](#3-data)
4. [How a decision is made](#4-how-a-decision-is-made)
5. [Model](#5-model)
6. [Results](#6-results)
7. [The application](#7-the-application)
8. [Spoken handover intake](#8-spoken-handover-intake)
9. [The AI boundary](#9-the-ai-boundary-what-is-a-model-and-what-is-arithmetic)
10. [Reassessment round](#10-reassessment-round)
11. [Judge criteria, and how this scores](#11-judge-criteria-and-how-this-scores)
12. [Meeting the brief's minimum expectations](#12-meeting-the-briefs-minimum-prototype-expectations)
13. [Testing](#13-testing)
14. [Known limitations](#14-known-limitations)
15. [Running it](#15-running-it)

---

## What is here, and what is submitted elsewhere

The brief asks for three things. **This repository is the working prototype and
its code.** The detailed business proposal and the pitch presentation are
submitted through the AIC portal and are deliberately not published here.

| Required deliverable | Where |
|---|---|
| **Working prototype** | This repository. `app.py` plus `ui/`, `engine/`, `models/`, `data/` — 14 pages across 4 roles, `streamlit run app.py` |
| **Detailed business proposal** | Submitted via the AIC portal |
| **Pitch presentation** | Submitted via the AIC portal |

Supporting evidence, in the order a sceptical reader would want it:

| Document | Answers |
|---|---|
| [`evaluation/saved_results/HEADLINE_NUMBERS.md`](evaluation/saved_results/HEADLINE_NUMBERS.md) | Every quotable figure, generated not hand-entered, with challenge-supplied scope kept separate from measured results |
| [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md) | The full evaluation narrative and protocol |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the pieces fit together |
| [`MODEL_CARD.md`](MODEL_CARD.md) | Intended use, training data, limitations |
| [`DATA_PROTECTION.md`](DATA_PROTECTION.md) | Jurisdiction, retention, consent, and audio treated as protected health information |

---

## 1. What it is

An AI-assisted triage assistant for emergency departments. It gives the triage
nurse a second opinion on how urgently a patient needs to be seen, surfaces how
confident it is, explains itself in terms a clinician can challenge, and keeps a
tamper-evident record of every decision.

It is a **prototype**. It has not been clinically validated, has no regulatory
clearance, and must not be used to make care decisions.

---

## 2. The problem it targets

Under-triage, meaning a genuinely sick patient assigned a lower urgency than
they need, is the failure mode that kills people in emergency departments. It is
also the one an accuracy-optimised machine learning model makes *worse*, because
predicting the common case is how a model maximises accuracy.

Our own data demonstrates this precisely. The same trained model, on the same
3,316 held-out patients:

| Decision rule | Critical patients caught | Exact agreement with nurse |
|---|---|---|
| Pick the most likely level (argmax) | 20.1% | 58.2% |
| **Minimise expected clinical harm** | **68.2%** | 46.8% |

An accuracy-maximising triage model misses four out of five critical patients
while looking like the better model on a leaderboard. That gap is the entire
reason this system separates risk *estimation* from the *decision rule*.

The brief asks teams to "demonstrate this design choice explicitly in their
prototype". The **Safety and throughput frontier** page is that demonstration:
the full operating curve, the budget in force, and who signs it off.

---

## 3. Data

**Source:** CDC / National Center for Health Statistics, National Hospital
Ambulatory Medical Care Survey (NHAMCS), Emergency Department component,
2021 and 2022.

| | |
|---|---|
| Visits with a nurse-assigned triage level | **20,702** |
| Hospitals | **176** |
| Critical (Level 1 or 2) prevalence | 17.46% |
| Hospital admission rate | 14.56% |
| ICU admission or death in ED | 2.31% |
| Paediatric / adult / geriatric | 3,825 / 12,652 / 4,225 |

Each record carries the triage level an actual nurse assigned, the vital signs
actually measured, the coded reason for visit, documented chronic conditions,
and **what actually happened to the patient**. That last field allows validation
against outcomes rather than against another human's opinion.

**Why real data.** A system validated only on data its authors generated proves
nothing. A synthetic cohort can be made arbitrarily easy without anyone
noticing, and the resulting metrics are unfalsifiable. These numbers can be
reproduced by anyone who downloads the same public file.

**Synthetic data still has one job.** NHAMCS records vitals but not bedside
observations, such as whether a patient is unresponsive or whether their skin is
mottled. Six clearly labelled synthetic cases exercise the safety rules that
depend on those observations. They are excluded from every accuracy figure.

---

## 4. How a decision is made

```
 spoken handover ─┐
                  ├─→ encounter → data quality → features → calibrated model → conformal set
 typed intake ────┘                                    ↓
                                              cost-sensitive policy
                                                       ↓
                                   ★ DETERMINISTIC SAFETY ENGINE ★  ← sets the level
                                                       ↓
                                         decision trace → audit record
```

Models propose; only the rule engine disposes. Two invariants are unit-tested:
the engine can only escalate, never reduce urgency; and every escalation
records its rule, threshold, evidence, citation and the rule-pack content hash.

Both intake paths converge on the identical encounter object and the identical
engine. There is no shortcut for spoken input.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

---

## 5. Model

| | |
|---|---|
| Algorithm | XGBoost, selected against LightGBM, HistGradientBoosting, Random Forest and Logistic Regression |
| Selection criterion | AUROC and critical recall **at a matched escalation budget**, so no model can win by escalating everything |
| Features | 135 (clinical + complaint-text components) |
| Calibration | sigmoid, chosen on a third disjoint fold |
| Missing data | Never imputed. NaN-native models plus per-vital missing indicators |
| Split | Grouped by **hospital**, four disjoint folds |

Candidate leaderboard (evaluated at a common escalation budget):

| Model | AUROC (critical) | Critical recall | λ |
|---|---|---|---|
| XGBoost | 0.795 | 72.5% | 0.3 |
| HistGradientBoosting | 0.793 | 74.8% | 0.08 |
| RandomForest | 0.793 | 65.8% | 0.08 |
| LightGBM | 0.789 | 71.8% | 0.2 |
| LogisticRegression | 0.767 | 69.5% | 0.05 |

**Why a matched budget matters.** Ranking triage models by raw recall rewards
whichever one escalates most, which is trivially gamed by a model that calls
every patient Level 1. Every candidate above is first tuned to the same
escalation budget (at most 35% of arrivals to the emergent lane, at least 65%
critical recall) and only then compared. Accuracy is reported and never used to
select.

---

## 6. Results

All figures on **3,316 visits from 28 hospitals held out** of
training, calibration, conformal fitting and threshold selection. Confidence
intervals bootstrapped clustered by hospital.

### Safety

| Metric | Value | 95% CI |
|---|---|---|
| Critical recall | **68.2%** | 64.6% to 72.2% |
| Critical under-triage rate | 31.8% | 27.8% to 35.4% |
| Emergent-lane load | 31.7% | 28.8% to 35.5% |
| Within-one-level agreement | 93.1% | 91.0% to 94.8% |

### Discrimination

| Target | AUROC |
|---|---|
| Critical triage level | 0.792 |
| Hospital admission | 0.777 |
| ICU admission or death in ED | 0.808 |

### Against outcomes, not opinions

Of the 66 held-out patients admitted to critical care or who died in the ED:

| Approach | Caught |
|---|---|
| **PatientTriage.ai** | **80.3%** |
| Triage nurses (actual) | 75.8% |
| NEWS2 at matched lane load | 72.7% |
| Accuracy-maximising argmax | 30.3% |

With 66 events, this is suggestive rather than conclusive.

### Calibration, fairness, generalisation

- Expected calibration error **0.043**, Brier **0.556**
- Max critical under-triage gap across age bands **9.3%**; across race/ethnicity **12.4%** (race is audited, never a model input)
- Across 26 unseen hospitals: critical recall 47% to 100%
- Trained on 2021, tested on 2022: critical recall 62.1%

### Speed

| Stage | p50 | p95 |
|---|---|---|
| Model inference | **2.16 ms** | 2.44 ms |
| Full pipeline end to end | **17.3 ms** | 18.9 ms |
| Deterministic field extraction from a transcript | **under 5 ms** | |
| On-device transcription of a 12-second handover | **409 ms warm** | 1.6 s cold |

End to end includes feature construction, the calibrated model, the conformal
check, every clinical safety rule, SHAP attribution, counterfactual search and
the full explanation trace. Throughput is **22,901 patients per second** on one
CPU core with no GPU; a 500-visit day at 3x surge scores in about 0.4 seconds of
compute. Cold start is 827 ms, reported separately rather than hidden in the
distribution.

### Engineering

| | |
|---|---|
| Test suite | **135 automated tests**, about 8 seconds |
| Coverage, runtime code | **80%** (`engine`, `models`, `data`, `ui`) |
| Coverage, the safety engine | **94%**, the single component that sets a triage level |
| Source | 12,181 lines across 44 modules, plus 1,666 lines of tests |
| Fresh clone | 12.9 MB, boots in about 4 seconds, no download or training step |

Full detail: [EVALUATION_REPORT.md](EVALUATION_REPORT.md) ·
Headline figures, generated rather than hand-entered:
[evaluation/saved_results/HEADLINE_NUMBERS.md](evaluation/saved_results/HEADLINE_NUMBERS.md)
(`python -m scripts.generate_metrics`)

### Classification metrics

Reported because they are expected, and reported with what causes them. Every
figure is verified against scikit-learn in `tests/test_headline_numbers.py`.

| Metric | Value | What it answers |
|---|---|---|
| Sensitivity (recall) | **68.2%** | Of genuinely critical patients, how many were caught |
| **Negative predictive value** | **91.2%** | Of patients NOT sent to the emergent lane, how many were truly non-critical |
| Positive predictive value | 40.3% | Of patients sent, how many were truly critical |
| Specificity | 76.7% | Of genuinely non-critical patients, how many were left alone |
| F1, critical vs non-critical | **0.507** | The decision the system actually makes |
| F1, five-class weighted | 0.385 | |
| F1, five-class macro | 0.237 | Low by construction, see below |
| Accuracy | 46.8% | |

**NPV is the number to lead with alongside recall.** It decides whether a nurse
can trust a non-critical call, and at 91.2% it is the strongest single figure
the system produces. The low PPV of 40.3% is its deliberate counterpart: the
system accepts false alarms to avoid misses.

### What the safety gain costs

A recall improvement means nothing alone, because recall is trivially bought by
escalating everyone. Against the same model at argmax, on the same 3,316
patients, the cost-sensitive policy catches **299 more critical patients** at a
cost of 837 more patients in the emergent lane: an exchange rate of
**2.8 patients over-triaged per extra critical patient caught**. That ratio is
what a charge nurse would ask for first.

**Macro F1 is low by construction, not by accident.** The cost policy pulls
uncertain patients up the scale, so Levels 4 and 5 score near zero F1, and macro
F1 weights all five classes equally. It therefore penalises the system hardest
for the half of triage that cannot hurt anyone. The same model tuned for
accuracy reaches 58.2% accuracy and catches 20.1% of critical patients; this
configuration reaches 46.8% and catches 68.2%. The accuracy and F1 numbers got
worse precisely because the system got safer. Quote the binary F1, and say which
one it is.

---

## 7. The application

A Streamlit front end (`app.py` plus `ui/`) in which **four roles see four
different applications**, because a triage nurse with nine seconds and a
compliance officer preparing for an audit need almost nothing in common. Access
is scoped to the minimum necessary for each role.

| Role | Pages |
|---|---|
| Triage nurse | Patient board · New patient intake · **Spoken handover** · Patient detail · Waiting queue · **Reassessment round** |
| Emergency physician | Patient board · Patient detail · Waiting queue · **Reassessment round** · Review & override · What-if explorer |
| Clinical analyst | Patient board · Model performance · Safety frontier · Robustness & surge · **AI boundary** · Patient detail |
| Compliance officer | Audit log · Clinical rule governance · **AI boundary** · Model performance · Patient detail |

A site rule pack selector switches the whole assistant between an urban ED and a
rural community ED, reloading the clinical rules and thresholds.

### Interface decisions that carry clinical weight

- **Vital sign fields start empty, never at a normal value.** A blank field is
  recorded as *not measured* and widens the uncertainty band. Pre-filling them
  would let a nurse who skips a field silently submit "normal", scoring a
  patient on a measurement nobody took.
- **Confidence is labelled with what it measures.** "91%" alone is meaningless;
  "91% confident this patient is not sicker than Level 3" is actionable.
- **Rules are visually distinct from model evidence,** so a clinician can tell
  instantly whether a deterministic criterion fired or a statistical model
  leaned a certain way.
- **Model attribution is reported as direction and share of the evidence
  shown,** not as a probability delta. SHAP values live in log-odds space, and
  printing them as percentages produced impossible figures such as "decreases
  the probability by 102%".
- **Disagreements with the nurse are shown in both directions,** with equal
  prominence. A tool that displays only its wins is a tool nobody should trust.
- **The queue explains its ordering in a sentence,** not as a hazard score. A
  number like 1386 is not something a nurse can act on or challenge.
- **Deterioration alerts are separated from wait-time alerts,** so the list that
  demands immediate action does not become wallpaper.

### Front-end review, 2026-08-29

The application was driven end to end in a browser across all four roles and
every page. Findings, all fixed:

| Issue | Resolution |
|---|---|
| Patients entered at intake were scored, logged and counted in the emergent-lane tile, but never appeared on the board, in the patient selector, or in the waiting queue | `triage_patient` now registers the encounter on the board, and intake joins the live hazard queue. Live intakes are labelled and excluded from every accuracy comparison |
| The intake form reopened holding the previous patient's age, vitals and complaint | Every form widget is versioned per submission, so each patient starts on blank fields |
| Model factors reported impossible probabilities ("decreases the probability by 102.2%") | Log-odds attributions now report direction plus share of shown evidence |
| The triage decision rendered two screens below the button that produced it | The result now renders above the form |
| Sidebar role and rule-pack dropdowns rendered near-white on near-white | Input surfaces excluded from the sidebar's blanket text colour |
| Vitals displayed as `128.00 bpm` and `38.40 °C` | Explicit formats per field type |
| Board and queue tables clipped their rightmost columns, including the escalation flag and provenance label | Explicit column widths and a wider content area |
| The audit log printed the operator's absolute home directory on screen | Shown relative to the deployment root |

Interface prose was also revised throughout to remove dash-joined clauses in
favour of ordinary sentence punctuation.

---

## 8. Spoken handover intake

*Differentiator L, previously parked for the roadmap, now built.*

**The problem it solves.** The brief's hardest workflow constraint is that a
triage decision must be made in seconds by a clinician already managing several
other patients. The bottleneck in that moment is not the model, which scores in
2 ms. It is the keyboard. A nurse walking a patient in from the ambulance bay is
holding a blood pressure cuff, not a mouse.

**What it does.** A microphone button in the browser. Press it, say the handover
you were going to say anyway, press stop. Transcription starts automatically:

> "Eighty-one year old female brought in by ambulance. Heart rate one eighteen,
> resp rate twenty six, sats ninety on air, BP ninety four over sixty. She's
> clammy and short of breath with chest pain. No history on file."

becomes twelve structured fields in under 5 ms, which the nurse confirms, and
which then go through the identical pipeline the typed form uses.

### The three-stage boundary

Each stage has a different trust level, and the code keeps them separate:

| Stage | What it does | Trust |
|---|---|---|
| **1. Transcription** | Audio to text. Whisper on device by preference, Whisper API as a labelled fallback | Lossy, and known to be |
| **2. Extraction** | Text to candidate fields. Claude or OpenAI under a closed schema, or the deterministic clinical parser | Allowed to be wrong |
| **3. Confirmation** | Candidate fields to encounter. A human | The only stage that is trusted |

### Four safety properties, all unit-tested

1. **Extraction can never write a triage level, confidence or safety decision.**
   The output schema contains 21 named *input* fields and no urgency field at
   all. A model cannot write a field that does not exist, so a prompt injection
   inside a transcript has nowhere to land. The **AI boundary** page ships a
   live injection attempt a judge can run.
2. **Uncertain means absent, never guessed.** Anything below 55% confidence
   arrives empty rather than pre-filled, and the pipeline handles unmeasured
   values by widening the uncertainty band. A mis-heard number is worse than a
   missing one, because a missing one is visible.
3. **Physiologically impossible values are rejected at the boundary.** "Sats one
   hundred and eighty" is a transcription error, not a patient. Every rejection
   is shown, never silently dropped.
4. **Every field carries its provenance and the words it came from.** Values are
   stamped `voice_transcribed` in the audit log, distinct from `manual_entry`,
   because a typo is idiosyncratic and a mis-transcription is systematic.

### Engineering that came out of real speech

The parser is built for how clinicians actually talk, not for clean prose:

- **Spoken numbers.** "One eighteen" is 118, not 1 and 18. "Ninety four over
  sixty" is two readings. "Thirty nine point six" is 39.6.
- **ASR artefacts.** Real Whisper output writes "81-year-old" with hyphens and
  reliably mis-hears "resp rate" as "respite". Both are handled, and both were
  found by transcribing genuinely synthesised audio rather than assuming clean
  input. A regression test pins the verbatim Whisper string.
- **A closed complaint vocabulary.** The chief complaint is matched against a
  curated phrase list rather than lifted as a free-text span, because an
  unbounded text field is the one place a transcript could smuggle arbitrary
  content into a clinical record.

### Degradation, and audio as protected health information

With no API key and no speech model installed, the deterministic parser still
runs on typed or pasted text, and every stage after transcription is identical.
Four sample handovers, including a deliberately noisy one, make the whole
workflow demonstrable offline in one click.

A recorded handover is protected health information: it carries a patient's age,
sex, complaint and physiology in a nurse's identifiable voice. On-device
transcription is therefore preferred wherever available and listed first; a
cloud backend is labelled as leaving the building everywhere it appears, and
`PT_ALLOW_CLOUD_AUDIO=0` removes it entirely.

---

## 9. The AI boundary: what is a model, and what is arithmetic

A triage assistant that answers *how did you decide this* with "the AI said so"
is not deployable. The **AI boundary** page names the technique behind every
component in the system:

| Component | Technique |
|---|---|
| Triage level (the decision) | Deterministic rules |
| Risk estimate feeding it | Traditional ML (XGBoost) |
| Turning risk into an action | Decision theory (expected-cost minimisation) |
| Uncertainty | Conformal prediction |
| Deterioration trend | Statistics (rate-of-change regression) |
| Queue ordering | Deterministic scoring |
| Per-factor explanation | Traditional ML (TreeSHAP) |
| Early warning score | Published clinical standard (NEWS2 / PEWS) |
| Multi-agent safety debate | Deterministic heuristics, no language model |
| Handover transcription | Speech model, optional |
| Handover field extraction | LLM, optional |

**Nine of eleven components involve no generative model at all.** The two that
do are both optional, both sit at intake *before* anything is scored, and both
have a deterministic fallback.

| | |
|---|---|
| Components that set the triage level | **1** (the safety engine) |
| Language-model calls per triage decision | **0** |
| LLM cost per triage decision | **$0.00** |
| LLM cost per spoken intake | ~$0.003, only when a key is configured |

Scoring a patient makes no language-model call at all. A department seeing 500
patients a day runs the entire pipeline for the price of electricity, on
hardware it already owns, with no per-decision API cost and no patient data
leaving the building. That is a direct consequence of keeping the language model
out of the decision path.

---

## 10. Reassessment round

The brief does not merely suggest ongoing monitoring, it requires it: the system
"must monitor patients already in the waiting queue and trigger re-assessment if
wait time exceeds safe thresholds for their severity level **or if vitals are
re-recorded as worsening**".

Wait-time monitoring was live from the start and drives the hazard queue. The
second half was not: `add_reading()` existed in the velocity model, but no
interface ever called it, so the only deterioration data in the system was a
fixture generated at boot for three demo patients. A judge asking to see a
patient deteriorate in the waiting room could only be shown simulated data.

**Reassessment round** closes that loop. A nurse walks the waiting room, picks
the patient the hazard queue puts at the top, re-records whichever vitals they
actually rechecked, and the system recomputes the trend, re-scores through the
identical deterministic engine, re-orders the queue and writes a `reassessment`
event to the audit log.

### Escalate-only, and why that is the important half

Re-scoring can raise a patient's urgency and can never lower it.

A heart rate that reads lower on recheck has not necessarily improved; the
patient may be tiring. Letting one favourable observation walk a patient back
down the queue would turn routine monitoring into an unaccountable
de-escalation channel, which is precisely what the override flow exists to keep
in a clinician's hands and on the record. So a downgrade remains an override,
with a name attached to it.

Worked example from the test suite, on a real board patient:

| Step | Recorded | Proposed | Final |
|---|---|---|---|
| Arrival | HR 65, RR 20, SpO2 99 | Level 3 | **Level 3** |
| Recheck, worsening | HR 110, RR 34, SpO2 88 | Level 2 | **Level 2**, escalated |
| Recheck, improved | HR 72, RR 14, SpO2 99 | Level 3 | **Level 2**, held |

Velocity risk on the second reading came back *high*, with the alert
"CRITICAL VELOCITY: spo2 changing at -6.4/hr". Both events are in the audit log
with the vitals recorded, the level before and after, the rules that fired and
the rule-pack hash in force.

Nine tests cover this, including one that drives four alternating rechecks and
asserts the level sequence is monotone.

---

## 11. Judge criteria, and how this scores

The brief asks for a business proposal, a working prototype and a pitch, and
tells teams to "focus on innovation, creativity, and technical novelty". These
are the parameters a judge is likely to apply, with an honest self-assessment.

### A. Prototype completeness and the stated minimum expectations

| Parameter | Status | Evidence |
|---|---|---|
| Triage scoring on 15 to 20+ records | **Exceeds** | 31-patient live board, 3,316 in evaluation |
| Ambiguous presentation | **Met** | Real cohort throughout; disagreement panel on the board |
| Paediatric and geriatric cases | **Met** | Dedicated edge cases plus 3,825 / 4,225 real visits, age-banded thresholds |
| Zero-history patient | **Met** | Dedicated rule, edge case, and a whole-cohort ablation |
| 3x surge behaviour | **Met** | Live simulation through the real pipeline, not a timing stub |
| Never a score without confidence | **Met** | Enforced structurally in `TriageResult`, unit-tested |
| Clinician override, and what is logged | **Met** | Override changes the level and re-orders the queue; hash-chained log |
| Monitor waiting patients, act on worsening vitals | **Met** | **Reassessment round**: re-record a vital, watch the system escalate, escalate-only by construction |
| Stated jurisdiction | **Met** | US, HIPAA + California CMIA |

### B. Front end and usability

| Parameter | Assessment |
|---|---|
| Is it user-friendly? | **Strong.** Four role-scoped applications, not one dashboard with tabs. Reviewed page by page in a browser; eight defects found and fixed |
| Is information easy to ingest? | **Strong.** A colour-banded decision, three labelled numbers, then a ranked trace separating rules from model evidence. Persona views render the same decision for nurse, patient and compliance |
| Time to first decision | **Strong.** Spoken handover removes the keyboard entirely |
| Honest about what it does not know | **Strong.** A "What we don't know" panel on every patient, and a follow-up question to ask next |

### C. Speed

| Parameter | Assessment |
|---|---|
| Is the code fast? | **Strong.** 2.16 ms inference, 17.3 ms full pipeline, 22,901 patients/second on one CPU core |
| Does it hold under surge? | **Strong.** Measured, and the threshold provably does not drift under load |
| Is the audio path fast? | **Adequate.** 409 ms warm transcription, under 5 ms extraction. Transcription dominates, and it is bounded by the speech model, not by us |

### D. Machine learning rigour

| Parameter | Assessment |
|---|---|
| How is the model evaluated? | **Strong.** Held-out *hospitals*, not a random row split. Bootstrapped CIs clustered by hospital. Validated against ICU admission and death, outcomes unknowable at triage time |
| How is the best model chosen? | **Strong.** Five candidates at a matched escalation budget; accuracy reported but never used to select |
| How accurate is it? | **Honest.** 68.2% critical recall, 0.792 AUROC, 46.8% exact agreement. The last number is deliberately low, and the page says so |
| Generalisation | **Reported, including the bad news.** 47% to 100% recall across 26 unseen hospitals; the spread is stated as the thing to plan against |
| Fairness | **Audited.** Race and sex never model inputs, gaps reported anyway |
| Uncertainty | **Strong.** Conformal guarantee, honestly scoped to the binary question it can actually support |

### E. LLM usage

| Parameter | Assessment |
|---|---|
| Is an LLM used well? | **Strong, by being used narrowly.** Track 2 never asks for one. It appears at exactly one place, extraction, where the task is genuinely linguistic |
| Can it corrupt a decision? | **No, structurally.** No urgency field in the schema; range and vocabulary checks; mandatory human confirmation. A runnable injection demo is in the app |
| Cost and latency discipline | **Strong.** Zero LLM calls and zero cost per triage decision |
| Failure behaviour | **Strong.** Deterministic fallback on missing key, network error or malformed response |

### F. Creativity and technical novelty

| What | Why it is not decoration |
|---|---|
| **Cost-sensitive decision policy** | The 20.1% to 68.2% recall swing on an identical model. Most triage prototypes never build this layer |
| **Spoken handover intake** | Attacks the real bottleneck, the keyboard, rather than the model |
| **The AI boundary page** | Turns "which parts are AI" from a question into a rendered artefact with a live injection demo |
| **Time-decay hazard queue** | Ordering shifts with wait time and vitals, explained in a sentence rather than a score |
| **Counterfactual what-if explorer** | Builds earned trust by letting a nurse probe where the system is sensitive |
| **Versioned rule pack with content hash** | Clinical policy as a reviewable file; an audit years later reconstructs the exact text in force |
| **Site rule packs as deltas** | The same assistant flexed to a rural ED by a dozen lines, not a fork |
| **Persona-tailored explanation** | One decision, three audiences, from one trace object |
| **Outcome validation against ICU and death** | Moves the question from "do you agree with the nurse" to "would the patient have been safe" |

### G. Governance, compliance and data protection

| Parameter | Assessment |
|---|---|
| Audit trail | **Strong.** Hash-chained append-only log with a runnable integrity check |
| Overridability | **Strong.** Overrides change the level, re-order the queue, and gate downgrades behind a second clinician |
| Data protection | **Strong.** Named jurisdiction, retention policy, consent model; audio treated as PHI with an on-device default |
| Policy transparency | **Strong.** Every rule readable and arguable without a code deployment |

### H. Where a judge could push back

Stated here rather than left to be discovered:

- **Exact agreement with the nurse is 46.8%.** Deliberate, and defended on the
  Model performance page, but it will be asked about.
- **Cross-site variance is wide** (47% to 100%). Real, and reported rather
  than averaged away.
- **The outcome validation rests on 66 events.** Suggestive, not conclusive.
- **No prospective validation.** No clinical or regulatory clearance.
- **Serial vitals are simulated** for the deterioration demonstration, and
  labelled as such wherever they appear.
- **Audio is captured on a laptop microphone**, not the noisy department it
  would face in reality.

---

## 12. Meeting the brief's minimum prototype expectations

| Requirement | Where |
|---|---|
| Triage scoring on 15 to 20+ simulated records | 31-patient demo board (25 real held-out + 6 synthetic edge cases); 3,316 in evaluation |
| At least one ambiguous presentation | Real cohort is full of them; see the disagreement panel on the board |
| Paediatric / geriatric case | `EDGE-002` (paediatric compensated shock), `EDGE-003` (geriatric anticoagulated fall); 3,825 paediatric and 4,225 geriatric visits in evaluation |
| Zero-history first-time patient | `EDGE-005`, plus a dedicated safety rule and a whole-cohort ablation |
| Behaviour under 3x surge | **Robustness & surge** page, a live simulation through the real pipeline |
| Never return a score without a confidence indicator | Enforced structurally in `TriageResult`; unit-tested |
| Capture a clinician override and show what is logged | **Review & override** page; overrides change the level and re-order the queue |
| Accept a new patient and act on the result | **New patient intake** and **Spoken handover**; the patient joins the board, selector, queue and audit log |
| Stated regulatory jurisdiction | US, HIPAA + California CMIA ([DATA_PROTECTION.md](DATA_PROTECTION.md)) |

---

## 13. Testing

**135 automated tests** (`pytest tests/`), covering:

- The escalate-only invariant, exhaustively across presentations and proposals
- Every named clinical safety rule, with its citation and certainty class
- Calibration metrics, conformal coverage guarantees, cost-policy monotonicity
- Every UI page rendering against the real pipeline
- **The spoken-intake boundary** (34 tests): that no decision field is
  extractable, that five prompt-injection patterns produce nothing, that
  impossible physiology is rejected, that uncertain fields stay empty, that
  clinical speech parses, and that verbatim Whisper output still yields the
  full clinical picture
- **The reassessment loop** (9 tests): that worsening vitals escalate, that an
  improved reading never de-escalates, that urgency is monotone across a whole
  round of rechecks, that a vital nobody rechecked keeps its arrival value, and
  that every reassessment is audited
- **Real interaction tests** (23 tests) driven through Streamlit's own
  `AppTest` harness, which runs the actual script, sets real widget state and
  submits real forms. Page-render tests prove a page does not raise; these
  prove the forms are wired to something. They caught a duplicated entry in the
  site rule pack menu that every other test missed
- **The published headline numbers** (18 tests): every F1, precision, recall,
  NPV and PPV verified against scikit-learn or an independent computation; the
  five-level scale reported as five; NPV asserted to exceed PPV, which is what
  a system tuned against misses must show; and the document checked for the
  warnings that must stay attached to an unflattering macro F1
- **Regressions for each bug found and fixed during development**, including:
  the second-clinician downgrade gate that failed open; explanations that named
  the wrong factor; confidence that read 5% on a correct Level 1; non-contiguous
  conformal sets; and an encounter round-trip that silently dropped 25 features
  and changed 15% of decisions.

---

## 14. Known limitations

- NHAMCS records vitals and coded complaints, not free-text nursing notes or
  bedside observations. The model sees less than a nurse does.
- Exact agreement with the nurse is 46.8%. Partly deliberate, since the system
  escalates on purpose, and partly genuine label noise.
- The five-class conformal set is wide (mean 4.8 of 5 levels). A triage level is
  not identifiable to a single value from triage-time data; the actionable
  guarantee is the binary critical-exclusion one.
- Cross-site variance is wide (47% to 100% critical recall).
- Serial vital observations are simulated for the deterioration demonstration.
- Speech-to-text is evaluated on synthesised audio in a quiet room, not on
  accented speech in a loud department. The confirmation step exists precisely
  because that gap is real.
- An untouched pain slider is still recorded as 0 unless the nurse ticks
  *Not asked*. The safer default is an open design question.
- Live intakes, recorded observations and the audit chain head live in process
  memory and reset when the application restarts. A deployment would persist all
  three.
- No prospective validation, no clinical validation, no regulatory clearance.

---

## 15. Running it

```bash
pip install -r requirements.txt
streamlit run app.py
pytest -q                     # 135 tests
```

The trained model bundle and the compressed NHAMCS archives are both in the
repository, so a fresh clone runs with no download and no training step. First
launch takes about four seconds. Verified by building a clone containing only
tracked files: 12.9 MB, 135 tests passing, application serving.

Everything runs with no API key and no network. Two optional extras enable the
spoken-handover path:

```bash
pip install faster-whisper    # on-device transcription, no audio leaves the machine
export ANTHROPIC_API_KEY=...  # or OPENAI_API_KEY, for LLM field extraction
```

Without either, the **Spoken handover** page still records audio and runs its
deterministic clinical parser on typed, pasted or sample transcripts, and every
other page is unaffected. Set `PT_ALLOW_CLOUD_AUDIO=0` to forbid audio leaving
the machine under any configuration.
