# PulseGuard

**Team PulseGuard — Submission README**

**Accenture Innovation Challenge 2026 · Round 2 · Problem Track 2 (PatientTriage.ai)**

**Repository:** `https://github.com/Vedant1e9/Pulseguard`
**Run it:** `pip install -r requirements.txt && streamlit run app.py` — no download step, no training step, ~2.5 s to first screen.

> ⚕️ A research prototype for demonstration. It does not diagnose, does not treat, and must not replace assessment by a qualified clinician. No prospective or regulatory validation has been performed.

---

## 1. The one-paragraph version

An AI-assisted triage assistant for emergency departments. It gives the triage nurse a second opinion on how urgently a patient needs to be seen, states how confident it is and in what terms, explains itself as a rule with a threshold rather than as a model output, and keeps a tamper-evident record of every decision. It is trained and evaluated on **20,702 real ED visits** from the CDC's National Hospital Ambulatory Medical Care Survey, not on data we generated ourselves.

---

## 2. The design decision the whole system rests on

Under-triage — a genuinely sick patient assigned a lower urgency than they need — is the failure mode that kills people in emergency departments. It is also the failure mode an accuracy-optimised model makes *worse*, because predicting the common case is how a model maximises accuracy.

The same trained model, on the same 3,316 held-out patients:

| Decision rule | Critical patients caught | Exact agreement with nurse |
|---|---|---|
| Pick the most likely level (argmax) | 20.1% | 58.2% |
| **Minimise expected clinical harm** | **68.2%** | 46.8% |

Identical model, identical data. The entire difference is in how a probability becomes an action — the layer most triage prototypes never build. This is why the system separates **risk estimation** from the **decision rule**, and publishes the full operating frontier rather than a single tuned point.

**What that safety gain costs, stated up front:** 837 additional patients in the emergent lane for 299 additional critical patients caught — an exchange rate of **2.8 over-triaged per extra critical patient found**. Exact agreement falls to 46.8% *because* the system got safer.

---

## 3. What is in this repository

**This repository is the code and the working prototype.** The business proposal
and the pitch presentation are submitted separately through the AIC portal and
are deliberately not published here.

| Component | Where | What it is |
|---|---|---|
| Working prototype | `app.py` + `ui/` | 14 pages across 4 clinical roles |
| Decision pipeline | `engine/` | Safety rules, orchestration, explanation, audit |
| Models | `models/` | Training, calibration, cost policy, uncertainty |
| Data layer | `data/` | NHAMCS parser, features, schema, demo cohort |
| Clinical policy | `config/` | Two versioned rule packs, content-hashed |
| Evaluation | `evaluation/`, `scripts/` | Full harness, and the regeneration of every published figure |
| Tests | `tests/` | 135 tests, including regressions for fixed bugs |

**Supporting evidence, in the order a sceptical reader would want it**

| Document | Answers |
|---|---|
| `evaluation/saved_results/HEADLINE_NUMBERS.md` | Every quotable figure, generated not hand-entered |
| `EVALUATION_REPORT.md` | Full evaluation protocol and narrative |
| `PROJECT_OVERVIEW.md` | The complete picture in one document |
| `ARCHITECTURE.md` | How the pieces fit together |
| `MODEL_CARD.md` | Intended use, training data, limitations |
| `DATA_PROTECTION.md` | Jurisdiction, retention, consent, audio as PHI |

---

## 4. Data

**CDC / NCHS National Hospital Ambulatory Medical Care Survey**, Emergency Department component, 2021–2022.

| | |
|---|---|
| Visits with a nurse-assigned triage level | 20,702 |
| Hospitals | 176 |
| Critical (Level 1–2) prevalence | 17.46% |
| ICU admission or death in ED | 2.31% |
| Paediatric / adult / geriatric | 3,825 / 12,652 / 4,225 |

Every record carries the level an actual nurse assigned, the vitals actually measured, **and what actually happened to the patient** — which is what allows validation against outcomes rather than against another human's opinion.

Six clearly labelled synthetic cases exercise the bedside-observation safety rules NHAMCS cannot supply. They are excluded from every accuracy figure.

---

## 5. Results

All figures on **3,316 visits from 28 hospitals held out** of training, calibration, conformal fitting and threshold selection. Confidence intervals bootstrapped clustered by hospital.

**Safety**

| Metric | Value | 95% CI |
|---|---|---|
| Critical recall | **68.2%** | 64.6–72.2% |
| Negative predictive value | **91.2%** | — |
| Within-one-level agreement | 93.1% | 91.0–94.8% |
| Emergent-lane load | 31.7% | 28.8–35.5% |

**Discrimination** — AUROC 0.792 (critical level), 0.777 (admission), **0.808 (ICU admission or death)**. The latter two were unknowable at triage time, which makes them the stronger evidence that the model learned clinical risk rather than documentation convention.

**Against outcomes, not opinions.** Of the 66 held-out patients admitted to critical care or who died in the ED:

| Approach | Caught |
|---|---|
| **PulseGuard** | **80.3%** |
| Triage nurses (actual) | 75.8% |
| NEWS2 at matched lane load | 72.7% |
| Accuracy-maximising argmax | 30.3% |

With 66 events this is **suggestive, not conclusive**. The honest reading is that the system is competitive with human triage on the outcome that matters, not that it is proven better.

**Calibration, fairness, generalisation** — ECE 0.043. Max critical under-triage gap 9.3% across age bands, 12.4% across race/ethnicity (race is audited, never a model input). Across 26 unseen hospitals, critical recall 47%–100%. Trained on 2021 and tested on 2022: 62.1%.

**Speed** — 2.16 ms p50 inference, 17.3 ms end-to-end including SHAP and counterfactual search, 22,901 patients/sec on one CPU core with no GPU.

**Engineering** — 135 automated tests (~8 s), 80% coverage on runtime code and 94% on the safety engine, 12,181 lines across 44 modules. A fresh clone is ~13 MB and boots in ~2.5 s.

---

## 6. How a decision is made

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

**Models propose; only the rule engine disposes.** Two invariants are unit-tested: the engine can only escalate, never reduce urgency; and every escalation records its rule, threshold, evidence, citation and the rule-pack content hash. Both intake paths converge on the identical encounter object and the identical engine — there is no shortcut for spoken input.

---

## 7. Which parts are AI, and which are arithmetic

A triage assistant that answers *how did you decide this* with "the AI said so" is not deployable. The **AI boundary** page in the app names the technique behind every component:

| Component | Technique |
|---|---|
| Triage level (the decision) | Deterministic rules |
| Risk estimate feeding it | Traditional ML (XGBoost) |
| Turning risk into an action | Decision theory (expected-cost minimisation) |
| Uncertainty | Conformal prediction |
| Per-factor explanation | Traditional ML (TreeSHAP) |
| Early warning score | Published clinical standard (NEWS2 / PEWS) |
| Multi-agent safety debate | Deterministic heuristics, **no language model** |
| Handover transcription | Speech model (Whisper), optional |
| Handover field extraction | LLM, optional |

**Nine of eleven components involve no generative model at all.** The two that do sit at intake, *before* anything is scored, and both have a deterministic fallback.

| | |
|---|---|
| Components that set the triage level | **1** (the safety engine) |
| Language-model calls per triage decision | **0** |
| LLM cost per triage decision | **$0.00** |

**The spoken-intake boundary.** The extraction schema contains 21 named *input* fields and no urgency field at all — a model cannot write a field that does not exist, so a prompt injection inside a transcript has nowhere to land. Anything below 55% confidence arrives empty rather than guessed. Physiologically impossible values are rejected and shown, never silently dropped. Every field is stamped `voice_transcribed`, distinct from `manual_entry`. A **runnable injection demo** ships in the app for a judge to try. 34 tests cover this boundary.

---

## 8. Meeting the brief's minimum expectations

| Requirement | Where |
|---|---|
| Triage scoring on 15–20+ records | 31-patient live board; 3,316 in evaluation |
| At least one ambiguous presentation | Real cohort throughout; disagreement panel on the board |
| Paediatric / geriatric case | `EDGE-002`, `EDGE-003`; 3,825 / 4,225 real visits, age-banded thresholds |
| Zero-history first-time patient | `EDGE-005`, a dedicated rule, and a whole-cohort ablation |
| Behaviour under 3× surge | **Robustness & surge** page, live through the real pipeline |
| Never a score without a confidence indicator | Enforced structurally in `TriageResult`, unit-tested |
| Clinician override, and what is logged | **Review & override**; hash-chained log, downgrades gated |
| Monitor the queue, act on worsening vitals | **Reassessment round**; escalate-only by construction |
| Stated regulatory jurisdiction | US — HIPAA + California CMIA |

---

## 9. Running it

```bash
pip install -r requirements.txt
streamlit run app.py          # ~2.5 s to the patient board
pytest -q                     # 135 tests, ~8 s
```

Everything runs with no API key and no network. Two optional extras enable the spoken-handover path:

```bash
pip install faster-whisper    # on-device transcription, no audio leaves the machine
export OPENAI_API_KEY=...     # or ANTHROPIC_API_KEY, for LLM field extraction
export PT_ALLOW_CLOUD_AUDIO=0 # forbid audio leaving the machine under any config
```

Without either, the **Spoken handover** page still records audio and runs its deterministic clinical parser on typed, pasted or sample transcripts, and every other page is unaffected.

Reproduce every published number:

```bash
python -m scripts.generate_metrics       # regenerate HEADLINE_NUMBERS.md
python -m evaluation.full_evaluation     # the full harness
```

---

## 10. Where a judge could push back

Stated here rather than left to be discovered.

- **Exact agreement with the nurse is 46.8%.** Deliberate, defended on the Model performance page, and the direct cost of the recall above.
- **Cross-site variance is wide** (47%–100% critical recall across 26 hospitals). Real, and reported rather than averaged away. A new deployment should plan against the lower end.
- **The outcome validation rests on 66 events.** Suggestive, not conclusive.
- **Serial vitals are simulated** for the deterioration demonstration, and labelled as such wherever they appear. NHAMCS records one observation per visit.
- **Speech-to-text is evaluated on synthesised audio in a quiet room**, not on accented speech in a loud department. The mandatory confirmation step exists precisely because that gap is real.
- **Live intakes and the audit chain head live in process memory** and reset on restart. A deployment would persist them.
- **No prospective validation, no clinical validation, no regulatory clearance.**

---

## 11. What is not in the repository, and why

- **Extracted NHAMCS survey files** (74 MB) are gitignored; the compressed archives (3.7 MB, US federal public-domain data) are tracked, and the loader extracts them on first run. Nothing to download.
- **The trained model bundle is tracked** (8.2 MB) so a fresh clone runs without a training step. Reproducible any time with `python -m scripts.train_model`.
- **Runtime audit logs** are gitignored, so a fresh clone starts with an empty audit trail. Use the app for a minute and the **Audit log** page fills, hash-chained, with a runnable integrity check.

---

**Team:** PulseGuard — Vedant Sewatkar ([@Vedant1e9](https://github.com/Vedant1e9))
**Repository:** https://github.com/Vedant1e9/Pulseguard
