# PatientTriage.ai: Headline Numbers

*Generated 2026-08-29T11:02:29 by `python -m scripts.generate_metrics`. Every figure below is computed from the saved evaluation artifacts or observed by booting the system in this process. None is hand-entered.*


---

## Part 1: Numbers given by the challenge

These describe **scope and required test conditions**. They are not results, and quoting them as achievements would misrepresent the work. A 3x surge is a scenario the brief asked us to survive; the measured outcome of running it is in Part 2.

| Parameter | Value | What it is | Source |
|---|---|---|---|
| Minimum patient records for prototype validation | **15 to 20+** | Required minimum | Brief, Minimum Prototype Expectations |
| Surge stress test | **3x normal volume** | Required scenario | Brief, Minimum Prototype Expectations |
| Triage scale | **5 levels** | Referenced framework | Brief, Reference Parameters |
| Target department scale | **100 to 500+ visits per day** | Illustrative environment, not achieved throughput | Brief, Reference Parameters |
| Assumed data availability | **About half of arrivals have prior records** | Stated assumption | Brief, Reference Parameters |

---

## Part 2: Numbers measured from this prototype

### The one-line summary

| Metric | Value | Measured on |
|---|---|---|
| Patient records tested | **3,316** (plus a 31-patient live board) | Held-out test fold |
| Critical-case recall | **68.2%** (95% CI 64.6% to 72.2%) | 3,316 visits, 28 held-out hospitals |
| Critical under-triage rate | **31.8%** | Same |
| F1, critical vs non-critical | **0.507** | Same |
| F1, five-class macro | **0.237** | Same |
| F1, five-class weighted | **0.385** | Same |
| AUROC, critical | **0.792** | Same |
| Inference latency | **2.16 ms** p50, 2.44 ms p95 | 300 sampled patients, one CPU core |
| End-to-end latency | **17.3 ms** p50 | Includes safety rules, SHAP and the full explanation trace |
| Surge capacity tested | **3x**, 0.07 s for a 1,500-patient day | Real pipeline, not a timing stub |
| Triage levels implemented | **5** (Levels 1 to 5) | `TriageLevel` enum, rule-pack targets, recommended actions |
| Input modalities | **2** (structured form entry, spoken handover (voice)) | Live |
| Automated tests | **135** | `pytest -q` |
| Test coverage | **80%** runtime code, **94%** on the safety engine | `pytest --cov` |
| Over-triage cost per extra critical catch | **2.8 patients** | vs the same model at argmax |

### Classification metrics in full

Reported because they are expected, and reported with the caveat that they are **not** what selects a model here. A five-class macro F1 rewards getting Levels 4 and 5 right, which is the half of triage that cannot hurt anyone. Critical recall governs the design.

| Level | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Level 1 | 0.500 | 0.054 | 0.097 | 56 |
| Level 2 | 0.366 | 0.677 | 0.475 | 566 |
| Level 3 | 0.514 | 0.693 | 0.590 | 1,666 |
| Level 4 | 0.625 | 0.011 | 0.021 | 925 |
| Level 5 | 0.000 | 0.000 | 0.000 | 103 |

- **Macro F1: 0.237** | **Weighted F1: 0.385** | Accuracy: 0.468
- **Critical vs non-critical: precision 0.403, recall 0.682, F1 0.507**, specificity 0.767
- Confusion counts: TP 424, FN 198, FP 628, TN 2,066

### What the safety gain actually costs

A recall improvement means nothing on its own, because recall is trivially bought by escalating everyone. This is the exchange rate.

Against the same model at argmax, on the same 3,316 patients:

- **299 additional critical patients** routed to the emergent lane who would otherwise have gone to the waiting room
- at a cost of **837 additional patients** in that lane in total
- an exchange rate of **2.8 patients over-triaged per extra critical patient caught**

That ratio is the number a charge nurse would ask for first, and it is the honest way to present the headline gain.

### Clinical predictive values

| Measure | Value | What it answers |
|---|---|---|
| Sensitivity (recall) | **68.2%** | Of genuinely critical patients, how many were caught |
| Specificity | 76.7% | Of genuinely non-critical patients, how many were left alone |
| Positive predictive value | 40.3% | Of patients sent to the emergent lane, how many were truly critical |
| **Negative predictive value** | **91.2%** | **Of patients NOT sent, how many were truly non-critical** |

NPV is the one to lead with alongside recall. It is the number that decides whether a nurse can trust a non-critical call, and at 91.2% it is the strongest single figure the system produces. Low PPV (40.3%) is the deliberate consequence: the system accepts false alarms to avoid misses.

### Validated against outcomes, not opinions

Of the **66 test-fold patients admitted to critical care or who died in the emergency department**, an outcome unknowable at triage time, this system routed **80.3%** to the emergent lane. The triage nurses who actually saw them routed **75.8%**.

*With 66 events this is suggestive rather than conclusive, and should be quoted that way.*

### Prototype scope, observed live

| Metric | Value |
|---|---|
| Live board size | 31 patients (25 real held-out visits + 6 synthetic edge cases) |
| Safety escalations fired on the board | 4 |
| High-uncertainty patients flagged | 1 |
| Cases where the system disagreed with the nurse | 13 |
| Patients scored / with a confidence attached | 31 / 31 (invariant holds) |
| Clinical safety rules enabled | 15 of 15 |
| Rule pack content hash | `977113d46708cf8b` |
| Cold boot to a fully scored board | 2.45 s |
| Audit events generated on boot | 33 (triage_decision: 32, reassessment: 1) |

### Waiting-room monitoring, demonstrated

Re-recording worsening vitals (HR 142, RR 34, SpO2 87) on waiting patient `ED-003` moved them from **Level 3 to Level 2**, with a deterioration risk of **high**. Re-scoring is escalate-only, so an improved reading can never walk a patient back down the queue.

### Spoken handover intake

| Metric | Value |
|---|---|
| Input fields the extractor may write | 21 |
| Decision fields the extractor may write | **0** (no triage level, confidence or urgency exists in its schema) |
| Mean extraction latency | 1.11 ms |
| Sample handovers shipped | 4, including a deliberately noisy one carrying a prompt injection |
|  Geriatric, ambulance, hypoxic | 12 fields extracted, 0 rejected, 2.7 ms |
|  Paediatric, walk-in, febrile | 11 fields extracted, 0 rejected, 0.59 ms |
|  Adult, minor, low acuity | 13 fields extracted, 0 rejected, 0.63 ms |
|  Deliberately noisy transcript | 4 fields extracted, 1 rejected, 0.51 ms |

### Engineering scale

- **12,181 lines** of source across 44 modules, plus 1,744 lines of tests across 12 files
- **80% test coverage** on runtime code (engine, models, data and ui; one-shot CLI scripts excluded)
- Coverage on the components that decide a triage level: safety engine **94%**, pipeline 86%, rule pack 89%
- 14 interface pages across 4 roles, 2 site rule packs

### Generalisation

- Across **26 unseen hospitals**, critical recall ranges **47.1% to 100.0%**. The spread, not the mean, is what a new deployment should be planned against.
- **Temporal:** trained on 2021, tested on 2022, critical recall **62.1%**.

---

## Ready-to-quote lines

Copy these. Each is accurate, each is defensible under questioning, and each leads with the number that reflects what the system was built to do.

**For a CV or a one-line summary**

> Built an ED triage decision-support system on 20,702 real CDC survey visits across 176 hospitals; a cost-sensitive decision policy raised critical-case recall from 20.1% to **68.2%** on 3,316 held-out visits from 28 unseen hospitals, at 2.16 ms inference.

**For a technical audience**

> Critical recall 68.2% (95% CI 64.6% to 72.2%), NPV 91.2%, AUROC 0.792, binary F1 0.507 on the critical vs non-critical decision the system actually makes. Hospital-grouped splits, bootstrapped CIs clustered by hospital, 135 automated tests, 80% coverage.

**For an engineering audience**

> 12,181 lines across 44 modules, 135 tests at 80% coverage (94% on the component that sets a triage level), 2.16 ms p50 inference, 17.3 ms end to end including explanation generation, 22,901.1 patients/second on one CPU core.

**For the trade-off, which is the real story**

> The cost-sensitive decision policy catches 299 more critical patients than the same model at argmax, at a cost of **2.8 patients over-triaged per extra critical patient caught**. Accuracy falls from 58.2% to 46.8% in exchange, which is the correct direction for triage.

**For an impact claim**

> Of 66 held-out patients who were admitted to critical care or died in the ED, the system routed **80.3%** to the emergent lane against **75.8%** by the triage nurses who saw them (n=66, suggestive not conclusive).


---

## Read this before quoting an F1

**Macro F1 is 0.237. Do not put that on a CV.** Not because it is wrong, it is computed correctly and verified against scikit-learn, but because quoting it without its cause invites exactly the wrong conclusion.

Here is what produces it, and it is a design decision rather than a defect:

- The cost policy deliberately pulls uncertain patients **up** the scale. On this cohort it routes most arrivals into Levels 2 and 3.
- Levels 4 and 5 therefore score near zero F1 (0.021 and 0.000), and macro F1 averages all five classes with equal weight.
- So macro F1 penalises the system hardest for the half of triage that **cannot hurt anyone**. A patient over-prioritised from Level 5 to Level 3 waits longer than necessary. A patient under-prioritised from Level 2 to Level 3 can die in the waiting room.
- The same model tuned to maximise accuracy reaches 58.2% accuracy and catches **20.1%** of critical patients. This configuration reaches 46.8% accuracy and catches **68.2%**. The F1 and accuracy numbers get worse precisely because the system got safer.

**If you need a single F1, quote the binary one: 0.507** for critical vs non-critical. That is the decision the system actually makes, so it is the F1 that corresponds to something real. Say which one it is, every time.

If an interviewer pushes on the macro F1, the honest answer is the strongest one available: *"it is low, deliberately, and here is the ablation showing what we bought with it."*


---

## What not to claim

- **Do not** quote "100 to 500 visits per day" as throughput this system delivered. It is the brief's description of a target environment. Measured throughput is 22,901.1 patients per second on one CPU core.
- **Do not** quote the 3x surge as a result. It is a required test condition; the result is that the emergent-lane rate did not drift under it.
- **Do not** lead with accuracy (46.8%) or macro F1 (0.237). Both are real, both are reported above, and both are the wrong objective for triage. The same model reaches 58.2% accuracy while catching only 20.1% of critical patients. That trade is the point of the project.
- **Do not** present outcome validation as conclusive. It rests on 66 events.
- No prospective or clinical validation has been performed. These are retrospective results on historical survey data.
