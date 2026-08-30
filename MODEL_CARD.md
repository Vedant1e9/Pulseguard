# Model Card: PatientTriage.ai

Following the model card framework (Mitchell et al., 2019), with the sections a
clinical reviewer would ask for.

---

## Model details

| | |
|---|---|
| **Name** | PatientTriage.ai triage risk model |
| **Version** | 2.1.0 |
| **Date** | 2026-08-28 |
| **Type** | Gradient-boosted decision trees (XGBoost), 5-class ordinal outcome |
| **Selected from** | XGBoost, LightGBM, HistGradientBoosting, Random Forest, Logistic Regression |
| **Features** | 135 (71 clinical + 64 complaint-text components) |
| **Calibration** | Platt scaling, selected against isotonic and no-calibration on a disjoint fold |
| **Uncertainty** | Split-conformal prediction, class-conditional, α = 0.10; binary critical-exclusion at α = 0.05 |
| **Decision rule** | Expected-cost minimisation under a versioned cost matrix |
| **Final authority** | A deterministic rule engine, not this model |

### Why boosted trees and not a neural network

With roughly 30 structured physiological features and 20,000 rows, boosted
trees are both the stronger performer and the auditable one. In a system whose
output a clinician must justify to a patient's family, auditability is not a
nice-to-have. The text branch (TF-IDF → truncated SVD) is deliberately simple
for the same reason: the complaint codes in NHAMCS are a controlled
vocabulary, not free prose, so a large language model would add opacity without
adding signal.

### No LLM is used in the decision path

This is a considered choice, not an omission. Triage decisions must be
deterministic, reproducible, offline-capable and explainable to a regulator. A
language model in the decision path would compromise all four. The natural
place for one is generating the patient-facing narrative, strictly downstream
of and unable to influence the level.

---

## Intended use

**Intended:** decision *support* for a trained triage nurse or emergency
physician in an emergency department, as a second opinion that surfaces
uncertainty and flags presentations that hide their severity.

**Explicitly not intended for:**
- Autonomous triage without a clinician in the loop
- Diagnosis, treatment selection or disposition decisions
- Use outside an emergency department (inpatient deterioration, primary care, telehealth)
- Any clinical use at all, this is an unvalidated prototype
- Populations materially unlike the training cohort (non-US EDs, without recalibration)

---

## Training data

| | |
|---|---|
| **Source** | CDC/NCHS National Hospital Ambulatory Medical Care Survey, ED component |
| **Years** | 2021, 2022 |
| **Records** | 20,702 visits with a nurse-assigned triage level (of 32,232 sampled) |
| **Hospitals** | 176 |
| **Label** | `IMMEDR`, the immediacy level assigned by the triage nurse at the visit |
| **Split** | Grouped by hospital: 55% train / 15% calibrate / 15% conformal / 15% test |

### Cohort composition

| Level | Share |
|---|---|
| 1, Immediate | 1.8% |
| 2, Emergent | 15.7% |
| 3, Urgent | 52.0% |
| 4, Semi-urgent | 27.0% |
| 5, Non-urgent | 3.5% |

Age: 3,825 paediatric · 12,652 adult · 4,225 geriatric.
Admission rate 14.6% · critical outcome (ICU or death in ED) 2.3%.

### Preprocessing

- Missing values are **preserved as NaN**, never imputed. The primary models
  are NaN-native, and every vital carries a companion missing-indicator.
- Temperature converted from Fahrenheit; documented sentinel codes (`-9` blank,
  `-8` unknown, `-7` not applicable, pulse `998` = Doppler) mapped to missing.
- Vitals additionally expressed as z-scores against published normal ranges for
  the patient's age band.
- NEWS2 (adults) and PEWS (children) computed and supplied as features.

### Race and ethnicity are audited, never used

Race/ethnicity is parsed from the survey and used **only** to audit the system
for disparate impact. It is never a model input. Proxies for it exist in the
remaining data, so the disparity is measured rather than assumed away.

---

## Evaluation

All figures on 3,316 visits from **28 hospitals held out** from training,
calibration, conformal fitting and threshold selection. Confidence intervals
bootstrapped clustered by hospital.

### Primary safety metrics

| Metric | Value | 95% CI |
|---|---|---|
| Critical recall (Level 1 to 2) | **68.2%** | 64.6 to 72.2% |
| Critical under-triage rate | **31.8%** | 27.8 to 35.4% |
| Emergent-lane load | **31.7%** | 28.8 to 35.5% |
| Within-one-level agreement | **93.1%** | 91.0 to 94.8% |
| Exact agreement | **46.8%** | 43.4 to 50.7% |

### Discrimination

| Target | AUROC |
|---|---|
| Critical triage level (1 to 2) | **0.792** |
| Hospital admission | **0.777** |
| ICU admission or death in ED | **0.808** |

The latter two are outcomes unknowable at triage time, which makes them the
more meaningful measures of whether the model has learned clinical risk rather
than documentation habits.

### Against alternatives, on identical patients

| Approach | Critical recall | Lane load |
|---|---|---|
| **This system** | **68.2%** | 31.7% |
| NEWS2/PEWS at matched lane load | 62.5% | 49.5% |
| Same model, accuracy-maximising argmax | 20.1% | 6.5% |
| Abnormal-vitals heuristic | 13.7% | 9.3% |

### Outcome-based validation

Of the 66 test-fold patients admitted to critical care or who died in the ED:

| Approach | Caught |
|---|---|
| **This system** | **80.3%** |
| **Triage nurses (actual)** | **75.8%** |
| NEWS2 at matched lane load | 72.7% |
| Accuracy-maximising argmax | 30.3% |

### Calibration

Expected calibration error **0.043**, Brier score **0.556**. Calibration
reduced ECE from 0.089 to 0.035 on the selection fold. The method (Platt) was
chosen on a fold used for neither training nor calibrator fitting.

### Fairness

| Age band | n | Critical recall | Critical under-triage |
|---|---|---|---|
| Paediatric | 599 | 63.3% | 36.7% |
| Adult | 2,084 | 66.8% | 33.3% |
| Geriatric | 633 | 72.7% | 27.3% |

Maximum critical under-triage gap across age bands **9.3%**; across
race/ethnicity groups **12.4%**. Both are reported rather than minimised: a
9-point gap in how reliably paediatric versus geriatric critical patients are
caught is a real limitation, and the paediatric band is the one to watch.

### Generalisation

- **Cross-site:** across 26 unseen hospitals, critical recall ranges
  **47.1% to 100%** (mean 70.9%, SD 12.7%). The spread is what a new deployment
  should be planned against.
- **Temporal:** trained on 2021 only, tested on all of 2022, critical recall
  **62.1%**, AUROC **0.778**. Degrades but does not collapse across a genuine
  distribution shift.

### Robustness

| Scenario | Critical recall | Change |
|---|---|---|
| Complete records | 68.2% |, |
| 10% of vitals missing | 66.7% | −1.5 pts |
| 25% missing | 66.6% | −1.6 pts |
| 50% missing | 61.1% | −7.1 pts |
| 75% missing | 62.5% | −5.7 pts |
| No medical history at all | 59.5% | −8.7 pts |

Performance degrades gradually and the emergent-lane load holds steady, the
system does not become quietly more permissive as its inputs thin out.

### Performance

p50 model inference **2.4 ms**, p95 5.6 ms. p50 **end-to-end 18.7 ms**
including all safety rules, SHAP attribution and the full explanation trace.
Batch throughput ~18,000 patients/second on a single CPU core. A full day of a
500-visit ED at 3× surge scores in well under a second.

---

## Ethical considerations

**Automation bias.** The most likely harm is not a wrong recommendation but a
clinician deferring to a right-looking one. Mitigations: the confidence figure
states what it measures; uncertainty is displayed prominently; the what-if
explorer exists specifically so staff learn where the system is unreliable;
overrides require a clinical note and are never discouraged.

**Alert fatigue.** Over-flagging pushes staff to ignore warnings. Deterioration
alerts are separated from wait-time alerts in the queue so the two do not merge
into undifferentiated noise, and the escalation budget caps how much of the
board can be escalated at all.

**Under-served groups.** Paediatric critical recall is the weakest band. Any
deployment must monitor it separately rather than relying on an aggregate.

**Data protection.** See [`DATA_PROTECTION.md`](DATA_PROTECTION.md).

---

## Environment this build was evaluated on

```
Python 3.13 · scikit-learn 1.6.1 · xgboost 3.4.1 · lightgbm 4.7.0
numpy 2.1.3 · pandas 2.2.3 · shap 0.52.0 · streamlit 1.45.1
```

---

## Caveats

- Retrospective evaluation on historical survey data. No prospective study, no
  clinical validation, no regulatory clearance.
- NHAMCS lacks bedside observations (AVPU, skin perfusion, bleeding control).
  Those drive several safety rules but cannot be learned from this data.
- The reference label is one nurse's judgement, itself imperfect. Agreement
  metrics inherit that noise, which is why outcome-based validation is
  reported alongside them.
- Survey weights (`PATWT`) are required for national estimates. Model training
  is unweighted, since each record is one real visit.
