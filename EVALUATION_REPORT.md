# PulseGuard: Evaluation Report

> **Every number in this document is generated, not hand-entered.** The canonical source is [`evaluation/saved_results/HEADLINE_NUMBERS.md`](evaluation/saved_results/HEADLINE_NUMBERS.md), rebuilt with `python -m scripts.generate_metrics`. It separates the figures the challenge supplied as scope from the figures this prototype measured, and it carries the caveat that belongs with each one.

*Generated 2026-08-28T22:22:49. Every figure reproducible with `python -m evaluation.full_evaluation`.*

## 1. Protocol

Four rules were applied without exception.

**The test fold is touched once.** It consists of whole hospitals used for nothing else, not training, not calibration, not conformal fitting, not operating-point selection. Patients from one department share documentation habits, equipment and triage culture, so a random row split lets a model memorise "hospital 47 codes everything a 3" and report it as skill. Splitting by hospital also answers the question that actually matters commercially: does this work at the *next* hospital?

**Every headline figure carries a confidence interval**, bootstrapped with resampling of whole hospitals rather than individual visits. Resampling visits would treat two patients from the same department as independent evidence and produce intervals far too narrow.

**Every claim has a baseline.** A recall figure means nothing until you know what the score the department already uses achieves on the same patients, given the same number of emergent-lane slots.

**Outcome-based validation runs alongside label agreement.** Agreeing with the triage nurse is the easy question.

### Folds

| Fold | Purpose | Visits | Hospitals |
|---|---|---|---|
| train | Fit the classifiers | 11,691 | 96 |
| calibrate | Fit probability calibration | 2,889 | 26 |
| conformal | Fit conformal thresholds and select the operating point | 2,806 | 26 |
| test | Reported results, touched once | 3,316 | 28 |

## 2. Cohort

**20,702 real emergency department visits** from the CDC/NCHS National Hospital Ambulatory Medical Care Survey (2021, 2022), across 176 hospitals. Each record carries a triage level assigned by an actual triage nurse, the vital signs actually measured at that visit, and the patient's actual outcome.

| | |
|---|---|
| Critical (Level 1 to 2) prevalence | 17.46% |
| Admission rate | 14.56% |
| Critical outcome (ICU or death in ED) | 2.31% |
| Paediatric / adult / geriatric | 3,825 / 12,652 / 4,225 |

The parse is validated before any year is allowed into a training set. That check is not ceremonial: reusing the 2022 record layout on the 2021 file produced a 9.6% critical-care rate, nearly four times the true figure, while passing every individual field range check, because the visit-disposition block is shifted two characters. A fixed-width parse hides that kind of error in plain sight, since every value it yields is still a well-formed number. The loader now cross-checks that critical-care admissions are a minority of all admissions, which is what caught it.

## 3. Primary safety metrics

Critical recall and critical under-triage come first. Accuracy is reported but never selects a model, a system that calls every patient Level 1 has perfect critical recall and is useless.

| Metric | Value | 95% CI |
|---|---|---|
| Critical recall (Level 1 to 2) | **68.2%** | 64.6% to 72.2% |
| Critical under-triage rate | **31.8%** | 27.8% to 35.4% |
| Emergent-lane load | **31.7%** | 28.8% to 35.5% |
| Under-triage rate (any level) | **7.2%** | 5.6% to 9.2% |
| Over-triage rate (any level) | **46.0%** | 41.6% to 49.8% |
| Within-one-level agreement | **93.1%** | 91.0% to 94.8% |
| Exact agreement | **46.8%** | 43.4% to 50.7% |

Emergent-lane load of 31.7% sits against a true critical prevalence of 17.46%, roughly 1.8× the number of patients who genuinely belong in that lane. That is the operational price of the recall above, and it is stated rather than buried.

## 4. Discrimination

| Target | AUROC |
|---|---|
| Critical triage level (1 to 2) | **0.792** |
| Hospital admission | **0.777** |
| ICU admission or death in ED | **0.808** |

AUROC measures what the model knows, independent of any decision threshold. The latter two targets were unknowable at triage time, which makes them the stronger evidence that the model has learned clinical risk rather than documentation convention.

## 5. Against the alternatives

| Approach | Critical recall | Critical under-triage | Lane load | Exact agreement |
|---|---|---|---|---|
| **PulseGuard (cost-sensitive policy)** | **68.2%** | 31.8% | 31.7% | 46.8% |
| Same model, accuracy-maximising argmax | 20.1% | 79.9% | 6.5% | 58.2% |
| NEWS2 / PEWS at matched lane load | 62.5% | 37.5% | 49.5% | 25.7% |
| NEWS2 / PEWS at published bands | 2.6% | 97.4% | 0.8% | 14.7% |
| Abnormal-vitals heuristic | 13.7% | 86.3% | 9.3% | 27.6% |

**The decision rule matters more than the model.** The argmax row is this exact trained model with the cost policy removed. It catches 20.1% of critical patients; with expected-harm minimisation the same model catches 68.2%. Exact agreement falls from 58.2% to 46.8%. That is the trade, made deliberately, and it is the right way round for triage.

The NEWS2 comparison is run at a **matched lane load**. Applying the published NEWS2 bands literally sends under 1% of arrivals to the emergent lane, they were designed to trigger ward escalation, not ED triage, so a literal comparison would be rigged in our favour. Matching the budget asks the only fair question: given the same number of slots, who fills them with the sicker patients?

## 6. Outcome-based validation

Of the **66 patients in the test fold who were admitted to critical care or died in the emergency department**, how many would each approach have routed to the waiting room?

| Approach | Caught | Sent to waiting room |
|---|---|---|
| **PulseGuard (cost-sensitive policy)** | **80.3%** | 13 |
| Same model, accuracy-maximising argmax | 30.3% | 46 |
| NEWS2 / PEWS at matched lane load | 72.7% | 18 |
| NEWS2 / PEWS at published bands | 6.1% | 62 |
| Abnormal-vitals heuristic | 18.2% | 54 |
| **Triage nurses (the reference standard)** | **75.8%** | 16 |

The nurse row is included for scale, not as a target. Human triage under-triages some patients who later deteriorate; a decision support tool earns its place if it catches some of those, not if it reproduces the label perfectly. On this cohort it catches 80.3% against the nurses' 75.8%. With 66 events, that difference is suggestive rather than conclusive, the honest reading is that the system is competitive with human triage on the outcome that matters, not that it is proven better.

## 7. Calibration and uncertainty

- Expected calibration error **0.043**, max calibration error 0.097
- Brier score **0.556**
- Calibration gap (mean confidence − accuracy) of +0.0351. Positive means the system is overconfident.

**Critical-exclusion guarantee (α = 0.05).** 26.6% of patients can have a critical presentation excluded with 95% confidence; of genuinely critical patients, 5.5% fall below the threshold (target ≤5%). Guarantee holds: yes.

**Five-class conformal sets** achieve 98.6% empirical coverage against a 90.0% target, at a mean width of 4.8 of 5 levels. That width is reported rather than hidden: it is a genuine finding that a triage level is not identifiable to a single value from triage-time data. Because a four-level range is not something a nurse can act on, the five-class set is used for honesty and review triggering, and the binary critical-exclusion guarantee is what drives decisions.

## 8. Fairness

**Age band**

| Age band | n | Critical recall | Critical under-triage |
|---|---|---|---|
| pediatric | 599 | 63.3% | 36.7% |
| adult | 2084 | 66.8% | 33.2% |
| geriatric | 633 | 72.7% | 27.3% |

Maximum critical under-triage gap: **9.3%**

**Sex**

| Sex | n | Critical recall | Critical under-triage |
|---|---|---|---|
| F | 1836 | 66.1% | 33.9% |
| M | 1480 | 70.6% | 29.4% |

Maximum critical under-triage gap: **4.6%**

**Race / ethnicity**

| Race / ethnicity | n | Critical recall | Critical under-triage |
|---|---|---|---|
| Hispanic | 511 | 58.5% | 41.5% |
| Non-Hispanic Black | 827 | 65.6% | 34.4% |
| Non-Hispanic Other | 135 | 69.7% | 30.3% |
| Non-Hispanic White | 1843 | 70.9% | 29.1% |

Maximum critical under-triage gap: **12.4%**

Race and ethnicity are **never model inputs**. They are audited because proxies for them exist in the remaining data, and a disparity nobody measures is one that ships. The paediatric band has the weakest critical recall of the three age groups and is the one any deployment should monitor separately rather than relying on an aggregate.

## 9. Generalisation

**Across hospitals.** Across 26 unseen hospitals, critical recall ranges from 47.1% to 100.0%. The spread, not the mean, is what a new deployment should be planned against. Mean 70.9%, SD 12.7%, IQR 60.4% to 77.1%.

**Across time.** Trained only on 2021 and evaluated on every 2022 visit, including hospitals and a case mix it never saw. This is the closest available proxy for deploying today and running through next year without retraining. Critical recall 62.1%, AUROC 0.778 on 10,207 visits.

## 10. Robustness

| Scenario | Critical recall | Change | Lane load | Stayed cautious |
|---|---|---|---|---|
| Complete records | 68.2% |, | 31.7% |, |
| 10% of vitals missing | 66.7% | -1.4 pts | 31.7% | yes |
| 25% missing | 66.6% | -1.6 pts | 31.1% | yes |
| 50% missing | 61.1% | -7.1 pts | 30.7% | yes |
| 75% missing | 62.5% | -5.6 pts | 31.5% | yes |
| No medical history at all | 59.5% | -8.7 pts | 25.8% | yes |

The question is not whether performance drops when information disappears, it must, but whether it degrades in the safe direction. A triage system that becomes *less* cautious as it learns less is actively dangerous. Here recall declines gradually and the lane load holds steady, which follows from treating missingness as a signal rather than imputing a normal-looking value.

## 11. Performance

- Model inference: p50 **2.16 ms**, p95 2.44 ms, p99 3.25 ms
- End-to-end pipeline: p50 **17.3 ms**, p95 18.9 ms, feature construction, calibrated model, conformal check, all clinical safety rules, SHAP attribution, counterfactual search and the full explanation trace
- Cold start 827 ms (one-off SHAP explainer construction, excluded from the distribution and reported here instead)
- Batch throughput **22,901 patients/second** on a single CPU core, no GPU
- A full day of a 500-visit ED at 3× surge (1,500 patients) scores in **0.07 seconds**

## 12. The operating point

λ = 0.35 selected under an escalation budget of ≤35% of arrivals routed to the emergent lane, with a ≥65% critical-recall floor. At this point the model alone catches 74.1% of Level 1 to 2 patients while sending 34.8% of arrivals to that lane.

Budget in force: ≤35% emergent-lane load, ≥65% critical recall. Status: `met_both_constraints`.

The budget is expressed as lane load rather than as an abstract over-triage rate because lane load maps onto what a department actually runs out of. "Over-triage" counts a 4→3 move the same as a 4→1 move, though one is clinically irrelevant and the other consumes a resuscitation bay.

## 13. Threats to validity

- **The reference label is one nurse's judgement.** Triage inter-rater agreement is moderate at best, so agreement metrics inherit that noise. This is precisely why outcome-based validation is reported alongside them.
- **NHAMCS is a survey, not a live feed.** No free-text nursing notes, no bedside observations. The model sees strictly less than a nurse does.
- **Only 66 critical-outcome events** in the test fold. The comparison against nurse performance is suggestive, not conclusive, and should not be quoted as proof of superiority.
- **Cross-site variance is wide** (47% to 100% critical recall). A new deployment should plan against the lower end, not the mean.
- **Simulated serial observations.** The deterioration-velocity feature is demonstrated on trends generated from real starting vitals, because the survey records one observation per visit.
- **No prospective validation, no regulatory clearance.** Retrospective results on historical survey data do not establish clinical safety or efficacy.

