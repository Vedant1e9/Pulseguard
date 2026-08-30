# PulseGuard: Data Protection Policy

## Jurisdiction

**United States, HIPAA + California CMIA (Confidentiality of Medical Information Act)**

All data protection measures are designed to comply with both federal (HIPAA) and state (California CMIA) requirements. Where requirements differ, the stricter standard applies.

## Data Classification

| Data Category | Classification | Handling |
|---|---|---|
| Patient vitals (real-time) | PHI, HIPAA Protected | Encrypted at rest and in transit; access logged; minimum necessary principle |
| Self-reported symptoms | PHI | Stored as part of encounter record; uncertain statements preserved as uncertain |
| Medical history | PHI | Retrieved via authenticated EHR integration; access logged |
| Staff observations | PHI | Attributed to observing clinician; immutable once recorded |
| **Spoken handover audio** | **PHI + Biometric identifier** | Transcribed on device by default; never persisted; destroyed the moment a transcript exists. See *Spoken handover audio* below |
| **Handover transcripts** | **PHI** | Retained as part of the encounter record; every field derived from one is tagged `voice_transcribed` |
| Triage recommendations | PHI + System Output | Linked to encounter; model version recorded; retained per policy |
| Override records | PHI + Clinical Decision | Immutable; extended retention for malpractice defense |
| Audit trail | System Metadata + PHI refs | Immutable; 6-year minimum retention |
| Aggregate analytics | De-identified | HIPAA Safe Harbor de-identification applied before aggregation |
| Model artifacts | Technical/Trade Secret | Version-controlled; training data snapshots retained for reproducibility |

## Consent Model

### Treatment Operations (Implicit Consent)
Under HIPAA's Treatment, Payment, and Health Care Operations (TPO) exception, patient data may be used for triage without separate consent. Triage is integral to emergency care delivery.

### Secondary Use (Explicit Consent Required)
Any use beyond immediate triage requires explicit, informed consent or IRB waiver:
- Model training and improvement
- Quality improvement studies
- Aggregate reporting beyond TPO scope
- Research (requires IRB approval under 45 CFR 46)

Under California CMIA, consent for secondary use must be **written and specific**, identifying the type of information, the parties authorized to disclose and receive, and the purpose.

### AI Transparency
Patients and their representatives are informed that an AI decision-support system assists in priority assessment. If a patient objects, the system's recommendation is suppressed, and purely manual triage is applied. This is logged.

## Data Retention Policy

| Data Category | Retention Period | Legal Basis |
|---|---|---|
| Individual triage records (PHI) | 7 years from date of service | HIPAA minimum; CMIA 7-year requirement |
| Clinician override records | 10 years | California medical malpractice statute of limitations |
| Audit trail logs | 6 years | HIPAA administrative requirement (45 CFR §164.530(j)) |
| De-identified aggregate data | Indefinite | No PHI; Safe Harbor de-identification |
| Model artifacts and training snapshots | Life of model version + 3 years | Reproducibility and regulatory audit |
| **Spoken handover audio** | **Not retained. Destroyed once transcribed** | Data minimisation (45 CFR §164.502(b)); no clinical purpose survives transcription |
| **Handover transcripts** | 7 years, as part of the encounter | Same basis as the triage record they produced |
| Backup copies | Same as source data | Consistent with primary retention |

**Disposal:** At the end of retention periods, data is securely destroyed using NIST SP 800-88 compliant methods (cryptographic erasure for encrypted storage, overwrite for unencrypted media).

## Spoken Handover Audio

Voice intake introduces a data category the rest of this policy did not
previously contemplate, and it carries risks that a typed form does not.

### Why audio is treated more strictly than the record it produces

A recording of a nursing handover is PHI twice over. It carries the patient's
age, sex, presenting complaint and physiology, and it carries the speaker's
voice, which is a biometric identifier capable of identifying the *clinician*
indefinitely. A transcript is a clinical record. The audio behind it is a
clinical record plus a voiceprint, and there is no clinical purpose that
survives transcription.

It also captures whatever else was audible. A handover spoken in a corridor may
pick up a second patient, a family member, or a colleague's unrelated
conversation. Nobody in that bystander group consented to anything.

### Controls

| Control | Implementation |
|---|---|
| **On-device by default** | Local Whisper is preferred and listed first wherever a backend is chosen. The recording never leaves the machine |
| **Cloud transcription is labelled, never silent** | If a cloud speech service is the only backend available, every surface that offers it states that audio leaves the building |
| **Hard off switch** | `PT_ALLOW_CLOUD_AUDIO=0` removes cloud transcription entirely. A deployment handling real patients sets this |
| **No persistence** | Audio is held in memory for the length of one transcription and never written to disk, logged, or included in a backup |
| **Provenance on every derived value** | Fields that came from speech are stamped `voice_transcribed`, distinct from `manual_entry`, so an auditor can find every value that passed through a speech model |
| **Human confirmation before use** | No transcribed value reaches a triage decision until a nurse has seen and confirmed it on screen |
| **Bounded extraction** | The extractor may write 21 named input fields and nothing else. Free text cannot enter the record except through a closed complaint vocabulary |

### What a real deployment would add

- **Explicit clinician notice and consent** for voice capture, separate from
  patient treatment consent, since the voiceprint is the clinician's.
- **Bystander mitigation:** a push-to-talk control rather than open-mic, and
  guidance to record at the bedside rather than in a corridor.
- **A documented BAA** with any cloud transcription vendor before that backend
  is enabled, and a DPIA if operating under GDPR rather than HIPAA.
- **Transcription accuracy monitoring by accent and first language.** A speech
  model that transcribes some clinicians less reliably than others is a
  fairness problem expressed as a data quality problem, and the confirmation
  step masks it rather than fixing it.

### Prototype status

The prototype does not persist audio and defaults to on-device transcription.
It has not been assessed against a real department's acoustics, accents or
bystander exposure, and no BAA exists for the optional cloud backend. Voice
intake should be treated as demonstrating a workflow, not as cleared for use
with real patients.

## Access Control

### Role-Based Access Control (RBAC)

| Role | Data Access | Justification |
|---|---|---|
| Triage Nurse | Current encounter vitals, symptoms, staff cues, triage recommendation | Minimum necessary for triage decision |
| ED Physician | Full current encounter + history (if available) + model details | Clinical review and override authority |
| Hospital Administrator | Aggregate dashboards only (de-identified) | Operational management; no individual PHI |
| Compliance Officer | Audit trails + override records (with PHI under logged access) | Regulatory compliance and audit |
| System Administrator | Technical logs only (no PHI) | System maintenance |
| Patient/Family | Simplified queue position and wait estimate only | Patient experience; no clinical data exposed |

### Authentication
- All users authenticate via hospital SSO integration
- Multi-factor authentication required for clinical systems access
- Session timeout: 15 minutes of inactivity for clinical users

### Access Logging
Every data access event is logged with:
- User identity (authenticated)
- Timestamp
- Data accessed (patient ID, data category)
- Purpose (triage, review, override, audit, administrative)
- Access method (UI, API)

Anomalous access patterns (accessing records outside assigned patients, bulk data retrieval, after-hours access) trigger automated alerts to the Information Security team.

## Encryption

| Layer | Standard | Implementation |
|---|---|---|
| Data at rest | AES-256 | Database-level encryption; encrypted storage volumes |
| Data in transit | TLS 1.3 | All API communication; no plaintext PHI transmission |
| Backup encryption | AES-256 | Encrypted backup media with separate key management |
| Key management | AWS KMS / Azure Key Vault / GCP Cloud KMS | Hardware security modules; automatic key rotation |

## Breach Response

In compliance with HIPAA Breach Notification Rule (45 CFR §§ 164.400-414) and CMIA breach provisions:

1. **Discovery:** Breach discovered or reported → Security team notified within 1 hour
2. **Assessment:** Risk assessment within 24 hours (nature, extent, mitigation, harm potential)
3. **Notification (if required):**
   - Individual notification: within 60 days of discovery (HIPAA)
   - HHS notification: within 60 days if ≥500 individuals; annual log if <500
   - California AG notification: if >500 California residents affected (CMIA)
   - Media notification: if >500 residents of a single state/jurisdiction affected
4. **Remediation:** Root cause analysis, vulnerability patching, policy update

## Fairness and Non-Discrimination

Patient data is protected from unfair use through:

1. **No discriminatory triage:** The system does not use race, ethnicity, national origin, religion, sexual orientation, gender identity, or insurance status as triage features. Age and sex are used only as clinically relevant variables (e.g., age-specific vital thresholds).

2. **Fairness monitoring:** Under-triage rates are monitored by age band (mandatory) and will be extended to additional demographic proxies in Phase 3 (Differentiator I) when multi-site data supports it. Systematic under-triage of any subgroup triggers model review.

3. **No secondary monetization:** Patient data is never sold, shared with advertisers, used for insurance scoring, or disclosed to employers. Use is strictly limited to clinical care and quality improvement within the covered entity.

4. **Bias audit:** The evaluation dashboard surfaces per-group performance metrics to enable proactive detection of algorithmic bias.

## Prototype Disclaimer

This data protection policy describes the **design intent** for a production deployment. The current prototype:

- Uses **synthetic data only**, no real patient data
- Runs **locally**, no cloud storage or network transmission of PHI
- Has **no real authentication**, demo uses simulated clinician IDs
- Has **no real EHR integration**, all patient data is generated in-memory
- **Does** implement a genuinely tamper-evident audit trail: every override and audit event is SHA-256 hash-chained and written to an append-only log file (`engine/override_audit.py`), with a one-click integrity check in the Audit Log screen. This is a real mechanism, not just a documented intention, though the chain currently resets on process restart and writes to a local file rather than a WORM store or database, which a production deployment would harden further.

A real deployment would require comprehensive security assessment, penetration testing, and compliance audit before handling any real patient data.

---

*PulseGuard — Accenture Innovation Challenge 2026, Round 2, Problem Track 2 (PatientTriage.ai)*

---

## Research Data Used to Build This Prototype

This section covers the data used to *develop* the system, which is distinct
from the patient data a deployment would process.

### Source

Model training and evaluation use the **National Hospital Ambulatory Medical
Care Survey (NHAMCS), Emergency Department component**, published by the
National Center for Health Statistics (NCHS), CDC, survey years 2021 and 2022.

### Legal basis and terms

NHAMCS public-use files are released by NCHS for statistical reporting and
analysis under the Public Health Service Act. All direct identifiers, and any
characteristics that might permit identification, are removed by NCHS before
release. The data use terms require that users:

1. Use the data for statistical reporting and analysis only;
2. Make no attempt to learn the identity of any person or establishment;
3. Not link the dataset with individually identifiable data from other sources;
4. Not attempt to assess or defeat the disclosure protections applied.

**This project complies with all four.** The data is used solely to train and
evaluate a statistical model; no re-identification is attempted; no linkage to
any other dataset is performed; and no disclosure-limitation methodology is
probed. Hospital identifiers in the file are NCHS-assigned masked codes used
only to group records for leakage-free splitting, they are not resolvable to
real institutions.

### Handling

| Aspect | Treatment |
|---|---|
| Classification | De-identified public-use research data, **not PHI** |
| Storage | Local to the development environment, under `data/real/` |
| Distribution | Not redistributed. The loader downloads from the CDC public mirror on demand |
| Retention | For the life of the model version, for reproducibility |
| Use in the running application | Held-out records are displayed on the demo board and clearly labelled as real de-identified survey records |

### Why real data was used at all

A triage system validated only on data its authors generated proves nothing: a
synthetic cohort can be made arbitrarily easy without anyone noticing, and the
resulting metrics are unfalsifiable. Using a public, de-identified, nationally
representative survey means the reported numbers can be independently
reproduced by anyone who downloads the same file, which is the point of
publishing them.

### Race and ethnicity

The race/ethnicity field is parsed and used **exclusively** to audit the system
for disparate impact. It is never a model input. Excluding it from the audit
would not make the system fairer; it would only make any unfairness
undetectable, since proxies for it exist throughout the remaining features.

### What a real deployment would change

Nothing in this section transfers to production. A deployed system processes
identifiable PHI under the HIPAA and CMIA controls described above: encrypted
at rest and in transit, access logged, minimum-necessary role-based access, and
the retention schedule set out earlier in this document. Model retraining on
live patient data would additionally require the explicit written consent or
IRB waiver described under **Secondary Use**, since improving a model is not a
treatment activity covered by the TPO exception.
