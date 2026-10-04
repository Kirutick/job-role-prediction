# Model Card: RoleSignal Classifier

## Model Details

- **System Name:** RoleSignal Resume Screening & Job-Role Prediction
- **Model Type:** TF-IDF + Calibrated LinearSVC (current demo) / TF-IDF + TruncatedSVD LSA (alternative)
- **Task:** Multi-class text classification (job role prediction from resume text)
- **Date:** October 2026
- **Status:** ⚠️ DEMO ONLY — No genuine training dataset available

---

## Intended Use

- **Primary use case:** Technical demonstration of an ML engineering pipeline — text preprocessing,
  feature extraction, leakage detection, and REST API integration.
- **Research/educational use:** Demonstrating how label leakage creates artificially inflated metrics,
  and how to detect and document it.

### Out-of-Scope Use Cases

This model is **NOT suitable for**:
- Making actual hiring or screening decisions in any environment
- Automating the rejection of job candidates
- Any production deployment without genuine labelled training data
- Any claim of accurate resume-to-role classification

---

## Dataset Limitations & Training Data

### CRITICAL — Dataset Status

**Three datasets were inspected. None qualified as genuine labelled data.**

#### `resume_data_labeled.csv` — REJECTED (Synthetic Labels)

9,544 rows across 28 job roles. Labels were assigned by appending a synthetic
"responsibility block" to each resume text. The block encoded the target role directly.
For example:

- Resume body: *"Fresher looking to join as a data analyst..."*
- Appended block: `Machine Learning Leadership / ML System Design / Algorithm Research`
- Assigned label: `Senior ML/NLP Engineer`

The model learned to classify the appended phrases, not the resume content.
This produced 100% test accuracy — entirely fabricated by the leakage.

**This dataset remains in the repository ONLY as a documented leakage demonstration.**

#### `resume_data_clean.csv` — REJECTED (Labels Remain Synthetic)

Same data with responsibility block stripped. Test accuracy drops to ~8.5%.
This is expected: the labels were never derived from the resume content — they
were derived from the block that was removed. The underlying resumes have no
genuine correspondence to the assigned roles.

**This dataset remains in the repository ONLY as a "cleaned leaked baseline" benchmark.**

#### `UpdatedResumeDataSet.csv` (Kaggle, external) — REJECTED (Insufficient Size)

962 rows, 25 categories. Audit revealed only **166 unique resumes** after deduplication.
Each unique resume appears ~6 times. A 70/15/15 split would yield approximately
4 training samples per class — completely insufficient for any meaningful classifier.

The category name appearing in resume text (84% of rows) was investigated and confirmed
to be **natural content** (job titles, education degrees), not appended leakage.
However, the duplication problem makes this dataset unusable regardless.

---

## Evaluation Methodology

### Honest Baseline (resume_data_clean.csv — demo only)

- **Splits:** 70% Train, 15% Validation, 15% Test. Stratified by class. Seed = 42.
- **Leakage check:** 0 train/test overlap confirmed.
- **Test Accuracy:** ~8.5%
- **Macro F1:** ~0.075

These metrics represent the **honest result after removing leakage**, not genuine classification.

### Genuine Model (pending)

No genuine model has been trained. Metrics will be reported here once a valid
dataset is obtained and training is completed. The test set will be evaluated exactly once.

---

## Performance Metrics

### Original (leaked baseline) — DO NOT CITE AS REAL PERFORMANCE
```
Test Accuracy: ~100%
Macro F1:      ~1.00
Root cause:    Label leakage (responsibility block encoded the target role)
Status:        INVALID
```

### Cleaned (demo only — labels still synthetic)
```
Test Accuracy: ~8.5%
Macro F1:      ~0.075
Root cause:    No genuine signal between resume text and assigned roles
Status:        Honest but not meaningful
```

### Genuine model
```
Test Accuracy: NOT YET MEASURED (no genuine dataset)
Macro F1:      NOT YET MEASURED
Status:        PENDING
```

---

## Known Biases & Limitations

- **Keyword reliance:** TF-IDF is sensitive to specific keywords and cannot understand
  semantic context. "Managed a team of software engineers" vs "worked as a software engineer"
  may produce very different classifications despite similar meaning.
- **English only:** The model only processes English-language resumes.
- **No genuine signal:** The current model was trained on synthetically labelled data.
  Its predictions are not meaningful.
- **Dataset origin unknown:** The underlying resume text in the original dataset was scraped
  from internet sources. Its origins, biases, and demographic distribution are unknown.
- **Coarse categories:** If a genuine dataset is used, the job categories may not match
  fine-grained real-world roles (e.g., "Data Science" covers many different actual jobs).

---

## Confidence Limitations

The model uses `CalibratedClassifierCV` to provide probabilities. Because the current
dataset lacks genuine signal, confidence values are low and not meaningful.

The UI explicitly states:

> *"Confidence represents model probability, not a guarantee of candidate suitability."*

If a genuine model is trained in the future, calibration quality should be evaluated
using reliability diagrams (calibration curves) before confidence values are displayed.

---

## Human Oversight Requirements

All automated screening tools must remain under human oversight. The screening breakdown
provided by the `/analyze` endpoint is a rigid heuristic and must be treated as a
demonstration, not a deterministic assessment of a person's capability.

Specifically:
- A skill absent from extracted text does not mean the candidate lacks it
- The screening score is labelled "demo heuristic" and cannot be used for hiring decisions
- Missing common skills are labelled "missing common skills", NOT "candidate weaknesses"
- Role prediction confidence is model output, not an assessment of candidate suitability

---

## Reproducibility

### To reproduce the original 100% accuracy (leakage demo):
1. Load `resume_data_labeled.csv`
2. Use `Resume_Text` column (includes responsibility block)
3. Train any TF-IDF classifier
4. Result: ~100% test accuracy

### To reproduce the honest 8.5% result:
1. Run `python build_clean_dataset.py` (strips responsibility block)
2. Run `python split_dataset.py`
3. Run `python train_tfidf.py`
4. Evaluate on `splits/test_ids.csv`
5. Result: ~8.5% test accuracy

---

## Files

| File | Purpose |
|---|---|
| `audit_dataset.py` | Full leakage detection audit |
| `inspect_datasets.py` | Phase 1 dataset inspection |
| `download_and_audit_genuine_dataset.py` | Downloads and audits external datasets |
| `PHASE1_FINDINGS.md` | Official Phase 1 findings report |
| `MODEL_AUDIT_REPORT.md` | Original detailed leakage audit report |
| `build_clean_dataset.py` | Strips responsibility block from original data |
| `train_tfidf.py` | TF-IDF baseline training |
| `train_semantic.py` | LSA (TF-IDF + TruncatedSVD) training |
| `split_dataset.py` | 70/15/15 stratified split generation |
