# Model Card: RoleSignal Resume Classifier

## Overview

- **Task:** Multi-class classification of resume text into occupational
  categories.
- **Dataset:** ResumeAtlas, `ahmedheakl/resume-atlas`, revision
  `3f80ca910fa9964890afb7845c09e239d07b9b1d`.
- **Input features:** Word-level TF-IDF unigrams and bigrams, sublinear term
  frequency, Unicode accent stripping, up to 10,000 features.
- **Classifier:** `LinearSVC(class_weight="balanced")`.
- **Classes:** 43 ResumeAtlas categories.
- **Model artifact:** `models/tfidf_model.pkl`.
- **Status:** Educational/demo classifier; not suitable for employment
  decisions.

## Dataset and filtering

ResumeAtlas contains 13,389 rows. The source schema was inspected and mapped
from `Text` and `Category` to `resume_text` and `job_role`; IDs identify source
row order. Audit results:

- 43 classes.
- No missing or empty resume texts or category labels.
- 1,304 duplicate normalized-text rows across 1,200 groups (the exact-duplicate
  count is also 1,304).
- 169 duplicate-text groups, comprising 381 rows, have conflicting categories.
  All rows in those groups were excluded rather than choosing an arbitrary
  label.
- Same-category duplicate texts were collapsed, keeping the first source row.
- Total excluded: 381 conflicting-label rows and 1,092 same-label duplicate
  copies. Final dataset: 11,916 unique resumes.
- Seven resumes shorter than 100 characters were flagged and retained.
- The audit found zero explicit appended `Category`/`job_role` marker blocks.
  Category words occurring in ordinary resume text are not by themselves
  considered evidence of label leakage.

The raw dataset is public under the dataset repository's MIT license. The
resume text may still contain personal or sensitive information; generated
misclassification reports are git-ignored and must not be published.

## Split and model selection

Duplicates were removed before splitting. A stratified split with random state
42 produced:

| Split | Samples |
|---|---:|
| Train | 8,340 |
| Validation | 1,788 |
| Test | 1,788 |

Normalized-text overlap was zero for train/test, train/validation, and
validation/test. Vectorizers and candidate models were fitted on train only.
The validation set selected the candidate by macro-F1; the test set was
evaluated once after selection.

Twelve candidates were compared: Logistic Regression, LinearSVC, calibrated
LinearSVC, and class-balanced LinearSVC, each with 10,000, 20,000, and 30,000
maximum TF-IDF features. The selected candidate was class-balanced LinearSVC
with 10,000 features.

## Performance

| Metric | Validation | Held-out test |
|---|---:|---:|
| Accuracy | 0.8272 | 0.8322 |
| Top-3 accuracy | — | 0.9379 |
| Macro precision | 0.8317 | 0.8327 |
| Macro recall | 0.8250 | 0.8304 |
| Macro F1 | 0.8232 | 0.8274 |
| Weighted F1 | 0.8220 | 0.8271 |

The test accuracy exceeds 80% on this particular held-out split. This is not
evidence of 80% performance on a different dataset, employer population,
future resumes, or real hiring outcomes. No additional model tuning was
performed against the test results.

## Confidence and intended use

The selected `LinearSVC` does not produce calibrated probabilities. The API
therefore returns `null` confidence rather than presenting a raw decision score
as probability. The existing frontend formats a missing confidence as `0%`
and shows no top-three entries for this classifier; that `0%` is a UI fallback,
not a model score. The model's class ranking is not a candidate suitability
assessment. The separate screening breakdown is a demo heuristic and is not a
probability or hiring score.

Do not use this model to rank, screen, reject, or select applicants. Dataset
categories are not verified employment outcomes. The dataset may have
representation, labeling, collection, and occupational coverage biases. Human
review and independent validation on appropriately governed, representative
data would be required before considering any consequential use.

## Reproducibility

```powershell
python audit_resume_atlas.py
python train_resume_atlas.py
```

The audit writes `resume_data_real.csv`, `resume_atlas_audit.json`, and
`splits/`. Training writes the runtime model artifacts, `metadata.json`,
`resume_atlas_results.json`, `confusion_matrix.png`, `class_performance.png`,
and `misclassified_resumes.csv`. Resume-containing CSV outputs are git-ignored.
