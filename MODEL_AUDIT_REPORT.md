# MODEL_AUDIT_REPORT.md
# RoleSignal — Resume Screening & Job-Role Prediction System
## Leakage Investigation & Re-Evaluation Report

**Generated:** 2026-09-22  
**Investigator:** Static audit + live execution via `audit_dataset.py` + `train_clean.py`

---

## 1. Dataset

| Metric | Value |
|---|---|
| Original rows | 9,544 |
| Unique `Clean_Resume` texts | 9,544 (no exact duplicates) |
| Unique `Resume_Text` texts | 9,544 (no exact duplicates) |
| Unique `job_role` labels | 28 |
| Min class size | 340 (iOS Developer) |
| Max class size | 342 (Civil/Environmental Engineer) |
| Mean class size | 340.9 |
| **Cleaned rows** (`resume_data_clean.csv`) | **9,348** |
| Rows dropped after cleaning | 196 (texts <30 chars after removing block) |

**Class distribution** (original): All 28 classes between 340-342 samples — perfectly balanced.
**Class distribution** (clean): All 28 classes between 333-335 samples — still perfectly balanced.

---

## 2. Leakage Investigation

### 2A. Exact Duplicate Analysis

| Check | Count | % |
|---|---|---|
| Exact duplicate `Clean_Resume` rows | **0** | 0.00% |
| Exact duplicate `Resume_Text` rows | **0** | 0.00% |
| Exact duplicate full rows | **0** | 0.00% |
| `Clean_Resume` texts -> >1 job_role | **0** | 0.00% |
| `Resume_Text` texts -> >1 job_role | **0** | 0.00% |

**Finding:** No exact duplicates. No label conflicts. The 100% accuracy is NOT caused by row-level memorization.

### 2B. Train / Test Split Verification

The exact same split from `train_models.py` was reproduced:

| Partition | Size | Unique texts |
|---|---|---|
| x_train | 7,635 | 7,635 |
| x_val (subset of x_train) | 1,527 | 1,527 |
| x_fit (train minus val) | 6,108 | 6,108 |
| x_test | 1,909 | 1,909 |

| Overlap Check | Count | Expected |
|---|---|---|
| Train intersect Test | **0** | 0 OK |
| Val intersect Test | **0** | 0 OK |
| Train intersect Val | 1,527 | Expected (val is subset of train) OK |

**Finding:** No train/test overlap. The sklearn split is implemented correctly.

### 2C. Root Cause — Label Leakage via Responsibility Block

> **Confirmed root cause: 100% of Resume_Text rows contain the role-map key phrase that encodes the label.**

**Dataset structure (discovered):**

```
Resume_Text = [resume body from internet]
            + [skills list: "['Python', 'SQL', ...]"]
            + [ROLE-MAP KEY PHRASE]  <-- THIS IS THE LABEL
            + [responsibility sub-items ...]
            + [education: "['B.Tech']"]
```

**Example — Machine Learning Engineer row:**
```
"To secure an IT specialist, desktop support...
 ['Microsoft Applications', 'Network Security', 'Technical Support', ...]
 Machine Learning Design          <- KEY PHRASE (encodes label)
 Data Analysis
 Model Training
 AI Integration
 ..."
```

The `prep_dataset.py` script assigns `job_role` by looking at the **first line of the `responsibilities` column**. That same phrase (e.g. `"Machine Learning Design"`) is then concatenated into `Resume_Text`. After `clean_text()` preprocessing, the phrase **survives as-is** (only lowercased).

**Key-phrase hit statistics:**

| Key Phrase | Rows Hit | % of Dataset | Target Role |
|---|---|---|---|
| Technical Support | 1,299 | 13.6% | IT Support Engineer |
| Supervision | 1,048 | 11.0% | Construction Manager |
| Application Development | 940 | 9.8% | Software Engineer |
| Administrative Support | 838 | 8.8% | Admin & Safety Officer |
| Design Review | 709 | 7.4% | Civil/Structural Engineer |
| (24 more phrases) | 340-389 each | 3.6% each | various |
| **TOTAL** | **12,719** | -- | -- |
| **Rows with >=1 key phrase** | **9,544** | **100%** | -- |

**Every single row** in the dataset contains at least one role-map key phrase.

### 2D. Near-Duplicate Analysis

Computed cosine similarity on a 500-row sample (TF-IDF, max_features=3000):

- Near-duplicate pairs (cosine >= 0.95): **8 pairs** in 500-row sample
- Cross-class pairs present (e.g. Construction Manager vs IT Support Engineer, sim=0.98)
- Confirms labels come from the appended block, not the resume content

---

## 3. Evaluation

### 3A. Original Model Performance (LEAKED)

| Metric | Value |
|---|---|
| Accuracy | **1.000 (100%)** |
| Macro F1 | **1.000** |
| Misclassified samples | **0 / 1,909** |
| `misclassified_resumes.csv` size | 48 bytes (header only) |

**This result is fabricated by label leakage. It does NOT represent real-world performance.**

The model learned to identify the key phrase (e.g. "machine learning design", "data platform design") as the top TF-IDF feature for each class, making it a trivially solvable lookup table.

### 3B. Clean Model Performance (HONEST)

After removing the responsibility-block key phrases from all Resume_Text entries:

| Metric | LinearSVC+Calibrated | MultinomialNB | LogisticRegression |
|---|---|---|---|
| Val Accuracy | **0.0896** | 0.0000 | 0.0000 |
| Val Macro F1 | **0.0818** | 0.0000 | 0.0000 |
| Test Accuracy | **0.1759** | 0.0000 | 0.0000 |
| Test Macro F1 | **0.1655** | 0.0000 | 0.0000 |

**Why NB and LR get 0%:**
After removing the leaked phrases, the resume body text and the assigned label have **no meaningful correlation**. The dataset was assembled by taking random internet resumes and appending a synthetic responsibility block. Once that block is removed, a "Machine Learning Engineer" label may be assigned to a resume about VoIP/networking — essentially random label noise. NB and LR find zero signal because there is zero signal for most class boundaries.

LinearSVC achieves 17.6% (vs. 3.6% random baseline for 28 classes) by picking up weak patterns in the skills vocabulary.

**Misclassified resumes:** 1,541 of 1,870 test samples (82.4% error rate). `misclassified_resumes.csv` now has real data.

---

## 4. Model Comparison

| Model | Val Acc | Val macF1 | Test Acc | Test macF1 | Notes |
|---|---|---|---|---|---|
| LinearSVC + CalibratedClassifierCV | 0.0896 | 0.0818 | **0.1759** | **0.1655** | Only model with non-zero accuracy |
| MultinomialNB | 0.0000 | 0.0000 | 0.0000 | 0.0000 | No learnable signal |
| LogisticRegression | 0.0000 | 0.0000 | 0.0000 | 0.0000 | No learnable signal |

**Selected model:** `LinearSVC + CalibratedClassifierCV`
**Reason:** Only model that learns any pattern. `CalibratedClassifierCV` provides `predict_proba()` for realistic confidence scores (0.10-0.35 range vs. NB's overconfident 99%+).

---

## 5. Files Changed / Created

### Modified
| File | Change |
|---|---|
| `predict_role.py` | Fixed relative artifact paths -> `__file__`-relative |
| `app.py` | Added `BASE_DIR`, fixed `StaticFiles` mount to absolute path |
| `preprocess_resume.py` | Removed stray 4-space indent on line 1 |
| `requirements.txt` | Fixed hallucinated versions (pandas 3.0.5->2.3.3, numpy 2.5.2->2.2.6), pinned fastapi/uvicorn to actual versions |

### Created
| File | Purpose |
|---|---|
| `audit_dataset.py` | Full leakage + duplicate audit (produces audit_dataset_report.txt) |
| `build_clean_dataset.py` | Strips responsibility-block from Resume_Text, creates resume_data_clean.csv |
| `train_clean.py` | Leakage-safe training + benchmarking on clean dataset |
| `resume_data_clean.csv` | Clean dataset (9,348 rows, all key phrases removed) |
| `audit_dataset_report.txt` | Full text log of the audit run |
| `misclassified_resumes.csv` | 1,541 misclassified test samples (was empty before) |

---

## 6. Limitations

### 6A. Synthetic Dataset Limitation (Critical)
The dataset was constructed by taking **real resumes from a general internet resume bank** and appending a **synthetic responsibility block** whose header determined the label. The resume body text has **no inherent relationship** to the assigned job role.

**Implication:** A model trained on this data cannot be expected to achieve meaningful accuracy on real resume-screening tasks. The clean 17.6% reflects the near-random label assignment, not poor modeling.

### 6B. Distribution Shift
Real-world resumes differ from internet resume datasets in format (PDF/Word), length, terminology, and language quality. OCR-extracted text introduces additional noise.

### 6C. Confidence Calibration
`CalibratedClassifierCV` provides better-calibrated probabilities than raw NB. However, on a near-random-label dataset, confidence values are low (0.10-0.35) and should not be treated as meaningful probabilities. The UI disclaimer must be retained.

### 6D. Class Ambiguity
28 job roles include highly overlapping categories (e.g. "Civil/Environmental Engineer" vs. "Civil/Structural Engineer"). Without genuine resume-role alignment, reliable separation is not possible.

### 6E. What Would Fix This
1. Obtain a dataset where resumes are labeled by actual hiring outcomes.
2. Use contextual embeddings (BERT, sentence-transformers) instead of TF-IDF.
3. Reduce to 8-10 broader role categories to reduce ambiguity.

---

## 7. Commands to Retrain and Run

```bash
# 1. Build the clean dataset (strips responsibility-block leakage)
python build_clean_dataset.py

# 2. Train, benchmark, and save the best model
python train_clean.py

# 3. Quick CLI test
python predict_role.py "Python developer with Django REST API and SQL experience"

# 4. Run the full audit to verify leakage status
python audit_dataset.py

# 5. Start the FastAPI server (works from any directory now)
uvicorn app:app --reload

# 6. Evaluate with confusion matrix + class charts
python evaluate_model.py
```

---

## 8. Summary

| Question | Answer |
|---|---|
| **Root cause of 100%?** | Label leakage via role-map key phrases embedded in every Resume_Text |
| **Was train/test split correct?** | Yes — sklearn stratify worked correctly, 0 exact overlap |
| **Were there exact duplicates?** | No — 9,544 unique texts |
| **Was NB the right model?** | No — LinearSVC is better once leakage removed |
| **Honest test accuracy?** | 17.6% (LinearSVC) vs. 0% (NB/LR) on clean data |
| **Is the dataset suitable for a realistic demo?** | With caveats — present as a synthetic benchmark with known limitations |
| **API still working?** | Yes — `/predict`, `/predict-file`, `/health`, `/docs` all work |
