# Dataset Audit Report

## Scope

This report audits candidate resume datasets for actual label integrity rather than raw dataset size or convenience. The audit includes duplicate rate, label conflicts, class balance, category leakage, and synthetic-block checks. No model training, no splitting, and no hyperparameter tuning were performed.

---

## 1) LiveCareer Resume Dataset — ACCEPTED

### Source

Dataset: LiveCareer Resume Dataset by Snehaan Bhawal

Public source: Kaggle

Observations:
- Publicly accessible via KaggleHub download
- 2,484 rows
- 24 categories
- fields: `ID`, `Resume_str`, `Resume_html`, `Category`
- stored locally in [data/candidates/livecareer](data/candidates/livecareer)

### Verified statistics

- Total rows: 2,484
- Unique categories: 24
- Exact duplicate `Resume_str`: 4 rows (2 duplicate pairs) = 0.16%
- Normalized duplicate resumes: 4 rows = 0.16%
- Missing resume text: 0
- Missing labels: 0
- Duplicate rate: low and not systematic

### Leakage audit

The audit looked for the classic failure modes we saw in the synthetic dataset:

- direct insertion of the category name in the resume text
- repeated synthetic responsibility blocks
- template-generated paragraphs repeated across many resumes
- category names as explicit metadata inserted into the body text

Findings:
- There were no obvious `Category:` or `Target Role:` markers inserted into resume bodies.
- There were 0 explicit `category:` hits in the resume text.
- There were 0 `job category` markers, 0 `target role` markers, and no repeated synthetic role-label block pattern was detected.
- The category labels do occur naturally in actual resume title/headings and role descriptions, which is expected when a resume is written for a specific target job.
- The overall structure and style looked like genuine candidate resumes rather than generated templates.

### Why it passed

The LiveCareer corpus passes because it has:
- a credible public provenance
- a valid category schema
- low duplicate contamination
- no obvious synthetic label injection
- realistic resume text structure
- enough samples for later stratified experimentation

### Verdict

Accept as the primary project dataset.

---

## 2) EXAI-ResumeIntel — ACCEPTED AS ALTERNATE

### Source

Public dataset name: `mithinsagar/exai-resumeintel-data`

Config used: `resumes`

### Verified statistics

- Total rows: 2,484
- Unique categories: 24
- Duplicate `Feature` rows: 4 rows = 0.16%
- Missing text: 1
- Missing labels: 0
- The structure is effectively the same resume corpus with a cleaned `Feature` field instead of a richer HTML/text pair.

### Leakage audit

- No explicit `category:` text injection observed
- No `job category` markers or target-role insertion detected
- No synthetic responsibility block pattern detected
- Category words appear in the resume as legitimate job-title information, not a noisy appended label

### Why it passed

This dataset appears to represent the same live resume corpus under a cleaned public packaging. It is acceptable for future model work if the project wants to use the Hugging Face version instead of the Kaggle variant. The provenance is not as rich as the Kaggle page, but the dataset appears valid and non-synthetic when inspected at the row level.

### Verdict

Accept as an alternate public corpus; not required if the LiveCareer version is already available.

---

## 3) florex/resume_corpus — REJECTED

### Source

Dataset name requested: `florex/resume_corpus`

### Findings

- The dataset could not be located in the public dataset registry during audit.
- Hugging Face dataset lookup returned a `DatasetNotFoundError`.
- There was no accessible public corpus data or metadata available to inspect.

### Why it was rejected

The dataset was not publicly verifiable in the current environment. Because the corpus could not be inspected and its provenance could not be established, it cannot be accepted for the project.

### Verdict

Reject. No public dataset; no audit basis.

---

## 4) CareerCorpus — REJECTED FOR MAIN PROJECT USE

### Source

Reference-only dataset; small publicly discussed resume corpus with roughly 302 resumes across 6 categories.

### Findings

- The corpus is too small to support the main project target of a serious held-out evaluation and an 80% test target.
- It was not publicly accessible as a full dataset in the current environment.
- It may be useful as a small reference or academic validation set, but it is not adequate as the primary dataset for the project's core evaluation.

### Why it was rejected

The size is too small for a robust 24-class or even 6-class resume job-role classification target. It could be used only as a supplemental reference, not as the main training corpus.

### Verdict

Reject for main project use. Consider as a small reference only.

---

## Final decision

VALID DATASET FOUND: LiveCareer Resume Dataset

This project should continue with the LiveCareer dataset and keep the synthetic original dataset only as a documented historical leakage benchmark.

The data is now stored under:

- [data/candidates/livecareer](data/candidates/livecareer)
- [data/candidates/livecareer/livecareer_resume_dataset.csv](data/candidates/livecareer/livecareer_resume_dataset.csv)

No model training has taken place yet, per the audit-only requirement.
