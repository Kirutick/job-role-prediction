# RoleSignal: Resume Screening & Job-Role Prediction

RoleSignal is a demonstration application that predicts one of 43 occupational
categories from resume text using a TF-IDF classifier trained on ResumeAtlas.
It also extracts skills and displays a separate heuristic screening breakdown.
It is not a hiring or candidate-suitability system.

## Dataset and model

The classifier uses the public
[ResumeAtlas dataset](https://huggingface.co/datasets/ahmedheakl/resume-atlas),
pinned to revision `3f80ca910fa9964890afb7845c09e239d07b9b1d`. The source has
13,389 rows and 43 categories. The audit found 1,304 repeated normalized-text
rows, including 381 rows in 169 duplicate-text groups with conflicting
categories. All rows in conflicting groups were excluded; same-label normalized
duplicates were collapsed, retaining the first source row. This leaves 11,916
unique resumes. Seven resumes shorter than 100 characters were flagged and
retained. The audit found no explicit category/role marker blocks appended to
resume text.

The unique resumes were split once, stratified by category with random state
42: 8,340 train, 1,788 validation, and 1,788 test. Normalized resume text has
zero overlap between any pair of splits. Model/vectorizer fitting uses the
training split only; validation macro-F1 selected the model, and the held-out
test split was evaluated once.

The selected model is **TF-IDF (10,000 maximum features) + class-balanced
LinearSVC**. Validation accuracy was **82.72%** and validation macro-F1 was
**0.8232**. Held-out test accuracy was **83.22%**, top-3 accuracy **93.79%**,
macro-F1 **0.8274**, and weighted F1 **0.8271**. These are results on this
dataset and split, not a guarantee of performance on other resumes or sources.
See [MODEL_CARD.md](MODEL_CARD.md), [DATA_STRATEGY.md](DATA_STRATEGY.md), and
`resume_atlas_results.json` for details.

The raw download is held in the Hugging Face cache. The normalized
`resume_data_real.csv`, split files, confusion-matrix images, and
`misclassified_resumes.csv` are local generated outputs and are git-ignored;
the misclassified CSV contains resume text and should not be published.
Historical synthetic datasets and their leakage-audit records remain unchanged
for reference; they are not used by the current runtime model.

## Local setup

Use Python 3.12 and install the runtime requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For dataset auditing, training, and report generation, install the development
dependencies:

```powershell
pip install -r requirements-dev.txt
```

Image OCR also requires the Tesseract system executable. On Windows, install
Tesseract OCR and make it available on PATH; the application can also locate
the standard Windows installation directory.

## Rebuild the dataset and model

Run these commands from the project root:

```powershell
python audit_resume_atlas.py
python train_resume_atlas.py
```

The audit downloads the pinned ResumeAtlas revision, identifies its text and
label columns, documents duplicates and filtering, normalizes the schema to
`resume_id`, `resume_text`, and `job_role`, and creates the stratified splits.
Training refuses to proceed if the audit/leakage checks fail. It compares the
specified TF-IDF classifier candidates on validation macro-F1, then evaluates
the selected model once on the held-out test set. It writes the model artifacts
to `models/`, metrics to `metadata.json` and `resume_atlas_results.json`, and
the requested evaluation plots and misclassification report.

## Run the application

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000`. API endpoints:

- `GET /health` — readiness and model status
- `GET /docs` — interactive API documentation
- `POST /predict` — predict a role from resume text
- `POST /analyze` — role prediction, skills, and screening breakdown
- `POST /predict-file` — extract text from a PDF or image, then analyze it

The model emits a confidence value only when its classifier provides
probabilities. The selected class-balanced LinearSVC is not calibrated, so its
current API confidence is `null`; raw decision scores are not presented as
calibrated confidence. The existing frontend renders a missing confidence as
`0%` and has no top-three entries for this model; that display is not a
calibrated score. The screening breakdown is an independent demo heuristic,
not model probability or hiring suitability.

## Limitations

- ResumeAtlas category labels are dataset annotations, not verified hiring
  outcomes or validated assessments of candidate suitability.
- The held-out test metrics describe only the audited ResumeAtlas split and
  may not generalize across employers, occupations, languages, resume formats,
  or populations.
- Duplicate-text groups with conflicting labels were excluded rather than
  assigning an arbitrary target. Seven very short records were retained.
- TF-IDF is based on word features and does not understand context or validate
  qualifications.
- PDF/image text extraction can omit or misread content.
- Never use these predictions or the heuristic breakdown to make hiring,
  rejection, or other consequential decisions.
