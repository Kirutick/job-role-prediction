# Vercel Bundle Audit

## Summary

The deployment failure was caused by the project packaging far more than the production inference stack.

The production bundle was effectively including:
- full development datasets (`resume_data_labeled.csv`, `resume_data_clean.csv`, `preprocessed_resume_data (1).csv`)
- notebook and audit artifacts (`resume_screening.ipynb`, `misclassified_resumes.csv`, `audit_dataset_report.txt`)
- large training and evaluation folders (`results/`, `eda_images/`, `splits/`, `tests/`)
- heavy development dependencies (`pandas`, `nltk`, `matplotlib`, `seaborn`, `jupyterlab`, `pytesseract`, `datasets`, `kagglehub`)

This combination explains why the Vercel deployment reported a function bundle above the limit even though the actual runtime model is small.

---

## 1) Local Vercel build status

I attempted to run the required local build command:

```bash
npx vercel build
```

but the environment is not authenticated to Vercel and the CLI exits with:

```text
Error: The specified token is not valid. Use `vercel login` to generate a new token.
```

Because of that, the actual Vercel output bundle could not be generated in this environment. The original 723.80 MB figure therefore remains the deployment-side figure reported by the failing project, not a fresh local measurement from this session.

---

## 2) Actual oversized files inside the repository

These are the largest runtime and development files currently present in the repo root and are the most obvious bundling offenders:

- `preprocessed_resume_data (1).csv` — 18.95 MB
- `resume_data_labeled.csv` — 15.82 MB
- `resume_data_clean.csv` — 14.85 MB
- `job_role_model.pkl` — 3.22 MB
- `resume_screening.ipynb` — 1.27 MB
- `misclassified_resumes.csv` — 1.11 MB
- `tfidf_vectorizer.pkl` — 0.20 MB

These files are not all required in production inference and should not be shipped with the serverless function.

---

## 3) Largest Python packages in the original requirements

The original `requirements.txt` was pulling in a large development-heavy stack, including:

- `pandas` — used by training, auditing, and data prep
- `numpy` — required for model inference but not large in isolation
- `scikit-learn` — required for inference and still acceptable
- `scipy` — transitive dependency of scikit-learn
- `nltk` — training and preprocessing tooling, not required in inference
- `matplotlib` and `seaborn` — plotting, not required at runtime
- `jupyterlab` — development-only
- `pytesseract` — optional OCR wrapper; not deployable without an external Tesseract binary
- `datasets`, `kagglehub`, and `huggingface_hub` — dataset download and audit tooling, not production inference

---

## 4) Largest files and unnecessary artifacts packed into the deployment

The following unnecessary or development-only files were present in the project root and would be included in a naive Vercel Python bundle:

- `resume_data_labeled.csv`
- `resume_data_clean.csv`
- `preprocessed_resume_data (1).csv`
- `misclassified_resumes.csv`
- `audit_dataset_report.txt`
- `resume_screening.ipynb`
- `results/`
- `eda_images/`
- `splits/`
- `tests/`
- `data/candidates/*` (dataset audit copies)
- `data/genuine/*` (historical dataset copies)

These are not required for inference at runtime and should not be included in the serverless bundle.

---

## 5) Model artifacts

The only model artifacts needed for production are:

- `job_role_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_classes.pkl`
- `metadata.json` (small)
- `data/role_requirements.json` and `data/screening_weights.json` (runtime screening configuration)

These are retained intentionally. Old training and audit artifacts were not kept in the deployment bundle.

---

## 6) OCR and PDF findings

OCR was present in the app as a direct dependency:

```python
import pytesseract
```

This is not a safe serverless deployment dependency because it requires the actual Tesseract executable on the runtime OS. Vercel does not provide that by default.

The correct production pattern is:
- keep PDF text extraction via `pypdf`
- keep image OCR as an optional local/dev feature only
- reject image OCR in serverless deployment when Tesseract is unavailable

That is now handled by guarded imports and explicit runtime errors in the application.

---

## 7) Root cause of the Vercel bundle issue

The root cause was not the actual model size. It was the combination of:
- large development data files in the repo root
- full training/evaluation dependencies in production
- no Vercel-specific bundle exclusion rules
- OCR dependency that is not serverless-safe by default

---

## 8) Changes applied to reduce the bundle size

The project now uses:
- a minimal production requirements file: [requirements.txt](requirements.txt)
- a dedicated dev-only file: [requirements-dev.txt](requirements-dev.txt)
- a minimal Vercel config: [vercel.json](vercel.json)
- an ignore file to exclude large development data: [.vercelignore](.vercelignore)
- lazy predictor initialization and optional OCR imports in [app.py](app.py)

These changes reduce the production payload to the runtime inference stack instead of the full development environment.

---

## 9) Verified local runtime checks

The app was smoke-tested after the production dependency cleanup:

```bash
python -c "from fastapi.testclient import TestClient; from app import app; client=TestClient(app); health=client.get('/health'); print('HEALTH', health.status_code, health.json()); pred=client.post('/predict', json={'resume_text':'Python developer with Django REST API and SQL experience'}); print('PREDICT_STATUS', pred.status_code); print(pred.json())"
```

Observed result:

- health status: 200
- model_loaded: True
- predict status: 200

The standalone CLI was also verified:

```bash
python predict_role.py "Python developer with Django REST API and SQL experience"
```

Observed result:

- Predicted job role: Mechanical Design Engineer
- Confidence: 0.1236

---

## 10) Current deployment status

This project is now structurally reduced to a production inference bundle, but the final Vercel CLI bundle cannot be proven in this environment because the required Vercel authentication token is missing.

The honest status is:

- bundle root cause identified and reduced
- serverless runtime path verified locally
- actual Vercel deployment build remains blocked by auth, not by Python code execution

This is the correct deployment-safe state before a real authenticated Vercel build is run.
