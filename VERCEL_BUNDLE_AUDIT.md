# Vercel Bundle Audit

## Summary

The production app is now importing correctly, and the remaining blocker is size, not code execution.

Fresh Vercel build output from the real project shows the current Python function bundle is:

- Total bundle size: 514.00 MB
- Limit: 500 MB
- Result: build failed with `Error: Total bundle size (514.00 MB) exceeds the maximum function size (500 MB).`

The large footprint is dominated by the compiled ML stack and by local build-cache artifacts, not by the model files themselves.

---

## 1) Actual build evidence

Command used:

```bash
npx vercel build --project job-role-prediction-smjf --debug
```

Observed result:

```text
Error: Total bundle size (514.00 MB) exceeds the maximum function size (500 MB).
```

This is the current measured size of the production function bundle, before any final deployment can succeed.

---

## 2) What is actually making it large

The main contributors are the runtime ML dependencies that Vercel installs into the function environment:

- `scikit-learn` — the largest pure-Python/compiled dependency in the inference stack
- `scipy` — large compiled numerical dependency
- `numpy` — compiled numerical dependency
- `uvicorn` and `fastapi` — lightweight compared to the ML stack, but still necessary
- `pillow` and `pypdf` — small but required for the upload/PDF path

The production model files themselves are not the real issue:

- `job_role_model.pkl` — approx 3.2 MB
- `tfidf_vectorizer.pkl` — approx 0.2 MB
- `label_classes.pkl` — small
- `metadata.json` — tiny

These are not large enough to explain a 500 MB build failure on their own.

---

## 3) What was removed from the deployable runtime footprint

The following developer-only artifacts were explicitly excluded from the deployable function bundle:

- `results/`
- `eda_images/`
- `splits/`
- `tests/`
- `data/candidates/`
- `data/genuine/`
- notebooks and large audit artifacts
- CSV datasets and duplicate cleaned copies
- local Vercel caches and virtualenv metadata

This is enforced in [.vercelignore](.vercelignore) and [.gitignore](.gitignore).

---

## 4) Production vs development dependency split

### Production requirements

The app imports only these runtime categories in [app.py](app.py):

- FastAPI / ASGI runtime
- scikit-learn model loading and prediction
- joblib model serialization
- NumPy / SciPy numerical support
- PDF extraction via `pypdf`
- basic upload handling via `python-multipart`
- optional image handling via `PIL` only when OCR is used

### Development-only requirements

These are intentionally separated into [requirements-dev.txt](requirements-dev.txt):

- `pandas`
- `matplotlib`
- `seaborn`
- `jupyterlab`
- `nltk`
- `datasets`
- `huggingface_hub`
- `kagglehub`
- `pytesseract`
- notebook and audit tooling

These are not needed for the serverless prediction API and should not be shipped to the deployed function.

---

## 5) Model artifact review

The runtime model artifacts kept in Git and required for inference are:

- `job_role_model.pkl`
- `tfidf_vectorizer.pkl`
- `label_classes.pkl`
- `metadata.json`
- `data/role_requirements.json`
- `data/screening_weights.json`

No retraining was performed. No additional model variants were added to the production deployment.

---

## 6) OCR and PDF handling

The critical safety constraint is that `pytesseract` requires a separate Tesseract binary and is not a safe default in Vercel serverless. This is why the app keeps OCR as optional and returns a controlled runtime error rather than crashing the whole function when OCR is not available.

The PDF path remains valid because `pypdf` is a runtime dependency and is not a platform-specific binary.

---

## 7) Vercel configuration review

Current [vercel.json](vercel.json) is minimal and points at the FastAPI entrypoint. The remaining problem is not route setup; it is dependency footprint. The remaining large consumers are still the compiled numerical libraries and cached build artifacts created during the Vercel install step.

The relevant safeguards now in place are:

- [.vercelignore](.vercelignore)
- [.gitignore](.gitignore)
- runtime-only [requirements.txt](requirements.txt)
- dev-only [requirements-dev.txt](requirements-dev.txt)

---

## 8) Current status

### Previous bundle

- 723.80 MB

### Current measured bundle

- 514.00 MB

### Result

- Still above Vercel’s 500 MB Python function limit

### What is still consuming the space

The remaining large footprint is primarily the compiled ML stack:

- SciPy
- scikit-learn
- NumPy

These dependencies are required for prediction and cannot be removed without changing the model or the inference stack. The project is therefore at the point where the bundle is already trimmed to the runtime minimum, but the platform still enforces a hard limit that is smaller than the installed footprint of the chosen dependency set.

At this point, the honest status is:

- runtime import issue: fixed
- Vercel bundle issue: still failing due to size cap
- no random redesign or retraining was done
- the function is still too large for the current Vercel limit

This is the exact remaining blocker.
