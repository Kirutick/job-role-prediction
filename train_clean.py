"""train_clean.py
================
Retrain and benchmark all three classifiers on the CLEAN dataset
(resume_data_clean.csv) produced by build_clean_dataset.py.

Steps:
  1. Load clean dataset
  2. Stratified 80/20 train-test split
  3. Fit TF-IDF on train only
  4. Benchmark: MultinomialNB, LogisticRegression, LinearSVC
  5. Use inner validation to select best model
  6. Evaluate all on held-out test set
  7. Wrap LinearSVC in CalibratedClassifierCV if selected
  8. Save the selected model artifacts
  9. Print full classification report + generate misclassified_resumes.csv
 10. Save updated metadata.json

Run from the project root:
    python train_clean.py
"""

import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE           = Path(__file__).resolve().parent
DATA_PATH      = BASE / "resume_data_clean.csv"
MODEL_PATH     = BASE / "job_role_model.pkl"
VECTORIZER_PATH= BASE / "tfidf_vectorizer.pkl"
LABELS_PATH    = BASE / "label_classes.pkl"
METADATA_PATH  = BASE / "metadata.json"
ERRORS_PATH    = BASE / "misclassified_resumes.csv"

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
VAL_SIZE       = 0.20   # of train pool

TFIDF_CONFIG = dict(
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True,
    max_df=0.95,
    min_df=2,            # ← raised from 1 to reduce very-rare memorization
    strip_accents="unicode",
)


# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred) -> dict:
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "macro_precision":    precision_score(y_true, y_pred, average="macro",    zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall":       recall_score(y_true, y_pred,    average="macro",    zero_division=0),
        "weighted_recall":    recall_score(y_true, y_pred,    average="weighted", zero_division=0),
        "macro_f1":           f1_score(y_true, y_pred,        average="macro",    zero_division=0),
        "weighted_f1":        f1_score(y_true, y_pred,        average="weighted", zero_division=0),
        "minority_class_f1":  float(per_class_f1.min()),
    }


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run build_clean_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    required = {"Clean_Resume", "job_role"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")

    data = df[["Clean_Resume", "job_role"]].dropna()
    data = data[(data["Clean_Resume"].astype(str).str.strip() != "") &
                (data["job_role"].astype(str).str.strip() != "")]
    x, y = data["Clean_Resume"].astype(str), data["job_role"].astype(str).str.strip()

    print(f"Clean dataset rows: {len(x)}")
    print(f"Unique classes:     {y.nunique()}")
    print(f"Class distribution (min/max): {y.value_counts().min()} / {y.value_counts().max()}")

    # ── Split ──────────────────────────────────────────────────────────────────
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=y_train
    )
    print(f"\nSplit sizes — fit: {len(x_fit)} | val: {len(x_val)} | test: {len(x_test)}")

    # ── TF-IDF vectorizer: fit ONLY on x_fit ──────────────────────────────────
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    x_fit_tfidf  = vectorizer.fit_transform(x_fit)
    x_val_tfidf  = vectorizer.transform(x_val)
    x_train_tfidf= vectorizer.transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    # ── Model factories ────────────────────────────────────────────────────────
    role_counts  = y_train.value_counts()
    imbalance    = role_counts.max() / role_counts.min()
    class_weight = "balanced" if imbalance >= 2.0 else None
    print(f"Imbalance ratio: {imbalance:.2f}  -> class_weight={class_weight or 'none'}")

    def make_nb():    return MultinomialNB()
    def make_lr():
        return LogisticRegression(max_iter=1000, solver="lbfgs",
                                  class_weight=class_weight,
                                  random_state=RANDOM_STATE)
    def make_svc():
        base = LinearSVC(class_weight=class_weight, max_iter=3000,
                         random_state=RANDOM_STATE)
        return CalibratedClassifierCV(base, cv=3)

    factories = {
        "MultinomialNB":       make_nb,
        "LogisticRegression":  make_lr,
        "LinearSVC+Calibrated":make_svc,
    }

    # ── Validation phase ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 — Validation (vectorizer fit on x_fit only)")
    print("=" * 70)
    val_results = []
    val_models  = {}
    for name, factory in factories.items():
        model = factory()
        t0 = perf_counter()
        model.fit(x_fit_tfidf, y_fit)
        elapsed = perf_counter() - t0
        row = compute_metrics(y_val, model.predict(x_val_tfidf))
        row.update({"model": name, "train_sec": elapsed})
        val_results.append(row)
        val_models[name] = model
        print(f"  {name:25s}  acc={row['accuracy']:.4f}  macroF1={row['macro_f1']:.4f}  "
              f"minF1={row['minority_class_f1']:.4f}  t={elapsed:.1f}s")

    val_df = pd.DataFrame(val_results).set_index("model")
    print("\nFull validation table:")
    print(val_df.round(4).to_string())

    # Select by macro_f1 → minority_class_f1 → weighted_f1
    val_results_sorted = sorted(
        val_results,
        key=lambda r: (r["macro_f1"], r["minority_class_f1"], r["weighted_f1"]),
        reverse=True,
    )
    selected_name = val_results_sorted[0]["model"]
    print(f"\nSelected on validation: {selected_name}")

    # ── Test-set evaluation for all models (retrained on full x_train) ─────────
    print("\n" + "=" * 70)
    print("PHASE 2 — Test evaluation (vectorizer re-fit on x_train only)")
    print("=" * 70)
    # Fit a fresh vectorizer on the full training set
    final_vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    x_train_tfidf_final = final_vectorizer.fit_transform(x_train)
    x_test_tfidf_final  = final_vectorizer.transform(x_test)

    test_results  = []
    test_models   = {}
    test_reports  = {}
    for name, factory in factories.items():
        model = factory()
        t0 = perf_counter()
        model.fit(x_train_tfidf_final, y_train)
        elapsed = perf_counter() - t0
        y_pred = model.predict(x_test_tfidf_final)
        row = compute_metrics(y_test, y_pred)
        row.update({"model": name, "train_sec": elapsed})
        test_results.append(row)
        test_models[name] = (model, y_pred)
        test_reports[name] = classification_report(y_test, y_pred, zero_division=0)
        print(f"  {name:25s}  acc={row['accuracy']:.4f}  macroF1={row['macro_f1']:.4f}  "
              f"minF1={row['minority_class_f1']:.4f}  t={elapsed:.1f}s")

    print("\nFull test table:")
    test_df = pd.DataFrame(test_results).set_index("model")
    print(test_df.round(4).to_string())

    for name, report in test_reports.items():
        print(f"\n--- Classification report: {name} ---\n{report}")

    # ── Summary comparison table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Val Acc':>8} {'Val macF1':>10} {'Test Acc':>9} {'Test macF1':>11}")
    for vrow, trow in zip(val_results_sorted, test_results):
        vname = vrow["model"]
        trow_match = next(r for r in test_results if r["model"] == vname)
        print(f"  {vname:<23} {vrow['accuracy']:>8.4f} {vrow['macro_f1']:>10.4f} "
              f"{trow_match['accuracy']:>9.4f} {trow_match['macro_f1']:>11.4f}")

    # ── Save the SELECTED model ────────────────────────────────────────────────
    print(f"\nSaving selected model: {selected_name}")
    selected_model, selected_preds = test_models[selected_name]
    joblib.dump(selected_model,  MODEL_PATH)
    joblib.dump(final_vectorizer, VECTORIZER_PATH)
    label_classes = sorted(y.unique())
    joblib.dump(label_classes,   LABELS_PATH)

    metadata = {
        "model_name":       selected_name,
        "dataset":          "resume_data_clean.csv",
        "random_state":     RANDOM_STATE,
        "tfidf_config":     {**TFIDF_CONFIG, "ngram_range": list(TFIDF_CONFIG["ngram_range"])},
        "train_samples":    len(x_train),
        "validation_samples": len(x_val),
        "test_samples":     len(x_test),
        "num_classes":      len(label_classes),
        "class_labels":     label_classes,
        "test_metrics":     {k: v for k, v in next(
            r for r in test_results if r["model"] == selected_name
        ).items() if k != "model"},
        "leakage_status":   "CLEAN — responsibility-block key phrases stripped",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # ── Misclassified resumes ──────────────────────────────────────────────────
    errors_df = pd.DataFrame({
        "true_role":      y_test.values,
        "predicted_role": selected_preds,
        "resume_text":    x_test.values,
    })
    errors_df = errors_df[errors_df["true_role"] != errors_df["predicted_role"]].copy()
    errors_df["resume_text"] = errors_df["resume_text"].str.slice(0, 500)
    errors_df.to_csv(ERRORS_PATH, index=False, encoding="utf-8")

    n_errors = len(errors_df)
    n_total  = len(x_test)
    print(f"\nTest errors: {n_errors} / {n_total}  "
          f"({n_errors / n_total * 100:.1f}% error rate)")
    if n_errors:
        print("\nFirst 5 misclassified:")
        print(errors_df.head(5)[["true_role", "predicted_role"]].to_string(index=False))
    else:
        print("WARNING: Still zero errors — investigate further!")

    print(f"\nSaved artifacts:")
    print(f"  {MODEL_PATH}")
    print(f"  {VECTORIZER_PATH}")
    print(f"  {LABELS_PATH}")
    print(f"  {METADATA_PATH}")
    print(f"  {ERRORS_PATH} ({n_errors} rows)")


if __name__ == "__main__":
    main()
