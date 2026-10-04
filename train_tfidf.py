"""train_tfidf.py
=================
TF-IDF baseline training for RoleSignal.

Benchmarks MultinomialNB, LogisticRegression, and LinearSVC on the same
stratified splits (loaded from splits/). The test set is touched only once,
at the end, for the winning model.

Usage:
    python split_dataset.py   # run first to create splits/
    python train_tfidf.py

Outputs:
    models/tfidf_model.pkl
    models/tfidf_vectorizer.pkl
    models/tfidf_label_classes.pkl
    results/tfidf_results.json
    confusion_matrix.png
    class_performance.png
    misclassified_resumes.csv
    metadata.json
"""

import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

BASE          = Path(__file__).resolve().parent
DATA_PATH     = BASE / "resume_data_clean.csv"
SPLITS_DIR    = BASE / "splits"
MODELS_DIR    = BASE / "models"
RESULTS_DIR   = BASE / "results"
ERRORS_PATH   = BASE / "misclassified_resumes.csv"
METADATA_PATH = BASE / "metadata.json"

RANDOM_STATE  = 42
TFIDF_CONFIG  = dict(
    ngram_range=(1, 2),
    max_features=10_000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataset and pre-made splits."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    if "resume_id" not in df.columns:
        df.insert(0, "resume_id", range(len(df)))

    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv")["resume_id"].tolist()
    val_ids   = pd.read_csv(SPLITS_DIR / "val_ids.csv"  )["resume_id"].tolist()
    test_ids  = pd.read_csv(SPLITS_DIR / "test_ids.csv" )["resume_id"].tolist()

    df_train = df[df["resume_id"].isin(train_ids)].copy()
    df_val   = df[df["resume_id"].isin(val_ids  )].copy()
    df_test  = df[df["resume_id"].isin(test_ids )].copy()

    print(f"Loaded splits: train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    return df_train, df_val, df_test, df


def compute_metrics(y_true, y_pred) -> dict:
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "macro_precision":    float(precision_score(y_true, y_pred, average="macro",    zero_division=0)),
        "macro_recall":       float(recall_score(   y_true, y_pred, average="macro",    zero_division=0)),
        "macro_f1":           float(f1_score(       y_true, y_pred, average="macro",    zero_division=0)),
        "weighted_f1":        float(f1_score(       y_true, y_pred, average="weighted", zero_division=0)),
        "minority_class_f1":  float(per_class_f1.min()),
    }


def plot_confusion_matrix(y_true, y_pred, labels: list[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(18, 16))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=90)
    ax.set_title("Confusion Matrix — TF-IDF Best Model", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(BASE / "confusion_matrix.png", dpi=120)
    plt.close()
    print("  Saved confusion_matrix.png")


def plot_class_performance(y_true, y_pred, labels: list[str]) -> None:
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rows = [(lbl, report[lbl]["precision"], report[lbl]["recall"], report[lbl]["f1-score"])
            for lbl in labels if lbl in report]
    df_p = pd.DataFrame(rows, columns=["role", "precision", "recall", "f1"])
    df_p = df_p.sort_values("f1", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 11))
    y_pos = np.arange(len(df_p))
    bar_w = 0.27
    ax.barh(y_pos - bar_w, df_p["precision"], bar_w, label="Precision", color="#4C9BE8")
    ax.barh(y_pos,         df_p["recall"],    bar_w, label="Recall",    color="#E8924C")
    ax.barh(y_pos + bar_w, df_p["f1"],        bar_w, label="F1",        color="#4CE88A")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_p["role"], fontsize=8)
    ax.set_xlabel("Score")
    ax.set_title("Per-Class Performance — TF-IDF Best Model", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(BASE / "class_performance.png", dpi=120)
    plt.close()
    print("  Saved class_performance.png")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df_train, df_val, df_test, df_all = load_splits()

    x_train = df_train["Clean_Resume"].astype(str)
    y_train = df_train["job_role"].astype(str).str.strip()
    x_val   = df_val  ["Clean_Resume"].astype(str)
    y_val   = df_val  ["job_role"].astype(str).str.strip()
    x_test  = df_test ["Clean_Resume"].astype(str)
    y_test  = df_test ["job_role"].astype(str).str.strip()

    label_classes = sorted(y_train.unique())

    # ── TF-IDF fitted on TRAIN only ──────────────────────────────────────────
    print("\nFitting TF-IDF vectorizer on train set only...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = vectorizer.fit_transform(x_train)
    X_val   = vectorizer.transform(x_val)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_):,}")

    # ── Model candidates ─────────────────────────────────────────────────────
    def make_nb():    return MultinomialNB()
    def make_lr():    return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
    def make_svc():
        base = LinearSVC(max_iter=5000, random_state=RANDOM_STATE)
        return CalibratedClassifierCV(base, cv=3)

    factories = {
        "MultinomialNB":       make_nb,
        "LogisticRegression":  make_lr,
        "LinearSVC+Calibrated": make_svc,
    }

    # ── Validation phase ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION PHASE (vectorizer fit on train only)")
    print("=" * 60)

    val_results = {}
    val_models  = {}
    for name, factory in factories.items():
        model = factory()
        t0 = perf_counter()
        model.fit(X_train, y_train)
        elapsed = perf_counter() - t0
        preds = model.predict(X_val)
        metrics = compute_metrics(y_val, preds)
        metrics["train_sec"] = elapsed
        val_results[name] = metrics
        val_models[name]  = model
        print(f"  {name:25s}  acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}  "
              f"minF1={metrics['minority_class_f1']:.4f}  t={elapsed:.1f}s")

    # Select best by macro_f1 on val
    best_name = max(val_results, key=lambda n: (val_results[n]["macro_f1"],
                                                 val_results[n]["minority_class_f1"],
                                                 val_results[n]["weighted_f1"]))
    print(f"\nSelected model: {best_name}")

    # ── Final vectorizer fitted on train only, test evaluation ────────────────
    print("\n" + "=" * 60)
    print("TEST PHASE (test set used ONCE for final best model)")
    print("=" * 60)

    X_test  = vectorizer.transform(x_test)
    best_model = val_models[best_name]
    y_pred     = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)
    print(f"\n  {best_name}:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")

    print("\nFull classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Save artifacts ────────────────────────────────────────────────────────
    joblib.dump(best_model,   MODELS_DIR / "tfidf_model.pkl")
    joblib.dump(vectorizer,   MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(label_classes, MODELS_DIR / "tfidf_label_classes.pkl")
    print(f"\nSaved model artifacts to {MODELS_DIR}/")

    # Plots
    print("Generating plots...")
    plot_confusion_matrix(y_test, y_pred, label_classes)
    plot_class_performance(y_test, y_pred, label_classes)

    # Misclassified
    errors = pd.DataFrame({
        "resume_id":      df_test["resume_id"].values,
        "true_role":      y_test.values,
        "predicted_role": y_pred,
        "resume_text":    x_test.str.slice(0, 500).values,
    })
    errors = errors[errors["true_role"] != errors["predicted_role"]]
    errors.to_csv(ERRORS_PATH, index=False, encoding="utf-8")
    print(f"Misclassified: {len(errors)} / {len(y_test)}  ({len(errors)/len(y_test)*100:.1f}% error rate)")
    print(f"Saved to {ERRORS_PATH.name}")

    # Results JSON
    results = {
        "model_type":    "TF-IDF baseline",
        "selected_model": best_name,
        "tfidf_config":  {**TFIDF_CONFIG, "ngram_range": list(TFIDF_CONFIG["ngram_range"])},
        "validation":    {name: {k: round(v, 6) for k, v in m.items()} for name, m in val_results.items()},
        "test_metrics":  {k: round(v, 6) for k, v in test_metrics.items()},
        "train_samples": len(x_train),
        "val_samples":   len(x_val),
        "test_samples":  len(x_test),
        "num_classes":   len(label_classes),
        "leakage_status": "CLEAN — responsibility-block stripped",
        "random_state":  RANDOM_STATE,
    }
    (RESULTS_DIR / "tfidf_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to results/tfidf_results.json")

    # Update metadata.json for API
    metadata = {
        "model_name":       best_name,
        "model_type":       "TF-IDF + LinearSVC (Calibrated)",
        "dataset":          "resume_data_clean.csv",
        "dataset_note":     "Synthetic labels; responsibility-block leakage removed",
        "random_state":     RANDOM_STATE,
        "tfidf_config":     {**TFIDF_CONFIG, "ngram_range": list(TFIDF_CONFIG["ngram_range"])},
        "train_samples":    len(x_train),
        "val_samples":      len(x_val),
        "test_samples":     len(x_test),
        "num_classes":      len(label_classes),
        "class_labels":     label_classes,
        "test_metrics":     {k: round(v, 6) for k, v in test_metrics.items()},
        "active_model":     "tfidf",
        "leakage_status":   "CLEAN",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nDone. Best model: {best_name}  |  Test accuracy: {test_metrics['accuracy']:.4f}  |  Macro F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
