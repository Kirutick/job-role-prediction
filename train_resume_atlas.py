"""Select and evaluate a TF-IDF role classifier using ResumeAtlas.

Run audit_resume_atlas.py first. All vectorizers and classifiers are fitted on
the training split only. Validation macro-F1 selects the model; the test split
is accessed only for the selected model's final evaluation.
"""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from audit_resume_atlas import (
    AUDIT_PATH,
    DATASET_ID,
    DATASET_REVISION,
    OUTPUT_PATH,
    RANDOM_STATE,
    SPLITS_DIR,
    normalize_for_deduplication,
)


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METADATA_PATH = BASE_DIR / "metadata.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
CLASS_PERFORMANCE_PATH = BASE_DIR / "class_performance.png"
MISCLASSIFIED_PATH = BASE_DIR / "misclassified_resumes.csv"
RESULTS_PATH = BASE_DIR / "resume_atlas_results.json"
MAX_FEATURES_CANDIDATES = (10_000, 20_000, 30_000)


def _load_and_validate_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not AUDIT_PATH.is_file() or not OUTPUT_PATH.is_file():
        raise FileNotFoundError(
            "ResumeAtlas audit outputs are missing. Run python audit_resume_atlas.py first."
        )
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if audit.get("leakage_status") != "CLEAN":
        raise ValueError("ResumeAtlas audit did not pass its leakage checks.")

    split_frames = {
        name: pd.read_csv(SPLITS_DIR / f"{name}.csv", encoding="utf-8")
        for name in ("train", "validation", "test")
    }
    full_data = pd.read_csv(OUTPUT_PATH, encoding="utf-8")
    if set(full_data.columns) != {"resume_id", "resume_text", "job_role"}:
        raise ValueError("resume_data_real.csv does not match the normalized schema.")

    expected_ids = set(full_data["resume_id"])
    actual_ids = set().union(*(set(frame["resume_id"]) for frame in split_frames.values()))
    if expected_ids != actual_ids or sum(map(len, split_frames.values())) != len(full_data):
        raise ValueError("Splits do not partition the complete audited dataset exactly once.")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ids_a = set(split_frames[first]["resume_id"])
        ids_b = set(split_frames[second]["resume_id"])
        if ids_a & ids_b:
            raise ValueError(f"Resume IDs overlap between {first} and {second}.")
        texts_a = set(
            split_frames[first]["resume_text"].astype(str).map(normalize_for_deduplication)
        )
        texts_b = set(
            split_frames[second]["resume_text"].astype(str).map(normalize_for_deduplication)
        )
        if texts_a & texts_b:
            raise ValueError(f"Normalized resume texts overlap between {first} and {second}.")
    return split_frames["train"], split_frames["validation"], split_frames["test"], audit


def _new_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=max_features,
    )


def _metric_values(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def _top3_predictions(model: Any, features: Any, labels: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)
    else:
        scores = model.decision_function(features)
        if scores.ndim == 1:
            scores = np.column_stack((-scores, scores))
    classes = np.asarray(model.classes_)
    top_indices = np.argsort(scores, axis=1)[:, -3:][:, ::-1]
    top_labels = classes[top_indices]
    return np.asarray(
        [actual in candidates for actual, candidates in zip(labels, top_labels)],
        dtype=bool,
    )


def _save_evaluation_outputs(
    y_test: pd.Series,
    y_pred: np.ndarray,
    label_classes: list[str],
    test_frame: pd.DataFrame,
) -> None:
    matrix = confusion_matrix(y_test, y_pred, labels=label_classes)
    figure_width = max(14, len(label_classes) * 0.42)
    figure, axis = plt.subplots(figsize=(figure_width, figure_width))
    sns.heatmap(
        matrix,
        cmap="Blues",
        annot=len(label_classes) <= 25,
        fmt="d",
        xticklabels=label_classes,
        yticklabels=label_classes,
        ax=axis,
    )
    axis.set(title="ResumeAtlas Test Confusion Matrix", xlabel="Predicted role", ylabel="True role")
    axis.tick_params(axis="x", rotation=90)
    axis.tick_params(axis="y", rotation=0)
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=140)
    plt.close(figure)

    report = classification_report(
        y_test,
        y_pred,
        labels=label_classes,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        [
            {
                "role": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in label_classes
        ]
    ).sort_values("f1", ascending=True)
    figure, axis = plt.subplots(figsize=(12, max(10, len(label_classes) * 0.34)))
    per_class.set_index("role")[["precision", "recall", "f1"]].plot.barh(
        ax=axis, width=0.82
    )
    axis.set(title="ResumeAtlas Test Performance by Class", xlabel="Score")
    figure.tight_layout()
    figure.savefig(CLASS_PERFORMANCE_PATH, dpi=140)
    plt.close(figure)

    errors = pd.DataFrame(
        {
            "resume_id": test_frame["resume_id"].to_numpy(),
            "true_role": y_test.to_numpy(),
            "predicted_role": y_pred,
            "resume_text": test_frame["resume_text"].to_numpy(),
        }
    )
    errors = errors.loc[errors["true_role"] != errors["predicted_role"]]
    errors.to_csv(MISCLASSIFIED_PATH, index=False, encoding="utf-8")


def main() -> None:
    train_frame, validation_frame, test_frame, audit = _load_and_validate_splits()
    x_train = train_frame["resume_text"].astype(str)
    y_train = train_frame["job_role"].astype(str)
    x_validation = validation_frame["resume_text"].astype(str)
    y_validation = validation_frame["job_role"].astype(str)
    x_test = test_frame["resume_text"].astype(str)
    y_test = test_frame["job_role"].astype(str)
    label_classes = sorted(y_train.unique())

    candidates: list[dict[str, Any]] = []
    for max_features in MAX_FEATURES_CANDIDATES:
        vectorizer = _new_vectorizer(max_features)
        x_train_features = vectorizer.fit_transform(x_train)
        x_validation_features = vectorizer.transform(x_validation)
        model_factories = (
            (
                "LogisticRegression",
                lambda: LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "LinearSVC",
                lambda: LinearSVC(max_iter=5000, random_state=RANDOM_STATE),
            ),
            (
                "Calibrated LinearSVC",
                lambda: CalibratedClassifierCV(
                    LinearSVC(max_iter=5000, random_state=RANDOM_STATE),
                    cv=3,
                ),
            ),
            (
                "LinearSVC balanced",
                lambda: LinearSVC(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        )
        for model_name, factory in model_factories:
            model = factory()
            started = perf_counter()
            model.fit(x_train_features, y_train)
            elapsed = perf_counter() - started
            prediction = model.predict(x_validation_features)
            metrics = _metric_values(y_validation, prediction)
            candidate = {
                "model": model_name,
                "max_features": max_features,
                "validation": metrics,
                "training_seconds": round(elapsed, 3),
                "vectorizer": vectorizer,
                "classifier": model,
            }
            candidates.append(candidate)
            print(
                f"{model_name:24s} max_features={max_features:5,d} "
                f"validation_accuracy={metrics['accuracy']:.4f} "
                f"validation_macro_f1={metrics['macro_f1']:.4f} "
                f"fit_seconds={elapsed:.1f}"
            )

    candidates.sort(
        key=lambda item: (
            item["validation"]["macro_f1"],
            item["validation"]["accuracy"],
            item["validation"]["weighted_f1"],
        ),
        reverse=True,
    )
    selected = candidates[0]
    model = selected["classifier"]
    vectorizer = selected["vectorizer"]
    print(
        f"\nSelected by validation macro-F1: {selected['model']} "
        f"(max_features={selected['max_features']})"
    )
    print(f"Validation metrics: {selected['validation']}")

    x_test_features = vectorizer.transform(x_test)
    y_pred = model.predict(x_test_features)
    test_metrics = _metric_values(y_test, y_pred)
    top3_correct = _top3_predictions(
        model,
        x_test_features,
        y_test.to_numpy(),
    )
    test_metrics["top3_accuracy"] = float(top3_correct.mean())
    _save_evaluation_outputs(y_test, y_pred, label_classes, test_frame)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "tfidf_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(label_classes, MODELS_DIR / "tfidf_label_classes.pkl")

    metadata = {
        "model_name": selected["model"],
        "model_type": "TF-IDF + " + selected["model"],
        "active_model": "tfidf",
        "dataset": "ResumeAtlas",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_dataset_rows": audit["total_rows"],
        "dataset_rows": audit["dataset_rows_after_filtering"],
        "num_classes": len(label_classes),
        "class_labels": label_classes,
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "test_samples": len(test_frame),
        "validation_accuracy": selected["validation"]["accuracy"],
        "validation_macro_f1": selected["validation"]["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "top3_accuracy": test_metrics["top3_accuracy"],
        "macro_precision": test_metrics["macro_precision"],
        "macro_recall": test_metrics["macro_recall"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "leakage_status": audit["leakage_status"],
        "random_state": RANDOM_STATE,
        "split_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "tfidf_config": {
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "strip_accents": "unicode",
            "max_features": selected["max_features"],
        },
        "model_selection": {
            "criterion": "validation macro-F1",
            "candidates": [
                {
                    "model": item["model"],
                    "max_features": item["max_features"],
                    "validation": item["validation"],
                    "training_seconds": item["training_seconds"],
                }
                for item in candidates
            ],
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results = {
        "dataset": metadata["dataset"],
        "source_dataset_rows": metadata["source_dataset_rows"],
        "dataset_rows": metadata["dataset_rows"],
        "num_classes": metadata["num_classes"],
        "filtering_decisions": audit["filtering_decisions"],
        "split_counts": audit["split_counts"],
        "selected_model": selected["model"],
        "selected_max_features": selected["max_features"],
        "validation": selected["validation"],
        "test": test_metrics,
        "leakage_status": audit["leakage_status"],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nHeld-out test metrics (evaluated once for selected model):")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.6f}")
    print(f"Saved model artifacts in {MODELS_DIR}")
    print(f"Saved {CONFUSION_MATRIX_PATH.name}, {CLASS_PERFORMANCE_PATH.name}")
    print(f"Saved {MISCLASSIFIED_PATH.name} ({len(test_frame) - int((y_test == y_pred).sum())} rows)")
    print(f"Saved {METADATA_PATH.name} and {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
