"""Train and compare classifiers for resume job-role prediction.

The vectorizer is fitted on the training partition only. An inner validation
partition selects the classifier; the untouched test partition is used only
for final comparison and reporting.
"""

import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
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

from preprocess_resume import clean_text


DATA_PATH = Path("resume_data_labeled.csv")
MODEL_PATH = Path("job_role_model.pkl")
LABELS_PATH = Path("label_classes.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")
METADATA_PATH = Path("metadata.json")
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# These settings match the feature-extraction stage and are suitable for
# resumes: unigrams capture skills and bigrams capture phrases such as "data
# science", while document-frequency limits remove noise and boilerplate.
TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 5000,
}


def load_data() -> tuple[pd.Series, pd.Series]:
    data = pd.read_csv(DATA_PATH, encoding="utf-8")
    required = {"Clean_Resume", "job_role"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = data[["Clean_Resume", "job_role"]].dropna()
    data["Clean_Resume"] = data["Clean_Resume"].astype(str).map(clean_text)
    data["job_role"] = data["job_role"].astype(str).str.strip()
    data = data[(data["Clean_Resume"] != "") & (data["job_role"] != "")]
    return data["Clean_Resume"], data["job_role"]


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        **TFIDF_CONFIG,
        strip_accents="unicode",
    )


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "minority_class_f1": per_class_f1.min(),
    }


def main() -> None:
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    # The test set is never passed to fit_transform or used for selection.
    vectorizer = make_vectorizer()
    x_fit_tfidf = vectorizer.fit_transform(x_fit)
    x_validation_tfidf = vectorizer.transform(x_validation)
    x_train_tfidf = vectorizer.transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    role_counts = y_train.value_counts()
    imbalance_ratio = role_counts.max() / role_counts.min()
    use_class_weight = imbalance_ratio >= 2.0
    class_weight = "balanced" if use_class_weight else None
    print(f"Training class imbalance ratio: {imbalance_ratio:.2f}")
    print(f"Class weighting for LR/LinearSVC: {class_weight or 'none'}")

    model_factories = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        ),
        "LinearSVC": lambda: LinearSVC(
            class_weight=class_weight,
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        "Multinomial Naive Bayes": lambda: MultinomialNB(),
    }

    validation_models = {}
    validation_results = []
    for name, factory in model_factories.items():
        model = factory()
        started = perf_counter()
        model.fit(x_fit_tfidf, y_fit)
        elapsed = perf_counter() - started
        validation_models[name] = model
        row = metrics(y_validation, model.predict(x_validation_tfidf))
        row.update({"model": name, "training_seconds": elapsed})
        validation_results.append(row)

    # Macro F1 gives every role equal influence; weighted F1 and minority F1
    # are tie-breakers so a large class cannot hide weak rare-role results.
    validation_results.sort(
        key=lambda row: (
            row["macro_f1"],
            row["minority_class_f1"],
            row["weighted_f1"],
            -row["training_seconds"],
        ),
        reverse=True,
    )
    selected_name = validation_results[0]["model"]
    print("\nValidation results used for model selection:")
    print(pd.DataFrame(validation_results).set_index("model").round(4).to_string())
    print(f"\nSelected model: {selected_name} (highest validation macro F1)")

    comparison = []
    reports = {}
    for name, factory in model_factories.items():
        model = factory()
        started = perf_counter()
        model.fit(x_train_tfidf, y_train)
        elapsed = perf_counter() - started
        y_pred = model.predict(x_test_tfidf)
        row = metrics(y_test, y_pred)
        row.update({"model": name, "training_seconds": elapsed})
        comparison.append(row)
        reports[name] = classification_report(y_test, y_pred, zero_division=0)

    comparison_df = pd.DataFrame(comparison).set_index("model")
    print("\nModel comparison on untouched test set:")
    print(comparison_df.round(4).to_string())
    for name, report in reports.items():
        print(f"\nClassification report: {name}\n{report}")

    # Refit the chosen vectorizer and model on all non-test training rows.
    final_vectorizer = make_vectorizer()
    final_x_train_tfidf = final_vectorizer.fit_transform(x_train)
    final_x_test_tfidf = final_vectorizer.transform(x_test)
    selected_model = model_factories[selected_name]()
    selected_model.fit(final_x_train_tfidf, y_train)
    joblib.dump(selected_model, MODEL_PATH)
    joblib.dump(final_vectorizer, VECTORIZER_PATH)
    joblib.dump(sorted(y.unique()), LABELS_PATH)
    metadata = {
        "model_name": selected_name,
        "random_state": RANDOM_STATE,
        "tfidf_config": {
            **TFIDF_CONFIG,
            "ngram_range": list(TFIDF_CONFIG["ngram_range"]),
        },
        "train_samples": len(x_train),
        "validation_samples": len(x_validation),
        "test_samples": len(x_test),
        "num_classes": len(label_classes := sorted(y.unique())),
        "class_labels": label_classes,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nSaved model files:")
    print(f"  {MODEL_PATH}")
    print("  tfidf_vectorizer.pkl")
    print(f"  {LABELS_PATH}")
    print(f"  {METADATA_PATH}")
    print("Final selection considers validation macro F1 first, minority-class F1, weighted F1, "
          "minority-class behavior, precision/recall balance, and training cost; "
          "accuracy is not used alone.")


if __name__ == "__main__":
    main()