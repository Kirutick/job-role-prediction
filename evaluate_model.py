"""Evaluate the saved resume job-role classifier on its held-out test set."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocess_resume import clean_text


DATA_PATH = Path("resume_data_labeled.csv")
MODEL_PATH = Path("job_role_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")
RANDOM_STATE = 42
TEST_SIZE = 0.20
CONFUSION_MATRIX_PATH = Path("confusion_matrix.png")
CLASS_PERFORMANCE_PATH = Path("class_performance.png")
ERRORS_PATH = Path("misclassified_resumes.csv")


def load_test_data() -> tuple[pd.Series, pd.Series]:
    data = pd.read_csv(DATA_PATH, encoding="utf-8")
    required = {"Clean_Resume", "job_role"}
    missing = required.difference(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = data[["Clean_Resume", "job_role"]].dropna()
    data["Clean_Resume"] = data["Clean_Resume"].astype(str).map(clean_text)
    data["job_role"] = data["job_role"].astype(str).str.strip()
    data = data[(data["Clean_Resume"] != "") & (data["job_role"] != "")]
    _, test_data = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["job_role"],
    )
    return test_data["Clean_Resume"], test_data["job_role"]


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    size = max(12, len(labels) // 2)
    plt.figure(figsize=(size, size))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Resume Job-Role Classification Confusion Matrix")
    plt.xlabel("Predicted role")
    plt.ylabel("Actual role")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=160)
    plt.close()


def save_class_performance(report: dict, labels: list[str]) -> None:
    class_rows = pd.DataFrame(report).T.loc[labels]
    class_rows = class_rows.sort_values("f1-score")
    plt.figure(figsize=(11, max(7, len(labels) * 0.28)))
    plt.barh(class_rows.index, class_rows["f1-score"], color="#1976a3")
    plt.xlim(0, 1.05)
    plt.xlabel("F1-score")
    plt.title("Class-wise F1-score on Held-out Test Set")
    plt.tight_layout()
    plt.savefig(CLASS_PERFORMANCE_PATH, dpi=160)
    plt.close()


def main() -> None:
    x_test, y_test = load_test_data()
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # transform() only: neither the model nor vectorizer sees test labels during fitting.
    x_test_tfidf = vectorizer.transform(x_test)
    y_pred = model.predict(x_test_tfidf)
    labels = sorted(y_test.unique())

    summary = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print("Held-out test metrics:")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")

    report = classification_report(
        y_test, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0
    )
    print("\nComplete classification report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=labels, zero_division=0))

    save_confusion_matrix(y_test, y_pred, labels)
    save_class_performance(report, labels)

    errors = pd.DataFrame(
        {
            "actual_job_role": y_test.to_numpy(),
            "predicted_job_role": y_pred,
            "resume_text": x_test.to_numpy(),
        }
    )
    errors = errors[errors["actual_job_role"] != errors["predicted_job_role"]]
    errors["resume_text"] = errors["resume_text"].str.slice(0, 500)
    errors.to_csv(ERRORS_PATH, index=False, encoding="utf-8")

    print(f"\nSaved confusion matrix: {CONFUSION_MATRIX_PATH}")
    print(f"Saved class performance chart: {CLASS_PERFORMANCE_PATH}")
    print(f"Saved error analysis data: {ERRORS_PATH}")
    print(f"Incorrect predictions: {len(errors)} of {len(y_test)}")
    if errors.empty:
        print("No incorrectly classified test resumes were found; role-level error analysis is not possible for this split.")
    else:
        print("\nIncorrect predictions (text truncated to 500 characters):")
        print(errors.to_string(index=False))


if __name__ == "__main__":
    main()