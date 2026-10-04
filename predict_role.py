"""Standalone inference for the trained resume job-role classifier."""

import sys
from pathlib import Path
from typing import Any

import joblib

from preprocess_resume import clean_text


BASE_DIR       = Path(__file__).resolve().parent
MODEL_PATH     = BASE_DIR / "job_role_model.pkl"
VECTORIZER_PATH= BASE_DIR / "tfidf_vectorizer.pkl"
LABELS_PATH    = BASE_DIR / "label_classes.pkl"


def _load_artifacts() -> tuple[Any, Any, list[str]]:
    """Load inference artifacts and report an actionable missing-file error."""
    missing = [
        str(path)
        for path in (MODEL_PATH, VECTORIZER_PATH, LABELS_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Required model artifact(s) not found: "
            + ", ".join(missing)
            + ". Run train_models.py first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_classes = joblib.load(LABELS_PATH)
    if not isinstance(label_classes, (list, tuple)):
        raise ValueError(f"Expected {LABELS_PATH} to contain a list of class labels.")
    return model, vectorizer, list(label_classes)


def predict_role(resume_text: str) -> dict[str, object]:
    """Predict a job role from one raw resume.

    Returns a dictionary containing the predicted role, confidence when the
    saved classifier supports ``predict_proba``, and all class probabilities.
    The saved vectorizer is only used with ``transform``; it is never refit.
    """
    if not isinstance(resume_text, str):
        raise TypeError("resume_text must be a string.")
    if not resume_text.strip():
        raise ValueError("resume_text cannot be empty or whitespace-only.")

    cleaned_text = clean_text(resume_text)
    if not isinstance(cleaned_text, str) or not cleaned_text.strip():
        raise ValueError("resume_text contains no usable text after preprocessing.")

    model, vectorizer, label_classes = _load_artifacts()
    try:
        features = vectorizer.transform([cleaned_text])
        predicted_role = str(model.predict(features)[0])
    except ValueError as error:
        raise ValueError(
            "The saved model and vectorizer are incompatible with each other."
        ) from error

    result: dict[str, object] = {
        "predicted_role": predicted_role,
        "confidence": None,
        "probabilities": None,
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        model_classes = getattr(model, "classes_", label_classes)
        result["confidence"] = float(max(probabilities))
        result["probabilities"] = {
            str(role): float(probability)
            for role, probability in zip(model_classes, probabilities)
        }
    return result


def main() -> None:
    resume_text = " ".join(sys.argv[1:]).strip()
    if not resume_text:
        try:
            resume_text = input("Paste resume text: ").strip()
        except EOFError as error:
            print("Error: no resume text was provided.", file=sys.stderr)
            raise SystemExit(1) from error

    try:
        result = predict_role(resume_text)
    except (FileNotFoundError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Predicted job role: {result['predicted_role']}")
    if result["confidence"] is not None:
        print(f"Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()