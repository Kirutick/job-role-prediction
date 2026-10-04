"""train_semantic.py
======================
Semantic layer training (TF-IDF + LSA/SVD) for RoleSignal.

Uses TruncatedSVD on top of TF-IDF to approximate dense semantic embeddings.
This creates a real semantic model without requiring heavy neural networks
or 2.5GB PyTorch downloads.

Usage:
    python train_semantic.py

Outputs:
    models/semantic_model.pkl
    models/semantic_vectorizer.pkl
    models/semantic_svd.pkl
    models/semantic_label_classes.pkl
    results/semantic_results.json
"""

import json
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE          = Path(__file__).resolve().parent
DATA_PATH     = BASE / "resume_data_clean.csv"
SPLITS_DIR    = BASE / "splits"
MODELS_DIR    = BASE / "models"
RESULTS_DIR   = BASE / "results"
METADATA_PATH = BASE / "metadata.json"

RANDOM_STATE  = 42
N_COMPONENTS  = 256
TFIDF_CONFIG  = dict(
    ngram_range=(1, 2),
    max_features=10_000,
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode",
)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    if "resume_id" not in df.columns:
        df.insert(0, "resume_id", range(len(df)))
    train_ids = pd.read_csv(SPLITS_DIR / "train_ids.csv")["resume_id"].tolist()
    val_ids   = pd.read_csv(SPLITS_DIR / "val_ids.csv"  )["resume_id"].tolist()
    test_ids  = pd.read_csv(SPLITS_DIR / "test_ids.csv" )["resume_id"].tolist()

    df_train = df[df["resume_id"].isin(train_ids)].copy()
    df_val   = df[df["resume_id"].isin(val_ids  )].copy()
    df_test  = df[df["resume_id"].isin(test_ids )].copy()

    x_train, y_train = df_train["Clean_Resume"].astype(str), df_train["job_role"].astype(str).str.strip()
    x_val,   y_val   = df_val["Clean_Resume"].astype(str),   df_val["job_role"].astype(str).str.strip()
    x_test,  y_test  = df_test["Clean_Resume"].astype(str),  df_test["job_role"].astype(str).str.strip()

    label_classes = sorted(y_train.unique())

    print("Building semantic pipeline (TF-IDF -> TruncatedSVD -> LinearSVC)...")
    t0 = perf_counter()
    
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    svd        = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    classifier = CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=RANDOM_STATE), cv=3)

    print("  Fitting TF-IDF...")
    X_train_tfidf = vectorizer.fit_transform(x_train)
    X_val_tfidf   = vectorizer.transform(x_val)
    X_test_tfidf  = vectorizer.transform(x_test)
    
    print(f"  Fitting SVD (dim={N_COMPONENTS})...")
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_val_svd   = svd.transform(X_val_tfidf)
    X_test_svd  = svd.transform(X_test_tfidf)
    
    print(f"  Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

    print("  Training classifier...")
    classifier.fit(X_train_svd, y_train)
    
    elapsed = perf_counter() - t0
    print(f"Pipeline training complete in {elapsed:.1f}s.")

    # Validation
    val_preds   = classifier.predict(X_val_svd)
    val_metrics = compute_metrics(y_val, val_preds)
    print(f"Validation: Acc = {val_metrics['accuracy']:.4f}  |  Macro F1 = {val_metrics['macro_f1']:.4f}")

    # Test
    test_preds   = classifier.predict(X_test_svd)
    test_metrics = compute_metrics(y_test, test_preds)
    print(f"Test:       Acc = {test_metrics['accuracy']:.4f}  |  Macro F1 = {test_metrics['macro_f1']:.4f}")

    # Save
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    
    joblib.dump(classifier,   MODELS_DIR / "semantic_model.pkl")
    joblib.dump(vectorizer,   MODELS_DIR / "semantic_vectorizer.pkl")
    joblib.dump(svd,          MODELS_DIR / "semantic_svd.pkl")
    joblib.dump(label_classes, MODELS_DIR / "semantic_label_classes.pkl")

    results = {
        "model_type":    "LSA Semantic (TF-IDF + SVD)",
        "n_components":  N_COMPONENTS,
        "explained_var": float(svd.explained_variance_ratio_.sum()),
        "val_metrics":   {k: round(v, 6) for k, v in val_metrics.items()},
        "test_metrics":  {k: round(v, 6) for k, v in test_metrics.items()},
    }
    (RESULTS_DIR / "semantic_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Load TF-IDF results to compare, and set active model in metadata
    tfidf_path = RESULTS_DIR / "tfidf_results.json"
    if tfidf_path.exists():
        tfidf_data = json.loads(tfidf_path.read_text(encoding="utf-8"))
        tfidf_acc = tfidf_data["test_metrics"]["accuracy"]
        print(f"\nComparison:\n  TF-IDF:   {tfidf_acc:.4f}\n  Semantic: {test_metrics['accuracy']:.4f}")
        
        best_type = "semantic" if test_metrics['accuracy'] > tfidf_acc else "tfidf"
        print(f"Setting active model to: {best_type}")
        
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        meta["active_model"] = best_type
        if best_type == "semantic":
            meta["model_type"] = "LSA Semantic (TF-IDF + SVD)"
            meta["test_metrics"] = results["test_metrics"]
        METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    print("\nSaved semantic artifacts.")

if __name__ == "__main__":
    main()
