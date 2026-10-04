# prep_dataset.py
"""
Dataset preparation and job‑role labeling for the AI‑powered resume screening project.

This script:
1️⃣ Loads `preprocessed_resume_data (1).csv` safely.
2️⃣ Inspects the dataset (columns, shape, missing/duplicate rows, dtypes, sample).
3️⃣ Verifies required columns exist.
4️⃣ Normalises the `responsibilities` text and maps the first line to a concise `job_role` using a robust ROLE_MAP.
5️⃣ Handles missing or unmappable records without silently dropping them.
6️⃣ Validates the resulting labels (counts per role, empty/unknown labels, tiny classes).
7️⃣ Saves the enriched CSV (`resume_data_labeled.csv`) and the role map (`role_map.json`).

Run from the project root:
    python prep_dataset.py
"""

import json
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("preprocessed_resume_data (1).csv")
OUTPUT_CSV = Path("resume_data_labeled.csv")
ROLE_MAP_PATH = Path("role_map.json")

# Human‑readable mapping from the first line of the responsibilities text
# to a concise job‑role label.
ROLE_MAP = {
    "Project Design": "Civil/Environmental Engineer",
    "Relationship Building": "HR Manager",
    "Design Review": "Civil/Structural Engineer",
    "Supervision": "Construction Manager",
    "Technical Support": "IT Support Engineer",
    "Trade Marketing Executive": "Marketing Manager",
    "Apparel Sourcing": "Apparel Sourcing Manager",
    "Machine Learning Design": "Machine Learning Engineer",
    "Machinery Maintenance": "Maintenance Engineer",
    "Design Creation": "Mechanical Design Engineer",
    "Administrative Support": "Admin & Safety Officer",
    "Database Design & Development": "Database Administrator",
    "Hardware & Network Installation": "IT Infrastructure Engineer",
    "Mikrotik Router Configuration": "Network Engineer",
    "Mushak Forms Maintenance": "VAT & Finance Officer",
    "Internal Audit Assistance": "Internal Auditor",
    "Management Trainee": "Management Trainee",
    "15+ Years Banking Experience": "Bank Audit Officer",
    "Recruitment Coordination": "HR & Recruitment Specialist",
    "Digital Marketing Strategy": "Digital Marketing Manager",
    "Machine Learning Leadership": "Senior ML/NLP Engineer",
    "iOS Lifecycle": "iOS Developer",
    "Hardware & Software Installation": "IT Systems Admin",
    "Generative AI Development": "AI & Deep Learning Engineer",
    "Open-Source Technologies": "DevOps Engineer",
    "Data Platform Design": "Data Engineer",
    "Full Stack Development": "Full Stack Developer",
    "Application Development": "Software Engineer",
}

# Normalise keys for case‑insensitive matching
NORMALIZED_ROLE_MAP = {k.strip().lower(): v for k, v in ROLE_MAP.items()}


def load_data(path: Path) -> pd.DataFrame:
    """Load the CSV, ensuring the original file is never altered."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path, encoding="utf-8")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick data‑quality overview.

    Includes column list, shape, missing values, duplicate rows,
    dtypes, and a few sample records.
    """
    print("--- Dataset Inspection ---")
    print("Columns:", list(df.columns))
    print("Shape (rows, cols):", df.shape)
    print("Missing values per column:\n", df.isnull().sum())
    duplicate_count = df.duplicated().sum()
    print("Duplicate rows:", duplicate_count)
    print("Data types:\n", df.dtypes)
    print("Sample records (first 3 rows):\n", df.head(3).to_string(index=False))
    print("--- End Inspection ---\n")


def verify_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def normalize_responsibility(text: str) -> str:
    """Return a lower‑cased, stripped version of the first line.
    Handles stray whitespace and ensures consistent matching.
    """
    first_line = text.split("\n")[0].strip()
    return first_line.lower()


def map_role(responsibility: str) -> str | None:
    """Map the responsibility text to a job role.

    Returns `None` for missing values and a string like
    "Unknown: <snippet>" for unmapped responsibilities.
    """
    if not isinstance(responsibility, str) or not responsibility.strip():
        return None
    normalized = normalize_responsibility(responsibility)
    role = NORMALIZED_ROLE_MAP.get(normalized)
    if role:
        return role
    # Keep a helpful placeholder for analysis
    snippet = responsibility.split("\n")[0].strip()[:40]
    return f"Unknown: {snippet}"


def validate_labels(df: pd.DataFrame, role_col: str = "job_role") -> None:
    print("--- Label Validation ---")
    total = len(df)
    unmapped = df[df[role_col].str.startswith("Unknown", na=True)]
    unknown_count = len(unmapped)
    print(f"Total records: {total}")
    print(f"Successfully labelled: {total - unknown_count}")
    print(f"Unlabelled (unknown) records: {unknown_count}")
    role_counts = df[role_col].value_counts()
    print(f"Number of unique roles: {role_counts.shape[0]}")
    print("Records per role:\n", role_counts.to_string())
    # Flag suspiciously small classes (threshold = 5)
    tiny = role_counts[role_counts < 5]
    if not tiny.empty:
        print("\nWarning: Roles with <5 records (may be too small for training):")
        print(tiny.to_string())
    print("--- End Validation ---\n")


def save_outputs(df: pd.DataFrame) -> None:
    # Keep only the essential columns for downstream modelling
    cols_to_save = ["Resume_Text", "Clean_Resume", "skills", "job_role"]
    df[cols_to_save].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved enriched CSV to {OUTPUT_CSV}")

    # Persist the (original) role map for reference
    with ROLE_MAP_PATH.open("w", encoding="utf-8") as f:
        json.dump(ROLE_MAP, f, indent=2, ensure_ascii=False)
    print(f"Saved role map to {ROLE_MAP_PATH}")


def main() -> None:
    # Step 1: Load data
    df = load_data(DATA_PATH)

    # Step 2: Inspect data (informational)
    inspect_data(df)

    # Step 3: Verify required columns exist
    required_columns = ["Resume_Text", "Clean_Resume", "skills", "responsibilities"]
    verify_required_columns(df, required_columns)

    # Step 4: Map responsibilities to job_role
    df["job_role"] = df["responsibilities"].apply(map_role)

    # Step 5: Validation & summary
    validate_labels(df)

    # Step 6: Save results
    save_outputs(df)


if __name__ == "__main__":
    main()
