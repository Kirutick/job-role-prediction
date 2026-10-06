"""Download, audit, normalize, and split the ResumeAtlas dataset.

Run from the project root:
    python audit_resume_atlas.py

The raw dataset is loaded from its pinned Hugging Face revision and is not
written into the repository. The normalized CSV and split CSV files are local,
git-ignored training inputs.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_ID = "ahmedheakl/resume-atlas"
DATASET_REVISION = "3f80ca910fa9964890afb7845c09e239d07b9b1d"
OUTPUT_PATH = BASE_DIR / "resume_data_real.csv"
AUDIT_PATH = BASE_DIR / "resume_atlas_audit.json"
SPLITS_DIR = BASE_DIR / "splits"
RANDOM_STATE = 42
SHORT_RESUME_CHARACTERS = 100

TEXT_COLUMNS = ("resume_text", "resumetext", "resume", "text", "content")
LABEL_COLUMNS = ("job_role", "jobrole", "category", "role", "label")


def _find_column(columns: list[str], candidates: tuple[str, ...], purpose: str) -> str:
    matches = {
        column.casefold().replace("_", "").replace(" ", ""): column
        for column in columns
    }
    for candidate in candidates:
        key = candidate.casefold().replace("_", "").replace(" ", "")
        if key in matches:
            return matches[key]
    raise KeyError(
        f"Could not identify the {purpose} column from dataset columns: {columns}"
    )


def normalize_for_deduplication(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _category_suffix_diagnostics(data: pd.DataFrame) -> dict[str, Any]:
    """Flag explicit label blocks, separately from natural label words in text."""
    explicit_block_count = 0
    normalized_label_at_end_count = 0
    label_at_end_examples: list[dict[str, str]] = []

    for row in data.itertuples(index=False):
        text = str(row.resume_text)
        label = str(row.job_role)
        escaped_label = re.escape(label)
        explicit_block = re.search(
            rf"(?:^|\n)\s*(?:category|job\s+category|target\s+role|"
            rf"classification)\s*[:=\-]\s*{escaped_label}\s*$",
            text,
            flags=re.IGNORECASE,
        )
        if explicit_block:
            explicit_block_count += 1

        normalized_text = normalize_for_deduplication(text)
        normalized_label = normalize_for_deduplication(label)
        if re.search(rf"(?:^|\s){re.escape(normalized_label)}$", normalized_text):
            normalized_label_at_end_count += 1
            if len(label_at_end_examples) < 10:
                label_at_end_examples.append(
                    {
                        "resume_id": str(row.resume_id),
                        "job_role": label,
                    }
                )

    return {
        "explicit_appended_category_block_rows": explicit_block_count,
        "rows_whose_normalized_text_ends_with_category_name": normalized_label_at_end_count,
        "category_suffix_examples_for_manual_review": label_at_end_examples,
        "interpretation": (
            "A category name at the end is a flag for review, not proof of an "
            "artificial block. Only an explicit category/role/classification "
            "marker is treated as direct evidence of an appended label."
        ),
    }


def _download_dataset() -> pd.DataFrame:
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
    )
    source = dataset.to_pandas()
    text_column = _find_column(list(source.columns), TEXT_COLUMNS, "resume text")
    label_column = _find_column(list(source.columns), LABEL_COLUMNS, "role label")
    normalized = pd.DataFrame(
        {
            "resume_id": [f"resume-atlas-{index:06d}" for index in range(len(source))],
            "resume_text": source[text_column],
            "job_role": source[label_column],
        }
    )
    normalized.attrs["source_columns"] = {
        "resume_text": text_column,
        "job_role": label_column,
    }
    return normalized


def _duplicate_audit(data: pd.DataFrame) -> dict[str, Any]:
    valid_text = data["resume_text"].notna() & data["resume_text"].astype(str).str.strip().ne("")
    rows = data.loc[valid_text].copy()
    normalized = rows["resume_text"].astype(str).map(normalize_for_deduplication)
    exact_duplicate_mask = rows["resume_text"].astype(str).duplicated(keep="first")
    normalized_duplicate_mask = normalized.duplicated(keep="first")
    normalized_groups = rows.assign(_normalized_text=normalized).groupby(
        "_normalized_text", sort=False
    )
    conflicting_groups = normalized_groups["job_role"].nunique()
    conflicting_keys = set(conflicting_groups[conflicting_groups > 1].index)
    conflicting_rows = rows.assign(_normalized_text=normalized)["_normalized_text"].isin(
        conflicting_keys
    )

    return {
        "exact_duplicate_rows_after_first_occurrence": int(exact_duplicate_mask.sum()),
        "exact_duplicate_groups": int(
            rows.loc[exact_duplicate_mask, "resume_text"].astype(str).nunique()
        ),
        "normalized_duplicate_rows_after_first_occurrence": int(
            normalized_duplicate_mask.sum()
        ),
        "normalized_duplicate_groups": int(
            normalized[normalized_duplicate_mask].nunique()
        ),
        "normalized_duplicate_groups_with_conflicting_labels": len(conflicting_keys),
        "rows_in_conflicting_label_groups": int(conflicting_rows.sum()),
    }


def _make_leak_free_splits(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    role_counts = data["job_role"].value_counts()
    if role_counts.min() < 3:
        raise ValueError(
            "Every role needs at least 3 unique resumes for stratified "
            f"train/validation/test splits; minimum class count is {role_counts.min()}."
        )

    train_validation, test = train_test_split(
        data,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=data["job_role"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=train_validation["job_role"],
    )
    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    normalized_sets = {
        name: set(frame["resume_text"].astype(str).map(normalize_for_deduplication))
        for name, frame in splits.items()
    }
    overlap = {
        "train_test": len(normalized_sets["train"] & normalized_sets["test"]),
        "train_validation": len(
            normalized_sets["train"] & normalized_sets["validation"]
        ),
        "validation_test": len(
            normalized_sets["validation"] & normalized_sets["test"]
        ),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Duplicate resume leakage remains between splits: {overlap}")
    return splits, overlap


def run_audit() -> dict[str, Any]:
    original = _download_dataset()
    original_rows = len(original)
    original["resume_text"] = original["resume_text"].map(
        lambda value: value if pd.isna(value) else str(value)
    )
    original["job_role"] = original["job_role"].map(
        lambda value: value if pd.isna(value) else str(value).strip()
    )

    missing_resume_rows = int(original["resume_text"].isna().sum())
    empty_resume_rows = int(
        (
            original["resume_text"].notna()
            & original["resume_text"].astype(str).str.strip().eq("")
        ).sum()
    )
    missing_label_rows = int(original["job_role"].isna().sum())
    empty_label_rows = int(
        (
            original["job_role"].notna()
            & original["job_role"].astype(str).str.strip().eq("")
        ).sum()
    )
    valid = original[
        original["resume_text"].notna()
        & original["resume_text"].astype(str).str.strip().ne("")
        & original["job_role"].notna()
        & original["job_role"].astype(str).str.strip().ne("")
    ].copy()

    original_class_distribution = {
        str(label): int(count)
        for label, count in original["job_role"].dropna().value_counts().items()
    }
    duplicate_metrics = _duplicate_audit(original)
    valid["_normalized_text"] = valid["resume_text"].map(normalize_for_deduplication)

    conflicting_label_counts = valid.groupby("_normalized_text")["job_role"].nunique()
    conflicting_keys = set(conflicting_label_counts[conflicting_label_counts > 1].index)
    conflicting_group_rows = valid["_normalized_text"].isin(conflicting_keys)
    conflicting_rows_removed = int(conflicting_group_rows.sum())
    unambiguous = valid.loc[~conflicting_group_rows].copy()
    before_dedup = len(unambiguous)
    unambiguous = unambiguous.drop_duplicates("_normalized_text", keep="first").copy()
    same_label_duplicate_rows_removed = before_dedup - len(unambiguous)
    unambiguous.drop(columns="_normalized_text", inplace=True)

    too_short = int(
        unambiguous["resume_text"].astype(str).str.len().lt(SHORT_RESUME_CHARACTERS).sum()
    )
    label_block_audit = _category_suffix_diagnostics(valid)
    if label_block_audit["explicit_appended_category_block_rows"]:
        raise ValueError(
            "Explicit appended category blocks were found; inspect "
            f"{label_block_audit['explicit_appended_category_block_rows']} rows "
            "before training."
        )

    if unambiguous.empty or unambiguous["job_role"].nunique() < 2:
        raise ValueError("The audited dataset has too few usable classes to train.")

    splits, overlap = _make_leak_free_splits(unambiguous)
    class_distribution = {
        str(label): int(count)
        for label, count in unambiguous["job_role"].value_counts().sort_index().items()
    }
    split_class_counts = {
        split_name: {
            str(label): int(count)
            for label, count in split["job_role"].value_counts().sort_index().items()
        }
        for split_name, split in splits.items()
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unambiguous.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split in splits.items():
        split.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False, encoding="utf-8")

    report = {
        "dataset": "ResumeAtlas",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_columns": original.attrs.get("source_columns", {}),
        "total_rows": original_rows,
        "number_of_classes": len(original_class_distribution),
        "original_class_distribution": original_class_distribution,
        "missing_resume_rows": missing_resume_rows,
        "empty_resume_rows": empty_resume_rows,
        "missing_label_rows": missing_label_rows,
        "empty_label_rows": empty_label_rows,
        "exact_duplicate_rows_after_first_occurrence": duplicate_metrics[
            "exact_duplicate_rows_after_first_occurrence"
        ],
        "exact_duplicate_groups": duplicate_metrics["exact_duplicate_groups"],
        "normalized_duplicate_rows_after_first_occurrence": duplicate_metrics[
            "normalized_duplicate_rows_after_first_occurrence"
        ],
        "normalized_duplicate_groups": duplicate_metrics["normalized_duplicate_groups"],
        "normalized_duplicate_groups_with_conflicting_labels": duplicate_metrics[
            "normalized_duplicate_groups_with_conflicting_labels"
        ],
        "rows_in_conflicting_label_groups": duplicate_metrics[
            "rows_in_conflicting_label_groups"
        ],
        "category_block_audit": label_block_audit,
        "extremely_short_threshold_characters": SHORT_RESUME_CHARACTERS,
        "extremely_short_rows_retained": too_short,
        "filtering_decisions": {
            "missing_or_empty_resume_or_label_rows_removed": (
                missing_resume_rows
                + empty_resume_rows
                + missing_label_rows
                + empty_label_rows
            ),
            "all_rows_in_conflicting_normalized_text_groups_removed": conflicting_rows_removed,
            "same_label_normalized_duplicates_removed_keep_first_source_row": (
                same_label_duplicate_rows_removed
            ),
            "extremely_short_resume_rows_removed": 0,
            "resume_text_cleaned_or_rewritten": False,
        },
        "dataset_rows_after_filtering": len(unambiguous),
        "class_distribution_after_filtering": class_distribution,
        "split_counts": {name: len(split) for name, split in splits.items()},
        "split_class_counts": split_class_counts,
        "random_state": RANDOM_STATE,
        "split_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "train_test_duplicate_overlap": overlap["train_test"],
        "train_validation_duplicate_overlap": overlap["train_validation"],
        "validation_test_duplicate_overlap": overlap["validation_test"],
        "leakage_status": "CLEAN" if not any(overlap.values()) else "FAILED",
    }
    AUDIT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Dataset: {DATASET_ID}@{DATASET_REVISION}")
    print(f"Total rows: {original_rows}")
    print(f"Identified columns: {original.attrs['source_columns']}")
    print(f"Classes: {len(original_class_distribution)}")
    print(f"Original class distribution: {original_class_distribution}")
    print(
        "Missing/empty resumes:",
        {
            "missing": missing_resume_rows,
            "empty": empty_resume_rows,
            "missing_labels": missing_label_rows,
            "empty_labels": empty_label_rows,
        },
    )
    print("Duplicate audit:", duplicate_metrics)
    print("Appended-label audit:", label_block_audit)
    print(f"Extremely short (<{SHORT_RESUME_CHARACTERS} chars), retained: {too_short}")
    print("Filtering decisions:", report["filtering_decisions"])
    print(f"Rows after filtering: {len(unambiguous)}")
    print(f"Filtered class distribution: {class_distribution}")
    print("Split counts:", report["split_counts"])
    print("Duplicate overlap:", overlap)
    print(f"Saved normalized data: {OUTPUT_PATH}")
    print(f"Saved audit report: {AUDIT_PATH}")
    print(f"Saved splits: {SPLITS_DIR}")
    return report


if __name__ == "__main__":
    run_audit()
