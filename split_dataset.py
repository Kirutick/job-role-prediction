"""split_dataset.py
==================
Creates a fixed 70/15/15 stratified train/val/test split from resume_data_clean.csv.
Saves resume_id lists to splits/ directory so every experiment uses identical partitions.

The test set is isolated and should NOT be touched until final evaluation.

Usage:
    python split_dataset.py
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE          = Path(__file__).resolve().parent
DATA_PATH     = BASE / "resume_data_clean.csv"
SPLITS_DIR    = BASE / "splits"

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_STATE = 42

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9


def main() -> None:
    SPLITS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"Loaded {len(df)} rows from {DATA_PATH.name}")

    # Add resume_id if not present
    if "resume_id" not in df.columns:
        df.insert(0, "resume_id", range(len(df)))

    x     = df["resume_id"]
    y     = df["job_role"].astype(str).str.strip()
    roles = y.unique()

    print(f"Classes: {len(roles)}")
    print(f"Class distribution: min={y.value_counts().min()} max={y.value_counts().max()}")

    # Split: train+val vs test
    val_test_ratio = VAL_RATIO + TEST_RATIO
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=val_test_ratio, random_state=RANDOM_STATE, stratify=y
    )

    # Split: train vs val (val is VAL_RATIO of total = VAL_RATIO/(VAL_RATIO+TEST_RATIO) of trainval)
    val_of_trainval = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval,
        test_size=val_of_trainval, random_state=RANDOM_STATE, stratify=y_trainval
    )

    train_df = pd.DataFrame({"resume_id": x_train.values, "job_role": y_train.values})
    val_df   = pd.DataFrame({"resume_id": x_val.values,   "job_role": y_val.values})
    test_df  = pd.DataFrame({"resume_id": x_test.values,  "job_role": y_test.values})

    train_df.to_csv(SPLITS_DIR / "train_ids.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val_ids.csv",     index=False)
    test_df.to_csv(SPLITS_DIR / "test_ids.csv",   index=False)

    # Verification
    train_set = set(x_train)
    val_set   = set(x_val)
    test_set  = set(x_test)

    print(f"\nSplit sizes:")
    print(f"  Train:      {len(train_df):5d}  ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):5d}  ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):5d}  ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Total:      {len(train_df)+len(val_df)+len(test_df):5d}")

    print(f"\nOverlap verification:")
    print(f"  Train intersect Test:  {len(train_set & test_set):4d}  (must be 0)")
    print(f"  Val intersect Test:    {len(val_set & test_set):4d}  (must be 0)")
    print(f"  Train intersect Val:   {len(train_set & val_set):4d}  (must be 0)")

    assert len(train_set & test_set) == 0,  "Train/test overlap detected!"
    assert len(val_set   & test_set) == 0,  "Val/test overlap detected!"
    assert len(train_set & val_set)  == 0,  "Train/val overlap detected!"
    print("  All overlap checks passed.")

    # Per-class verification
    print("\nPer-class distribution check (min/max across train/val/test):")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        vc = split_df["job_role"].value_counts()
        print(f"  {split_name}: min={vc.min()} max={vc.max()} classes={split_df['job_role'].nunique()}")

    print(f"\nSplit files saved to {SPLITS_DIR}/")


if __name__ == "__main__":
    main()
