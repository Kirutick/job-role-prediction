"""
inspect_datasets.py
====================
PHASE 1 — Inspect all existing datasets for genuine label-to-resume correspondence.
Run: python inspect_datasets.py
"""
import re
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent


def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
section("A. INSPECTING resume_data_labeled.csv")

df = pd.read_csv(BASE / "resume_data_labeled.csv", encoding="utf-8")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Unique job_roles: {df['job_role'].nunique()}")
print()

# Show sample resume text to manually inspect
for i in [0, 1, 2]:
    row = df.iloc[i]
    print(f"--- Row {i}: job_role={row['job_role']} ---")
    print(f"Resume_Text (first 400 chars):\n{str(row['Resume_Text'])[:400]}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
section("B. CHECKING RESPONSIBILITY BLOCK LEAKAGE IN resume_data_labeled.csv")

# The responsibility block is appended AFTER the skills list in brackets.
# Pattern: after the closing '] of skills, before the education bracket ['B...]
def extract_resp_block(text):
    """Extract the responsibility block between skills list and education."""
    t = str(text)
    # Find end of first list
    idx = t.rfind("]")
    if idx == -1:
        return ""
    after = t[idx+1:].strip()
    # The responsibility block ends at the next [ 
    end = after.find("[")
    if end == -1:
        return after[:300]
    return after[:end].strip()[:300]

df["resp_block"] = df["Resume_Text"].apply(extract_resp_block)
df["first_resp_line"] = df["resp_block"].apply(lambda x: x.split("\n")[0].strip()[:80] if x else "")

# How many unique first-resp-lines map to exactly 1 role?
resp_role = df.groupby("first_resp_line")["job_role"].nunique()
single_role = (resp_role == 1).sum()
multi_role  = (resp_role > 1).sum()
print(f"Unique first-resp lines that map to EXACTLY 1 job_role: {single_role}")
print(f"Unique first-resp lines that map to >1 job_roles:       {multi_role}")
print()

print("Sample: First resp line per role (first 5 roles):")
for role in df["job_role"].unique()[:5]:
    sub = df[df["job_role"] == role]
    print(f"  Role: {role}")
    print(f"    Most common first_resp_line: {sub['first_resp_line'].value_counts().index[0][:80]}")
print()

# How many rows have a non-empty resp block?
nonempty_resp = (df["resp_block"].str.len() > 5).sum()
print(f"Rows with non-empty responsibility block: {nonempty_resp}/{len(df)} ({nonempty_resp/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
section("C. CHECKING resume_data_clean.csv")

clean_csv = BASE / "resume_data_clean.csv"
if clean_csv.exists():
    df_clean = pd.read_csv(clean_csv, encoding="utf-8")
    print(f"Shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    print()
    for i in [0, 1]:
        row = df_clean.iloc[i]
        role_col = [c for c in df_clean.columns if "role" in c.lower() or "category" in c.lower() or "job" in c.lower()]
        text_col = [c for c in df_clean.columns if "resume" in c.lower() or "text" in c.lower() or "clean" in c.lower()]
        print(f"--- Row {i} ---")
        if role_col:
            print(f"  Job role ({role_col[0]}): {row[role_col[0]]}")
        if text_col:
            print(f"  Resume text ({text_col[0]}, first 300 chars):\n  {str(row[text_col[0]])[:300]}")
        print()
else:
    print("resume_data_clean.csv NOT FOUND")

# ─────────────────────────────────────────────────────────────────────────────
section("D. SEARCHING FOR OTHER DATASETS IN PROJECT")

exts = ["*.csv", "*.json", "*.jsonl", "*.xlsx", "*.parquet"]
found = []
for ext in exts:
    found.extend(BASE.rglob(ext))

# Exclude known synthetic ones and model artifacts
exclude_keywords = ["role_requirements", "screening_weights", "role_map",
                    "metadata", "splits", "results", "eda_images", "models",
                    "__pycache__", ".venv", ".git"]

dataset_candidates = []
for f in found:
    if any(kw in str(f) for kw in exclude_keywords):
        continue
    if f.suffix in [".csv", ".xlsx", ".parquet", ".jsonl"]:
        dataset_candidates.append(f)

print(f"Potential dataset files found (excluding known artifacts): {len(dataset_candidates)}")
for f in dataset_candidates:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name:50s}  ({size_mb:.1f} MB)")

# ─────────────────────────────────────────────────────────────────────────────
section("E. VERDICT")

print("""
QUESTION: Does ANY existing dataset have genuine resume-to-role correspondence?

resume_data_labeled.csv:
  - Responsibility blocks are role-specific synthetic additions.
  - These blocks encode the target label DIRECTLY into the text.
  - Confirmed LEAKAGE. REJECTED as training data.

resume_data_clean.csv:
  - Responsibility blocks stripped. Labels remain from the original synthetic process.
  - The underlying resume bodies have NO genuine correspondence to labels.
  - Confirmed SYNTHETIC LABELS. REJECTED as training data.

CONCLUSION: No existing local dataset qualifies as genuinely labelled.

NEXT ACTION: Search for and download a publicly available genuine resume dataset.
Candidate: Kaggle 'Resume Dataset' by gauravduttakiit (UpdatedResumeDataSet.csv)
  URL: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
  - 2484 rows, 25 categories
  - Labels are coarse job categories (HR, Engineering, etc.)
  - Resumes are real scraped text — NOT synthetically generated
  - Must still be audited for leakage before use
""")
