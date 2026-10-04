# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
download_and_audit_genuine_dataset.py
======================================
PHASE 1 — Download UpdatedResumeDataSet.csv and run full leakage audit.
Run: python download_and_audit_genuine_dataset.py
"""
import re
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATASET_URL = "https://raw.githubusercontent.com/2003vivek/TalentTorch/main/UpdatedResumeDataSet.csv"
OUTPUT_PATH = BASE / "data" / "genuine" / "UpdatedResumeDataSet.csv"


def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
section("STEP 1 — DOWNLOADING DATASET")

print(f"URL: {DATASET_URL}")
req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    raw_bytes = r.read()

raw = raw_bytes.decode("utf-8", errors="replace")
print(f"Downloaded: {len(raw_bytes):,} bytes")

df = pd.read_csv(StringIO(raw))
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 2 — BASIC STATISTICS")

category_col = "Category"
resume_col   = "Resume"

print(f"\nTotal rows:     {len(df)}")
print(f"Unique categories: {df[category_col].nunique()}")
print(f"\nCategory distribution:")
vc = df[category_col].value_counts()
for cat, cnt in vc.items():
    print(f"  {cnt:4d}  {cat}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 3 — SAMPLE INSPECTION (Manual)")

for i in [0, 1, 2]:
    row = df.iloc[i]
    print(f"\n--- Row {i}: Category={row[category_col]} ---")
    print(f"Resume (first 500 chars):\n{str(row[resume_col])[:500]}")

# Also sample one from each of 3 different categories
for cat in ["HR", "Java Developer", "Civil Engineer"]:
    subset = df[df[category_col] == cat]
    if not subset.empty:
        row = subset.iloc[0]
        print(f"\n--- Sample: Category={cat} ---")
        snippet = str(row[resume_col])[:400].encode('ascii', errors='replace').decode('ascii')
        print(f"Resume (first 400 chars):\n{snippet}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 4 — LEAKAGE CHECK: Does category name appear in resume text?")

# Normalize for comparison
df["resume_lower"] = df[resume_col].str.lower().fillna("")
df["category_lower"] = df[category_col].str.lower().fillna("")

# Check 1: Full category string in resume
df["exact_cat_in_resume"] = df.apply(
    lambda r: r["category_lower"] in r["resume_lower"], axis=1
)
exact_hits = df["exact_cat_in_resume"].sum()
print(f"\nCategory string found verbatim in resume: {exact_hits}/{len(df)} ({exact_hits/len(df)*100:.1f}%)")

if exact_hits > 0:
    print("Examples of verbatim hits:")
    for _, row in df[df["exact_cat_in_resume"]].head(3).iterrows():
        print(f"  Cat='{row[category_col]}'  Resume snippet: {row[resume_col][:200]}")

# Check 2: All words of category (len > 3) in resume
def all_words_present(cat, text):
    words = [w for w in cat.split() if len(w) > 3]
    if not words:
        return False
    return all(w in text for w in words)

df["cat_words_in_resume"] = df.apply(
    lambda r: all_words_present(r["category_lower"], r["resume_lower"]), axis=1
)
word_hits = df["cat_words_in_resume"].sum()
print(f"\nAll category words (len>3) found in resume: {word_hits}/{len(df)} ({word_hits/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 5 — LEAKAGE CHECK: Synthetic responsibility blocks?")

# Look for repeated identical blocks across resumes
# Check if any resume text is mostly a template
def check_template_markers(text):
    """Look for signs of synthetic/template generation."""
    markers = [
        r"target role\s*:",
        r"job role\s*:",
        r"responsibility block",
        r"machine learning design",
        r"data platform design",
    ]
    t = str(text).lower()
    return any(re.search(m, t) for m in markers)

df["has_template_marker"] = df[resume_col].apply(check_template_markers)
template_hits = df["has_template_marker"].sum()
print(f"\nResumes with synthetic template markers: {template_hits}/{len(df)}")

# Check for unusually repeated blocks (same paragraph in many resumes)
# Sample: check if a 50-char substring appears in >50% of resumes for a given class
print("\nChecking for repeated boilerplate blocks per category (top suspicious):")
suspicious_found = False
for cat in df[category_col].unique():
    subset = df[df[category_col] == cat][resume_col].tolist()
    if len(subset) < 5:
        continue
    # Take 50-char windows from first resume and count occurrences
    anchor = str(subset[0])
    windows = [anchor[i:i+60] for i in range(0, min(len(anchor), 300), 60) if len(anchor[i:i+60]) == 60]
    for window in windows[:3]:
        count = sum(1 for r in subset if window.lower() in str(r).lower())
        pct = count / len(subset) * 100
        if pct > 70:
            print(f"  [{cat}] Block appears in {pct:.0f}% of resumes: '{window[:50]}'")
            suspicious_found = True
if not suspicious_found:
    print("  No suspicious repeated boilerplate blocks detected.")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 6 — EXACT DUPLICATES")

dup_resume = df[resume_col].duplicated(keep=False).sum()
dup_full   = df.duplicated(keep=False).sum()
print(f"\nExact duplicate resumes: {dup_resume}/{len(df)} ({dup_resume/len(df)*100:.1f}%)")
print(f"Exact duplicate full rows: {dup_full}/{len(df)} ({dup_full/len(df)*100:.1f}%)")

# Cross-label conflicts
conflicts = df.groupby(resume_col)[category_col].nunique()
conflict_count = (conflicts > 1).sum()
print(f"\nResumes assigned to >1 category: {conflict_count}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 7 — RESUME TEXT QUALITY CHECK")

df["resume_len"] = df[resume_col].str.len()
print(f"\nResume length stats:")
print(f"  Min:    {df['resume_len'].min()}")
print(f"  Max:    {df['resume_len'].max()}")
print(f"  Mean:   {df['resume_len'].mean():.0f}")
print(f"  Median: {df['resume_len'].median():.0f}")

very_short = (df["resume_len"] < 100).sum()
print(f"\nResumes shorter than 100 chars: {very_short}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 8 — SAVE DATASET IF CLEAN")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Drop helper columns before saving
save_df = df[[category_col, resume_col]].copy()
save_df.columns = ["job_role", "resume_text"]

# Remove duplicates
save_df = save_df.drop_duplicates(subset="resume_text").reset_index(drop=True)
save_df["resume_id"] = save_df.index

print(f"\nFinal cleaned shape: {save_df.shape}")
save_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"Saved to: {OUTPUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
section("STEP 9 — AUDIT VERDICT")

leakage_score = 0
issues = []

if exact_hits > len(df) * 0.3:
    leakage_score += 3
    issues.append(f"HIGH LEAKAGE: {exact_hits} rows have category verbatim in resume text")
elif exact_hits > len(df) * 0.05:
    leakage_score += 1
    issues.append(f"MEDIUM: {exact_hits} rows have category in resume text (possible noise)")
else:
    issues.append(f"OK: Only {exact_hits} rows have category verbatim in resume text")

if template_hits > 0:
    leakage_score += 2
    issues.append(f"WARNING: {template_hits} resumes have synthetic template markers")
else:
    issues.append("OK: No synthetic template markers detected")

if dup_resume > len(df) * 0.1:
    leakage_score += 1
    issues.append(f"WARNING: {dup_resume} exact duplicate resumes ({dup_resume/len(df)*100:.1f}%)")
else:
    issues.append(f"OK: {dup_resume} exact duplicates")

if conflict_count > 0:
    leakage_score += 2
    issues.append(f"WARNING: {conflict_count} resumes assigned to multiple categories")
else:
    issues.append("OK: No cross-label conflicts")

print()
for issue in issues:
    flag = "  [!!]" if "HIGH" in issue or "WARNING" in issue else "  [ OK]"
    print(f"{flag} {issue}")

print()
if leakage_score == 0:
    print("VERDICT: DATASET PASSES INITIAL LEAKAGE AUDIT.")
    print("         Proceed to PHASE 2 (detailed audit) and PHASE 3 (splitting).")
elif leakage_score <= 2:
    print("VERDICT: MINOR ISSUES DETECTED. Dataset may be usable with caution.")
    print("         Review the flagged items above before proceeding.")
else:
    print("[LEAKAGE WARNING] DATASET HAS SIGNIFICANT LEAKAGE.")
    print("         Do NOT use as training data without resolving issues.")
