"""
phase1_deep_inspection.py
==========================
Deep inspection of UpdatedResumeDataSet.csv after initial audit raised two flags:
1. 84% of resumes contain the category name verbatim
2. 99.6% "exact duplicates"

This script investigates WHAT the duplication really means and WHETHER the
category name appearing is actual leakage or simply a natural occurrence
(e.g., a "Data Scientist" resume naturally mentions "data science" in work history).

Run: python phase1_deep_inspection.py
"""
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import re
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


section("LOADING DATASET")
import urllib.request
from io import StringIO

URL = "https://raw.githubusercontent.com/2003vivek/TalentTorch/main/UpdatedResumeDataSet.csv"
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    raw = r.read().decode("utf-8", errors="replace")

df = pd.read_csv(StringIO(raw))
print(f"Shape: {df.shape}")

# ─────────────────────────────────────────────────────────────────────────────
section("INVESTIGATING THE DUPLICATE PROBLEM")

# The audit said 958/962 are "exact duplicates" — that's almost everything.
# This is very suspicious. Let's investigate.

print("\nFirst 10 rows of Resume column lengths and first 50 chars:")
for i in range(10):
    row = df.iloc[i]
    print(f"  [{i}] Cat={row['Category']:20s}  len={len(str(row['Resume'])):5d}  text='{str(row['Resume'])[:60]}'")

# Check if it's the same text repeated
print("\nAre rows 0 and 1 identical?", df.iloc[0]['Resume'] == df.iloc[1]['Resume'])
print("Are rows 0 and 2 identical?", df.iloc[0]['Resume'] == df.iloc[2]['Resume'])

# Count by category
print("\nUnique resumes per category:")
for cat in df['Category'].unique():
    sub = df[df['Category'] == cat]
    unique_in_cat = sub['Resume'].nunique()
    total_in_cat = len(sub)
    print(f"  {cat:30s}: {unique_in_cat:3d} unique / {total_in_cat:3d} total")

# The 99.6% duplicate number is suspicious — it means almost every resume
# appears more than once. Let's find the actual unique resumes.
print(f"\nTotal rows: {len(df)}")
print(f"Unique Resume texts: {df['Resume'].nunique()}")
print(f"True duplicates (same text, same category): {df.duplicated(subset=['Category','Resume']).sum()}")
print(f"True duplicates (same text only, any cat):  {df.duplicated(subset=['Resume']).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
section("INVESTIGATING THE CATEGORY-IN-RESUME LEAKAGE")

# Is finding "Data Science" in a Data Science resume actually leakage?
# Example: A data scientist's resume WILL mention "data science" in their job title.
# That is NOT leakage — it's natural content.
# Leakage would be if the LABEL was APPENDED to the resume to encode the target.

print("\nManual analysis of category appearance in resumes:")

def check_category_context(resume_text, category):
    """Find WHERE the category appears: beginning, job title, or appended at end."""
    text = str(resume_text)
    cat_lower = category.lower()
    text_lower = text.lower()
    
    if cat_lower not in text_lower:
        return "NOT FOUND"
    
    idx = text_lower.find(cat_lower)
    total_len = len(text)
    position_pct = idx / total_len * 100
    
    # Get context around the match
    start = max(0, idx - 50)
    end = min(len(text), idx + len(category) + 50)
    context = text[start:end].replace('\n', ' ')
    
    return f"pos={idx} ({position_pct:.0f}% through), context: '...{context}...'"

# Sample 3 from each of 5 categories
for cat in ["Data Science", "HR", "Java Developer", "Civil Engineer", "DevOps Engineer"]:
    subset = df[df['Category'] == cat].head(3)
    print(f"\n  Category: {cat}")
    for _, row in subset.iterrows():
        result = check_category_context(row['Resume'], cat)
        print(f"    {result[:120]}")

# ─────────────────────────────────────────────────────────────────────────────
section("CHECKING IF CATEGORY APPEARS AS JOB TITLE (natural) vs APPENDED (leakage)")

# If a Java Developer's resume mentions "Java Developer" because that's their 
# job title, it is NATURAL. If it was APPENDED at the end as a label, it's leakage.

print("\nFor each category, checking last 200 chars of resumes to see if category is appended:")
appended_count = 0
total_with_cat = 0
for _, row in df.iterrows():
    cat = str(row['Category']).lower()
    text = str(row['Resume'])
    text_lower = text.lower()
    if cat in text_lower:
        total_with_cat += 1
        # Check if it's in the last 10% of the text
        last_portion = text_lower[int(len(text_lower) * 0.9):]
        if cat in last_portion:
            appended_count += 1

print(f"Rows where category appears in resume: {total_with_cat}")
print(f"Of those, rows where it also appears in the LAST 10% of text: {appended_count}")
print(f"(High last-10% count would suggest it was appended as a label)")

# ─────────────────────────────────────────────────────────────────────────────
section("DETAILED DUPLICATE ANALYSIS")

# The 99.6% duplicate figure is clearly wrong given we have 962 rows and
# 25 categories with 20-84 resumes each. Let's understand it.

# Perhaps the CSV was read with duplicate index or pandas is comparing
# the category column which has many repeats.

# Let's do it manually
from collections import Counter
text_counter = Counter(df['Resume'].tolist())
most_common_texts = text_counter.most_common(10)
print("\nTop 10 most repeated Resume texts (and how many times each appears):")
for text, count in most_common_texts:
    snippet = str(text)[:80].replace('\n', ' ')
    print(f"  count={count}  text='{snippet}'")

# What fraction of resumes appear exactly once?
appears_once = sum(1 for c in text_counter.values() if c == 1)
appears_more = sum(1 for c in text_counter.values() if c > 1)
print(f"\nUnique resume texts appearing EXACTLY ONCE: {appears_once}")
print(f"Unique resume texts appearing MORE THAN ONCE: {appears_more}")
print(f"Total unique resume texts: {len(text_counter)}")

# ─────────────────────────────────────────────────────────────────────────────
section("FINAL ASSESSMENT")

print("""
KEY QUESTIONS TO RESOLVE:
1. Are the 958/962 'duplicates' real data duplication, or an artifact of CSV parsing?
2. Is the category name appearing in resumes natural content (job title) or appended leakage?
   - NATURAL: A Java Developer's resume mentions "Java Developer" as their job title
   - LEAKAGE: The label "Java Developer" was appended at the end of every resume in that class

Preliminary assessment from sample inspection above:
- Row 1 (Data Science): The resume contains "Data Scientist" as a JOB TITLE — this is natural
- HR sample: The resume contains "HR" in "HR / Skill Details" — looks like a template field
- Java Developer: "Java developer" appears as job title — natural

This is DIFFERENT from the synthetic dataset where responsibility blocks like
"Machine Learning Design", "ML System Design" were APPENDED to encode the label.

The critical question is now the DUPLICATE problem: 958/962 duplicate resumes
would make this dataset essentially unusable. This needs to be understood.
""")
