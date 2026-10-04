# preprocess_resume.py
"""Preprocess the `Clean_Resume` column of `resume_data_labeled.csv`.
The script:
1️⃣ Loads the labelled dataset.
2️⃣ Prints basic quality diagnostics (missing, empty, duplicate, length extremes).
3️⃣ Applies a robust cleaning pipeline preserving technical terms.
4️⃣ Optionally evaluates stop‑word removal and lemmatization (disabled by default).
5️⃣ Saves the cleaned text to `resume_data_clean.csv` ready for TF‑IDF.
"""

import re
import json
import pandas as pd
from pathlib import Path

# Optional NLTK imports (only used if the user enables them)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:
    nltk = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("resume_data_labeled.csv")
OUTPUT_PATH = Path("resume_data_clean.csv")
ROLE_MAP_PATH = Path("role_map.json")  # kept for reference – not altered here

# Toggle advanced NLTK processing – keep False unless the user decides otherwise
USE_STOPWORDS = False
USE_LEMMATIZATION = False

# ---------------------------------------------------------------------------
# Helper regex patterns
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?|\d{2,4}[\s-])?\d{3,4}[\s-]?\d{3,4}\b")
# Simple address heuristic – looks for numbers followed by street‑type words
ADDRESS_PATTERN = re.compile(r"\b\d{1,5}\s+(?:[A-Za-z0-9]+\s){0,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Way|Court|Ct)\b", flags=re.IGNORECASE)
# Preserve typical programming language symbols (e.g., C+, C#, .NET) – we will NOT delete '+' or '#'
SPECIAL_CHAR_PATTERN = re.compile(r"[^\w\s\+\#\.\-&]", flags=re.UNICODE)

def clean_text(text: str) -> str:
    """Apply the preprocessing pipeline to a single resume string.
    Returns the text unchanged for non‑string inputs (e.g., NaN).
    """
    if not isinstance(text, str):
        return text
    # 1. lower‑case
    text = text.lower()
    # 2. remove URLs
    text = URL_PATTERN.sub(" ", text)
    # 3. strip HTML/markup
    text = HTML_TAG_PATTERN.sub(" ", text)
    # 4. remove email addresses
    text = EMAIL_PATTERN.sub(" ", text)
    # 5. remove phone numbers
    text = PHONE_PATTERN.sub(" ", text)
    # 6. remove crude physical addresses
    text = ADDRESS_PATTERN.sub(" ", text)
    # 7. replace undesired special characters (keep +, #, ., &, - for tech terms)
    text = SPECIAL_CHAR_PATTERN.sub(" ", text)
    # 8. collapse multiple whitespace into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def optional_nltk_processing(series: pd.Series) -> pd.Series:
    """If the toggles are enabled, apply stop‑word removal / lemmatization.
    This function is deliberately separate so the default pipeline stays lightweight.
    """
    if not nltk:
        return series
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    stop_words = set(stopwords.words("english")) if USE_STOPWORDS else set()
    lemmatizer = WordNetLemmatizer() if USE_LEMMATIZATION else None
    def process(text: str) -> str:
        if not isinstance(text, str):
            return text
        tokens = text.split()
        if USE_STOPWORDS:
            tokens = [t for t in tokens if t not in stop_words]
        if lemmatizer:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)
    return series.apply(process)

def main() -> None:
    # -------------------------------------------------------------------
    # 1️⃣ Load dataset
    # -------------------------------------------------------------------
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print("--- Pre‑processing diagnostics (pre‑clean) ---")
    total = len(df)
    missing = df["Clean_Resume"].isnull().sum()
    empty = (df["Clean_Resume"].fillna("") == "").sum()
    dup = df["Clean_Resume"].duplicated().sum()
    lengths = df["Clean_Resume"].fillna("").str.len()
    short_thr, long_thr = 30, 3000
    short = (lengths < short_thr).sum()
    long = (lengths > long_thr).sum()
    print(f"Total records               : {total}")
    print(f"Missing Clean_Resume values  : {missing}")
    print(f"Empty Clean_Resume strings   : {empty}")
    print(f"Duplicate Clean_Resume entries: {dup}")
    print(f"Extremely short (<{short_thr} chars): {short}")
    print(f"Extremely long  (>{long_thr} chars): {long}\n")

    # -------------------------------------------------------------------
    # 2️⃣ Apply cleaning pipeline
    # -------------------------------------------------------------------
    df["Clean_Resume"] = df["Clean_Resume"].apply(clean_text)
    if USE_STOPWORDS or USE_LEMMATIZATION:
        df["Clean_Resume"] = optional_nltk_processing(df["Clean_Resume"])

    # -------------------------------------------------------------------
    # 3️⃣ Post‑clean diagnostics
    # -------------------------------------------------------------------
    print("--- Diagnostics after cleaning ---")
    missing2 = df["Clean_Resume"].isnull().sum()
    empty2 = (df["Clean_Resume"].fillna("") == "").sum()
    dup2 = df["Clean_Resume"].duplicated().sum()
    lengths2 = df["Clean_Resume"].fillna("").str.len()
    short2 = (lengths2 < short_thr).sum()
    long2 = (lengths2 > long_thr).sum()
    print(f"Missing after clean          : {missing2}")
    print(f"Empty after clean            : {empty2}")
    print(f"Duplicate after clean        : {dup2}")
    print(f"Extremely short after clean (<{short_thr} chars): {short2}")
    print(f"Extremely long after clean (>{long_thr} chars): {long2}\n")

    # -------------------------------------------------------------------
    # 4️⃣ Save cleaned CSV (keep columns needed for TF‑IDF)
    # -------------------------------------------------------------------
    cols = ["Resume_Text", "Clean_Resume", "skills", "job_role"]
    df[cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved cleaned data to {OUTPUT_PATH}")

    # -------------------------------------------------------------------
    # 5️⃣ Show a few before/after examples (first two non‑null rows)
    # -------------------------------------------------------------------
    examples_idx = df[df["Clean_Resume"].notnull()].head(2).index
    raw_df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print("--- Example transformations ---")
    for i, idx in enumerate(examples_idx, start=1):
        raw = raw_df.at[idx, "Clean_Resume"]
        cleaned = df.at[idx, "Clean_Resume"]
        print(f"Example {i}:\n  RAW   : {raw[:200]}...\n  CLEAN : {cleaned[:200]}...\n")

if __name__ == "__main__":
    main()
