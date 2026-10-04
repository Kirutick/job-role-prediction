"""build_clean_dataset.py  (v2)
================================
Creates resume_data_clean.csv that eliminates the responsibility-block
label leakage while maximising the real signal in the data.

DATASET STRUCTURE (discovered by audit):
  Resume_Text = [resume body text] + [skills list] + [responsibility block] + [education]
  e.g.:
    "To secure an IT specialist... ['Microsoft', 'Network Security', ...]
     Machine Learning Design\nData Analysis\nModel Training...\n ['B.Sc']"

  The responsibility block (starting at the role-map key phrase) encodes the
  label directly.  After removing it, the remaining text (resume body + skills
  list) is genuine signal.

STRATEGY:
  1. Truncate Resume_Text at the first occurrence of any role-map key phrase.
  2. Also parse the skills list from Resume_Text and prepend it to the cleaned
     text so TF-IDF can pick up skill-level features.
  3. Re-apply clean_text() on the combined text.
  4. Save as resume_data_clean.csv (Clean_Resume column only).

The original files are NEVER modified.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

BASE          = Path(__file__).resolve().parent
DATA_PATH     = BASE / "resume_data_labeled.csv"
ROLE_MAP_PATH = BASE / "role_map.json"
OUTPUT_PATH   = BASE / "resume_data_clean.csv"

sys.path.insert(0, str(BASE))
from preprocess_resume import clean_text  # noqa: E402


def load_role_map() -> dict[str, str]:
    with open(ROLE_MAP_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_truncation_patterns(role_map: dict[str, str]) -> list[tuple[str, re.Pattern]]:
    """Build patterns that match the role-map key phrase so we know where to cut."""
    pats = []
    for phrase in role_map:
        escaped = re.escape(phrase)
        pats.append((phrase, re.compile(escaped, re.IGNORECASE)))
    return pats


def extract_skills_from_text(text: str) -> str:
    """Extract the skills list (inside square brackets) from Resume_Text.
    Returns them as a space-separated string of skill words."""
    if not isinstance(text, str):
        return ""
    # Skills appear as ['skill1', 'skill2', ...] or ["skill1", ...]
    skills = re.findall(r"'([^']{2,80})'", text)
    # Also grab double-quoted variants
    skills += re.findall(r'"([^"]{2,80})"', text)
    # Join and return
    return " ".join(skills)


def truncate_at_responsibility_block(
    text: str, patterns: list[tuple[str, re.Pattern]]
) -> str:
    """Find the earliest role-map phrase in text and truncate from there."""
    if not isinstance(text, str):
        return text
    earliest_pos = len(text)
    for _, pat in patterns:
        m = pat.search(text)
        if m and m.start() < earliest_pos:
            earliest_pos = m.start()
    truncated = text[:earliest_pos]
    # Strip trailing list-close artifacts  e.g.   "] "  "'] "
    truncated = re.sub(r"[\['\]\s]+$", "", truncated).strip()
    return truncated


def build_combined_text(raw_text: str,
                        patterns: list[tuple[str, re.Pattern]]) -> str:
    """Truncate the resume at the responsibility block, then extract skills
    from the TRUNCATED portion so skill extraction never sees the leaked text."""
    truncated = truncate_at_responsibility_block(raw_text, patterns)
    skills_str = extract_skills_from_text(truncated)
    # Combine: body + skills repeated for TF-IDF weight
    combined = truncated.strip() + " " + skills_str.strip()
    return combined.strip()


def verify_no_leakage(series: pd.Series, role_map: dict[str, str]) -> dict:
    """Return counts of any surviving role-map key phrases per phrase."""
    results = {}
    for phrase in role_map:
        cp = clean_text(phrase).strip()
        if not cp:
            continue
        count = series.str.contains(re.escape(cp), case=False, na=False).sum()
        if count:
            results[phrase] = count
    return results


def main() -> None:
    role_map = load_role_map()
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"Loaded: {len(df)} rows")

    patterns = build_truncation_patterns(role_map)

    print("Building combined (truncated body + skills) text ...")
    df["Combined_Text"] = df["Resume_Text"].apply(
        lambda t: build_combined_text(t, patterns)
    )

    print("Re-applying clean_text() ...")
    df["Clean_Resume_New"] = df["Combined_Text"].apply(clean_text)

    orig_count = len(df)
    df = df[df["Clean_Resume_New"].str.len() >= 30].copy()
    print(f"Dropped {orig_count - len(df)} rows with <30 chars after cleaning.")

    print("\nVerifying no role-map key phrases survive ...")
    remaining = verify_no_leakage(df["Clean_Resume_New"], role_map)
    if not remaining:
        print("  OK: zero key-phrase hits remain.")
    else:
        print("  Remaining phrase hits (checking if benign):")
        for phrase, cnt in remaining.items():
            cp = clean_text(phrase).strip()
            dist = df[df["Clean_Resume_New"].str.contains(re.escape(cp), case=False, na=False)]["job_role"].value_counts()
            is_uniform = dist.std() < 2.0
            print(f"    '{phrase}' ({cp}): {cnt} rows, uniform={is_uniform}")
            print(f"      {dist.to_string()}")

    # Save
    out = df[["Clean_Resume_New", "skills", "job_role"]].rename(
        columns={"Clean_Resume_New": "Clean_Resume"}
    )
    # Also save the original Resume_Text truncated (as Resume_Text column)
    out["Resume_Text"] = df["Combined_Text"]
    out = out[["Resume_Text", "Clean_Resume", "skills", "job_role"]]
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nSaved {len(out)} rows to {OUTPUT_PATH}")
    print("\nClass distribution:")
    vc = df["job_role"].value_counts()
    print(vc.to_string())
    print(f"\nMin class: {vc.min()} | Max class: {vc.max()}")

    # Sample
    print("\n=== Before / After (first 2 rows) ===")
    orig_df = pd.read_csv(DATA_PATH, encoding="utf-8")
    for i in [0, 1]:
        idx = df.index[i]
        role = df.at[idx, "job_role"]
        orig_text = str(orig_df.at[idx, "Resume_Text"])
        new_text  = str(df.at[idx, "Clean_Resume_New"])
        print(f"\nRow {idx} | Role: {role}")
        print(f"  ORIG tail: {repr(orig_text[-200:])}")
        print(f"  CLEAN first 200: {repr(new_text[:200])}")


if __name__ == "__main__":
    main()
