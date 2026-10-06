# ResumeAtlas Data Strategy

## Dataset source

The current classifier is trained on the public Hugging Face dataset
[`ahmedheakl/resume-atlas`](https://huggingface.co/datasets/ahmedheakl/resume-atlas),
pinned at revision `3f80ca910fa9964890afb7845c09e239d07b9b1d`. The audit inspects
the source schema instead of assuming column names; at this revision the
resume-text column is `Text` and the target is `Category`.

The source contains 13,389 resumes in 43 occupational categories. Its raw file
is loaded through Hugging Face Datasets and kept in the local Hugging Face
cache, not committed to the repository.

## Audit and filtering decisions

Run `python audit_resume_atlas.py` to regenerate the audit, normalized
`resume_data_real.csv`, and splits. The audit writes
`resume_atlas_audit.json`, including source columns, original and retained
class counts, duplicate findings, exclusions, and overlap checks.

Audit findings and decisions:

1. There were no missing or empty resume texts or category values.
2. 1,304 rows repeat normalized resume text across 1,200 duplicate groups.
   Exact-duplicate and normalized-duplicate counts are both 1,304.
3. 169 duplicate-text groups have inconsistent category labels (381 rows).
   Every row in each such group is excluded; none is arbitrarily relabeled.
4. The remaining same-label duplicates are collapsed by normalized text,
   retaining the earliest source row. This removes 1,092 duplicate copies.
5. Seven resumes are shorter than 100 characters. They are reported and
   retained; no minimum-length filter is applied.
6. Resume text is preserved as provided; normalization is used only to detect
   duplicates and check leakage.
7. The audit looks for explicit category/role marker blocks appended to the
   text and finds none. An occurrence of a category word in normal resume
   content is not treated as artificial leakage.

The resulting normalized dataset contains 11,916 unique resumes. Missing,
conflicting-label, duplicate, and short-record decisions are all recorded.

## Split and leakage checks

Duplicate resolution happens before splitting. `train_test_split` creates
stratified 70% train, 15% validation, and 15% test partitions using random
state 42. The generated split sizes are 8,340 / 1,788 / 1,788. The audit
normalizes text across each partition and requires zero overlap for all three
pairwise comparisons before it writes the split files.

## Training and evaluation

Run `python train_resume_atlas.py` after a successful audit. TF-IDF uses
unigrams/bigrams, `sublinear_tf=True`, `strip_accents="unicode"`, and 10,000,
20,000, or 30,000 feature limits. Four requested classifier families are
compared at each feature limit. Only the training set fits vectorizers and
classifiers. Validation macro-F1 chooses the winner; test metrics are generated
once for that selected candidate.

The selected model is TF-IDF + class-balanced LinearSVC with 10,000 features.
Validation accuracy/macro-F1 are 0.8272/0.8232. Held-out test
accuracy/top-3 accuracy/macro-F1/weighted-F1 are
0.8322/0.9379/0.8274/0.8271. These are dataset-specific results, not a
guarantee of generalization or hiring suitability.

ResumeAtlas annotations are not independently verified employment outcomes.
Resume text may contain personal data. Keep normalized and split CSV files and
`misclassified_resumes.csv` local; they are git-ignored. Do not use the model
or the separate application screening heuristic for consequential hiring
decisions.

Historical synthetic datasets and audit reports are retained for provenance;
they are not training inputs for the current ResumeAtlas model.
