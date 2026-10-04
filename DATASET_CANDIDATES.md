# Dataset Candidate Audit Summary

This audit was conducted without model training or splitting. The goal was to identify a genuinely categorized resume dataset suitable for later 70/15/15 evaluation and final held-out testing.

| Dataset | Rows | Classes | Unique | Duplicate % | Label Leakage | Provenance | License | Verdict |
|---|---:|---:|---:|---:|---|---|---|---|
| LiveCareer Resume Dataset | 2,484 | 24 | 2,482 | 0.16% | No systematic target-label injection. Category names appear naturally in resume titles and experience sections; no explicit `Category:` or synthetic responsibility block was found. | Public Kaggle dataset by Snehaan Bhawal, described as resumes collected from LiveCareer and stored with PDFs and text exports. | Kaggle page indicates CC0 / Public Domain. | Accept |
| EXAI-ResumeIntel | 2,484 | 24 | 2,482 | 0.16% | Same corpus characteristics as LiveCareer; category words appear in natural resume headings, but no direct label injection or synthetic role block was detected. | Public dataset on Hugging Face: `mithinsagar/exai-resumeintel-data` with `resumes` config. | Public metadata does not clearly state a license, but the corpus appears to be the same public LiveCareer-derived collection. | Accept as alternate public corpus |
| florex/resume_corpus | Not accessible | N/A | N/A | N/A | N/A | Not publicly accessible from the dataset registry in the current environment. | N/A | Reject |
| CareerCorpus | 302 (reference only) | 6 | Unverified | Unknown | Not audited because not publicly accessible in current environment. | Reference dataset only; too small for the main 80% target and not accessible as a project dataset here. | Unknown | Reject for main project use |

## Final Decision

VALID DATASET FOUND: LiveCareer Resume Dataset

The LiveCareer corpus is the primary candidate for the project because it is public, has 2,484 resumes across 24 categories, has low duplicate contamination, and does not show the synthetic label-injection patterns that invalidated the original synthetic dataset.

The dataset has been stored locally under:

- [data/candidates/livecareer](data/candidates/livecareer)
- [data/candidates/livecareer/livecareer_resume_dataset.csv](data/candidates/livecareer/livecareer_resume_dataset.csv)

The EXAI-ResumeIntel corpus was also inspected and appears to be the same public resume corpus under a different packaging. It is a valid alternate, but the LiveCareer dataset remains the primary accepted source.

The project must still keep the original synthetic dataset documented as a historical leakage benchmark only. It is not the usable training corpus.
