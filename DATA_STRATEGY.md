# Dataset Strategy

## The Lack of Genuine Data
Upon auditing the `RoleSignal` directory, no genuine role-labeled resume dataset was found.
The original `resume_data_labeled.csv` assigned labels by taking internet resumes and appending a synthetic "responsibility block" that contained the exact target role (e.g., "Machine Learning Design" = Machine Learning Engineer). This resulted in massive label leakage.

## Current Working Dataset
To proceed with the engineering architecture without fabricating data, the system uses `resume_data_clean.csv`. 

This file is created by stripping the synthetic responsibility block from the original data. 

**Limitations:**
Because the labels were originally determined *only* by the synthetic block, the underlying resume text now has almost no genuine correlation with the assigned `job_role`. 

This means that a "Machine Learning Engineer" label might now be assigned to a resume that describes a traditional IT Support role. 

**Conclusion on Baseline Accuracy:**
Because the labels are functionally random noise relative to the text, the model achieves a ~8.5% accuracy (slightly above random chance for 28 classes, due to minor residual correlations). This is the honest, defensible baseline for this dataset.

## How to Obtain Genuine Data
To make this system practically useful in the real world, the `resume_data_clean.csv` must be replaced. A genuine dataset must satisfy these properties:
1. Resumes must be linked to the role the candidate was *actually hired for* or *successfully screened for*.
2. The target label must NOT be appended into the text of the resume itself.
3. The classes should ideally be reduced from 28 highly overlapping roles (e.g., "Civil/Environmental Engineer" vs "Civil/Structural Engineer") down to 8-10 broader role families to reduce ambiguity.
