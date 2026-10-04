# Model Card: RoleSignal Classifier

## Model Details
- **Model Type:** LinearSVC (CalibratedClassifierCV) over TF-IDF features.
- **Task:** Multi-class text classification (28 job roles).
- **Date:** September 2026.

## Intended Use
- **Primary Use Case:** Technical demonstration of a machine learning pipeline, text preprocessing, feature extraction, and REST API integration.
- **Out-of-Scope Use Cases:** This model is **NOT** suitable for making actual hiring or screening decisions in a real-world enterprise environment. It should not be used to automate the rejection of candidates.

## Dataset Limitations & Training Data
The model was trained on a cleaned version (`resume_data_clean.csv`, 9,348 rows) of an internet resume dataset. 

**CRITICAL LIMITATION**: The original dataset assigned labels by appending a synthetic "responsibility block" to the end of randomly selected internet resumes (e.g., appending "Machine Learning Design" to encode the ML Engineer label). While this leakage block has been stripped from the training text, the underlying resumes have *no genuine correlation* with the assigned labels. 
Therefore, the model's accuracy on the test set is low (approx. 8.5%), which is entirely expected because the assigned labels are essentially random noise relative to the resume body text. 

## Evaluation Methodology
- **Splits:** 70% Train, 15% Validation, 15% Test. Stratified by class. Fixed random seed (42).
- **Leakage Check:** The test set is isolated. Overlap between Train/Val/Test is exactly 0.

## Performance Metrics (Honest Baseline)
- **Test Accuracy:** ~8.5%
- **Macro F1:** ~0.075

*Note: The original reported 100% accuracy was entirely fabricated by the label leakage. The current metrics reflect the true (lack of) signal in the synthetic dataset.*

## Known Biases
- **Keyword Reliance:** Because it relies on TF-IDF, the model is highly sensitive to specific keywords and cannot understand semantic context (e.g., "managed a team of software engineers" vs "worked as a software engineer").
- **Language:** English only.

## Confidence Limitations
The model uses `CalibratedClassifierCV` to provide probabilities. Because the dataset lacks genuine signal, the maximum confidence is usually low (10% - 35%). The UI explicitly states that the confidence represents model probability, not candidate suitability.

## Human Oversight Requirements
All automated screening tools must remain under human oversight. The screening breakdown provided by the `/analyze` endpoint is a rigid heuristic and must be treated as a demonstration, not a deterministic assessment of a person's capability.
