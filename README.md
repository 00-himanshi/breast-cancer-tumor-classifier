<div align="center"> 

# 🩺 Breast Cancer Prediction — SVM (RBF Kernel)

**A leakage-free, nested-CV machine learning pipeline for classifying breast tumors as Malignant or Benign**

[Python](https://img.shields.io/badge/Python-3.12%2B-blue)
[scikit--learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange)
[Streamlit](https://img.shields.io/badge/Streamlit-1.61.1-red)
[Status](https://img.shields.io/badge/status-production--ready-brightgreen)
[License](https://img.shields.io/badge/license-MIT-lightgrey)

 </div> 

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Model Selection Methodology](#model-selection-methodology)
- [Why SVM RBF Was Selected](#why-svm-rbf-was-selected)
- [Threshold Optimization](#threshold-optimization)
- [Final Test Results](#final-test-results)
- [Data Leakage Audit](#data-leakage-audit)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [API Reference](#api-reference)
- [Requirements](#requirements)

---

## Overview

This project implements a machine learning pipeline that classifies breast tumors as **Malignant (1)** or **Benign (0)** using 30 numerical features derived from a digitized image of a Fine Needle Aspirate (FNA) of a breast mass — the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The project's core design principle is **methodological correctness over inflated metrics**. Every stage of the pipeline — scaling, resampling, feature selection, dimensionality reduction, hyperparameter search, and threshold selection — is strictly isolated inside a **Nested Cross-Validation** framework to prevent data leakage.

> The final model is not the one with the highest raw accuracy — it's the one selected by a **recall-weighted composite score**, because in a cancer-screening context, missing a malignant case (false negative) is far costlier than a false alarm.

---

## Dataset

| Property Value                       |                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Total samples                        | 569                                                                                                          |
| Features used                        | 30 (numeric)                                                                                                 |
| Missing values                       | 0                                                                                                            |
| Class distribution                   | Benign: 357 (62.7%) · Malignant: 212 (37.3%)                                                                 |
| Most correlated features with target | `concave points_worst` (0.79), `perimeter_worst` (0.78), `concave points_mean` (0.78), `radius_worst` (0.78) |

Because of the class imbalance and the presence of outliers (e.g. `area_se` has \~11.4% IQR-based outliers), the pipeline pairs **SMOTE** with **class-weighted classifiers** and uses **RobustScaler** instead of StandardScaler.

Full EDA output — near-zero-variance features, outlier tables, correlation heatmap, class-wise distributions — is available in [`eda_report.md`](eda_report.md).

---

## Pipeline Architecture

```
RobustScaler → SMOTE (searched) → Feature Selection (searched) → PCA (searched) → Classifier

```

| Stage Role                |                                                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `RobustScaler`            | Fixed. Scales using median/IQR, robust to the outliers found in EDA                                                     |
| `SMOTE`                   | Searched (on/off). Applied **only inside training folds** via `imblearn.Pipeline`, never touching validation/test folds |
| `SelectKBest (f_classif)` | Searched: off / k=10 / k=20                                                                                             |
| `PCA`                     | Searched: off / n=10 / n=15                                                                                             |
| Classifier                | 10 candidate algorithms benchmarked (see below)                                                                         |

The pipeline uses `imblearn.Pipeline` rather than a plain `sklearn.Pipeline` specifically so that resampling logic respects fold boundaries — this is what prevents **SMOTE leakage**.

**Final selected configuration:** SMOTE = ON, Feature Selection = OFF, PCA = OFF.

---

## Model Selection Methodology

**Evaluation strategy: Nested Cross-Validation**

- **Outer loop** — 5-fold `StratifiedKFold`: measures generalization performance
- **Inner loop** — 3-fold `StratifiedKFold` + `RandomizedSearchCV` (15 iterations, `scoring='roc_auc'`): finds the best hyperparameters *and* the best SMOTE/FeatureSelection/PCA combination, independently within each outer fold

Because hyperparameter search is re-run inside every outer fold, hyperparameters themselves never see the held-out data — this eliminates **hyperparameter leakage**, not just data leakage.

### Model Comparison

| Model CV Accuracy CV Recall CV F1 CV ROC-AUC Overfit Gap **Selection Score**  |        |            |        |        |        |            |
| ----------------------------------------------------------------------------- | ------ | ---------- | ------ | ------ | ------ | ---------- |
| Logistic Regression                                                           | 0.9604 | 0.9294     | 0.9460 | 0.9946 | 0.0198 | 0.9554     |
| Linear SVM                                                                    | 0.9736 | 0.9471     | 0.9641 | 0.9948 | 0.0071 | 0.9675     |
| **SVM RBF** ✅                                                                 | 0.9692 | **0.9529** | 0.9583 | 0.9933 | 0.0198 | **0.9677** |
| AdaBoost                                                                      | 0.9648 | 0.9412     | 0.9523 | 0.9955 | 0.0352 | 0.9093     |
| KNN                                                                           | 0.9648 | 0.9294     | 0.9515 | 0.9896 | 0.0099 | 0.9554     |
| HistGradientBoosting                                                          | 0.9626 | 0.9353     | 0.9494 | 0.9909 | 0.0374 | 0.9015     |
| Extra Trees                                                                   | 0.9626 | 0.9412     | 0.9493 | 0.9915 | 0.0341 | 0.9090     |
| Random Forest                                                                 | 0.9516 | 0.9471     | 0.9362 | 0.9841 | 0.0456 | 0.8880     |
| Gradient Boosting                                                             | 0.9516 | 0.9353     | 0.9359 | 0.9869 | 0.0484 | 0.8800     |
| Decision Tree                                                                 | 0.9077 | 0.8824     | 0.8780 | 0.9221 | 0.0681 | 0.7938     |

*(Full raw output:* [*`model_comparison.csv`*](model_comparison.csv) */* [*`nested_cv_results.csv`*](nested_cv_results.csv)*)*

---

## Why SVM RBF Was Selected

### The Selection Score Formula

Model selection was **not** based on raw accuracy. A weighted composite score was used instead:

```python
selection_score = (recall * 0.4) + (roc_auc * 0.3) + (f1 * 0.2) + (accuracy * 0.1)

if overfit_gap > 0.03:
    selection_score -= (overfit_gap * 1.5)   # overfitting penalty

```

**Recall carries 40% of the weight — the highest of any metric.** This is intentional: in a cancer-screening pipeline, a **false negative** (calling a malignant tumor benign) is the most dangerous failure mode. The scoring function was deliberately biased toward recall rather than optimizing for accuracy or precision alone.

### SVM RBF vs. Linear SVM — the closest contest

These two models were nearly tied, so the arithmetic matters:

**SVM RBF**

```
= (0.9529 × 0.4) + (0.9933 × 0.3) + (0.9583 × 0.2) + (0.9692 × 0.1)
= 0.38118 + 0.29800 + 0.19166 + 0.09692
= 0.9678

```

Overfit gap = 0.0198 → below the 0.03 threshold, no penalty applied.

**Linear SVM**

```
= (0.9471 × 0.4) + (0.9948 × 0.3) + (0.9641 × 0.2) + (0.9736 × 0.1)
= 0.37882 + 0.29845 + 0.19282 + 0.09736
= 0.9675

```

Overfit gap = 0.0071 → the best generalization gap of any model tested, no penalty applied.

**The two scores differ by only \~0.0003** — a statistical near-tie. SVM RBF won because:

1. **It has the higher recall** (0.9529 vs 0.9471), and recall is weighted at 40% — the single most influential term in the formula.
2. Linear SVM had a better ROC-AUC (0.9948 vs 0.9933) and a much tighter generalization gap (0.0071 vs 0.0198), but neither difference was large enough (nor penalized, since both were under the 0.03 threshold) to offset SVM RBF's recall edge.
3. The **RBF kernel captures non-linear decision boundaries** — some features (e.g. `area_worst`, `concavity_worst`) show class distributions that aren't perfectly linearly separable, which the RBF kernel handles slightly better, reflected in its higher recall.

> **In short:** if the selection metric had been plain accuracy, Linear SVM (0.9736) would have won. Because the scoring function was deliberately recall-first — appropriate for a diagnostic screening task — SVM RBF took the top spot by a narrow but principled margin.

### Final Selected Hyperparameters

After model selection, a fresh `RandomizedSearchCV` (20 iterations, 5-fold, `scoring='roc_auc'`) was run on the **full training set** to lock in final hyperparameters:

| Parameter Value   |              |
| ----------------- | ------------ |
| `C`               | `10`         |
| `gamma`           | `0.01`       |
| `probability`     | `True`       |
| `class_weight`    | `'balanced'` |
| SMOTE             | ON           |
| Feature Selection | OFF          |
| PCA               | OFF          |

---

## Threshold Optimization

Rather than using the default 0.5 probability cutoff, a **leakage-free custom threshold** was derived:

- **Method:** 5-fold independent Out-of-Fold (OOF) estimation. The model is trained 5 times (holding out a different fold each time), so every training sample gets a prediction from a fold it was never trained on — giving "as-if-unseen" probabilities across the entire training set without ever touching the test set.
- **Selection rule:** Using a precision-recall curve over the OOF probabilities, the threshold was chosen to **maximize F1-score subject to Recall ≥ 95%**.
- **Result:** `threshold = 0.6479`, achieving **OOF Recall = 0.9529** and **OOF F1 = 0.9701**.

A higher-than-default threshold may seem counterintuitive (lowering the threshold is the usual way to boost recall), but `class_weight='balanced'` combined with SMOTE already skewed the raw predicted probabilities toward the malignant class — so a 0.65 cutoff was where the 95%+ recall constraint and optimal F1 actually intersected.

---

## Final Test Results

The 20% stratified hold-out test set was used **exactly once**, at the very end — never during hyperparameter tuning or threshold selection.

| Metric Value              |        |
| ------------------------- | ------ |
| Test Accuracy             | 0.9737 |
| Test Precision            | 1.0000 |
| Test Recall / Sensitivity | 0.9286 |
| Test Specificity          | 1.0000 |
| Test F1                   | 0.9630 |
| Test ROC-AUC              | 0.9974 |
| False Positive Rate       | 0.0000 |
| False Negative Rate       | 0.0714 |

> Test recall (92.86%) is slightly lower than CV recall (95.29%) — this is expected sampling variance from a small test set (\~114 samples), not a sign of leakage or overfitting, given the audit results below.

---

## Data Leakage Audit

| Check Status              |        |
| ------------------------- | ------ |
| Train/Test Leakage        | ✅ PASS |
| Nested CV                 | ✅ PASS |
| Scaler Leakage            | ✅ PASS |
| SMOTE Leakage             | ✅ PASS |
| Feature Selection Leakage | ✅ PASS |
| PCA Leakage               | ✅ PASS |
| Hyperparameter Leakage    | ✅ PASS |
| Threshold Leakage         | ✅ PASS |
| Final Test Isolation      | ✅ PASS |

---

## Project Structure

```
.
├── 01_eda.py / 01_eda.ipynb            # Exploratory data analysis
├── 02_model_training.py / .ipynb       # Nested-CV model comparison, training, thresholding
├── prediction.py                       # predict_breast_cancer() — validation + safe inference
├── app.py                              # Streamlit web app
├── final_model_pipeline.joblib         # Trained SVM RBF pipeline
├── feature_names.joblib                # Ordered list of 30 expected features
├── model_metadata.json                 # Threshold, CV/test metrics, class mapping
├── model_comparison.csv                # All 10 models — nested CV results
├── nested_cv_results.csv               # Raw CV run output
├── eda_report.md                       # Full EDA writeup with plots
├── breast-cancer-wisconsin-data.csv    # Source dataset
└── requirements.txt

```

---

## Installation & Usage

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py

```

---

## API Reference

```python
from prediction import predict_breast_cancer

patient_data = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    # ... all 30 features (order is handled automatically by validate_input)
}

result = predict_breast_cancer(patient_data)
print(result)
# {'prediction': 1, 'probability': 0.89, 'threshold': 0.6479, 'class_label': 'Malignant'}

```

`prediction.py` enforces strict input validation: exactly 30 numeric features, no missing/extra/NaN/infinite values — otherwise it raises a clear, typed error.

---

## Requirements

```
pandas==3.0.5
numpy==2.5.2
scikit-learn==1.9.0
joblib==1.5.3
imbalanced-learn==0.14.2
streamlit==1.61.1

```

---

 <div align="center"> 

*Built with an emphasis on methodological rigor: nested CV, leak-free thresholding, and a clinically-motivated recall-weighted model selection criterion.*

 </div>  
