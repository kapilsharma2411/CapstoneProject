# CapstoneProject - 🩺 Diabetes Prediction — ML Pipeline

A supervised machine learning project to predict diabetes onset using the
**Healthcare-Diabetes dataset**. The pipeline covers data cleaning,
exploratory analysis, model comparison, hyperparameter tuning, and
overfitting diagnostics.

Jupyter Notebook Link:
https://github.com/kapilsharma2411/CapstoneProject/blob/main/Capstone-Project.ipynb

Dataset Link:
https://github.com/kapilsharma2411/CapstoneProject/blob/main/data/Healthcare-Diabetes.csv

Dataset originally downloaded from :
https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes


## Below are some conlusive points after EDA step: 

- Glucose shows strongest correlation with diabetes
- BMI and Age show moderate positive relationship
- Dataset is moderately imbalanced as ~65% of records represent non-diabetic population while ~35% is diabetic, so this can bring some biasing while training the models.
- Older and obese individuals have higher diabetes prevalence
- Some medical variables required cleaning (zero replacement)


## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Setup & Installation](#4-setup--installation)
5. [Running the Pipeline](#5-running-the-pipeline)
6. [Pipeline Walkthrough](#6-pipeline-walkthrough)
7. [Results & Model Comparison](#7-results--model-comparison)
8. [Key Visualisations](#8-key-visualisations)
9. [Feature Importance](#9-feature-importance)
10. [Lessons Learned](#10-lessons-learned)
11. [Next Steps](#11-next-steps)
12. [References](#12-References)

## 1. Project Overview

| Property | Value |
|---|---|
| **Task** | Binary classification |
| **Target** | `Outcome` — 1 = Diabetic, 0 = Non-diabetic |
| **Best model** | Gradient Boosting (tuned) |
| **Best ROC-AUC** | **0.835** |
| **Best Accuracy** | 74.4% |

The goal is not only to build accurate models, but to build them **correctly** —
avoiding common pitfalls like data leakage, improper scaling, and overfitting.

---

## 2. Dataset

**Source:** [Kaggle — Healthcare Diabetes Dataset](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes)
(Based on the original UCI Pima Indians Diabetes Database)

**Raw shape:** 2,768 rows × 10 columns  
**After deduplication:** 778 rows × 9 columns (see Section 3)

### Features

| Feature | Description | Notes |
|---|---|---|
| `Pregnancies` | Number of times pregnant | |
| `Glucose` | Plasma glucose concentration (2hr OGTT) | Most predictive feature |
| `BloodPressure` | Diastolic blood pressure (mm Hg) | |
| `SkinThickness` | Triceps skinfold thickness (mm) | High zero rate |
| `Insulin` | 2-hour serum insulin (mu U/ml) | High zero rate |
| `BMI` | Body mass index (kg/m²) | |
| `DiabetesPedigreeFunction` | Genetic risk score | |
| `Age` | Age in years | |
| `Outcome` | **Target** — 1 = Diabetes, 0 = No Diabetes | 35% positive rate |

---

## 3. Project Structure

```
diabetes-prediction/
│
├── data/
│   └── Healthcare-Diabetes.csv      # Raw dataset
│
├── diabetes_prediction.py           # Full ML pipeline (this project)
│
├── eda_outcome_distribution.png     # Class balance chart
├── eda_feature_distributions.png    # Per-feature histograms by outcome
├── eda_correlation_heatmap.png      # Feature correlation matrix
├── eda_boxplots.png                 # Boxplots per feature / outcome
├── eda_pairplot.png                 # Pairplot of top features
│
├── eval_confusion_matrix.png        # Best model confusion matrix
├── eval_roc_curves.png              # ROC curves for all models
├── eval_feature_importance.png      # Permutation feature importance
├── eval_learning_curves.png         # Learning curves (overfit check)
├── model_comparison_default.png     # Comparison of various models used on Accuracy/ROC-AUC/F1
│
└── README.md                        # This file
```

---

## 4. Setup & Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Python version:** 3.8+

---

## 5. Running the Pipeline

```bash
# Place the CSV in a data/ subdirectory, then run:
python diabetes_prediction.py
```

The script will print progress to the terminal and save all plots as PNG files
in the working directory.

---

## 6. Pipeline Walkthrough

```
Raw CSV (2,768 rows)
        │
        ▼
 Drop 'Id' column
        │
        ▼
 Replace impossible zeros with NaN
 Impute NaN with column median
        │
        ▼
 Deduplicate (2,768 → 778 rows)          ← Prevents data leakage
        │
        ▼
 Exploratory Data Analysis
 (distributions, correlations, boxplots)
        │
        ▼
 Stratified Train / Test Split (80/20)
        │
        ▼
 StandardScaler fitted on train only     ← Prevents leakage
        │
        ├── Default model comparison (7 models)
        │       └── Cross-validated ROC-AUC
        │
        ├── Hyperparameter tuning (GridSearchCV, StratifiedKFold)
        │       └── 4 models tuned
        │
        ├── Best model detailed evaluation
        │       ├── Classification report
        │       ├── Confusion matrix
        │       └── ROC curves
        │
        ├── Permutation feature importance
        │
        └── Learning curves (overfitting diagnostics)
```

---

## 7. Results & Model Comparison

### Default Hyperparameters

| Model | Accuracy | ROC-AUC | F1 Score | CV ROC-AUC |
|---|---|---|---|---|
| Gradient Boosting | 0.750 | **0.827** | 0.621 | **0.834** |
| Random Forest | 0.737 | 0.821 | 0.624 | 0.825 |
| K-Nearest Neighbors | 0.750 | 0.815 | 0.621 | 0.799 |
| Logistic Regression | 0.712 | 0.814 | 0.571 | 0.831 |
| SVM (RBF) | 0.750 | 0.812 | 0.614 | 0.829 |
| Decision Tree | 0.731 | 0.713 | 0.632 | 0.672 |
| Dummy Baseline | 0.647 | 0.500 | 0.000 | 0.500 |

### After Hyperparameter Tuning

| Model | Accuracy | ROC-AUC | F1 Score | Best Parameters |
|---|---|---|---|---|
| **Gradient Boosting** | 0.744 | **0.835** | 0.630 | lr=0.05, depth=4, n=100, subsample=0.8 |
| Random Forest | 0.744 | 0.823 | 0.600 | depth=5, leaf=10, features=log2, n=100 |
| Decision Tree | 0.750 | 0.805 | 0.636 | entropy, depth=5, leaf=20 |
| Logistic Regression | 0.712 | 0.814 | 0.571 | C=10, solver=liblinear |

### Best Model: Gradient Boosting (tuned)

```
Classification Report:
              precision    recall  f1-score   support
 No Diabetes       0.80      0.81      0.80       101
    Diabetes       0.64      0.62      0.63        55
    accuracy                           0.74       156
```

> **ROC-AUC of 0.835** is the most meaningful metric here — it measures
> ranking ability across all thresholds, which matters more than raw accuracy
> on an imbalanced dataset.

---

## 8. Key Visualisations

| File | What It Shows |
|---|---|
| `eda_outcome_distribution.png` | Class imbalance — 65% non-diabetic, 35% diabetic |
| `eda_feature_distributions.png` | Clear separation for Glucose and BMI by outcome |
| `eda_correlation_heatmap.png` | Glucose (0.49) and Age (0.24) most correlated with Outcome |
| `eda_boxplots.png` | Outliers in Insulin; Glucose/BMI shift clearly with outcome |
| `eda_pairplot.png` | Glucose vs BMI shows best visual class separation |
| `eval_roc_curves.png` | Gradient Boosting and Random Forest dominate |
| `eval_confusion_matrix.png` | Best model: 82 TN, 34 TP, 21 FN, 19 FP |
| `eval_learning_curves.png` | GB train ≈ CV AUC — no overfitting after tuning |
| `eval_feature_importance.png` | Glucose alone accounts for ~60% of AUC contribution |

---

## 9. Feature Importance

Based on **permutation importance** (model-agnostic, measured on test set):

| Rank | Feature | Importance (ΔAUC) |
|---|---|---|
| 1 | Glucose | **0.149** |
| 2 | BMI | 0.052 |
| 3 | Age | 0.026 |
| 4 | Pregnancies | 0.015 |
| 5 | DiabetesPedigreeFunction | 0.013 |
| 6 | Insulin | 0.011 |
| 7 | SkinThickness | 0.004 |
| 8 | BloodPressure | ~0.000 |

**Glucose is by far the most important predictor** — a clinically expected
result, as elevated blood glucose is the primary diagnostic marker for diabetes.
BloodPressure shows near-zero importance and could be dropped without
meaningful loss.

---

## 10. Lessons Learned

### What went wrong in the original code

| Issue | Consequence | Fix |
|---|---|---|
| Duplicate rows not removed | Data leakage inflated accuracy to 99.6% | Deduplicate before split |
| Scaler fit on full dataset | Test statistics leak into training | Fit scaler on train only |
| No `stratify=y` | Class ratio may differ across splits | Add `stratify=y` |
| Missing import | `train_test_split` not imported | Added to imports |

### Why accuracy alone is misleading here

The dummy classifier (always predict "No Diabetes") achieves **64.7% accuracy**
just by exploiting class imbalance. A model with 70% accuracy that ignores the
minority class entirely would appear better than random. Always report
**ROC-AUC and F1** alongside accuracy for imbalanced datasets.

### Why cross-validation matters

A single train/test split on 778 rows is noisy. With stratified 5-fold CV,
the variance in estimates drops significantly and the ranking of models
becomes more reliable.

---

## 11. Next Steps

| Improvement | Expected Benefit |
|---|---|
| **SMOTE or class weighting** | Better recall on the minority (diabetic) class |
| **XGBoost / LightGBM** | Likely 2–4% AUC improvement over sklearn GBM |
| **Feature engineering** | `Glucose × BMI`, `Age²`, insulin-resistance proxies |
| **Threshold tuning** | Trade precision/recall depending on clinical cost of FN vs FP |
| **SHAP values** | Per-sample explanations for clinical interpretability |
| **Pipeline with `ColumnTransformer`** | Cleaner, leakage-proof preprocessing |
| **External validation** | Evaluate on a held-out dataset from a different source |

## 12. References

- Pima Indians Diabetes Database — UCI Machine Learning Repository
- Smith, J.W. et al. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*
- Scikit-learn documentation — [https://scikit-learn.org](https://scikit-learn.org)
