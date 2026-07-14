# Stroke Prediction — End-to-End Machine Learning Pipeline

A complete data science project predicting stroke risk from patient demographic and clinical data. Built as a portfolio project demonstrating the full machine learning workflow from raw data to a deployable model.

---

## The Problem

Stroke is one of the leading causes of death and long-term disability worldwide. Many of the risk factors — age, hypertension, glucose levels, BMI — are measurable before a stroke occurs. This project builds a machine learning pipeline that identifies high-risk patients early, giving healthcare professionals the opportunity to intervene before it is too late.

---

## Dataset

**Stroke Prediction Dataset** sourced from Kaggle (fedesoriano, 2021).

- 5,110 patient records
- 11 clinical and demographic features including age, BMI, average glucose level, hypertension, heart disease, smoking status, and work type
- Target variable: binary stroke indicator (1 = stroke, 0 = no stroke)
- Severe class imbalance: 95% no stroke, 5% stroke

Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

---

## Project Structure

```
stroke_prediction/
│
├── data/
│   ├── raw/                    ← original dataset
│   └── processed/              ← cleaned dataset
│
├── notebooks/
│   └── stroke_dataset.ipynb    ← full end-to-end pipeline
│
├── src/
│   └── cleaning_utilitis.py    ← reusable data cleaning toolkit
│
├── outputs/
│   ├── figures/                ← all charts and visualizations
│   └── models/                 ← saved model and scaler
│
├── requirements.txt
└── README.md
```

---

## What Was Built

### 1. Data Foundation
- Loaded dataset and pushed into SQLite database
- Ran SQL queries to extract initial clinical insights
- Performed comprehensive EDA including target distribution, feature distributions, and correlation analysis

### 2. Data Cleaning
- Filled 201 missing BMI values using median imputation
- Removed duplicate rows
- Cleaned string columns for consistency
- Removed BMI outliers using IQR method
- Retained high glucose values as clinically legitimate diabetic readings

### 3. Feature Engineering
Six new features were created from existing columns:

| Feature | Type | Reasoning |
|---|---|---|
| `age_group` | Binning | Stroke rate analysis showed meaningful risk increases at specific age boundaries |
| `bmi_group` | Binning | WHO BMI categories with data-driven ordering based on observed stroke rates |
| `glucose_group` | Binning | Clinical glucose ranges — Normal, Prediabetic, Diabetic |
| `age_bmi_risk` | Interaction | Combined age and BMI risk score — single strongest PCA contributor at 0.688 |
| `combined_condition` | Aggregation | Patients with both hypertension AND heart disease show 20.3% stroke rate vs 4.7% for others |
| `ever_smoked` | Transformation | Binary flag capturing cumulative smoking exposure — formerly smoked has highest stroke rate at 7.9% |

### 4. Clustering Analysis
- K-Means clustering with K=2 to 10 evaluated using elbow curve and silhouette scores
- K=4 selected balancing statistical validity with clinical interpretability
- PCA visualization confirmed well-separated clusters explaining 64.1% of total variance

**Four patient personas identified:**

| Cluster | Name | Avg Age | Stroke Rate |
|---|---|---|---|
| 0 | Young Children | 8 | 0.0% |
| 1 | Healthy Young Adults | 32 | 0.6% |
| 2 | Older Adults with CV Risk | 61 | 7.6% |
| 3 | High Risk Metabolic Patients | 61 | 12.8% |

### 5. Classification
Three models trained with class imbalance handling:

| Model | Recall (Stroke) | Precision (Stroke) | AUC-ROC |
|---|---|---|---|
| Logistic Regression | **0.80** | 0.14 | **0.84** |
| Random Forest | 0.56 | 0.21 | 0.83 |
| XGBoost | 0.46 | 0.18 | 0.79 |

**Logistic Regression with balanced class weights is the recommended model** — it correctly identified 80% of actual stroke patients, outperforming more complex models on the metric that matters most in a clinical context.

### 6. Key Finding — The Tie In
K-Means clustering discovered patient risk structure without ever seeing the stroke label. Among older patients (similar average age of 61), glucose level was the critical differentiator — Cluster 3 patients with high glucose had 12.8% stroke rate vs 7.6% for Cluster 2 with normal glucose. The classifier treats glucose as one feature among many. Clustering revealed it as the defining separator within the highest risk age group.

---

## Results Summary

- **Best model:** Logistic Regression with balanced class weights
- **Recall:** 0.80 — correctly identified 40 out of 50 stroke patients in the test set
- **AUC-ROC:** 0.84
- **Key insight:** Simple models outperform complex ones on small, severely imbalanced medical datasets
- **Counterintuitive finding:** Overweight patients showed higher stroke rate (7.14%) than obese patients (5.07%) in this dataset

---

## Tech Stack

- **Python 3.13**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — visualization
- **scikit-learn** — preprocessing, clustering, classification
- **xgboost** — gradient boosting
- **sqlite3** — SQL exploration
- **joblib** — model persistence

---

## How to Run

```bash
# clone the repo
git clone https://github.com/maxintelligence/stroke-prediction.git
cd stroke-prediction

# create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# install dependencies
pip install -r requirements.txt

# open the notebook
jupyter notebook notebooks/stroke_dataset.ipynb
```

---

## What is Next

- Streamlit app for interactive stroke risk prediction
- Cross-validation across all models for more robust evaluation
- Investigation of the counterintuitive overweight vs obese stroke rate finding

---

## Author

Built by UGORJI MAXWELL 
Connect on LinkedIn: https://www.linkedin.com/in/maxwell-ugorji-136999313/  
GitHub: https://github.com/maxintelligence