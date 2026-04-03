# ⏳ Predict Customer Churn — Kaggle Playground Series S6E3

<p align="left">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg" alt="Made with Python">
  </a>
  <img src="https://img.shields.io/badge/Scikit-learn%20?label=Python%20Library&color=green" alt="Scikit-learn Badge">
</p>

An end-to-end machine learning pipeline for the **Kaggle Playground Series Season 6, Episode 3** competition, predicting whether a telecom customer will churn using gradient-boosted decision trees. The project is split across two notebooks: a **data processing pipeline** and a **modelling & evaluation** pipeline.

> 🏆 Competition: [Kaggle — Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3/overview)  
> 📏 Evaluation Metric: **ROC-AUC**

---

## 📌 Project Overview

The goal is to predict binary customer churn (`1` = churned, `0` = retained) from a telecom dataset of **~594,000 records**. The pipeline focuses on intelligent feature engineering tailored for gradient-boosted trees, followed by a **CatBoost** baseline model with early stopping and class balancing.

---

## 📁 Project Structure

```
├── S6_E3_Predict_Customer_Churn_Data_Processing.ipynb        # Notebook 1: Data pipeline
├── S6_E3_Predict_Customer_Churn__Modeling___Evaluation_.ipynb # Notebook 2: Modelling & submission
├── data/
│   ├── train.csv                  # Raw training data (from Kaggle)
│   ├── test.csv                   # Raw test data (from Kaggle)
│   ├── sample_submission.csv      # Kaggle submission format
│   ├── train_modified.csv         # Processed training data (output of Notebook 1)
│   └── test_modified.csv          # Processed test data (output of Notebook 1)
├── catboost_model.cbm             # Saved CatBoost model
├── predictions.npy                # Saved prediction probabilities
├── meine_submission.csv           # Final Kaggle submission file
└── README.md
```

---

## 🗂️ Dataset

**Source:** [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3/data)

| Feature | Description |
|---|---|
| `Gender` | Customer gender |
| `Tenure` | Months as a customer |
| `Partner` | Has a partner (Yes/No) |
| `Dependents` | Has dependents (Yes/No) |
| `PhoneService` | Phone service subscription |
| `MultipleLines` | Multiple phone lines |
| `InternetService` | Internet service type (DSL / Fiber / None) |
| `OnlineSecurity` / `OnlineBackup` | Add-on services |
| `DeviceProtection` / `TechSupport` | Support services |
| `StreamingTV` / `StreamingMovies` | Streaming subscriptions |
| `Contract` | Month-to-month / One year / Two year |
| `PaperlessBilling` | Paperless billing flag |
| `PaymentMethod` | Payment method used |
| `MonthlyCharges` / `TotalCharges` | Billing amounts |
| `Churn` | **Target** — 1 = churned, 0 = retained |

---

## 🔬 Notebook 1 — Data Processing

**File:** `S6_E3_Predict_Customer_Churn_Data_Processing.ipynb`

### Key Steps

**1. Data Loading**
- Downloaded directly from Kaggle API (`kaggle competitions download`)
- Loaded `train.csv` and `test.csv` into pandas DataFrames

**2. Feature Engineering — Service Outage Flags**

Two new binary features created to distinguish between "not subscribed" vs "service unavailable":

| New Feature | Logic | Rationale |
|---|---|---|
| `InternetOutage` | 1 if ANY internet service column = `'No internet service'` | Differentiates users with no internet access vs no subscription |
| `PhoneServiceOutage` | 1 if `MultipleLines` = `'No phone service'` | Flags phone service gaps independently |

> This distinction matters — a user without internet due to a technical issue is fundamentally different from one who simply chose not to subscribe.

**3. Encoding**
- **One-hot encoding** on `PaymentMethod` (4 dummy columns, `drop_first=False` — all 4 retained as each carries meaningful signal)
- **Binary mapping** (`Yes` → 1, `No` → 0) for 12 binary object columns
- `'No internet service'` and `'No phone service'` replaced with `'No'` after outage flags were created
- Dropped original `Gender`, `InternetService`, `Contract` after transformation

**4. Output**
- Saved `train_modified.csv` and `test_modified.csv` — clean, fully numeric datasets ready for modelling

---

## 🤖 Notebook 2 — Modelling & Evaluation

**File:** `S6_E3_Predict_Customer_Churn__Modeling___Evaluation_.ipynb`

### Models Considered

| Model | Notes |
|---|---|
| **CatBoost** ✅ | Baseline — handles categorical features natively, minimal preprocessing required |
| LightGBM | Planned |
| XGBoost | Planned |
| Random Forest | Planned |
| Naive Bayes | Suitable for binary target prediction |
| HistGradientBoosting | Hill-climbing approach |

> Gradient Boosted Decision Trees (GBDTs) are the primary focus — empirically proven to outperform on tabular classification problems of this type.

### CatBoost Baseline Configuration

```python
CatBoostClassifier(
    iterations           = 3000,
    early_stopping_rounds = 100,
    learning_rate        = 0.05,
    loss_function        = 'Logloss',
    eval_metric          = 'AUC',
    depth                = 6,
    auto_class_weights   = 'Balanced',   # handles class imbalance
    bagging_temperature  = 0.8,
    random_strength      = 1,
    l2_leaf_reg          = 5
)
```

**Train/test split:** Stratified split with `test_size=254,655` records, `random_state=0`

### Evaluation

```python
roc_auc = roc_auc_score(y_test, y_proba)
```

**Submission validation:**
- Verified `id` alignment between `sample_submission.csv` and `meine_submission.csv`
- Computed MAE between submission and sample as a sanity check

### Saved Artefacts

| File | Description |
|---|---|
| `catboost_model.cbm` | Trained CatBoost model (reload with `model.load_model()`) |
| `predictions.npy` | Prediction probabilities array |
| `meine_submission.csv` | Final Kaggle submission |

---

## 🚀 How to Run

### On Google Colab (recommended — matches original environment)

**Notebook 1 — Data Processing:**
1. Upload your `kaggle.json` API key when prompted
2. Run all cells — downloads competition data, processes it, saves `train_modified.csv` and `test_modified.csv`

**Notebook 2 — Modelling:**
1. Data is loaded directly from the GitHub raw URL (no local file needed)
2. Run all cells — trains CatBoost, evaluates ROC-AUC, generates `meine_submission.csv`

### Install dependencies
```bash
pip install catboost kaggle pandas numpy scikit-learn matplotlib seaborn
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data wrangling and feature engineering |
| `CatBoost` | Primary gradient-boosted classifier |
| `scikit-learn` | Train/test split, OrdinalEncoder, ROC-AUC metric |
| `matplotlib` / `seaborn` | Visualisation |
| `Kaggle API` | Dataset download automation |

---

## 📝 Notes & Limitations

- `auto_class_weights='Balanced'` is applied to handle the natural class imbalance in churn datasets (churned customers are typically a minority)
- `drop_first=False` was deliberately chosen for `PaymentMethod` one-hot encoding — all four payment categories carry distinct signal worth preserving despite the multicollinearity risk
- The `InternetOutage` and `PhoneServiceOutage` flags are the key novel features — they prevent the model from conflating service unavailability with active opt-out decisions
- LightGBM, XGBoost, and Random Forest benchmarks are planned as next steps to compare against the CatBoost baseline

---

## 👤 Author

**Rony Soliman** — Kaggle competition submission for Playground Series S6E3.  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-RonySoliman-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/RonySoliman)
[![YouTube](https://img.shields.io/badge/YouTube-RonyMLE-FF0000?style=flat&logo=youtube)](https://www.youtube.com/@RonyMLE)
