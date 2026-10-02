# Hotel Booking Prediction 🏨

A machine learning pipeline for predicting hotel booking cancellations using the `hotel_bookings_updated_2024.csv` dataset.

## 📋 Lab 2 — Dataset Benchmarking & Research Paper Reproduction

### Dataset
- **Source**: `hotel_bookings_updated_2024.csv` (Bedmutha, K. S., 2025, Kaggle)
- **Records**: 119,390 rows
- **Features**: 33 columns (+ 3 derived features)
- **Target**: `is_canceled` (Binary: 0 = Not Canceled, 1 = Canceled)

### Research Paper
> **Determinants of Hotel Booking Cancellations: Evidence from Reservation Analytics**
> Dr. Vinay Raj R — *International Journal for Research in Tourism and Hospitality*, Vol. 01, Issue 02, July 2026
> DOI: [10.53555/th.v1i2.2575](https://doi.org/10.53555/th.v1i2.2575)

---

## 🤖 Models Implemented

| Model | Scenario | Notes |
|---|---|---|
| **Logistic Regression** | Scenario A (matches paper) | Binary LR with StandardScaler + OneHotEncoder |
| **Random Forest** | Ensemble Benchmark | 200 trees, Gini importance |

---

## 📊 Benchmark Results (vs Research Paper)

| Metric | Research Paper | Our LR | Our RF |
|---|---|---|---|
| Accuracy | 81.62% | 81.59% | 87.13% |
| Precision | 83.73% | 81.19% | 88.13% |
| Sensitivity | 62.89% | 65.46% | 75.40% |
| Specificity | 92.74% | 91.08% | 94.03% |
| F1-Score | 71.82% | 72.48% | 81.27% |
| ROC-AUC | 0.866 | 0.894 | 0.945 |

---

## 📁 Files

| File | Description |
|---|---|
| `ml_pipeline.py` | Main ML pipeline (LR + RF training, evaluation, visualizations) |
| `benchmark_report_generator.py` | Loads saved models and generates comparison table |
| `generate_word_report.py` | Generates full `.docx` lab report with all charts embedded |
| `model_comparison_benchmark.csv` | Exported benchmark comparison table |
| `confusion_matrices.png` | Confusion matrices for both models |
| `roc_curves.png` | ROC curves with AUC scores |
| `lr_feature_coefficients.png` | Logistic Regression log-odds coefficients |
| `rf_feature_importance.png` | Random Forest Gini feature importance |
| `Lab2_Hotel_Cancellation_Report.docx` | Full Word report (submitted for lab) |
| `logistic_regression_pipeline.joblib` | Saved trained LR pipeline |
| `random_forest_pipeline.joblib` | Saved trained RF pipeline |

---

## ⚙️ How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn joblib python-docx

# Run full ML pipeline (trains & saves models)
python ml_pipeline.py

# Generate benchmark comparison table
python benchmark_report_generator.py

# Generate Word report
python generate_word_report.py
```

---

## 🔑 Key Findings

- **Non-Refundable Deposits** are the strongest cancellation predictor (OR = 235.6×)
- **Previous Cancellations** increase odds by 9.4×
- **Special Requests** are the strongest protective factor (reduces odds by 43%)
- Random Forest outperforms LR by +5.5% Accuracy and +0.051 AUC due to non-linear feature interactions
