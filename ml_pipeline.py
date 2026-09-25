import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

print("=== STEP 1: LOAD AND INSPECT DATASET ===")
data_path = 'hotel_bookings_updated_2024.csv'
df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nColumn Names and Data Types:")
print(df.dtypes)

print("\nMissing Values:")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0])

print(f"\nDuplicate Rows: {df.duplicated().sum()}")

print("\nBasic Statistical Summary (Numerical):")
print(df.describe().T)

print("\nTarget Variable ('is_canceled') Distribution:")
print(df['is_canceled'].value_counts())
print(df['is_canceled'].value_counts(normalize=True) * 100)

print("\n=== STEP 2: DATA CLEANING & LEAKAGE DETECTION ===")
# Identify leakage columns
leakage_cols = ['reservation_status', 'reservation_status_date']
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print(f"Direct leakage columns identified: {leakage_cols}")
print(f"Constant/Zero-variance columns identified: {constant_cols}")

# Explanation of leakage columns
print("\nLeakage Explanation:")
print("1. 'reservation_status': Contains values ('Canceled', 'Check-Out', 'No-Show') directly indicating the outcome.")
print("2. 'reservation_status_date': Date of the last status update, recorded when cancellation/checkout occurred.")
print("3. Constant columns (e.g., 'arrival_date_year'): Only 1 unique value, provides zero predictive variance.")

# Drop target and leakage/constant columns from features X
drop_cols = ['is_canceled'] + leakage_cols + constant_cols
X = df.drop(columns=drop_cols)
y = df['is_canceled']

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical Features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")

print("\n=== STEP 3: TRAIN / TEST SPLIT ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Train target distribution: {np.bincount(y_train) / len(y_train)}")
print(f"Test target distribution: {np.bincount(y_test) / len(y_test)}")

print("\n=== STEP 4 & 5: PREPROCESSING PIPELINES & MODEL TRAINING ===")

# Preprocessing for Logistic Regression (requires scaling for numeric features)
num_transformer_lr = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for Random Forest (tree-based models do not require scaling)
num_transformer_rf = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_lr = ColumnTransformer(transformers=[
    ('num', num_transformer_lr, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

preprocessor_rf = ColumnTransformer(transformers=[
    ('num', num_transformer_rf, numerical_cols),
    ('cat', cat_transformer, categorical_cols)
])

# Define models
lr_model = LogisticRegression(max_iter=1000, random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Full pipelines
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor_lr),
    ('classifier', lr_model)
])

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor_rf),
    ('classifier', rf_model)
])

print("Fitting Logistic Regression Pipeline...")
pipeline_lr.fit(X_train, y_train)

print("Fitting Random Forest Pipeline...")
pipeline_rf.fit(X_train, y_train)

print("\n=== STEP 6: EVALUATION ON TEST SET ===")

# Predictions & Probabilities
y_pred_lr = pipeline_lr.predict(X_test)
y_prob_lr = pipeline_lr.predict_proba(X_test)[:, 1]

y_pred_rf = pipeline_rf.predict(X_test)
y_prob_rf = pipeline_rf.predict_proba(X_test)[:, 1]

def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob)
    }

metrics_lr = get_metrics(y_test, y_pred_lr, y_prob_lr)
metrics_rf = get_metrics(y_test, y_pred_rf, y_prob_rf)

print("\n--- Logistic Regression Metrics ---")
for k, v in metrics_lr.items():
    print(f"{k}: {v:.4f}")
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))

print("\n--- Random Forest Metrics ---")
for k, v in metrics_rf.items():
    print(f"{k}: {v:.4f}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

print("\nConfusion Matrix - Logistic Regression:")
print(cm_lr)
print("\nConfusion Matrix - Random Forest:")
print(cm_rf)

print("\n=== STEP 7: GENERATING PLOTS & VISUALIZATIONS ===")

# Plot 1: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
axes[0].set_title('Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
            xticklabels=['Not Canceled', 'Canceled'], yticklabels=['Not Canceled', 'Canceled'])
axes[1].set_title('Random Forest - Confusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Plot 2: ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {metrics_lr["ROC-AUC"]:.4f})', color='#1f77b4', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {metrics_rf["ROC-AUC"]:.4f})', color='#2ca02c', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.5000)')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.savefig('roc_curves.png')
plt.close()

# Extract Feature Names from preprocessor
cat_encoder = pipeline_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
all_feature_names = numerical_cols + cat_feature_names

# Plot 3: Random Forest Feature Importances (Top 20)
rf_importances = pipeline_rf.named_steps['classifier'].feature_importances_
rf_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': rf_importances
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=rf_imp_df.head(20), palette='viridis')
plt.title('Random Forest - Top 20 Most Important Features', fontsize=13, fontweight='bold')
plt.xlabel('Feature Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()

# Plot 4: Logistic Regression Top Positive and Negative Coefficients
lr_coefs = pipeline_lr.named_steps['classifier'].coef_[0]
lr_coef_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': lr_coefs
}).sort_values('Coefficient', ascending=False)

top_pos_coef = lr_coef_df.head(10)
top_neg_coef = lr_coef_df.tail(10).iloc[::-1]
top_coefs = pd.concat([top_pos_coef, top_neg_coef])

plt.figure(figsize=(10, 8))
colors = ['#d62728' if c < 0 else '#1f77b4' for c in top_coefs['Coefficient']]
sns.barplot(x='Coefficient', y='Feature', data=top_coefs, palette=colors)
plt.title('Logistic Regression - Top Positive (Risk) & Negative (Protective) Coefficients', fontsize=12, fontweight='bold')
plt.xlabel('Coefficient Value (Log-Odds)')
plt.ylabel('Feature')
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig('lr_feature_coefficients.png')
plt.close()

print("Visualizations saved successfully!")

print("\n=== STEP 9: 5-FOLD CROSS-VALIDATION (ON TRAIN SET) ===")
cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring_metrics = ['accuracy', 'f1', 'roc_auc']

print("Running 5-Fold CV for Logistic Regression...")
cv_results_lr = cross_validate(pipeline_lr, X_train, y_train, cv=cv_strat, scoring=scoring_metrics, n_jobs=-1)

print("Running 5-Fold CV for Random Forest...")
cv_results_rf = cross_validate(pipeline_rf, X_train, y_train, cv=cv_strat, scoring=scoring_metrics, n_jobs=-1)

print("\n--- Cross-Validation Results ---")
print("Logistic Regression CV Results:")
print(f"Mean CV Accuracy: {cv_results_lr['test_accuracy'].mean():.4f} +/- {cv_results_lr['test_accuracy'].std():.4f}")
print(f"Mean CV F1-Score: {cv_results_lr['test_f1'].mean():.4f} +/- {cv_results_lr['test_f1'].std():.4f}")
print(f"Mean CV ROC-AUC:  {cv_results_lr['test_roc_auc'].mean():.4f} +/- {cv_results_lr['test_roc_auc'].std():.4f}")

print("\nRandom Forest CV Results:")
print(f"Mean CV Accuracy: {cv_results_rf['test_accuracy'].mean():.4f} +/- {cv_results_rf['test_accuracy'].std():.4f}")
print(f"Mean CV F1-Score: {cv_results_rf['test_f1'].mean():.4f} +/- {cv_results_rf['test_f1'].std():.4f}")
print(f"Mean CV ROC-AUC:  {cv_results_rf['test_roc_auc'].mean():.4f} +/- {cv_results_rf['test_roc_auc'].std():.4f}")

print("\n=== SAVE TRAINED PIPELINES WITH JOBLIB ===")
joblib.dump(pipeline_lr, 'logistic_regression_pipeline.joblib')
joblib.dump(pipeline_rf, 'random_forest_pipeline.joblib')
print("Saved models: 'logistic_regression_pipeline.joblib' and 'random_forest_pipeline.joblib'")

print("\nPipeline Execution Complete!")
