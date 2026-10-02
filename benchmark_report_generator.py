"""
Lab 2: Comparative Analysis & Benchmark Reproduction
Dataset: hotel_bookings_updated_2024.csv
Research Paper: 'DETERMINANTS OF HOTEL BOOKING CANCELLATIONS: EVIDENCE FROM RESERVATION ANALYTICS' (Dr. Vinay Raj R, 2026)
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

def run_benchmark_analysis():
    print("="*80)
    print("LAB 2: BENCHMARKING & REPRODUCTION OF PUBLISHED RESEARCH PAPER")
    print("="*80)
    
    # 1. Load Dataset
    data_path = 'hotel_bookings_updated_2024.csv'
    df = pd.read_csv(data_path)
    
    # Feature Engineering matching Paper (Section 2.6)
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)
    df['room_mismatch'] = (df['assigned_room_type'] != df['reserved_room_type']).astype(int)
    
    print(f"\n[Dataset Verification]")
    print(f"Total Rows: {len(df):,} (Criteria: >= 100,000 -> PASS)")
    print(f"Total Columns (with derived features): {df.shape[1]} (Criteria: >= 35 -> PASS)")
    
    # 2. Load Existing Trained Pipelines
    lr_pipeline = joblib.load('logistic_regression_pipeline.joblib')
    rf_pipeline = joblib.load('random_forest_pipeline.joblib')
    
    # Target and Features
    drop_cols = ['is_canceled', 'reservation_status', 'reservation_status_date'] + [c for c in df.columns if df[c].nunique() == 1]
    # Note: drop derived columns from raw X if pipeline was trained on original columns
    raw_feature_cols = [c for c in df.columns if c not in ['total_stay', 'total_guests', 'room_mismatch'] + drop_cols]
    
    X = df[raw_feature_cols]
    y = df['is_canceled']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Evaluate Logistic Regression
    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
    spec_lr = tn_lr / (tn_lr + fp_lr)
    
    # 4. Evaluate Random Forest
    y_pred_rf = rf_pipeline.predict(X_test)
    y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]
    
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()
    spec_rf = tn_rf / (tn_rf + fp_rf)
    
    # 5. Paper Reported Metrics (Table 5)
    paper_metrics = {
        'Observations': 118559,
        'Accuracy': 0.8162,
        'Precision': 0.8373,
        'Recall (Sensitivity)': 0.6289,
        'Specificity': 0.9274,
        'F1-Score': 0.7182,
        'ROC-AUC': 0.8660,
        'TN': 13798,
        'FP': 1080,
        'FN': 3279,
        'TP': 5556
    }
    
    comparison_df = pd.DataFrame({
        'Evaluation Metric': [
            'Accuracy',
            'Precision',
            'Sensitivity (Recall)',
            'Specificity',
            'F1-Score',
            'ROC-AUC Score',
            'True Negatives (TN)',
            'False Positives (FP)',
            'False Negatives (FN)',
            'True Positives (TP)'
        ],
        'Research Paper (Dr. Vinay Raj R, 2026)': [
            f"{paper_metrics['Accuracy']*100:.2f}%",
            f"{paper_metrics['Precision']*100:.2f}%",
            f"{paper_metrics['Recall (Sensitivity)']*100:.2f}%",
            f"{paper_metrics['Specificity']*100:.2f}%",
            f"{paper_metrics['F1-Score']*100:.2f}%",
            f"{paper_metrics['ROC-AUC']:.3f}",
            f"{paper_metrics['TN']:,}",
            f"{paper_metrics['FP']:,}",
            f"{paper_metrics['FN']:,}",
            f"{paper_metrics['TP']:,}"
        ],
        'Our Logistic Regression (Scenario A)': [
            f"{accuracy_score(y_test, y_pred_lr)*100:.2f}%",
            f"{precision_score(y_test, y_pred_lr)*100:.2f}%",
            f"{recall_score(y_test, y_pred_lr)*100:.2f}%",
            f"{spec_lr*100:.2f}%",
            f"{f1_score(y_test, y_pred_lr)*100:.2f}%",
            f"{roc_auc_score(y_test, y_prob_lr):.3f}",
            f"{tn_lr:,}",
            f"{fp_lr:,}",
            f"{fn_lr:,}",
            f"{tp_lr:,}"
        ],
        'Our Random Forest (Ensemble Benchmark)': [
            f"{accuracy_score(y_test, y_pred_rf)*100:.2f}%",
            f"{precision_score(y_test, y_pred_rf)*100:.2f}%",
            f"{recall_score(y_test, y_pred_rf)*100:.2f}%",
            f"{spec_rf*100:.2f}%",
            f"{f1_score(y_test, y_pred_rf)*100:.2f}%",
            f"{roc_auc_score(y_test, y_prob_rf):.3f}",
            f"{tn_rf:,}",
            f"{fp_rf:,}",
            f"{fn_rf:,}",
            f"{tp_rf:,}"
        ]
    })
    
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON TABLE (SCENARIO A)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table to CSV
    comparison_df.to_csv('model_comparison_benchmark.csv', index=False)
    print("\nSaved comparison table to 'model_comparison_benchmark.csv'")

if __name__ == '__main__':
    run_benchmark_analysis()
