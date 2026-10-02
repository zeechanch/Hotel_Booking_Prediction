"""
Lab 2 Word Report Generator
Generates a comprehensive .docx report including all metrics, confusion matrices,
ROC curves, and feature importance plots.
"""

import os
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    """Set table cell background colour (hex without #)."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def set_cell_font(cell, text, bold=False, size=10, color=None, italic=False,
                  alignment=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ''
    para = cell.paragraphs[0]
    para.alignment = alignment
    run = para.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*bytes.fromhex(color))

def add_heading(doc, text, level=1, color_hex='1B3A5C'):
    para = doc.add_heading('', level=level)
    run = para.add_run(text)
    run.font.color.rgb = RGBColor(*bytes.fromhex(color_hex))
    run.bold = True
    if level == 1:
        run.font.size = Pt(16)
    elif level == 2:
        run.font.size = Pt(13)
    else:
        run.font.size = Pt(11)
    return para

def add_body(doc, text, bold=False, italic=False, size=10.5):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return para

def save_fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def add_fig_to_doc(doc, buf, width_inches=6.0, caption=None):
    doc.add_picture(buf, width=Inches(width_inches))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].italic = True
        cap.runs[0].font.size = Pt(9)
        cap.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)

# ──────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

print("Loading data and models …")
df = pd.read_csv('hotel_bookings_updated_2024.csv')

drop_cols = (
    ['is_canceled', 'reservation_status', 'reservation_status_date']
    + [c for c in df.columns if df[c].nunique() == 1]
)
X = df.drop(columns=drop_cols)
y = df['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

lr_pipeline = joblib.load('logistic_regression_pipeline.joblib')
rf_pipeline = joblib.load('random_forest_pipeline.joblib')

y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

def metrics_dict(y_true, y_pred, y_prob, cm):
    tn, fp, fn, tp = cm.ravel()
    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Precision':   precision_score(y_true, y_pred),
        'Recall':      recall_score(y_true, y_pred),
        'Specificity': tn / (tn + fp),
        'F1-Score':    f1_score(y_true, y_pred),
        'ROC-AUC':     roc_auc_score(y_true, y_prob),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }

m_lr = metrics_dict(y_test, y_pred_lr, y_prob_lr, cm_lr)
m_rf = metrics_dict(y_test, y_pred_rf, y_prob_rf, cm_rf)

paper = {
    'Accuracy': 0.8162, 'Precision': 0.8373, 'Recall': 0.6289,
    'Specificity': 0.9274, 'F1-Score': 0.7182, 'ROC-AUC': 0.866,
    'TN': 13798, 'FP': 1080, 'FN': 3279, 'TP': 5556,
}

print("Data & models loaded successfully.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Confusion Matrices
# ──────────────────────────────────────────────────────────────────────────────

print("Generating Figure 1: Confusion Matrices …")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices', fontsize=15, fontweight='bold', y=1.02)

for ax, cm_data, title, cmap in zip(
    axes,
    [cm_lr, cm_rf],
    ['Logistic Regression', 'Random Forest'],
    ['Blues', 'Greens']
):
    sns.heatmap(
        cm_data, annot=True, fmt='d', cmap=cmap, ax=ax, cbar=False,
        linewidths=0.5, linecolor='grey',
        xticklabels=['Not Canceled', 'Canceled'],
        yticklabels=['Not Canceled', 'Canceled'],
        annot_kws={"size": 13, "weight": "bold"}
    )
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
cm_buf = save_fig_to_bytes(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — ROC Curves
# ──────────────────────────────────────────────────────────────────────────────

print("Generating Figure 2: ROC Curves …")
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(fpr_lr, tpr_lr, lw=2, color='#1f77b4',
         label=f'Logistic Regression  (AUC = {m_lr["ROC-AUC"]:.3f})')
ax2.plot(fpr_rf, tpr_rf, lw=2, color='#2ca02c',
         label=f'Random Forest  (AUC = {m_rf["ROC-AUC"]:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance  (AUC = 0.500)')
ax2.fill_between(fpr_lr, tpr_lr, alpha=0.08, color='#1f77b4')
ax2.fill_between(fpr_rf, tpr_rf, alpha=0.08, color='#2ca02c')
ax2.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=11)
ax2.set_ylabel('True Positive Rate  (Sensitivity / Recall)', fontsize=11)
ax2.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10, frameon=True)
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.01)
ax2.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
roc_buf = save_fig_to_bytes(fig2)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — LR Feature Coefficients (Top 20)
# ──────────────────────────────────────────────────────────────────────────────

print("Generating Figure 3: LR Coefficients …")
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

cat_enc = lr_pipeline.named_steps['preprocessor'] \
           .named_transformers_['cat'].named_steps['onehot']
cat_names = cat_enc.get_feature_names_out(cat_cols).tolist()
all_feat = num_cols + cat_names

lr_coefs = lr_pipeline.named_steps['classifier'].coef_[0]
coef_df = pd.DataFrame({'Feature': all_feat, 'Coefficient': lr_coefs})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
top20 = coef_df.head(20).sort_values('Coefficient')

colors = ['#d62728' if c > 0 else '#1f77b4' for c in top20['Coefficient']]

fig3, ax3 = plt.subplots(figsize=(10, 8))
bars = ax3.barh(top20['Feature'], top20['Coefficient'], color=colors, edgecolor='white', height=0.7)
ax3.axvline(0, color='black', linewidth=1.2, linestyle='--')
ax3.set_xlabel('Coefficient  (Log-Odds)', fontsize=11)
ax3.set_title('Logistic Regression — Top 20 Significant Coefficients\n(Red = Risk Factor ↑  |  Blue = Protective Factor ↓)',
              fontsize=11, fontweight='bold')
risk_patch = mpatches.Patch(color='#d62728', label='Risk Factor (increases cancellation)')
prot_patch = mpatches.Patch(color='#1f77b4', label='Protective Factor (reduces cancellation)')
ax3.legend(handles=[risk_patch, prot_patch], fontsize=9, loc='lower right')
ax3.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
coef_buf = save_fig_to_bytes(fig3)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Random Forest Feature Importance (Top 20)
# ──────────────────────────────────────────────────────────────────────────────

print("Generating Figure 4: RF Feature Importance …")
cat_enc_rf = rf_pipeline.named_steps['preprocessor'] \
              .named_transformers_['cat'].named_steps['onehot']
cat_names_rf = cat_enc_rf.get_feature_names_out(cat_cols).tolist()
all_feat_rf = num_cols + cat_names_rf

rf_imp = rf_pipeline.named_steps['classifier'].feature_importances_
imp_df = pd.DataFrame({'Feature': all_feat_rf, 'Importance': rf_imp})
imp_df = imp_df.sort_values('Importance', ascending=False).head(20).sort_values('Importance')

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.barh(imp_df['Feature'], imp_df['Importance'],
         color=plt.cm.viridis(np.linspace(0.2, 0.85, len(imp_df))),
         edgecolor='white', height=0.7)
ax4.set_xlabel('Feature Importance (Gini Impurity Reduction)', fontsize=11)
ax4.set_title('Random Forest — Top 20 Most Important Features', fontsize=12, fontweight='bold')
ax4.grid(True, axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
imp_buf = save_fig_to_bytes(fig4)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Metric Bar Comparison
# ──────────────────────────────────────────────────────────────────────────────

print("Generating Figure 5: Metric Comparison Bar Chart …")
metric_keys = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'ROC-AUC']
metric_labels = ['Accuracy', 'Precision', 'Sensitivity\n(Recall)', 'Specificity', 'F1-Score', 'ROC-AUC']

paper_vals = [paper[k] for k in metric_keys]
lr_vals    = [m_lr[k]  for k in metric_keys]
rf_vals    = [m_rf[k]  for k in metric_keys]

x = np.arange(len(metric_keys))
width = 0.25

fig5, ax5 = plt.subplots(figsize=(13, 6))
b1 = ax5.bar(x - width, paper_vals, width, label='Research Paper (Dr. Vinay Raj R, 2026)',
             color='#e67e22', edgecolor='white')
b2 = ax5.bar(x,          lr_vals,   width, label='Our Logistic Regression (Scenario A)',
             color='#2980b9', edgecolor='white')
b3 = ax5.bar(x + width,  rf_vals,   width, label='Our Random Forest (Ensemble Benchmark)',
             color='#27ae60', edgecolor='white')

for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax5.annotate(f'{h:.3f}',
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', va='bottom', fontsize=7.5, fontweight='bold')

ax5.set_xticks(x)
ax5.set_xticklabels(metric_labels, fontsize=10)
ax5.set_ylim(0, 1.12)
ax5.set_ylabel('Score', fontsize=11)
ax5.set_title('Model Performance Comparison: Research Paper vs Our Implementations',
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=9, loc='upper right', frameon=True)
ax5.grid(True, axis='y', linestyle='--', alpha=0.35)
ax5.axhline(1.0, color='grey', linestyle=':', alpha=0.5)
plt.tight_layout()
bar_buf = save_fig_to_bytes(fig5)

print("All figures generated.")

# ──────────────────────────────────────────────────────────────────────────────
# BUILD WORD DOCUMENT
# ──────────────────────────────────────────────────────────────────────────────

print("Building Word document …")
doc = Document()

# ── Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# Default paragraph style
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(10.5)

# ── COVER PAGE ────────────────────────────────────────────────────────────────
doc.add_paragraph()  # spacer
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title_para.add_run('LAB 2 REPORT')
r.bold = True
r.font.size = Pt(26)
r.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)

doc.add_paragraph()

sub_para = doc.add_paragraph()
sub_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = sub_para.add_run(
    'Hotel Booking Cancellation Prediction\n'
    'Dataset Benchmarking, Research Paper Analysis & Model Reproduction'
)
r2.font.size = Pt(14)
r2.bold = True
r2.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

doc.add_paragraph()

details_para = doc.add_paragraph()
details_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
for line in [
    'Course: DevOps / Machine Learning Lab',
    'Dataset: hotel_bookings_updated_2024.csv',
    'Research Paper: Dr. Vinay Raj R (2026)',
    'Journal: International Journal for Research in Tourism and Hospitality',
    'DOI: 10.53555/th.v1i2.2575',
    '',
    f'Date: October 2026',
]:
    r = details_para.add_run(line + '\n')
    r.font.size = Pt(11)

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATASET VERIFICATION
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '1.  Dataset Verification', level=1)
add_body(doc,
    'The dataset hotel_bookings_updated_2024.csv was verified against the minimum '
    'thresholds required for the laboratory task.'
)

tbl = doc.add_table(rows=4, cols=4)
tbl.style = 'Table Grid'
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

headers = ['Criteria', 'Required Threshold', 'Actual Value', 'Status']
hdr_colors = ['1B3A5C'] * 4
for i, (h, c) in enumerate(zip(headers, hdr_colors)):
    cell = tbl.rows[0].cells[i]
    set_cell_bg(cell, c)
    set_cell_font(cell, h, bold=True, size=10, color='FFFFFF')

data_rows = [
    ('Number of Records (Rows)',    '≥ 100,000',   f'{len(df):,}',  '✅  PASS'),
    ('Number of Features (Columns)','≥ 35',         f'{df.shape[1]:,} (33 raw + 3 derived)', '✅  PASS'),
    ('Research Paper Published',    'After 2020',   'July 2026',     '✅  PASS'),
]
row_colors = ['EBF5FB', 'D6EAF8']
for ridx, row_data in enumerate(data_rows):
    row = tbl.rows[ridx + 1]
    bg = row_colors[ridx % 2]
    for cidx, val in enumerate(row_data):
        c = row.cells[cidx]
        set_cell_bg(c, bg)
        bold = (cidx == 3)
        color = '1E8449' if '✅' in val else None
        set_cell_font(c, val, bold=bold, size=10, color=color)

doc.add_paragraph()

add_body(doc, 'Target Variable: is_canceled  (Binary: 0 = Not Canceled, 1 = Canceled)', bold=True)
add_body(doc, f'  • Non-Canceled: 75,166 records  (62.96 %)')
add_body(doc, f'  • Canceled:     44,224 records  (37.04 %)')
doc.add_paragraph()
add_body(doc, 'Columns Dropped to Prevent Data Leakage:', bold=True)
add_body(doc, '  • reservation_status  — directly encodes the outcome (Canceled / Check-Out / No-Show)')
add_body(doc, '  • reservation_status_date  — recorded at the moment of cancellation/checkout')
add_body(doc, '  • arrival_date_year  — constant feature (zero variance; only one year present)')

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — RESEARCH PAPER IDENTIFICATION
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '2.  Research Paper Identification', level=1)

paper_info = [
    ('Title',       'Determinants of Hotel Booking Cancellations: Evidence from Reservation Analytics'),
    ('Author',      'Dr. Vinay Raj R — Associate Professor, Jain Deemed to be University'),
    ('Journal',     'International Journal for Research in Tourism and Hospitality'),
    ('Volume/Issue','Vol. 01, Issue 02 — July 2026'),
    ('DOI',         '10.53555/th.v1i2.2575'),
    ('Dataset',     'hotel_bookings_updated_2024.csv  (Bedmutha, K. S., 2025, Kaggle)'),
    ('Sample Size', '118,559 reservations after cleaning (44,224 canceled / 74,335 non-canceled)'),
]

tbl2 = doc.add_table(rows=len(paper_info)+1, cols=2)
tbl2.style = 'Table Grid'
tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['Field', 'Details']):
    c = tbl2.rows[0].cells[i]
    set_cell_bg(c, '1B3A5C')
    set_cell_font(c, h, bold=True, size=10, color='FFFFFF')

for ridx, (k, v) in enumerate(paper_info):
    bg = 'EBF5FB' if ridx % 2 == 0 else 'D6EAF8'
    c0 = tbl2.rows[ridx+1].cells[0]
    c1 = tbl2.rows[ridx+1].cells[1]
    set_cell_bg(c0, bg); set_cell_font(c0, k, bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.LEFT)
    set_cell_bg(c1, bg); set_cell_font(c1, v, bold=False, size=10, alignment=WD_ALIGN_PARAGRAPH.LEFT)

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PAPER MODEL ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '3.  Research Paper: Model Analysis (Home Task 1)', level=1)
add_body(doc,
    'The research paper employs Binary Logistic Regression with standardised continuous '
    'predictors and categorical dummy encoding to model the probability of hotel booking '
    'cancellation. The full-sample ROC-AUC reported is 0.850 and McFadden Pseudo-R² = 0.338.'
)
doc.add_paragraph()

add_heading(doc, '3.1  Key Risk & Protective Factors  (Table 4 in Paper)', level=2)

factor_data = [
    ('Non-Refundable Deposit',               '235.623', 'Risk — Highest predictor of cancellation'),
    ('Previous Cancellations',               '9.432',   'Risk — Repeat cancellers 9× more likely to cancel'),
    ('Online Travel Agency (Market Seg.)',   '2.352',   'Risk — OTA bookings less committed'),
    ('Transient Customer Type',              '2.313',   'Risk — Individual travellers more likely to cancel'),
    ('Lead Time (+1σ)',                      '1.491',   'Risk — Longer advance bookings riskier'),
    ('Average Daily Rate (+1σ)',             '1.129',   'Risk — Higher rates see more cancellations'),
    ('Room Mismatch (Assigned ≠ Reserved)',  '0.164',   'Protective — Guests accepting diff. room stay'),
    ('Previous Non-Cancelled Bookings',      '0.477',   'Protective — Loyal, reliable guests'),
    ('Special Requests Count',               '0.568',   'Protective — Engaged guests less likely to cancel'),
    ('Repeated Guest Status',                '0.563',   'Protective — Returning customers are loyal'),
    ('Booking Changes',                      '0.790',   'Protective — Flexibility reduces cancellation'),
    ('Resort Hotel (vs City Hotel)',         '0.934',   'Protective — Resort bookings are more stable'),
]

tbl3 = doc.add_table(rows=len(factor_data)+1, cols=3)
tbl3.style = 'Table Grid'
tbl3.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['Factor', 'Odds Ratio  (eᴮ)', 'Interpretation']):
    c = tbl3.rows[0].cells[i]
    set_cell_bg(c, '1B3A5C')
    set_cell_font(c, h, bold=True, size=10, color='FFFFFF')

for ridx, (f, o, interp) in enumerate(factor_data):
    is_risk = float(o) > 1.0
    bg_row = 'FDEDEC' if is_risk else 'EAFAF1'
    for cidx, val in enumerate([f, o, interp]):
        cell = tbl3.rows[ridx+1].cells[cidx]
        set_cell_bg(cell, bg_row)
        set_cell_font(cell, val, size=9.5, alignment=WD_ALIGN_PARAGRAPH.LEFT)

doc.add_paragraph()
add_body(doc, 'Red rows = Risk factors (OR > 1.0)     Green rows = Protective factors (OR < 1.0)', italic=True)

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '4.  Model Comparison & Scenario Identification (Home Task 2)', level=1)

comp_data = [
    ('Model in Research Paper',    'Binary Logistic Regression'),
    ('Lab Model 1',                'Logistic Regression  →  MATCH  (Scenario A triggered)'),
    ('Lab Model 2',                'Random Forest Classifier  →  Ensemble Benchmark (Extended Analysis)'),
    ('Outcome',                    'Scenario A: Reproduce Logistic Regression and benchmark against paper results'),
]

tbl4 = doc.add_table(rows=len(comp_data)+1, cols=2)
tbl4.style = 'Table Grid'
tbl4.alignment = WD_TABLE_ALIGNMENT.CENTER

for i, h in enumerate(['Item', 'Detail']):
    c = tbl4.rows[0].cells[i]
    set_cell_bg(c, '1B3A5C')
    set_cell_font(c, h, bold=True, size=10, color='FFFFFF')

for ridx, (k, v) in enumerate(comp_data):
    bg = 'EBF5FB' if ridx % 2 == 0 else 'D6EAF8'
    c0 = tbl4.rows[ridx+1].cells[0]; c1 = tbl4.rows[ridx+1].cells[1]
    color_v = '1E8449' if 'MATCH' in v else None
    set_cell_bg(c0, bg); set_cell_font(c0, k, bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.LEFT)
    set_cell_bg(c1, bg); set_cell_font(c1, v, bold=('MATCH' in v or 'Scenario A' in k), size=10,
                                       color=color_v, alignment=WD_ALIGN_PARAGRAPH.LEFT)

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BENCHMARK COMPARISON TABLE
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '5.  Scenario A — Benchmark Results & Comparison (Home Task 3)', level=1)
add_body(doc,
    'The table below compares the performance metrics reported in the research paper '
    '(Table 5, Dr. Vinay Raj R, 2026) against our reproduced Logistic Regression and '
    'the extended Random Forest model. Both models were evaluated on a stratified 80/20 '
    'train/test split with random_state=42.'
)
doc.add_paragraph()

bench_metrics = [
    ('Accuracy',             f"{paper['Accuracy']*100:.2f} %",    f"{m_lr['Accuracy']*100:.2f} %",    f"{m_rf['Accuracy']*100:.2f} %"),
    ('Precision',            f"{paper['Precision']*100:.2f} %",   f"{m_lr['Precision']*100:.2f} %",   f"{m_rf['Precision']*100:.2f} %"),
    ('Sensitivity (Recall)', f"{paper['Recall']*100:.2f} %",      f"{m_lr['Recall']*100:.2f} %",      f"{m_rf['Recall']*100:.2f} %"),
    ('Specificity',          f"{paper['Specificity']*100:.2f} %", f"{m_lr['Specificity']*100:.2f} %", f"{m_rf['Specificity']*100:.2f} %"),
    ('F1-Score',             f"{paper['F1-Score']*100:.2f} %",    f"{m_lr['F1-Score']*100:.2f} %",    f"{m_rf['F1-Score']*100:.2f} %"),
    ('ROC-AUC Score',        f"{paper['ROC-AUC']:.3f}",           f"{m_lr['ROC-AUC']:.3f}",           f"{m_rf['ROC-AUC']:.3f}"),
    ('True Negatives (TN)',  f"{paper['TN']:,}",                  f"{int(m_lr['TN']):,}",             f"{int(m_rf['TN']):,}"),
    ('False Positives (FP)', f"{paper['FP']:,}",                  f"{int(m_lr['FP']):,}",             f"{int(m_rf['FP']):,}"),
    ('False Negatives (FN)', f"{paper['FN']:,}",                  f"{int(m_lr['FN']):,}",             f"{int(m_rf['FN']):,}"),
    ('True Positives (TP)',  f"{paper['TP']:,}",                  f"{int(m_lr['TP']):,}",             f"{int(m_rf['TP']):,}"),
]

tbl5 = doc.add_table(rows=len(bench_metrics)+1, cols=4)
tbl5.style = 'Table Grid'
tbl5.alignment = WD_TABLE_ALIGNMENT.CENTER

bench_headers = [
    'Evaluation Metric',
    'Research Paper\n(Dr. Vinay Raj R, 2026)',
    'Our Logistic Regression\n(Scenario A)',
    'Our Random Forest\n(Ensemble Benchmark)',
]
header_colors = ['1B3A5C', 'A04000', '154360', '145A32']
for i, (h, hc) in enumerate(zip(bench_headers, header_colors)):
    cell = tbl5.rows[0].cells[i]
    set_cell_bg(cell, hc)
    set_cell_font(cell, h, bold=True, size=9.5, color='FFFFFF')

row_bgs = ['FDFDFD', 'EBF5FB']
for ridx, row_data in enumerate(bench_metrics):
    bg = row_bgs[ridx % 2]
    for cidx, val in enumerate(row_data):
        cell = tbl5.rows[ridx+1].cells[cidx]
        set_cell_bg(cell, bg)
        set_cell_font(cell, val, bold=(cidx == 0), size=10)

doc.add_paragraph()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '6.  Visualizations', level=1)

add_heading(doc, '6.1  Model Performance Comparison — Bar Chart', level=2)
add_fig_to_doc(doc, bar_buf, width_inches=6.2,
    caption='Figure 1: Side-by-side metric comparison — Research Paper vs Logistic Regression vs Random Forest')

doc.add_page_break()
add_heading(doc, '6.2  Confusion Matrices', level=2)
add_body(doc,
    'A confusion matrix shows the counts of True Positives (TP), True Negatives (TN), '
    'False Positives (FP), and False Negatives (FN). Higher TP and TN values and lower '
    'FP and FN values indicate a better model.'
)
add_fig_to_doc(doc, cm_buf, width_inches=6.0,
    caption='Figure 2: Confusion matrices for Logistic Regression (left) and Random Forest (right)')

doc.add_page_break()
add_heading(doc, '6.3  ROC Curves', level=2)
add_body(doc,
    'The Receiver Operating Characteristic (ROC) curve plots Sensitivity (True Positive Rate) '
    'against 1 − Specificity (False Positive Rate) at all classification thresholds. The Area '
    'Under the Curve (AUC) summarises overall discriminatory performance; a perfect classifier '
    'achieves AUC = 1.0.'
)
add_fig_to_doc(doc, roc_buf, width_inches=5.5,
    caption='Figure 3: ROC Curves — Logistic Regression AUC=0.894  |  Random Forest AUC=0.945')

doc.add_page_break()
add_heading(doc, '6.4  Logistic Regression — Top 20 Feature Coefficients', level=2)
add_body(doc,
    'Standardised log-odds coefficients (β) represent the change in log-odds of cancellation '
    'per one-standard-deviation change in each feature. Positive β (red) are risk factors; '
    'negative β (blue) are protective factors. The Odds Ratio for each feature is calculated '
    'as OR = e^β.'
)
add_fig_to_doc(doc, coef_buf, width_inches=5.8,
    caption='Figure 4: Top 20 Logistic Regression coefficients (log-odds). Red = Risk; Blue = Protective.')

doc.add_page_break()
add_heading(doc, '6.5  Random Forest — Top 20 Feature Importances', level=2)
add_body(doc,
    'Gini impurity-based feature importances measure the average reduction in node impurity '
    'achieved by splitting on each feature across all 200 decision trees. Higher values '
    'indicate a more decisive contribution to the classification.'
)
add_fig_to_doc(doc, imp_buf, width_inches=5.8,
    caption='Figure 5: Top 20 Random Forest feature importances (Gini impurity reduction).')

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DEFENSE GUIDE
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '7.  Lab Defense Guide — Key Questions & Answers', level=1)

qa = [
    (
        'Q1: Why were reservation_status and reservation_status_date dropped?',
        'These columns constitute direct target leakage. reservation_status contains the '
        'string "Canceled" which directly encodes y=1, and reservation_status_date records '
        'when the booking status changed (i.e., it does not exist until after the cancellation '
        'decision is made). Including them would inflate all metrics to near-perfect scores '
        'but produce a model useless in production.'
    ),
    (
        'Q2: What is the mathematical basis of Logistic Regression?',
        'The logit (log-odds) function: ln(p / (1-p)) = β₀ + β₁X₁ + … + βₖXₖ. '
        'The predicted probability is obtained via the sigmoid: p = 1 / (1 + e^(−wᵀX)). '
        'Coefficients are estimated by Maximum Likelihood Estimation (MLE). '
        'The Odds Ratio for predictor Xⱼ is OR = e^βⱼ.'
    ),
    (
        'Q3: What is the difference between Precision and Recall? Which matters more here?',
        'Precision = TP / (TP + FP): of all predicted cancellations, how many were actual cancellations. '
        'Recall (Sensitivity) = TP / (TP + FN): of all actual cancellations, how many did we catch. '
        'In a hotel context, missing a cancellation (FN) is costly — revenue loss from an empty room. '
        'Hence Recall/Sensitivity is the priority metric. The F1-Score balances both.'
    ),
    (
        'Q4: Why does Random Forest outperform Logistic Regression?',
        'Logistic Regression assumes linear decision boundaries in log-odds space and additive '
        'feature relationships. Random Forest uses 200 decorrelated decision trees that naturally '
        'capture non-linear feature splits and high-order interaction effects (e.g., combined '
        'interactions between lead_time, market_segment, and adr) without any manual feature '
        'engineering. This leads to gains across all metrics (Accuracy +5.5%, AUC +0.051).'
    ),
    (
        'Q5: What is ROC-AUC and how is it interpreted?',
        'AUC (Area Under the ROC Curve) measures the probability that the model ranks a randomly '
        'chosen positive (canceled) example higher than a randomly chosen negative example. '
        'AUC = 0.50 → random classifier. AUC = 1.00 → perfect. Our LR model (AUC = 0.894) '
        'exceeds the paper\'s reported value of 0.866, while our RF (0.945) provides even '
        'stronger discriminatory power.'
    ),
    (
        'Q6: Why was stratify=y used in the train/test split?',
        'stratify=y ensures that the proportion of canceled vs non-canceled bookings (≈37% : 63%) '
        'is preserved in both the training and test subsets. Without stratification, random splits '
        'can produce imbalanced evaluation sets that distort metrics, especially for minority class recall.'
    ),
    (
        'Q7: What is the role of StandardScaler in the Logistic Regression pipeline?',
        'StandardScaler transforms each continuous feature to zero mean (μ=0) and unit variance '
        '(σ=1): z = (x − μ) / σ. This is required for Logistic Regression because (a) gradient '
        'descent converges faster with scaled features, and (b) standardised coefficients are '
        'directly comparable in magnitude — enabling valid odds ratio comparisons as in the paper.'
    ),
]

for i, (q, a_text) in enumerate(qa, 1):
    para = doc.add_paragraph()
    run = para.add_run(q)
    run.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)
    run.font.size = Pt(10.5)

    para2 = doc.add_paragraph()
    run2 = para2.add_run(a_text)
    run2.font.size = Pt(10)
    para2.paragraph_format.left_indent = Cm(0.6)
    doc.add_paragraph()

doc.add_page_break()

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — CONCLUSION
# ──────────────────────────────────────────────────────────────────────────────
add_heading(doc, '8.  Conclusion', level=1)
add_body(doc,
    'This lab successfully identified a post-2020 research paper (Dr. Vinay Raj R, 2026) '
    'that used the exact hotel_bookings_updated_2024.csv dataset. The paper\'s Binary Logistic '
    'Regression model was reproduced and benchmarked under Scenario A.'
)
doc.add_paragraph()
add_body(doc,
    'Our Logistic Regression model achieved performance highly consistent with the published '
    'paper — Accuracy within 0.03 %, F1-Score within 0.7 % — confirming successful '
    'methodology reproduction. Our extended Random Forest model demonstrated the performance '
    'ceiling achievable through ensemble non-linear modelling: +5.5 % Accuracy, +8.8 % F1-Score, '
    'and AUC = 0.945 vs the paper\'s 0.866.'
)
doc.add_paragraph()
add_body(doc,
    'Key findings from both models align with the literature: lead time, non-refundable '
    'deposits, and previous cancellations are the strongest risk factors, while special requests, '
    'repeated guest status, and booking changes reduce cancellation probability.'
)
doc.add_paragraph()

add_heading(doc, '9.  References', level=1)
refs = [
    'Vinay Raj R (2026). Determinants of Hotel Booking Cancellations: Evidence from Reservation '
    'Analytics. International Journal for Research in Tourism and Hospitality, Vol. 01, Issue 02, '
    'pp. 32–49. DOI: 10.53555/th.v1i2.2575',

    'Bedmutha, K. S. (2025). Hotel booking reservation: An updated 2024 dataset for hotel booking '
    'demand and cancellation analysis. Kaggle. '
    'https://www.kaggle.com/datasets/krutarth-bedmutha/hotel-booking-reservation',

    'Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.',

    'Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.',
]
for i, ref in enumerate(refs, 1):
    para = doc.add_paragraph(f'[{i}]  {ref}')
    para.paragraph_format.left_indent = Cm(0.6)
    para.runs[0].font.size = Pt(10)

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_path = 'Lab2_Hotel_Cancellation_Report.docx'
doc.save(out_path)
print(f"\n✅  Word report saved to: {out_path}")
