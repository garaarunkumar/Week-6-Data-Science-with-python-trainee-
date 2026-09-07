"""
Week 6 Capstone Project — Employee Attrition: A Full Data Science Pipeline
Dataset: HR Analytics.xlsx (cleaned, continued from Weeks 1-4)

This script combines every stage of the internship into one integrated
pipeline:
  1. Data acquisition & cleaning (Week 1)
  2. Exploratory Data Analysis (Week 2)
  3. Unsupervised learning — employee segmentation via K-Means (Week 3 skills)
  4. Supervised learning — attrition prediction via classification (Week 4)
  5. Combined insight: linking employee segments to attrition risk
  6. Final evaluation, business recommendations, and reflection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, roc_auc_score, roc_curve)

sns.set_style('whitegrid')

# ===========================================================
# STAGE 1: DATA ACQUISITION AND CLEANING (recap of Week 1)
# ===========================================================
df = pd.read_csv('HR_Analytics_analysis_ready.csv')
print("=== Stage 1: Data ===")
print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# ===========================================================
# STAGE 2: EXPLORATORY DATA ANALYSIS (recap of Week 2)
# ===========================================================
print("\n=== Stage 2: EDA ===")
attrition_rate = (df['Attrition'] == 'Yes').mean() * 100
print(f"Overall attrition rate: {attrition_rate:.2f}%")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
sns.barplot(x=df.groupby('OverTime')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100).index,
            y=df.groupby('OverTime')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100).values,
            ax=axes[0], hue=df.groupby('OverTime')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100).index,
            palette=['#55A868', '#C44E52'], legend=False)
axes[0].set_title('Attrition by Overtime')
axes[0].set_ylabel('Attrition Rate (%)')

order = ['18-25', '26-35', '36-45', '46-55', '55+']
ag = df.groupby('AgeGroup')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100).reindex(order)
sns.barplot(x=ag.index, y=ag.values, ax=axes[1], hue=ag.index, palette='rocket', legend=False)
axes[1].set_title('Attrition by Age Group')
axes[1].set_ylabel('Attrition Rate (%)')
axes[1].tick_params(axis='x', rotation=30)

dept = df.groupby('Department')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100).sort_values(ascending=False)
sns.barplot(x=dept.index, y=dept.values, ax=axes[2], hue=dept.index, palette='viridis', legend=False)
axes[2].set_title('Attrition by Department')
axes[2].set_ylabel('Attrition Rate (%)')
axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('charts/00_eda_recap.png', dpi=140)
plt.close()

# ===========================================================
# STAGE 3: UNSUPERVISED LEARNING — Employee Segmentation
# ===========================================================
print("\n=== Stage 3: Unsupervised Learning (K-Means Segmentation) ===")
cluster_features = ['Age', 'MonthlyIncome', 'DistanceFromHome (km)', 'JobSatisfaction',
                     'EnvironmentSatisfaction', 'NumCompaniesWorked', 'PercentSalaryHike',
                     'JobInvolvement']
X_cluster = df[cluster_features].copy()
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

sil_scores = {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    sil_scores[k] = silhouette_score(X_cluster_scaled, labels)
print("Silhouette scores by k:", {k: round(v, 3) for k, v in sil_scores.items()})

K = 4
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(X_cluster_scaled)
print(f"Final K={K}, silhouette = {silhouette_score(X_cluster_scaled, df['Segment']):.3f}")

seg_profile = df.groupby('Segment')[cluster_features].mean().round(1)
seg_profile['Count'] = df['Segment'].value_counts().sort_index()
seg_profile['AttritionRate%'] = df.groupby('Segment')['Attrition'].apply(lambda s: (s == 'Yes').mean() * 100).round(2)
print("\nSegment profiles:\n", seg_profile)

plt.figure(figsize=(8, 4.5))
sns.heatmap(seg_profile[cluster_features], annot=True, fmt='.1f', cmap='coolwarm', center=0)
plt.title('Employee Segment Profiles (Average Feature Values)')
plt.ylabel('Segment')
plt.tight_layout()
plt.savefig('charts/01_segment_profiles.png', dpi=140)
plt.close()

plt.figure(figsize=(6, 4.5))
attr_by_seg = seg_profile['AttritionRate%']
ax = sns.barplot(x=attr_by_seg.index, y=attr_by_seg.values, hue=attr_by_seg.index, palette='rocket', legend=False)
plt.title('Attrition Rate by Employee Segment')
plt.xlabel('Segment')
plt.ylabel('Attrition Rate (%)')
for i, v in enumerate(attr_by_seg.values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('charts/02_attrition_by_segment.png', dpi=140)
plt.close()

# ===========================================================
# STAGE 4: SUPERVISED LEARNING — Attrition Prediction
# ===========================================================
print("\n=== Stage 4: Supervised Learning (Classification) ===")
y = (df['Attrition'] == 'Yes').astype(int)
drop_cols = ['EmployeeID_Code', 'EmployeeNumber', 'Attrition', 'AgeGroup', 'Over18',
             'EmployeeCount', 'StandardHours', 'SalarySlab', 'Salary Hike', 'Segment']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(exclude='object').columns.tolist()
X_enc = X.copy()
for c in cat_cols:
    X_enc[c] = LabelEncoder().fit_transform(X_enc[c].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.25,
                                                      random_state=42, stratify=y)
scaler2 = StandardScaler()
X_train_scaled = X_train.copy(); X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler2.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler2.transform(X_test[num_cols])

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

def evaluate(name, y_true, y_pred, y_prob):
    m = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
    }
    print(f"{name}: {m}")
    return m

metrics_lr = evaluate("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
metrics_rf = evaluate("Random Forest", y_test, y_pred_rf, y_prob_rf)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (name, y_pred) in zip(axes, [('Logistic Regression', y_pred_lr), ('Random Forest', y_pred_rf)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    ax.set_title(f'{name}')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('charts/03_confusion_matrices.png', dpi=140)
plt.close()

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={metrics_lr['roc_auc']:.3f})", color='#4C72B0')
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={metrics_rf['roc_auc']:.3f})", color='#DD8452')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('charts/04_roc_curves.png', dpi=140)
plt.close()

# ===========================================================
# STAGE 5: COMBINED INSIGHT — Segments x Predicted Risk
# ===========================================================
print("\n=== Stage 5: Combining Segmentation + Prediction ===")
# Get predicted probability for the FULL dataset using the RF model (trained on X_enc/y)
full_prob = rf.predict_proba(X_enc)[:, 1]
df['PredictedAttritionRisk'] = full_prob

risk_by_segment = df.groupby('Segment')['PredictedAttritionRisk'].mean().round(3) * 100
print("Average predicted attrition risk (%) by segment:\n", risk_by_segment)

plt.figure(figsize=(6.5, 4.5))
ax = sns.barplot(x=risk_by_segment.index, y=risk_by_segment.values, hue=risk_by_segment.index,
                  palette='mako', legend=False)
plt.title('Average Model-Predicted Attrition Risk by Segment')
plt.xlabel('Segment')
plt.ylabel('Average Predicted Risk (%)')
for i, v in enumerate(risk_by_segment.values):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('charts/05_predicted_risk_by_segment.png', dpi=140)
plt.close()

# Save final combined dataset
df.to_csv('HR_Analytics_capstone_final.csv', index=False)

comparison = pd.DataFrame({'Logistic Regression': metrics_lr, 'Random Forest': metrics_rf}).round(3)
comparison.to_csv('capstone_model_comparison.csv')

print("\nAll stages complete. Charts saved to 'charts/'.")
