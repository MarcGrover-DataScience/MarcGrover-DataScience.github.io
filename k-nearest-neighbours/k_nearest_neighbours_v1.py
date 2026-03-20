# K-nearest neighbours project
# Using physicochemical measurements (acidity, alcohol, sulphates, etc.) to predict wine quality
# Wine Quality Dataset downloaded from UCI at:  https://archive.ics.uci.edu/dataset/186/wine+quality

# =============================================================================
# K-Nearest Neighbours (KNN) Classification — Wine Quality
# Portfolio Project | Marc Grover | Data Science Portfolio
# Dataset: Wine Quality (UCI / Kaggle)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Update this path to match your local file location ---
FILE_PATH = "winequality-red.csv"      # "winequality-red.csv" Or "winequality-white.csv"
WINE_TYPE = "Red"                       # Label used in chart titles
RANDOM_STATE = 42
TEST_SIZE = 0.2
K_RANGE = range(1, 31)                  # Range of K values to evaluate

# Seaborn theme
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = "Greens"

# =============================================================================
# 1. DATA LOADING & VALIDATION
# =============================================================================

print("\nK-NEAREST NEIGHBOURS — WINE QUALITY CLASSIFICATION\n")

# Load — UCI/Kaggle Wine Quality files are semicolon-delimited
# try:
#     df = pd.read_csv(FILE_PATH, sep=";")
#     print(f"\n[✓] Data loaded successfully: {FILE_PATH}")
# except FileNotFoundError:
#     raise FileNotFoundError(
#         f"\n[✗] File not found: '{FILE_PATH}'\n"
#         "    Please update FILE_PATH to point to your downloaded CSV."
#     )

df = pd.read_csv(FILE_PATH, sep=";")

# --- Basic shape & structure ---
print("\nDATASET OVERVIEW:")
print(f"Rows         : {df.shape[0]}")
print(f"Columns      : {df.shape[1]}")
print(f"\nColumn names:\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")

# --- Missing values ---
print("\nMISSING VALUE AUDIT:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values detected")
else:
    print(missing[missing > 0])

# --- Duplicate rows ---
n_dupes = df.duplicated().sum()
print("\nDUPLICATE ROW AUDIT:")
if n_dupes > 0:
    print(f"{n_dupes} duplicate rows found — removing")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows after deduplication: {df.shape[0]}")
else:
    print("No duplicate rows detected")

# --- Descriptive statistics ---
print("\nDESCRIPTIVE STATISTICS:")
print(df.describe().round(3).to_string())

# --- Quality score distribution (raw) ---
print("\nRAW QUALITY SCORE DISTRIBUTION:")
print(df["quality"].value_counts().sort_index())

# =============================================================================
# 2. FEATURE ENGINEERING — Quality Banding
# =============================================================================
# Bin raw quality scores (3–9) into three interpretable classes:
#   Low    : 3–4
#   Medium : 5–6
#   High   : 7–9
# This addresses class imbalance in the raw scores while keeping
# the classification task meaningful and non-trivial.

def band_quality(score):
    if score <= 4:
        return "Low"
    elif score <= 5:
        return "Medium"
    else:
        return "High"

df["quality_band"] = df["quality"].apply(band_quality)

quality_order = ["Low", "Medium", "High"]

print("\nQUALITY BAND DISTRIBUTION (after banding)")
band_counts = df["quality_band"].value_counts()[quality_order]
band_pct = (band_counts / len(df) * 100).round(1)
for band, count, pct in zip(band_counts.index, band_counts.values, band_pct.values):
    print(f"{band:<8}: {count:>4}  ({pct}%)")

# =============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

features = [c for c in df.columns if c not in ["quality", "quality_band"]]

# --- PLOT 1: Quality Band Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(
    data=df,
    x="quality_band",
    order=quality_order,
    palette=PALETTE,
    ax=ax
)
ax.set_title(f"{WINE_TYPE} Wine — Quality Band Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Quality Band", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
for bar in ax.patches:
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{int(bar.get_height())}",
        ha="center", va="bottom", fontsize=11
    )
plt.tight_layout()
plt.savefig("plot_01_quality_band_distribution.png", dpi=150)
plt.show()

# --- PLOT 2: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(11, 8))
corr_matrix = df[features + ["quality"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    ax=ax
)
ax.set_title(f"{WINE_TYPE} Wine — Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("plot_02_correlation_heatmap.png", dpi=150)
plt.show()

# --- PLOT 3–13: Individual Feature Distributions by Quality Band ---
# Key features selected for portfolio-relevant narrative
key_features = ["alcohol", "volatile acidity", "sulphates", "citric acid", "density"]

for i, feat in enumerate(key_features, start=3):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="quality_band",
        y=feat,
        order=quality_order,
        palette=PALETTE,
        ax=ax,
        width=0.5,
        linewidth=1.5
    )
    ax.set_title(
        f"{WINE_TYPE} Wine — {feat.title()} by Quality Band",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Quality Band", fontsize=12)
    ax.set_ylabel(feat.title(), fontsize=12)
    plt.tight_layout()
    safe_name = feat.replace(" ", "_")
    plt.savefig(f"plot_{i:02d}_boxplot_{safe_name}.png", dpi=150)
    plt.show()

# =============================================================================
# 4. PRE-PROCESSING
# =============================================================================

print(f"\n{'─'*40}")
print("PRE-PROCESSING")
print(f"{'─'*40}")

X = df[features]
y = df["quality_band"]

# Train / test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"  Training samples  : {X_train.shape[0]}")
print(f"  Test samples      : {X_test.shape[0]}")

# Feature scaling — essential for KNN (distance-based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("\n  StandardScaler applied. ✓")
print("  (KNN is sensitive to feature scale — scaling is mandatory)")

# =============================================================================
# 5. OPTIMAL K SELECTION
# =============================================================================

print(f"\n{'─'*40}")
print("OPTIMAL K SELECTION")
print(f"{'─'*40}")

train_accuracies = []
test_accuracies  = []

for k in K_RANGE:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

best_k     = K_RANGE.start + int(np.argmax(test_accuracies))
best_score = max(test_accuracies)

print(f"  Optimal K (test accuracy) : K = {best_k}")
print(f"  Test accuracy at K={best_k}  : {best_score:.4f} ({best_score*100:.2f}%)")

# --- PLOT 8: Accuracy vs K ---
k_list = list(K_RANGE)
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=k_list, y=train_accuracies, label="Training Accuracy", marker="o", markersize=5, ax=ax)
sns.lineplot(x=k_list, y=test_accuracies,  label="Test Accuracy",     marker="o", markersize=5, ax=ax)
ax.axvline(x=best_k, color="crimson", linestyle="--", linewidth=1.5, label=f"Optimal K = {best_k}")
ax.set_title(f"{WINE_TYPE} Wine KNN — Accuracy vs Number of Neighbours (K)", fontsize=14, fontweight="bold")
ax.set_xlabel("K (Number of Neighbours)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xticks(k_list)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("plot_08_accuracy_vs_k.png", dpi=150)
plt.show()

# =============================================================================
# 6. FINAL MODEL — FIT & PREDICT
# =============================================================================

print(f"\n{'─'*40}")
print(f"FINAL MODEL — KNN (K={best_k})")
print(f"{'─'*40}")

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train_scaled, y_train)
y_pred = knn_final.predict(X_test_scaled)

# =============================================================================
# 7. MODEL EVALUATION
# =============================================================================

# --- Classification report ---
print("\nCLASSIFICATION REPORT")
print(f"{'─'*40}")
# print(classification_report(y_test, y_pred, target_names=quality_order))
print(classification_report(y_test, y_pred, labels=quality_order, target_names=quality_order))

overall_accuracy = accuracy_score(y_test, y_pred)
print(f"  Overall Test Accuracy : {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")

# --- PLOT 9: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred, labels=quality_order)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=quality_order,
    yticklabels=quality_order,
    linewidths=0.5,
    ax=ax
)
ax.set_title(f"{WINE_TYPE} Wine KNN (K={best_k}) — Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("plot_09_confusion_matrix.png", dpi=150)
plt.show()

# =============================================================================
# 8. PERMUTATION FEATURE IMPORTANCE
# =============================================================================

print(f"\n{'─'*40}")
print("PERMUTATION FEATURE IMPORTANCE")
print(f"{'─'*40}")

perm_result = permutation_importance(
    knn_final, X_test_scaled, y_test,
    n_repeats=20,
    random_state=RANDOM_STATE,
    scoring="accuracy"
)

importance_df = pd.DataFrame({
    "Feature"   : features,
    "Importance": perm_result.importances_mean,
    "Std"       : perm_result.importances_std
}).sort_values("Importance", ascending=False).reset_index(drop=True)

print(importance_df.to_string(index=False))

# --- PLOT 10: Feature Importance ---
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    palette="Greens_r",
    errorbar=None,
    ax=ax
)
# Add error bars manually
ax.errorbar(
    x=importance_df["Importance"],
    y=range(len(importance_df)),
    xerr=importance_df["Std"],
    fmt="none",
    color="black",
    capsize=4,
    linewidth=1.2
)
ax.set_title(
    f"{WINE_TYPE} Wine KNN (K={best_k}) — Permutation Feature Importance",
    fontsize=14, fontweight="bold"
)
ax.set_xlabel("Mean Accuracy Decrease (Permutation Importance)", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.axvline(x=0, color="grey", linestyle="--", linewidth=1)
plt.tight_layout()
plt.savefig("plot_10_feature_importance.png", dpi=150)
plt.show()

# =============================================================================
# 9. SUMMARY
# =============================================================================

print(f"\n{'=' * 65}")
print(" MODEL SUMMARY")
print(f"{'=' * 65}")
print(f"  Dataset            : {WINE_TYPE} Wine Quality")
print(f"  Observations       : {df.shape[0]}")
print(f"  Features used      : {len(features)}")
print(f"  Target classes     : Low / Medium / High")
print(f"  Optimal K          : {best_k}")
print(f"  Overall Accuracy   : {overall_accuracy*100:.2f}%")
print(f"\n  Top 3 predictive features:")
for _, row in importance_df.head(3).iterrows():
    print(f"    {row['Feature']:<30} importance = {row['Importance']:.4f}")
print(f"\n{'=' * 65}")
print(" Plots saved to working directory.")
print(f"{'=' * 65}")