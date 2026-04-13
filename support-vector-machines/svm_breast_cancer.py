# Support Vector Machine — Breast Cancer Classification
#
# Purpose:
#   End-to-end SVM classification pipeline applied to the Wisconsin Breast
#   Cancer Diagnostic dataset (available directly from scikit-learn).
#   The objective is to classify tumours as malignant or benign from 30
#   continuous cell nucleus measurements, and to compare the result against
#   the Decision Tree (93.86%), Random Forest (95.61%), and Gradient Boosted
#   Trees (97.37%) results previously established in this portfolio.
#
#   Full data validation and exploratory analysis for this dataset were
#   conducted in the Decision Tree project and are not repeated here.
#
#   This script should be read after svm_kernel_illustration.py, which
#   establishes the conceptual foundation for kernel selection.
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    f1_score
)

# =============================================================================
# 0. GLOBAL PLOT SETTINGS
# =============================================================================

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# COLOUR_MALIGNANT = "#E05C5C"   # red
# COLOUR_BENIGN    = "#5B8DB8"   # blue
COLOUR_LINEAR    = "#A8C5DA"   # light blue — linear kernel bars
COLOUR_RBF       = "#5B8DB8"   # mid blue   — RBF kernel bars
COLOUR_HEATMAP   = "Blues"
COLOUR_CM        = "Blues"
COLOUR_ROC       = "#5B8DB8"

RANDOM_STATE     = 42
FIGURE_DPI       = 150

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("=" * 60)
print("\nSECTION 1: DATA LOADING")
print("=" * 60)

data        = load_breast_cancer()
X           = pd.DataFrame(data.data,   columns=data.feature_names)
y           = pd.Series(data.target,    name="target")
target_names = data.target_names        # 0 = malignant, 1 = benign

print(f"\nDataset          : Wisconsin Breast Cancer Diagnostic (scikit-learn)")
print(f"Observations     : {X.shape[0]}")
print(f"Features         : {X.shape[1]}")
# print(f"Target classes   : {list(target_names)}  (0 = malignant, 1 = benign)")

print("\n--- Class Distribution ---")
class_counts = y.value_counts().sort_index()
class_labels = [target_names[i] for i in class_counts.index]
for label, count in zip(class_labels, class_counts):
    print(f"  {label:<12}: {count}  ({count / len(y) * 100:.1f}%)")

print("\nNote: Full data validation and exploratory analysis for this dataset were conducted in the Decision Tree project and are not repeated here.")

# =============================================================================
# 2. PLOT 01 — CLASS DISTRIBUTION
# =============================================================================

class_df = pd.DataFrame({
    "Class" : [target_names[i] for i in class_counts.index],
    "Count" : class_counts.values
})

fig, ax = plt.subplots(figsize=(7, 5))

sns.barplot(
    data     = class_df,
    x        = "Class",
    y        = "Count",
    palette  = 'Greens',
    errorbar = None,
    ax       = ax,
    hue      = "Class"
)

for idx, row in class_df.iterrows():
    ax.text(
        idx, row["Count"] + 4,
        f"{row['Count']}  ({row['Count'] / len(y) * 100:.1f}%)",
        ha="center", va="bottom", fontsize=11, fontweight="bold"
    )

ax.set_title("Class Distribution — Breast Cancer Dataset",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Tumour Class", fontsize=11)
ax.set_ylabel("Number of Observations", fontsize=11)
ax.set_ylim(0, class_df["Count"].max() * 1.18)

plt.tight_layout()
plt.savefig("plot_01_class_distribution.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 3. TRAIN / TEST SPLIT AND FEATURE SCALING
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: PRE-PROCESSING")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    stratify     = y,
    random_state = RANDOM_STATE
)

scaler   = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain / Test split : 80% / 20%")
print(f"Training set       : {X_train_scaled.shape[0]} observations")
print(f"Test set           : {X_test_scaled.shape[0]}  observations")
print(f"\nFeature scaling    : StandardScaler fitted on training set, applied to training and test sets.")
print(f"Scaling is mandatory for SVM — the algorithm computes distances between observations and the hyperplane.")
print(f"Without scaling, features measured on large numerical ranges dominate distance calculations as an artefact of their units rather than their predictive value.")

# =============================================================================
# 4. KERNEL COMPARISON — LINEAR VS RBF (DEFAULT HYPERPARAMETERS)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: KERNEL COMPARISON (DEFAULT HYPERPARAMETERS)")
print("=" * 60)

kernels = {
    "Linear": SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE),
    "RBF"   : SVC(kernel="rbf",    C=1.0, gamma="scale", random_state=RANDOM_STATE)
}

kernel_results = {}

for name, model in kernels.items():
    model.fit(X_train_scaled, y_train)
    y_pred     = model.predict(X_test_scaled)
    acc        = accuracy_score(y_test, y_pred)
    f1_mal     = f1_score(y_test, y_pred, pos_label=0)
    f1_ben     = f1_score(y_test, y_pred, pos_label=1)
    f1_macro   = f1_score(y_test, y_pred, average="macro")
    kernel_results[name] = {
        "model"    : model,
        "accuracy" : acc,
        "f1_mal"   : f1_mal,
        "f1_ben"   : f1_ben,
        "f1_macro" : f1_macro
    }
    print(f"\n{name} SVM")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  F1 Malignant  : {f1_mal:.4f}")
    print(f"  F1 Benign     : {f1_ben:.4f}")
    print(f"  F1 Macro      : {f1_macro:.4f}")
    print(f"  Support Vectors: {model.n_support_}  (malignant, benign)")

# =============================================================================
# 5. PLOT 02 — KERNEL COMPARISON BAR CHART (F1-SCORES)
# =============================================================================

kernel_comp_data = []
for kernel_name, res in kernel_results.items():
    kernel_comp_data.append({"Kernel": kernel_name, "Class": "Malignant", "F1-Score": res["f1_mal"]})
    kernel_comp_data.append({"Kernel": kernel_name, "Class": "Benign",    "F1-Score": res["f1_ben"]})
    kernel_comp_data.append({"Kernel": kernel_name, "Class": "Macro Avg", "F1-Score": res["f1_macro"]})

comp_df = pd.DataFrame(kernel_comp_data)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data     = comp_df,
    x        = "Class",
    y        = "F1-Score",
    hue      = "Kernel",
    # palette  = {"Linear": COLOUR_LINEAR, "RBF": COLOUR_RBF},
    palette  = "YlGn_r",
    errorbar = None,
    ax       = ax
)

for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

ax.set_ylim(0, 1.12)
ax.set_title("Kernel Comparison — F1-Score by Class (Default Hyperparameters)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Class / Average", fontsize=11)
ax.set_ylabel("F1-Score", fontsize=11)
ax.legend(title="Kernel", fontsize=10, title_fontsize=10)

plt.tight_layout()
plt.savefig("plot_02_kernel_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 6. HYPERPARAMETER TUNING — GRIDSEARCHCV (C AND GAMMA, RBF KERNEL)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: HYPERPARAMETER TUNING — GRIDSEARCHCV")
print("=" * 60)

param_grid = {
    "C"    : [0.1, 1, 10, 100, 1000],
    "gamma": [0.0001, 0.001, 0.01, 0.1, 1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator  = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    param_grid = param_grid,
    cv         = cv,
    scoring    = "accuracy",
    n_jobs     = -1,
    verbose    = 1
)

print("\nRunning GridSearchCV over:")
print(f"  C     : {param_grid['C']}")
print(f"  gamma : {param_grid['gamma']}")
print(f"  CV    : 5-fold stratified  |  Scoring: accuracy")
print(f"  Total fits: {len(param_grid['C']) * len(param_grid['gamma']) * 5}\n")

grid_search.fit(X_train_scaled, y_train)

best_C     = grid_search.best_params_["C"]
best_gamma = grid_search.best_params_["gamma"]
best_cv_acc = grid_search.best_score_

print(f"\nOptimal C           : {best_C}")
print(f"Optimal gamma       : {best_gamma}")
print(f"Best CV accuracy    : {best_cv_acc * 100:.2f}%")

# =============================================================================
# 7. PLOT 03 — GRIDSEARCH HEATMAP (C x GAMMA)
# =============================================================================

results_df = pd.DataFrame(grid_search.cv_results_)
pivot_df   = results_df.pivot_table(
    index   = "param_gamma",
    columns = "param_C",
    values  = "mean_test_score"
)

# Format axis labels
col_labels = [str(c) for c in param_grid["C"]]
row_labels = [str(g) for g in param_grid["gamma"]]
pivot_df.columns = col_labels
pivot_df.index   = row_labels

fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(
    pivot_df,
    annot      = True,
    fmt        = ".4f",
    cmap       = COLOUR_HEATMAP,
    linewidths = 0.5,
    linecolor  = "white",
    annot_kws  = {"size": 10},
    cbar_kws   = {"label": "Mean CV Accuracy"},
    ax         = ax
)

# Highlight optimal cell
gamma_idx = param_grid["gamma"].index(best_gamma)
c_idx     = param_grid["C"].index(best_C)
ax.add_patch(plt.Rectangle(
    (c_idx, gamma_idx), 1, 1,
    fill=False, edgecolor="#E05C5C", lw=3
))

ax.set_title("GridSearchCV — Mean CV Accuracy by C and Gamma (RBF Kernel)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("C  (Regularisation Parameter)", fontsize=11)
ax.set_ylabel("Gamma  (Kernel Coefficient)", fontsize=11)
ax.tick_params(axis="x", rotation=0)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("plot_03_gridsearch_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 8. FINAL MODEL — FIT WITH OPTIMAL HYPERPARAMETERS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: FINAL MODEL FITTING AND EVALUATION")
print("=" * 60)

svm_final = SVC(
    kernel       = "rbf",
    C            = best_C,
    gamma        = best_gamma,
    probability  = True,
    random_state = RANDOM_STATE
)

svm_final.fit(X_train_scaled, y_train)
y_pred_final  = svm_final.predict(X_test_scaled)
y_prob_final  = svm_final.predict_proba(X_test_scaled)[:, 1]

acc_final  = accuracy_score(y_test, y_pred_final)
auc_final  = roc_auc_score(y_test, y_prob_final)

print(f"\nFinal Model — RBF SVM  (C={best_C}, gamma={best_gamma})")
print(f"\nTest Accuracy : {acc_final * 100:.2f}%")
print(f"ROC-AUC       : {auc_final:.4f}")

print("\n--- Classification Report ---")
print(classification_report(
    y_test, y_pred_final,
    target_names=["Malignant", "Benign"]
))

print("--- Portfolio Accuracy Comparison ---")
print(f"  Decision Tree            :  93.86%")
print(f"  Random Forest            :  95.61%")
print(f"  Gradient Boosted Trees   :  97.37%")
print(f"  SVM (RBF, C={best_C}, gamma={best_gamma}) :  {acc_final * 100:.2f}%")

# =============================================================================
# 9. PLOT 04 — CONFUSION MATRIX
# =============================================================================

cm = confusion_matrix(y_test, y_pred_final)
cm_df = pd.DataFrame(
    cm,
    index   = ["Actual: Malignant", "Actual: Benign"],
    columns = ["Predicted: Malignant", "Predicted: Benign"]
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    cm_df,
    annot      = True,
    fmt        = "d",
    cmap       = COLOUR_CM,
    linewidths = 0.8,
    linecolor  = "white",
    annot_kws  = {"size": 14, "weight": "bold"},
    cbar       = False,
    ax         = ax
)

ax.set_title(
    f"Confusion Matrix — RBF SVM  (C={best_C}, gamma={best_gamma})\n"
    f"Test Accuracy: {acc_final * 100:.2f}%",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Predicted Class", fontsize=11)
ax.set_ylabel("Actual Class", fontsize=11)
ax.tick_params(axis="x", rotation=15)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("plot_04_confusion_matrix.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 10. PLOT 05 — ROC CURVE
# =============================================================================

fpr, tpr, thresholds = roc_curve(y_test, y_prob_final)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(fpr, tpr, color=COLOUR_ROC, lw=2.5,
        label=f"RBF SVM  (AUC = {auc_final:.4f})")
ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.5,
        linestyle="--", label="Random Classifier (AUC = 0.50)")

ax.fill_between(fpr, tpr, alpha=0.08, color=COLOUR_ROC)

ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
ax.set_title("ROC Curve — RBF Kernel SVM",
             fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
ax.legend(fontsize=10, loc="lower right")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

plt.tight_layout()
plt.savefig("plot_05_roc_curve.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 11. SUPPORT VECTOR ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: SUPPORT VECTOR ANALYSIS")
print("=" * 60)

n_sv        = svm_final.n_support_
total_sv    = sum(n_sv)
pct_sv      = total_sv / X_train_scaled.shape[0] * 100

print(f"\nSupport vectors (malignant class) : {n_sv[0]}")
print(f"Support vectors (benign class)    : {n_sv[1]}")
print(f"Total support vectors             : {total_sv}")
print(f"Training set size                 : {X_train_scaled.shape[0]}")
print(f"Support vectors as % of training  : {pct_sv:.1f}%")
print(
    f"\nInterpretation: {pct_sv:.1f}% of the training observations directly "
    f"determine\nthe position of the decision hyperplane. All remaining "
    f"training\nobservations have no influence on the boundary whatsoever."
)

# =============================================================================
# 12. PLOT 06 — SUPPORT VECTOR COUNTS BY CLASS
# =============================================================================

sv_df = pd.DataFrame({
    "Class"            : ["Malignant", "Benign"],
    "Support Vectors"  : n_sv,
    "Total Observations": [
        int((y_train == 0).sum()),
        int((y_train == 1).sum())
    ]
})
sv_df["Non-Support Vectors"] = sv_df["Total Observations"] - sv_df["Support Vectors"]

sv_plot_df = sv_df.melt(
    id_vars   = "Class",
    value_vars = ["Support Vectors", "Non-Support Vectors"],
    var_name  = "Type",
    value_name= "Count"
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data     = sv_plot_df,
    x        = "Class",
    y        = "Count",
    hue      = "Type",
    palette  = {"Support Vectors": "#5B8DB8", "Non-Support Vectors": "#D4E6F1"},
    errorbar = None,
    ax       = ax
)

for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                str(int(height)),
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

ax.set_title(
    f"Support Vectors vs Non-Support Vectors by Class\n"
    f"Total Support Vectors: {total_sv}  ({pct_sv:.1f}% of training set)",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Tumour Class", fontsize=11)
ax.set_ylabel("Number of Observations", fontsize=11)
ax.legend(title="Observation Type", fontsize=10, title_fontsize=10)

plt.tight_layout()
plt.savefig("plot_06_support_vectors.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nDataset              : Wisconsin Breast Cancer Diagnostic")
print(f"Observations         : {X.shape[0]}  |  Features: {X.shape[1]}")
print(f"Train / Test split   : 80% / 20%  "
      f"({X_train_scaled.shape[0]} / {X_test_scaled.shape[0]})")
print(f"\nOptimal Kernel       : RBF")
print(f"Optimal C            : {best_C}")
print(f"Optimal gamma        : {best_gamma}")
print(f"Best CV Accuracy     : {best_cv_acc * 100:.2f}%")
print(f"\nFinal Test Accuracy  : {acc_final * 100:.2f}%")
print(f"ROC-AUC              : {auc_final:.4f}")
print(f"Support Vectors      : {total_sv}  ({pct_sv:.1f}% of training set)")
print(f"\nPortfolio Comparison:")
print(f"  Decision Tree          :  93.86%")
print(f"  Random Forest          :  95.61%")
print(f"  Gradient Boosted Trees :  97.37%")
print(f"  SVM (RBF)              :  {acc_final * 100:.2f}%")
print("\nAll plots saved.")
print("=" * 60)