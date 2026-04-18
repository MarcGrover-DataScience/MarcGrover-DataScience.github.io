# SHAP and Model Interpretability — Breast Cancer Random Forest
#
# Purpose:
#   Applies SHAP (SHapley Additive exPlanations) to the Random Forest classifier previously developed in the Random Forest project,
#   using the Wisconsin Breast Cancer Diagnostic dataset.
#   The objective is to move beyond predictive accuracy and explain why the model produces the predictions it does
#   — both at the global dataset level and at the level of individual patient observations.
#
#   The Random Forest model is rebuilt here using the optimal hyperparameters established in the prior project
#   (n_estimators=150, max_depth=10), ensuring full reproducibility without dependency on saved model files.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =============================================================================
# 0. GLOBAL PLOT SETTINGS
# =============================================================================

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

COLOUR_MALIGNANT = "#E05C5C"    # red
COLOUR_BENIGN    = "#5B8DB8"    # blue
COLOUR_BAR       = "#5B8DB8"    # bar chart fill
FIGURE_DPI       = 150
RANDOM_STATE     = 42

# SHAP uses its own colour scheme for most plots, but matplotlib figures
# produced via SHAP are restyled below to align with portfolio conventions.

# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("=" * 60)
print("SECTION 1: DATA LOADING")
print("=" * 60)

data         = load_breast_cancer()
X            = pd.DataFrame(data.data,  columns=data.feature_names)
y            = pd.Series(data.target,   name="target")
target_names = data.target_names        # 0 = malignant, 1 = benign

print(f"\nObservations    : {X.shape[0]}")
print(f"Features        : {X.shape[1]}")
print(f"Target classes  : {list(target_names)}  (0 = malignant, 1 = benign)")

print("\nClass Distribution")
class_counts = y.value_counts().sort_index()
for idx, count in class_counts.items():
    print(f"{target_names[idx]:<12}: {count}  ({count / len(y) * 100:.1f}%)")

# =============================================================================
# 2. TRAIN / TEST SPLIT
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 2: TRAIN / TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    stratify     = y,
    random_state = RANDOM_STATE
)

print(f"\nTrain / Test split : 80% / 20%")
print(f"Training set       : {X_train.shape[0]} observations")
print(f"Test set           : {X_test.shape[0]}  observations")
print(f"\nNote: No feature scaling is applied. Random Forest is a tree-based method and is invariant to the scale of input features.")

# =============================================================================
# 3. REBUILD RANDOM FOREST (OPTIMAL HYPERPARAMETERS FROM PRIOR PROJECT)
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 3: RANDOM FOREST MODEL (n_estimators=150, max_depth=10)")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators = 150,
    max_depth    = 10,
    random_state = RANDOM_STATE
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\nModel            : Random Forest")
print(f"n_estimators     : 150")
print(f"max_depth        : 10")
print(f"Test Accuracy    : {acc * 100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

# =============================================================================
# 4. SHAP EXPLAINER — TreeSHAP
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 4: SHAP EXPLAINER")
print("=" * 60)

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test)

# shap_values is a list of two arrays for binary classification:
#   shap_values[0] — SHAP values for class 0 (malignant)
#   shap_values[1] — SHAP values for class 1 (benign)
# We use class 1 (benign) as the positive class throughout,
# consistent with scikit-learn's convention (target=1 is benign).


# #### OLD VERSION
# shap_vals_benign    = shap_values[1]    # shape: (n_test, n_features)
# shap_vals_malignant = shap_values[0]    # shape: (n_test, n_features)
#
# # Expected value (baseline) for the benign class
# expected_value_benign = explainer.expected_value[1]
#
# print(f"\nExpected value (baseline, benign class) : {expected_value_benign:.4f}")
# print(f"SHAP value array shape (test set)       : {shap_vals_benign.shape}")
# print(f"\nSHAP values computed successfully.")


#### NEW VERSION
# Handle both old API (list of arrays) and new API (single 3D array)
if isinstance(shap_values, list):
    # Older SHAP versions: list of 2D arrays, one per class
    # shap_values[0] = malignant, shap_values[1] = benign
    shap_vals_malignant    = shap_values[0]   # shape: (n_test, n_features)
    shap_vals_benign       = shap_values[1]   # shape: (n_test, n_features)
    expected_value_benign    = explainer.expected_value[1]
    expected_value_malignant = explainer.expected_value[0]
else:
    # Newer SHAP versions: single 3D array (n_test, n_features, n_classes)
    shap_vals_malignant    = shap_values[:, :, 0]   # shape: (n_test, n_features)
    shap_vals_benign       = shap_values[:, :, 1]   # shape: (n_test, n_features)
    if hasattr(explainer.expected_value, '__len__'):
        expected_value_benign    = explainer.expected_value[1]
        expected_value_malignant = explainer.expected_value[0]
    else:
        expected_value_benign    = explainer.expected_value
        expected_value_malignant = explainer.expected_value

print(f"\nSHAP values format   : {'list (older API)' if isinstance(shap_values, list) else '3D array (newer API)'}")
print(f"Expected value (baseline, benign class) : {expected_value_benign:.4f}")
print(f"SHAP value array shape (test set)       : {shap_vals_benign.shape}")

# =============================================================================
# 5. GLOBAL SHAP ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 5: GLOBAL SHAP ANALYSIS")
print("=" * 60)

# --- Mean absolute SHAP values (global feature importance) ---
mean_abs_shap = pd.DataFrame({
    "Feature"   : X_test.columns,
    "Mean |SHAP|": np.abs(shap_vals_benign).mean(axis=0)
}).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)

print("\nTop 15 Features by Mean |SHAP Value| (Global Importance)")
print(mean_abs_shap.head(15).to_string(index=False))

top5_sum   = mean_abs_shap["Mean |SHAP|"].head(5).sum()
total_sum  = mean_abs_shap["Mean |SHAP|"].sum()
print(f"\nTop 5 features — sum of mean |SHAP|  : {top5_sum:.4f}")
print(f"All 30 features — sum of mean |SHAP| : {total_sum:.4f}")
print(f"Top 5 as % of total                  : {top5_sum / total_sum * 100:.1f}%")

# =============================================================================
# 6. PLOT 01 — SHAP SUMMARY BEESWARM PLOT (GLOBAL)
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

shap.summary_plot(
    shap_vals_benign,
    X_test,
    plot_type   = "dot",
    max_display = 15,
    show        = False,
    color_bar   = True
)

fig = plt.gcf()
fig.set_size_inches(10, 8)

plt.title(
    "SHAP Summary Plot — Feature Impact on Benign Classification\n"
    "Random Forest  |  Wisconsin Breast Cancer Dataset",
    fontsize=13, fontweight="bold", pad=14
)
plt.xlabel("SHAP Value  (impact on model output)", fontsize=11)
plt.tight_layout()
plt.savefig("plot_01_shap_summary_beeswarm.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 7. PLOT 02 — GLOBAL MEAN |SHAP| BAR CHART
# =============================================================================

top15 = mean_abs_shap.head(15).sort_values("Mean |SHAP|", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))

bars = ax.barh(
    top15["Feature"],
    top15["Mean |SHAP|"],
    color     = "seagreen",
    edgecolor = "white",
    linewidth = 0.5
)

for bar, val in zip(bars, top15["Mean |SHAP|"]):
    ax.text(
        val + 0.001, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}",
        va="center", ha="left", fontsize=9, fontweight="bold"
    )

ax.set_title(
    "Global Feature Importance — Mean Absolute SHAP Value\n"
    "Random Forest  |  Wisconsin Breast Cancer Dataset",
    fontsize=13, fontweight="bold", pad=12
)
ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
ax.set_xlim(0, top15["Mean |SHAP|"].max() * 1.20)

plt.tight_layout()
plt.savefig("plot_02_shap_bar_global.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 8. LOCAL SHAP ANALYSIS — OBSERVATION SELECTION
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 6: LOCAL SHAP ANALYSIS")
print("=" * 60)

# Identify a high-confidence malignant and a high-confidence benign prediction
# from the test set, selecting correctly classified observations in each case.

y_prob      = rf_model.predict_proba(X_test)
y_pred_arr  = rf_model.predict(X_test)
y_test_arr  = y_test.values

# High-confidence correctly predicted malignant (class 0)
malignant_candidates = np.where(
    (y_test_arr == 0) & (y_pred_arr == 0) & (y_prob[:, 0] >= 0.80)
)[0]
print(malignant_candidates)
malignant_idx = malignant_candidates[np.argmax(y_prob[malignant_candidates, 0])]

# High-confidence correctly predicted benign (class 1)
benign_candidates = np.where(
    (y_test_arr == 1) & (y_pred_arr == 1) & (y_prob[:, 1] >= 0.80)
)[0]
print(benign_candidates)
benign_idx = benign_candidates[np.argmax(y_prob[benign_candidates, 1])]

print(f"\nSelected observations for local explanation:")
print(f"  Malignant case — test index: {malignant_idx}  "
      f"| Predicted probability (malignant): {y_prob[malignant_idx, 0]:.4f}")
print(f"  Benign case    — test index: {benign_idx}  "
      f"| Predicted probability (benign):    {y_prob[benign_idx, 1]:.4f}")

# =============================================================================
# 9. PLOT 03 — WATERFALL PLOT: MALIGNANT CASE
# =============================================================================

print("\n--- Generating Plot 03: Waterfall Plot — Malignant Case ---")

# # Build SHAP Explanation object for the malignant class (class 0)
# explanation_malignant = shap.Explanation(
#     values         = shap_values[0][malignant_idx],
#     base_values    = explainer.expected_value[0],
#     data           = X_test.iloc[malignant_idx].values,
#     feature_names  = list(X_test.columns)
# )


# Build SHAP Explanation object for the malignant class (class 0)
explanation_malignant = shap.Explanation(
    values        = shap_vals_malignant[malignant_idx],
    base_values   = expected_value_malignant,
    data          = X_test.iloc[malignant_idx].values,
    feature_names = list(X_test.columns)
)

fig, ax = plt.subplots(figsize=(10, 7))
shap.waterfall_plot(explanation_malignant, max_display=12, show=False)

fig = plt.gcf()
fig.set_size_inches(10, 7)
plt.title(
    f"SHAP Waterfall Plot — Individual Malignant Prediction\n"
    f"Predicted probability (malignant): {y_prob[malignant_idx, 0]:.4f}",
    fontsize=12, fontweight="bold", pad=12
)
plt.tight_layout()
plt.savefig("plot_03_shap_waterfall_malignant.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 10. PLOT 04 — WATERFALL PLOT: BENIGN CASE
# =============================================================================

print("--- Generating Plot 04: Waterfall Plot — Benign Case ---")

# explanation_benign = shap.Explanation(
#     values         = shap_values[1][benign_idx],
#     base_values    = explainer.expected_value[1],
#     data           = X_test.iloc[benign_idx].values,
#     feature_names  = list(X_test.columns)
# )

# Build SHAP Explanation object for the benign class (class 1)
explanation_benign = shap.Explanation(
    values        = shap_vals_benign[benign_idx],
    base_values   = expected_value_benign,
    data          = X_test.iloc[benign_idx].values,
    feature_names = list(X_test.columns)
)

fig, ax = plt.subplots(figsize=(10, 7))
shap.waterfall_plot(explanation_benign, max_display=12, show=False)

fig = plt.gcf()
fig.set_size_inches(10, 7)
plt.title(
    f"SHAP Waterfall Plot — Individual Benign Prediction\n"
    f"Predicted probability (benign): {y_prob[benign_idx, 1]:.4f}",
    fontsize=12, fontweight="bold", pad=12
)
plt.tight_layout()
plt.savefig("plot_04_shap_waterfall_benign.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 11. SHAP DEPENDENCE PLOTS — TOP 3 FEATURES
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 7: SHAP DEPENDENCE PLOTS")
print("=" * 60)

# Use the top 3 features from global importance ranking
top3_features = mean_abs_shap["Feature"].head(3).tolist()
print(f"\nTop 3 features selected for dependence plots:")
for i, f in enumerate(top3_features, 1):
    print(f"  {i}. {f}")

dependence_filenames = [
    "plot_05_shap_dependence_feature1.png",
    "plot_06_shap_dependence_feature2.png",
    "plot_07_shap_dependence_feature3.png"
]

for feature, filename in zip(top3_features, dependence_filenames):
    print(f"\n--- Generating dependence plot: {feature} ---")

    fig, ax = plt.subplots(figsize=(8, 6))

    shap.dependence_plot(
        feature,
        shap_vals_benign,
        X_test,
        interaction_index = "auto",
        ax                = ax,
        show              = False
    )

    ax.set_title(
        f"SHAP Dependence Plot — {feature}\n"
        f"Impact on Benign Classification Probability",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel("SHAP Value", fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.show()

# =============================================================================
# 12. PLOT 08 — SHAP HEATMAP (TEST SET OVERVIEW)
# =============================================================================

print("\n--- Generating Plot 08: SHAP Heatmap ---")

# Sort test observations by predicted probability (benign) for visual clarity
sort_order   = np.argsort(y_prob[:, 1])
shap_sorted  = shap_vals_benign[sort_order, :]
X_test_sorted = X_test.iloc[sort_order]

# Retain top 10 features only for readability
top10_features = mean_abs_shap["Feature"].head(10).tolist()
top10_idx      = [list(X_test.columns).index(f) for f in top10_features]

shap_top10 = shap_sorted[:, top10_idx]

fig, ax = plt.subplots(figsize=(12, 7))

sns.heatmap(
    shap_top10.T,
    cmap       = "RdBu_r",
    center     = 0,
    xticklabels= False,
    yticklabels= top10_features,
    linewidths = 0,
    cbar_kws   = {"label": "SHAP Value (positive = towards benign)"},
    ax         = ax
)

ax.set_title(
    "SHAP Value Heatmap — Top 10 Features Across All Test Observations\n"
    "Observations sorted left to right by predicted benign probability",
    fontsize=12, fontweight="bold", pad=12
)
ax.set_xlabel("Test Observations (sorted by predicted benign probability)", fontsize=10)
ax.set_ylabel("Feature", fontsize=11)
ax.tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig("plot_08_shap_heatmap.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.show()

# =============================================================================
# 13. FEATURE COMPARISON — SHAP VS RANDOM FOREST NATIVE IMPORTANCE
# =============================================================================

print("\n" + "=" * 60)
print("SECTION 8: SHAP VS NATIVE FEATURE IMPORTANCE COMPARISON")
print("=" * 60)

rf_importance = pd.DataFrame({
    "Feature"       : X_train.columns,
    "RF Importance" : rf_model.feature_importances_
}).sort_values("RF Importance", ascending=False).reset_index(drop=True)

comparison = mean_abs_shap.merge(rf_importance, on="Feature")
comparison["SHAP Rank"] = comparison["Mean |SHAP|"].rank(ascending=False).astype(int)
comparison["RF Rank"]   = comparison["RF Importance"].rank(ascending=False).astype(int)
comparison["Rank Delta"]= abs(comparison["SHAP Rank"] - comparison["RF Rank"])
comparison = comparison.sort_values("SHAP Rank").reset_index(drop=True)

print("\n--- Top 15 Features: SHAP vs Native Importance Rank ---")
print(comparison[["Feature", "Mean |SHAP|", "SHAP Rank",
                   "RF Importance", "RF Rank", "Rank Delta"]].head(15).to_string(index=False))

print(
    "\nNote: Differences between SHAP and native importance rankings are expected.\n"
    "Native Random Forest importance (mean impurity decrease) is computed on the\n"
    "training set and can be biased towards high-cardinality features. SHAP values\n"
    "are computed on the test set and reflect the actual marginal contribution of\n"
    "each feature to individual predictions, making them a more reliable indicator\n"
    "of true feature influence."
)

# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nModel                    : Random Forest")
print(f"n_estimators             : 150  |  max_depth: 10")
print(f"Test Accuracy            : {acc * 100:.2f}%")
print(f"SHAP Explainer           : TreeSHAP (exact, tree-native)")
print(f"Test observations        : {X_test.shape[0]}")
print(f"Features explained       : {X_test.shape[1]}")
print(f"\nTop 5 features by mean |SHAP value| (benign class):")
for _, row in mean_abs_shap.head(5).iterrows():
    print(f"  {row['Feature']:<30} {row['Mean |SHAP|']:.4f}")
print(f"\nLocal explanations produced for:")
print(f"  Malignant case — test index {malignant_idx}  "
      f"(P(malignant) = {y_prob[malignant_idx, 0]:.4f})")
print(f"  Benign case    — test index {benign_idx}  "
      f"(P(benign)    = {y_prob[benign_idx, 1]:.4f})")
print("=" * 60)
