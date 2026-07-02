"""
Feedforward Neural Network (Multi-Layer Perceptron)
====================================================
Dataset : UCI Adult Income ("Census Income") Dataset - cleaned extract
Source  : Output of the EDA project (adult_income_cleaned.csv), itself
          derived from UCI Machine Learning Repository, id=2
          https://archive.ics.uci.edu/dataset/2/adult

Goal    : Train a feedforward neural network (MLP) to predict whether an
          individual earns more or less than $50,000/year, building
          directly on the validated, cleaned dataset produced by the EDA
          project. This project resolves three issues flagged in that
          EDA as deliberately out of scope at the time: the
          education / education_num redundancy, the need for explicit
          encoding and scaling (irrelevant for the tree-based benchmark
          models used elsewhere in the portfolio, but essential here),
          and an explicit, justified strategy for the 76/24 class
          imbalance in the target.

Author  : Marc Grover
Portfolio: https://marcgrover-datascience.github.io/

Environment note: this script was developed and tested against
scikit-learn 1.8.0. OneHotEncoder's `sparse_output` parameter requires
scikit-learn >= 1.2; if running against an older environment, replace
`sparse_output=False` with `sparse=False` in the ColumnTransformer
definition below.

Input note: this script expects `adult_income_cleaned.csv`, produced by
the companion EDA project script, to be present in the same directory
(or at the path set in CLEANED_DATA_PATH below).
"""

import copy
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample, shuffle

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "adult_income_cleaned.csv")
PLOT_DPI = 150
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = PLOT_DPI
plt.rcParams["savefig.bbox"] = "tight"

plot_counter = 1


def save_plot(fig, name):
    """Save a figure as a sequentially numbered PNG and close it."""
    global plot_counter
    filename = f"plot_{plot_counter:02d}_{name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=PLOT_DPI)
    plt.show()
    plt.close(fig)
    print(f"Saved: {filename}")
    plot_counter += 1


# ----------------------------------------------------------------------------
# 1. Data Loading
# ----------------------------------------------------------------------------

print("=" * 80)
print("1. DATA LOADING")
print("=" * 80)

if not os.path.exists(CLEANED_DATA_PATH):
    raise FileNotFoundError(
        f"Could not find '{CLEANED_DATA_PATH}'. This script expects the "
        f"cleaned dataset produced by the EDA project (adult_income_cleaned.csv) "
        f"to be present in the same directory, or for CLEANED_DATA_PATH to be "
        f"updated to point to it."
    )

df = pd.read_csv(CLEANED_DATA_PATH)

print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ----------------------------------------------------------------------------
# 2. Data Validation (refresh)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. DATA VALIDATION (REFRESH)")
print("=" * 80)

# This dataset was already validated and cleaned in the EDA project. The
# checks below are a deliberately lightweight re-verification - confirming
# the handoff between the two projects was clean - rather than a repeat of
# the full EDA validation process.

print(f"\nData types:\n{df.dtypes}")

# --- 2.1 No disguised or standard missingness should remain ---
total_missing = df.isnull().sum().sum()
n_placeholder = (df.select_dtypes(include="object") == "?").sum().sum()
print(f"\nStandard (NaN) missing values: {total_missing}")
print(f"Literal '?' placeholder values remaining: {n_placeholder}")
if total_missing > 0 or n_placeholder > 0:
    raise ValueError(
        "Unexpected missing values found in the cleaned dataset. Re-run the "
        "EDA project script to regenerate adult_income_cleaned.csv."
    )

# --- 2.2 Target classes ---
target_col = "income"
print(f"\nTarget classes: {df[target_col].unique().tolist()}")
income_counts = df[target_col].value_counts()
income_props = (100 * income_counts / len(df)).round(2)
print(f"\nClass distribution:\n{income_counts}")
print(f"\nClass proportions (%):\n{income_props}")

# --- 2.3 education / education_num redundancy ---
# Confirm the two columns are in a strict 1:1 (bijective) relationship before
# relying on that assumption to justify dropping one of them.
mapping_check = df.groupby("education")["education_num"].nunique()
n_inconsistent = (mapping_check > 1).sum()
print(
    f"\neducation -> education_num mapping: {len(mapping_check)} distinct "
    f"education labels, {n_inconsistent} with inconsistent education_num "
    f"values (0 expected if the redundancy is a clean 1:1 encoding)."
)
if n_inconsistent > 0:
    raise ValueError(
        "education and education_num are not in a clean 1:1 relationship - "
        "re-examine before dropping either column."
    )

# --- 2.4 Capital gain / loss zero-inflation ---
pct_zero_gain = 100 * (df["capital_gain"] == 0).mean()
pct_zero_loss = 100 * (df["capital_loss"] == 0).mean()
print(
    f"\nZero-inflation: capital_gain {pct_zero_gain:.1f}% zero, "
    f"capital_loss {pct_zero_loss:.1f}% zero "
    f"(consistent with the EDA finding of ~91.7% / ~95.3%)."
)

# ----------------------------------------------------------------------------
# 3. Feature Engineering
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("3. FEATURE ENGINEERING")
print("=" * 80)

# --- 3.1 Resolve the education / education_num redundancy ---
# Both columns encode identical information (confirmed above). education_num
# is retained because it is already a clean ordinal integer, requiring no
# further encoding; the categorical education column is dropped to avoid
# feeding the network two perfectly correlated representations of the same
# feature, which would be redundant at best and could distort the relative
# influence of education in the learned weights at worst.
print("Dropping 'education' (redundant with 'education_num').")

# --- 3.2 Drop fnlwgt ---
# fnlwgt is the Census Bureau's final sampling weight, describing how many
# people in the broader US population each record is estimated to represent
# for survey-weighting purposes. It is not a property of the individual and
# carries no genuine income signal; it is dropped here rather than passed to
# the network as a spurious numeric feature.
print("Dropping 'fnlwgt' (a survey sampling weight, not an individual-level feature).")

# --- 3.3 Capital gain / loss: two-part feature engineering ---
# As flagged in the EDA Next Steps, capital_gain and capital_loss are heavily
# zero-inflated (~91.7% / ~95.3% zeros). Feeding the raw, heavily-skewed
# continuous values directly into a scaler would compress almost all of the
# population into a narrow band near zero, while the rare non-zero values -
# which are likely to carry a disproportionate amount of income signal -
# become extreme outliers after scaling. An explicit binary "has capital
# activity" flag is added alongside each continuous amount, giving the
# network a clean, low-noise signal for the common case (no capital
# activity) independent of the scaled magnitude for the rarer case where it
# occurred.
df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
print(
    f"Added 'has_capital_gain' ({df['has_capital_gain'].mean() * 100:.1f}% positive) "
    f"and 'has_capital_loss' ({df['has_capital_loss'].mean() * 100:.1f}% positive) flags."
)

# --- 3.4 Target encoding ---
df["target"] = (df[target_col] == ">50K").astype(int)
print(f"\nTarget encoded: 0 = '<=50K', 1 = '>50K'")

# --- 3.5 Final feature set ---
numeric_features = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
binary_features = ["has_capital_gain", "has_capital_loss"]
categorical_features = [
    "workclass", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
all_features = numeric_features + binary_features + categorical_features

X = df[all_features].copy()
y = df["target"].copy()

print(f"\nFinal feature set ({len(all_features)} pre-encoding columns):")
print(f"  Numeric (to scale)    : {numeric_features}")
print(f"  Binary (pass through) : {binary_features}")
print(f"  Categorical (one-hot) : {categorical_features}")

# ----------------------------------------------------------------------------
# 4. Train / Validation / Test Split
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("4. TRAIN / VALIDATION / TEST SPLIT")
print("=" * 80)

# A three-way split is used rather than the more common train/test split,
# because the validation portion below is used for manual early stopping
# (Section 7) and must remain genuinely untouched by the class-imbalance
# resampling applied in Section 6. If a validation set were instead carved
# out of the training data *after* oversampling, duplicated rows could end
# up split across both the training and validation portions - letting the
# model "validate" against near-copies of records it was trained on, and
# producing an artificially optimistic early-stopping signal. Splitting
# first and resampling only the final training portion avoids this leakage
# entirely.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE,
)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
    X_train_full, y_train_full, test_size=0.15, stratify=y_train_full, random_state=RANDOM_STATE,
)

print(f"Training set   (pre-resampling): {X_train_raw.shape[0]:>6,} rows")
print(f"Validation set (early stopping): {X_val_raw.shape[0]:>6,} rows")
print(f"Test set       (held out)      : {X_test.shape[0]:>6,} rows")
print(
    f"\nClass balance check - train: {y_train_raw.mean():.4f}, "
    f"val: {y_val_raw.mean():.4f}, test: {y_test.mean():.4f} "
    f"(proportion earning >50K; should be consistent across all three splits)"
)

# ----------------------------------------------------------------------------
# 5. Preprocessing (Scaling & Encoding)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("5. PREPROCESSING (SCALING & ENCODING)")
print("=" * 80)

# Unlike the tree-based models used elsewhere in this portfolio (which are
# invariant to monotonic feature transforms and split categorical features
# natively), an MLP is sensitive to both feature scale - large-magnitude
# inputs like capital_gain would otherwise dominate the first layer's
# gradients - and to any ordinal structure implied by raw category codes.
# Numeric features are therefore standardised, and categorical features are
# one-hot encoded so the network sees no false ordinal relationship between,
# for example, "Private" and "Self-emp-not-inc" workclass values.
#
# The preprocessor is fit on the training portion ONLY, before any
# resampling, so that the learned scaling parameters and the encoder's known
# category list both reflect a single, unduplicated, untouched sample of the
# population - avoiding any leakage from the validation or test sets, and
# avoiding distortion of the scaler's mean/variance estimates by the
# duplicated rows introduced during oversampling in Section 6.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("bin", "passthrough", binary_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_val_proc = preprocessor.transform(X_val_raw)
X_test_proc = preprocessor.transform(X_test)

encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
feature_names = numeric_features + binary_features + list(encoded_cat_names)

print(f"Preprocessor fitted on training data only ({X_train_raw.shape[0]:,} rows).")
print(f"Feature count after encoding: {X_train_proc.shape[1]} (from {len(all_features)} raw columns)")

# ----------------------------------------------------------------------------
# 6. Class Imbalance Handling (Training Set Only)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("6. CLASS IMBALANCE HANDLING (TRAINING SET ONLY)")
print("=" * 80)

# Design decision: random oversampling of the minority class, applied to the
# training portion only.
#
# Three alternatives were considered and ruled out:
#   1. Class weighting - scikit-learn's MLPClassifier does not expose a
#      class_weight or sample_weight parameter (unlike LogisticRegression or
#      tree-based estimators), so this option is not directly available.
#   2. Random undersampling - with a ~3.2:1 imbalance ratio, undersampling
#      the majority class to match the minority would discard roughly
#      26,000 majority-class training records. Given the dataset's already
#      modest size and feature count, that is a significant loss of signal
#      for no clear benefit over oversampling.
#   3. SMOTE - generates synthetic minority examples by interpolating
#      between real ones. This works naturally for continuous features, but
#      after one-hot encoding most of this feature space is binary; SMOTE
#      would interpolate those binary columns into invalid fractional
#      values (e.g. workclass_Private = 0.43), which a categorical feature
#      can never genuinely take. Random oversampling (exact duplication)
#      avoids fabricating such values.
#
# Oversampling is applied AFTER the train/validation/test split and AFTER
# the preprocessor is fit (Section 5), so neither the validation set, the
# test set, nor the scaler's learned statistics are affected by duplicated
# rows - only the gradient updates the network actually trains on see the
# rebalanced distribution.
train_majority_mask = (y_train_raw == 0).to_numpy()
train_minority_mask = (y_train_raw == 1).to_numpy()

X_train_majority = X_train_proc[train_majority_mask]
X_train_minority = X_train_proc[train_minority_mask]

n_majority = X_train_majority.shape[0]
n_minority = X_train_minority.shape[0]

X_train_minority_upsampled = resample(
    X_train_minority, replace=True, n_samples=n_majority, random_state=RANDOM_STATE,
)

X_train_balanced = np.vstack([X_train_majority, X_train_minority_upsampled])
y_train_balanced = np.concatenate([np.zeros(n_majority), np.ones(n_majority)])
X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=RANDOM_STATE)

print(f"Pre-resampling training set : {n_majority:,} majority / {n_minority:,} minority "
      f"({100 * n_minority / (n_majority + n_minority):.1f}% positive)")
print(f"Post-resampling training set: {n_majority:,} majority / {n_majority:,} minority (50.0% positive)")
print(f"Total training rows after oversampling: {X_train_balanced.shape[0]:,} "
      f"(validation and test sets are untouched and remain at their original imbalance)")

# --- Plot 1: Training class balance, before resampling ---
fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(x=["<=50K", ">50K"], y=[n_majority, n_minority], ax=ax)
ax.set_title("Training Set Class Balance — Before Resampling")
ax.set_xlabel("Income Band")
ax.set_ylabel("Number of Records")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
save_plot(fig, "train_class_balance_before")

# --- Plot 2: Training class balance, after resampling ---
fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(x=["<=50K", ">50K"], y=[n_majority, n_majority], ax=ax)
ax.set_title("Training Set Class Balance — After Random Oversampling")
ax.set_xlabel("Income Band")
ax.set_ylabel("Number of Records")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
save_plot(fig, "train_class_balance_after")

# ----------------------------------------------------------------------------
# 7. Model Architecture & Manual Early-Stopping Training Loop
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("7. MODEL ARCHITECTURE & MANUAL EARLY-STOPPING TRAINING LOOP")
print("=" * 80)

# Architecture and training design choices are made deliberately conservative
# throughout, given this script runs on a CPU-only laptop rather than a GPU
# workstation:
#   - Two modest hidden layers (64, 32 units) rather than a deep or wide
#     network - the engineered feature space (~90 columns after encoding) is
#     small enough that a larger network would risk overfitting the
#     duplicated training rows before adding meaningful capacity.
#   - A moderate mini-batch size (256) balances gradient stability against
#     the number of optimiser steps per epoch on a single CPU core.
#   - Manual early stopping (implemented below via partial_fit, rather than
#     MLPClassifier's built-in early_stopping=True) is used specifically
#     because the built-in option would carve its validation split from
#     whatever data is passed to fit() - which, if that were the
#     already-oversampled training set, would reintroduce exactly the
#     train/validation leakage discussed in Section 4. Monitoring against
#     the untouched, pre-resampling validation set instead gives an honest
#     stopping signal.
hidden_layer_sizes = (64, 32)
batch_size = 256
learning_rate_init = 0.001
alpha = 1e-4
max_epochs = 150
patience = 15

model = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation="relu",
    solver="adam",
    alpha=alpha,
    batch_size=batch_size,
    learning_rate_init=learning_rate_init,
    max_iter=1,            # one call to partial_fit = one epoch
    warm_start=True,
    random_state=RANDOM_STATE,
)

print(f"Architecture: input -> {hidden_layer_sizes[0]} -> {hidden_layer_sizes[1]} -> output (sigmoid)")
print(f"Batch size: {batch_size} | Learning rate: {learning_rate_init} | L2 alpha: {alpha}")
print(f"Max epochs: {max_epochs} | Early-stopping patience: {patience} epochs (monitored on held-out validation ROC-AUC)")

train_loss_history = []
val_auc_history = []
best_val_auc = -np.inf
best_epoch = 0
best_model = None
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(1, max_epochs + 1):
    model.partial_fit(X_train_balanced, y_train_balanced, classes=np.array([0, 1]))

    epoch_train_loss = model.loss_curve_[-1]
    epoch_val_auc = roc_auc_score(y_val_raw, model.predict_proba(X_val_proc)[:, 1])

    train_loss_history.append(epoch_train_loss)
    val_auc_history.append(epoch_val_auc)

    if epoch_val_auc > best_val_auc:
        best_val_auc = epoch_val_auc
        best_epoch = epoch
        best_model = copy.deepcopy(model)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(f"  Epoch {epoch:>3d}: train loss = {epoch_train_loss:.4f}, val ROC-AUC = {epoch_val_auc:.4f}")

    if epochs_without_improvement >= patience:
        print(f"  Early stopping triggered at epoch {epoch} "
              f"(no improvement for {patience} epochs; best was epoch {best_epoch}, val ROC-AUC = {best_val_auc:.4f}).")
        break

training_time = time.time() - start_time
model = best_model

print(f"\nTraining complete in {training_time:.1f} seconds ({len(train_loss_history)} epochs run).")
print(f"Best epoch: {best_epoch} (validation ROC-AUC = {best_val_auc:.4f}); weights from this epoch are used for evaluation.")

# --- Plot 3: Training loss and validation ROC-AUC by epoch ---
fig, ax1 = plt.subplots(figsize=(9, 6))
epochs_range = range(1, len(train_loss_history) + 1)
ax1.plot(epochs_range, train_loss_history, color="tab:blue", label="Training loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Training Loss", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(epochs_range, val_auc_history, color="tab:orange", label="Validation ROC-AUC")
ax2.set_ylabel("Validation ROC-AUC", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.axvline(best_epoch, color="grey", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("Training Loss and Validation ROC-AUC by Epoch")
save_plot(fig, "training_history")

# ----------------------------------------------------------------------------
# 8. Model Evaluation on Test Set
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("8. MODEL EVALUATION ON TEST SET")
print("=" * 80)

y_test_pred = model.predict(X_test_proc)
y_test_proba = model.predict_proba(X_test_proc)[:, 1]

print(f"\nClassification report (default 0.5 threshold):\n")
print(classification_report(y_test, y_test_pred, target_names=["<=50K", ">50K"]))

test_roc_auc = roc_auc_score(y_test, y_test_proba)
test_pr_auc = average_precision_score(y_test, y_test_proba)
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"Test PR-AUC (average precision): {test_pr_auc:.4f}")
print(f"(Test set positive class prevalence — the PR-AUC baseline for a random "
      f"classifier — is {y_test.mean():.4f})")

# --- Plot 4: Confusion matrix (default threshold) ---
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["<=50K", ">50K"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.grid(False)
ax.set_title("Confusion Matrix — Test Set (0.5 threshold)")
save_plot(fig, "confusion_matrix_default_threshold")

# --- Plot 5: ROC curve ---
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="tab:blue", label=f"MLP (AUC = {test_roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random classifier")
ax.set_title("ROC Curve — Test Set")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
save_plot(fig, "mlp_roc_curve")

# --- Plot 6: Precision-Recall curve ---
precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_test_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(recall_vals, precision_vals, color="tab:blue", label=f"MLP (PR-AUC = {test_pr_auc:.3f})")
ax.axhline(y_test.mean(), color="grey", linestyle="--", label=f"Random classifier ({y_test.mean():.3f})")
ax.set_title("Precision-Recall Curve — Test Set")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()
save_plot(fig, "precision_recall_curve")

# ----------------------------------------------------------------------------
# 9. Decision Threshold Tuning
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("9. DECISION THRESHOLD TUNING")
print("=" * 80)

# A 0.5 classification threshold is an arbitrary default, not a justified
# choice, particularly on an imbalanced target. The F1-optimal threshold is
# identified directly from the precision-recall curve computed above.
f1_scores = np.divide(
    2 * precision_vals * recall_vals,
    precision_vals + recall_vals,
    out=np.zeros_like(precision_vals),
    where=(precision_vals + recall_vals) != 0,
)
# precision_recall_curve returns one more precision/recall point than
# thresholds; drop the last point (which has no corresponding threshold) to
# align the arrays.
best_idx = np.argmax(f1_scores[:-1])
best_threshold = pr_thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"F1-optimal threshold: {best_threshold:.3f} (F1 = {best_f1:.4f}, "
      f"vs F1 = {f1_score(y_test, y_test_pred):.4f} at the default 0.5 threshold)")

y_test_pred_tuned = (y_test_proba >= best_threshold).astype(int)
print(f"\nClassification report (tuned {best_threshold:.3f} threshold):\n")
print(classification_report(y_test, y_test_pred_tuned, target_names=["<=50K", ">50K"]))

# --- Plot 7: Precision, recall and F1 vs threshold ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(pr_thresholds, precision_vals[:-1], label="Precision")
ax.plot(pr_thresholds, recall_vals[:-1], label="Recall")
ax.plot(pr_thresholds, f1_scores[:-1], label="F1 score")
ax.axvline(best_threshold, color="grey", linestyle="--", label=f"F1-optimal threshold ({best_threshold:.3f})")
ax.axvline(0.5, color="black", linestyle=":", alpha=0.6, label="Default threshold (0.5)")
ax.set_title("Precision, Recall and F1 Score vs Decision Threshold")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.legend()
save_plot(fig, "threshold_tuning")

# --- Plot 8: Confusion matrix (tuned threshold) ---
cm_tuned = confusion_matrix(y_test, y_test_pred_tuned)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm_tuned, display_labels=["<=50K", ">50K"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.grid(False)
ax.set_title(f"Confusion Matrix — Test Set ({best_threshold:.3f} threshold)")
save_plot(fig, "confusion_matrix_tuned_threshold")

# ----------------------------------------------------------------------------
# 10. Baseline Comparison (Logistic Regression)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("10. BASELINE COMPARISON (LOGISTIC REGRESSION)")
print("=" * 80)

# A simple linear baseline establishes whether the added complexity, training
# time, and reduced interpretability of an MLP is actually justified by a
# meaningful performance gain. Unlike MLPClassifier, LogisticRegression
# natively supports class_weight="balanced", so the baseline is trained
# directly on the original (pre-oversampling) training data using that
# native mechanism, rather than on the duplicated rows used for the network
# - giving each model its own, appropriately-matched imbalance strategy
# rather than forcing an artificial like-for-like comparison.
baseline_start = time.time()
baseline_model = LogisticRegression(
    class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
)
baseline_model.fit(X_train_proc, y_train_raw)
baseline_time = time.time() - baseline_start

baseline_pred = baseline_model.predict(X_test_proc)
baseline_proba = baseline_model.predict_proba(X_test_proc)[:, 1]
baseline_roc_auc = roc_auc_score(y_test, baseline_proba)
baseline_pr_auc = average_precision_score(y_test, baseline_proba)
baseline_f1 = f1_score(y_test, baseline_pred)

mlp_f1_default = f1_score(y_test, y_test_pred)

print(f"Logistic Regression baseline trained in {baseline_time:.2f}s "
      f"(MLP took {training_time:.1f}s, {training_time / max(baseline_time, 1e-9):.0f}x longer).")
print(f"\n{'Metric':<20s}{'MLP (0.5 thr.)':>18s}{'Logistic Reg.':>18s}")
print(f"{'ROC-AUC':<20s}{test_roc_auc:>18.4f}{baseline_roc_auc:>18.4f}")
print(f"{'PR-AUC':<20s}{test_pr_auc:>18.4f}{baseline_pr_auc:>18.4f}")
print(f"{'F1 score':<20s}{mlp_f1_default:>18.4f}{baseline_f1:>18.4f}")

# --- Plot 9: MLP vs Logistic Regression benchmark comparison ---
comparison_df = pd.DataFrame({
    "Metric": ["ROC-AUC", "PR-AUC", "F1 Score"] * 2,
    "Score": [test_roc_auc, test_pr_auc, mlp_f1_default,
              baseline_roc_auc, baseline_pr_auc, baseline_f1],
    "Model": ["MLP"] * 3 + ["Logistic Regression"] * 3,
})
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=comparison_df, x="Metric", y="Score", hue="Model", ax=ax)
ax.set_title("MLP vs Logistic Regression Baseline — Test Set Performance")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=9)
save_plot(fig, "mlp_vs_logistic_regression_benchmark")

# ----------------------------------------------------------------------------
# 11. Artifact Export (for downstream LIME explainability project)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("11. ARTIFACT EXPORT")
print("=" * 80)

# This project is the second step in an EDA -> MLP -> LIME portfolio arc. The
# fitted preprocessor and model, along with the raw (untransformed) test
# split, are persisted here so the LIME project can load them directly
# without repeating any of the cleaning, splitting, or training steps above.
model_path = os.path.join(OUTPUT_DIR, "mlp_model.joblib")
preprocessor_path = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
X_test_path = os.path.join(OUTPUT_DIR, "X_test_raw.csv")
y_test_path = os.path.join(OUTPUT_DIR, "y_test.csv")

joblib.dump(model, model_path)
joblib.dump(preprocessor, preprocessor_path)
X_test.to_csv(X_test_path, index=False)
y_test.to_csv(y_test_path, index=False)

print(f"Saved trained model to       : {model_path}")
print(f"Saved fitted preprocessor to : {preprocessor_path}")
print(f"Saved raw test features to   : {X_test_path}")
print(f"Saved test labels to         : {y_test_path}")

print("\n" + "=" * 80)
print("SCRIPT COMPLETE")
print("=" * 80)