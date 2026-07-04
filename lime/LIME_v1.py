"""
Model Interpretability with LIME (Local Interpretable Model-agnostic Explanations)
====================================================================================
Dataset : UCI Adult Income ("Census Income") Dataset - cleaned extract
Source  : Artifacts exported by the MLP project (mlp_model.joblib,
          preprocessor.joblib, X_test_raw.csv, y_test.csv), itself built on
          the cleaned dataset produced by the EDA project.

Goal    : Generate individual-level, human-readable explanations for the
          feedforward neural network's predictions, using LIME. Four
          representative test-set cases are explained: a confident true
          positive, a confident false positive, a confident false negative,
          and a case sitting right on the model's tuned decision threshold.

Design note: LIME explanations are generated in the ORIGINAL 14 raw feature
columns (age, workclass, education_num, ... native_country), not the 92
one-hot/scaled columns the network actually receives. This is a deliberate
choice for readability - an explanation phrased as "occupation =
Exec-managerial" is meaningfully more useful to a human reader than one
phrased in terms of a one-hot column name. This requires a small amount of
extra plumbing (Section 3 below) to translate between LIME's internal
numeric representation and the raw feature values, and between raw features
and the encoded representation the model was actually trained on.

Author  : Marc Grover
Portfolio: https://marcgrover-datascience.github.io/

Environment note: this script requires the `lime` package
(`pip install lime`), in addition to the scikit-learn / pandas / seaborn
stack used elsewhere in this portfolio.

Input note: this script expects mlp_model.joblib, preprocessor.joblib,
X_test_raw.csv, and y_test.csv - all produced by the companion MLP project
script - to be present in the same directory (or at the paths set in the
Configuration section below).
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUTPUT_DIR, "mlp_model.joblib")
PREPROCESSOR_PATH = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test_raw.csv")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.csv")

PLOT_DPI = 150
RANDOM_STATE = 42

# The F1-optimal threshold identified in the MLP project. Used here, rather
# than the default 0.5, so that the true/false positive/negative examples
# selected below reflect the model's actual recommended operating point.
DECISION_THRESHOLD = 0.626

# Raw (pre-encoding) feature groups, matching the MLP project exactly.
NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
BINARY_FEATURES = ["has_capital_gain", "has_capital_loss"]
CATEGORICAL_FEATURES = [
    "workclass", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
CLASS_NAMES = ["<=50K", ">50K"]

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = PLOT_DPI
plt.rcParams["savefig.bbox"] = "tight"

plot_counter = 1


def save_plot(fig, name):
    """Save a figure as a sequentially numbered PNG, display it, and close it."""
    global plot_counter
    filename = f"plot_{plot_counter:02d}_{name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=PLOT_DPI)
    plt.show()
    plt.close(fig)
    print(f"Saved: {filename}")
    plot_counter += 1


# ----------------------------------------------------------------------------
# 1. Data & Artifact Loading
# ----------------------------------------------------------------------------

print("=" * 80)
print("1. DATA & ARTIFACT LOADING")
print("=" * 80)

for path, description in [
    (MODEL_PATH, "trained MLP model"),
    (PREPROCESSOR_PATH, "fitted preprocessor"),
    (X_TEST_PATH, "raw test features"),
    (Y_TEST_PATH, "test labels"),
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find the {description} at '{path}'. This script expects "
            f"the artifacts exported by the MLP project to be present in the "
            f"same directory, or for the relevant path constant to be updated."
        )

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
X_test_raw = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

# Column order matters throughout this script - the raw CSV columns are
# re-ordered here to guarantee they match ALL_FEATURES exactly, since LIME's
# internal representation is a plain numeric array with no column names.
X_test_raw = X_test_raw[ALL_FEATURES]

print(f"Loaded MLP model: {model.hidden_layer_sizes} hidden layers")
print(f"Loaded test set: {X_test_raw.shape[0]} rows, {X_test_raw.shape[1]} raw features")
print(f"Decision threshold in use: {DECISION_THRESHOLD} (F1-optimal, from the MLP project)")

# ----------------------------------------------------------------------------
# 2. Data Validation
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. DATA VALIDATION")
print("=" * 80)

assert X_test_raw.shape[0] == y_test.shape[0], "Feature and label row counts do not match."
assert X_test_raw.isnull().sum().sum() == 0, "Unexpected missing values in the loaded test features."
assert set(y_test.unique()) <= {0, 1}, "Unexpected label values in y_test (expected 0/1)."

n_expected_raw_cols = len(ALL_FEATURES)
assert X_test_raw.shape[1] == n_expected_raw_cols, (
    f"Expected {n_expected_raw_cols} raw feature columns, found {X_test_raw.shape[1]}."
)
print(f"Row count match, no missing values, and expected column count all confirmed "
      f"({X_test_raw.shape[0]} rows x {n_expected_raw_cols} columns).")

# Sanity-check the model/preprocessor pairing: the preprocessor's expected
# output width must match what the model was actually trained on.
n_encoded_cols = preprocessor.transform(X_test_raw.head(1)).shape[1]
assert n_encoded_cols == model.n_features_in_, (
    f"Preprocessor output ({n_encoded_cols} columns) does not match the "
    f"model's expected input ({model.n_features_in_} columns) - the model "
    f"and preprocessor artifacts do not appear to be a matching pair."
)
print(f"Model/preprocessor compatibility confirmed: {n_encoded_cols} encoded columns.")

# ----------------------------------------------------------------------------
# 3. Prediction Function & LIME-Compatible Feature Representation
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("3. PREDICTION FUNCTION & LIME-COMPATIBLE FEATURE REPRESENTATION")
print("=" * 80)

# LIME's tabular explainer perturbs instances in a single numeric array, and
# requires categorical columns to already be integer-coded (with a lookup
# table back to their original string labels for display purposes). The raw
# test set is therefore label-encoded here purely for LIME's internal use;
# the model itself never sees this representation.
category_lookup = {}          # {feature_name: [category_0, category_1, ...]}
X_test_lime = X_test_raw.copy()

for col in CATEGORICAL_FEATURES:
    codes, uniques = pd.factorize(X_test_raw[col], sort=True)
    X_test_lime[col] = codes
    category_lookup[col] = list(uniques)

X_test_lime_array = X_test_lime.to_numpy(dtype=float)
feature_index = {name: i for i, name in enumerate(ALL_FEATURES)}
categorical_feature_indices = [feature_index[c] for c in CATEGORICAL_FEATURES + BINARY_FEATURES]

# categorical_names must cover every index passed in categorical_features,
# including the two already-binary flag columns (given readable labels here
# rather than left as bare 0/1 integers).
categorical_names = {feature_index[c]: [str(v) for v in category_lookup[c]] for c in CATEGORICAL_FEATURES}
categorical_names[feature_index["has_capital_gain"]] = ["No", "Yes"]
categorical_names[feature_index["has_capital_loss"]] = ["No", "Yes"]

print(f"Label-encoded {len(CATEGORICAL_FEATURES)} categorical columns for LIME's internal use.")
print(f"Categorical + binary feature indices passed to LIME: {categorical_feature_indices}")


def predict_fn(lime_array):
    """
    Translate LIME's perturbed numeric array back into model-ready
    predictions. LIME calls this function repeatedly with batches of
    synthetic, perturbed instances during explanation generation.

    Every perturbed instance arrives in the same numeric layout as
    X_test_lime_array: numeric/binary columns as raw floats, categorical
    columns as integer codes. This function reverses the categorical
    encoding to recover the original string categories, rounds the binary
    flags back to clean 0/1 integers, reassembles a properly-typed
    DataFrame in the exact column order the fitted preprocessor expects,
    and returns class probabilities from the trained MLP.
    """
    df = pd.DataFrame(lime_array, columns=ALL_FEATURES)

    for col in CATEGORICAL_FEATURES:
        categories = category_lookup[col]
        # Perturbed codes are continuous-valued in general; round and clip
        # to the nearest valid category index before mapping back to a label.
        codes = df[col].round().clip(0, len(categories) - 1).astype(int)
        df[col] = codes.map(lambda i: categories[i])

    for col in BINARY_FEATURES:
        df[col] = df[col].round().clip(0, 1).astype(int)

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)

    X_encoded = preprocessor.transform(df)
    return model.predict_proba(X_encoded)


# Quick round-trip check: predictions from the reconstructed representation
# should exactly match predictions from the original raw data, confirming
# predict_fn introduces no distortion.
proba_direct = model.predict_proba(preprocessor.transform(X_test_raw))[:, 1]
proba_roundtrip = predict_fn(X_test_lime_array)[:, 1]
max_discrepancy = np.abs(proba_direct - proba_roundtrip).max()
print(f"Round-trip check (raw prediction vs. predict_fn(lime_array)): "
      f"max discrepancy = {max_discrepancy:.2e} (should be ~0).")
assert max_discrepancy < 1e-6, "predict_fn round-trip does not match direct predictions - check encoding logic."

# ----------------------------------------------------------------------------
# 4. LIME Explainer Setup
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("4. LIME EXPLAINER SETUP")
print("=" * 80)

# The explainer is fitted using the test set itself as its reference
# distribution for perturbation statistics (feature means/stds for numeric
# columns, category frequencies for categorical columns). Ideally this would
# be the original training set, but only the test split was persisted by the
# MLP project; at 9,763 rows, the test set is a large enough, representative
# enough sample of the same underlying population to serve this purpose
# reasonably well, and - being held out from training - carries no risk of
# LIME's perturbation statistics being distorted by the earlier oversampling
# applied to the training set only.
explainer = LimeTabularExplainer(
    training_data=X_test_lime_array,
    feature_names=ALL_FEATURES,
    class_names=CLASS_NAMES,
    categorical_features=categorical_feature_indices,
    categorical_names=categorical_names,
    discretize_continuous=True,
    random_state=RANDOM_STATE,
)

print(f"Explainer initialised on {X_test_lime_array.shape[0]} reference rows "
      f"({len(ALL_FEATURES)} features, {len(categorical_feature_indices)} treated as categorical).")

# ----------------------------------------------------------------------------
# 5. Selecting Representative Cases
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("5. SELECTING REPRESENTATIVE CASES")
print("=" * 80)

y_test_array = y_test.to_numpy()
proba_positive = proba_direct  # predicted P(income > 50K), computed in Section 3
y_pred = (proba_positive >= DECISION_THRESHOLD).astype(int)

is_tp = (y_test_array == 1) & (y_pred == 1)
is_fp = (y_test_array == 0) & (y_pred == 1)
is_fn = (y_test_array == 1) & (y_pred == 0)

# True positive and false positive: the most CONFIDENT example of each,
# i.e. furthest from the threshold, to give the clearest, least ambiguous
# illustration of what drives that kind of prediction.
idx_tp = np.where(is_tp)[0][np.argmax(proba_positive[is_tp])]
idx_fp = np.where(is_fp)[0][np.argmax(proba_positive[is_fp])]

# False negative: the most confidently WRONG example - i.e. the true >50K
# case the model was most confident did NOT earn >50K - since this is the
# most informative failure case to explain.
idx_fn = np.where(is_fn)[0][np.argmin(proba_positive[is_fn])]

# Borderline case: whichever test-set row sits closest to the decision
# threshold itself, regardless of which side it falls on or whether the
# prediction was correct.
idx_borderline = np.argmin(np.abs(proba_positive - DECISION_THRESHOLD))

selected_cases = {
    "true_positive": idx_tp,
    "false_positive": idx_fp,
    "false_negative": idx_fn,
    "borderline_case": idx_borderline,
}

print("Selected cases (row index, true label, predicted probability of >50K):")
for case_name, idx in selected_cases.items():
    print(f"  {case_name:<16s}: row {idx:>5d} | true = {CLASS_NAMES[y_test_array[idx]]:>6s} "
          f"| P(>50K) = {proba_positive[idx]:.3f} "
          f"| predicted = {CLASS_NAMES[int(proba_positive[idx] >= DECISION_THRESHOLD)]}")

# ----------------------------------------------------------------------------
# 6. Generating and Visualising Explanations
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("6. GENERATING AND VISUALISING EXPLANATIONS")
print("=" * 80)

case_titles = {
    "true_positive": "True Positive — Correctly Predicted >50K",
    "false_positive": "False Positive — Incorrectly Predicted >50K",
    "false_negative": "False Negative — Incorrectly Predicted <=50K",
    "borderline_case": f"Borderline Case — Near the {DECISION_THRESHOLD} Decision Threshold",
}

explanation_records = []

for case_name, idx in selected_cases.items():
    instance = X_test_lime_array[idx]

    explanation = explainer.explain_instance(
        instance,
        predict_fn,
        num_features=10,
        num_samples=5000,
    )

    predicted_proba = proba_positive[idx]
    true_label = CLASS_NAMES[y_test_array[idx]]
    predicted_label = CLASS_NAMES[int(predicted_proba >= DECISION_THRESHOLD)]

    print(f"\n--- {case_titles[case_name]} ---")
    print(f"True label: {true_label} | Predicted: {predicted_label} "
          f"(P(>50K) = {predicted_proba:.3f}, threshold = {DECISION_THRESHOLD})")
    print("Raw feature values for this individual:")
    print(X_test_raw.iloc[idx].to_string())
    print("\nTop contributing factors (feature: weight toward >50K):")
    for feature_description, weight in explanation.as_list():
        print(f"  {feature_description:<45s} {weight:+.4f}")

    # --- Plot: LIME explanation as a horizontal bar chart ---
    exp_list = explanation.as_list()
    labels = [item[0] for item in exp_list]
    weights = [item[1] for item in exp_list]
    colors = ["#2b6f39" if w > 0 else "#a83232" for w in weights]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, weights, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to P(income > $50K)")
    ax.set_title(
        f"{case_titles[case_name]}\n"
        f"True: {true_label} | Predicted: {predicted_label} | P(>50K) = {predicted_proba:.3f}"
    )
    save_plot(fig, f"lime_explanation_{case_name}")

    explanation_records.append({
        "case": case_name,
        "row_index": idx,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "predicted_probability": predicted_proba,
        "top_factors": exp_list,
    })

# ----------------------------------------------------------------------------
# 7. Explanation Fidelity Check
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("7. EXPLANATION FIDELITY CHECK")
print("=" * 80)

# LIME's explanations come from a simple, local linear surrogate model
# fitted only around each individual instance - it is important to check
# that surrogate actually fits well locally, rather than assuming every
# generated explanation is trustworthy by default. R^2 below this level
# indicates the local surrogate is a poor local approximation of the
# network's true decision boundary at that point, and the explanation
# should be treated with more caution.
FIDELITY_WARNING_THRESHOLD = 0.5

fidelity_scores = []
for case_name, idx in selected_cases.items():
    instance = X_test_lime_array[idx]
    explanation = explainer.explain_instance(instance, predict_fn, num_features=10, num_samples=5000)
    score = explanation.score
    fidelity_scores.append(score)
    flag = "" if score >= FIDELITY_WARNING_THRESHOLD else "  <-- below fidelity threshold, interpret with caution"
    print(f"  {case_name:<16s}: local surrogate R^2 = {score:.3f}{flag}")

# --- Plot: fidelity scores across the four explanations ---
fig, ax = plt.subplots(figsize=(7, 5))
case_labels = [case_titles[c].split(" — ")[0] for c in selected_cases.keys()]
bar_colors = ["#3b6fa0" if s >= FIDELITY_WARNING_THRESHOLD else "#c98a2a" for s in fidelity_scores]
ax.bar(case_labels, fidelity_scores, color=bar_colors)
ax.axhline(FIDELITY_WARNING_THRESHOLD, color="grey", linestyle="--",
           label=f"Caution threshold ({FIDELITY_WARNING_THRESHOLD})")
ax.set_ylim(0, 1)
ax.set_ylabel("Local Surrogate Model R²")
ax.set_title("LIME Explanation Fidelity by Case")
ax.legend()
plt.xticks(rotation=15, ha="right")
save_plot(fig, "explanation_fidelity")

# ----------------------------------------------------------------------------
# 8. Summary Export
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("8. SUMMARY EXPORT")
print("=" * 80)

# A flat CSV summarising every explanation is exported for reference and for
# potential reuse in the project write-up, alongside the plots above.
summary_rows = []
for record in explanation_records:
    for feature_description, weight in record["top_factors"]:
        summary_rows.append({
            "case": record["case"],
            "row_index": record["row_index"],
            "true_label": record["true_label"],
            "predicted_label": record["predicted_label"],
            "predicted_probability": record["predicted_probability"],
            "feature_description": feature_description,
            "weight": weight,
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "lime_explanations_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved explanation summary to: {summary_path}")

print("SCRIPT COMPLETE")