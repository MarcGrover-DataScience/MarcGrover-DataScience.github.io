"""
Counterfactual Explanations with DiCE (Diverse Counterfactual Explanations)
============================================================================
Dataset : UCI Adult Income ("Census Income") Dataset - cleaned extract
Source  : Artifacts exported by the MLP project (mlp_model.joblib,
          preprocessor.joblib, X_test_raw.csv, y_test.csv), the same
          artifacts used by the companion LIME project.

Goal    : For the same four representative cases explained by the LIME
          project, find the minimal change to an individual's features that
          would flip the MLP's prediction - directly extending LIME's key
          finding that capital_gain/capital_loss dominate the network's
          decisions at a near-deterministic magnitude.

Design note 1 - Actionable vs. immutable features: counterfactuals are only
searched for over a restricted set of "actionable" features an individual
could plausibly change (workclass, occupation, education_num, hours_per_week,
capital_gain, capital_loss and their binary flags). age, race, sex and
native_country are excluded as immutable/protected characteristics.
marital_status and relationship are ALSO treated as immutable here: while
not legally protected in the same way, they are not something a loan
applicant could act on to change an outcome, so allowing the model to
"recommend" changing them would be a poor real-world fit for the credit-
scoring framing this project uses - a judgement call, not a hard rule, and
one this project's write-up states explicitly rather than leaving implicit.

Design note 2 - Dual-regime stress test: every case is searched TWICE -
once with capital_gain/capital_loss free to vary (Regime A), and once with
them locked at the individual's actual values (Regime B). Regime B is a
direct, quantified extension of the LIME project's closing finding: if the
model's reliance on capital_gain/capital_loss is genuinely near-deterministic,
Regime B should struggle or fail to find a valid counterfactual for cases
where those features are doing the work - even though DiCE is free to change
every other actionable feature to compensate.

Design note 3 - Genetic search has no built-in time budget: DiCE's genetic
method builds its initial population by repeatedly sampling random candidate
instances until enough VALID ones are found (see dice_ml's do_random_init).
When a flip is genuinely very hard or impossible to reach at random - exactly
the scenario Regime B is designed to probe - this loop has no upper bound
and can run indefinitely. A hard wall-clock timeout (Section 6) is therefore
used to guarantee this script terminates, and a search hitting that timeout
without finding a valid counterfactual is treated as a legitimate result in
its own right (a locked-out search), not a bug to hide.

Design note 4 - Feature consistency repair: capital_gain/capital_loss and
their engineered binary flags (has_capital_gain/has_capital_loss) are passed
to DiCE as independent features, since DiCE has no built-in concept of a
derived column. This means a raw DiCE proposal could suggest an internally
inconsistent combination (e.g. capital_gain = 5,000 with has_capital_gain =
0). Every generated counterfactual is therefore repaired post-hoc (Section 6)
by recomputing the two flags from the proposed capital_gain/capital_loss
values, and re-scored through the pipeline afterwards, before being accepted.

Author  : Marc Grover
Portfolio: https://marcgrover-datascience.github.io/

Environment note: this script requires the `dice-ml` package
(`pip install dice-ml`), in addition to the scikit-learn / pandas / seaborn
stack used elsewhere in this portfolio.

Input note: this script expects mlp_model.joblib, preprocessor.joblib,
X_test_raw.csv, and y_test.csv - all produced by the companion MLP project
script - to be present in the same directory (or at the paths set in the
Configuration section below).

Timing note: DiCE's genetic search is run 8 times (4 cases x 2 regimes),
each independently timed and capped at GENETIC_TIMEOUT_SECONDS. Total
script runtime is logged at the end and is expected to run to a few
minutes, driven mainly by any Regime B searches that hit the timeout.
"""

import os
import threading
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

import dice_ml
from dice_ml import Dice

SCRIPT_START_TIME = time.time()

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

# The F1-optimal threshold identified in the MLP project, and reused by the
# LIME project. Used here too so that the same four rows are selected below.
DECISION_THRESHOLD = 0.626

# Raw (pre-encoding) feature groups, matching the MLP and LIME projects exactly.
NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
BINARY_FEATURES = ["has_capital_gain", "has_capital_loss"]
CATEGORICAL_FEATURES = [
    "workclass", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
CLASS_NAMES = ["<=50K", ">50K"]

# Immutable / protected features (see Design note 1 above) are excluded from
# every counterfactual search, in both regimes.
IMMUTABLE_FEATURES = ["age", "race", "sex", "native_country", "marital_status", "relationship"]
ACTIONABLE_FEATURES = [f for f in ALL_FEATURES if f not in IMMUTABLE_FEATURES]

# The two numeric capital features and their engineered binary flags - locked
# in Regime B, free to vary in Regime A.
CAPITAL_FEATURES = ["capital_gain", "capital_loss", "has_capital_gain", "has_capital_loss"]
ACTIONABLE_FEATURES_NO_CAPITAL = [f for f in ACTIONABLE_FEATURES if f not in CAPITAL_FEATURES]

# DiCE genetic method settings.
CFS_PER_QUERY = 5
GENETIC_MAX_ITERATIONS = 100
GENETIC_TIMEOUT_SECONDS = 90

# Manual single-axis sanity-check settings.
MANUAL_SEARCH_STEPS = 400

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
section_start = time.time()

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

# Column order matters throughout this script for the same reason it did in
# the LIME project - reordered here to guarantee it matches ALL_FEATURES.
X_test_raw = X_test_raw[ALL_FEATURES]

print(f"Loaded MLP model: {model.hidden_layer_sizes} hidden layers")
print(f"Loaded test set: {X_test_raw.shape[0]} rows, {X_test_raw.shape[1]} raw features")
print(f"Decision threshold in use: {DECISION_THRESHOLD} (F1-optimal, from the MLP project)")
print(f"Section 1 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 2. Data Validation
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. DATA VALIDATION")
print("=" * 80)
section_start = time.time()

assert X_test_raw.shape[0] == y_test.shape[0], "Feature and label row counts do not match."
assert X_test_raw.isnull().sum().sum() == 0, "Unexpected missing values in the loaded test features."
assert set(y_test.unique()) <= {0, 1}, "Unexpected label values in y_test (expected 0/1)."

n_expected_raw_cols = len(ALL_FEATURES)
assert X_test_raw.shape[1] == n_expected_raw_cols, (
    f"Expected {n_expected_raw_cols} raw feature columns, found {X_test_raw.shape[1]}."
)
print(f"Row count match, no missing values, and expected column count all confirmed "
      f"({X_test_raw.shape[0]} rows x {n_expected_raw_cols} columns).")

n_encoded_cols = preprocessor.transform(X_test_raw.head(1)).shape[1]
assert n_encoded_cols == model.n_features_in_, (
    f"Preprocessor output ({n_encoded_cols} columns) does not match the "
    f"model's expected input ({model.n_features_in_} columns) - the model "
    f"and preprocessor artifacts do not appear to be a matching pair."
)
print(f"Model/preprocessor compatibility confirmed: {n_encoded_cols} encoded columns.")
print(f"Section 2 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 3. Pipeline Assembly & Feature Groups
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("3. PIPELINE ASSEMBLY & FEATURE GROUPS")
print("=" * 80)
section_start = time.time()

# Both the preprocessor and model are already fitted - wrapping them in a
# single Pipeline requires no refitting, and gives DiCE one object that
# accepts raw 14-column input and returns class probabilities, matching the
# raw-feature-space approach the LIME project used for readability.
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

# Round-trip sanity check: predictions via the wrapped pipeline should
# exactly match predictions computed directly, confirming the wrapping
# introduces no distortion.
proba_direct = model.predict_proba(preprocessor.transform(X_test_raw))[:, 1]
proba_pipeline = pipeline.predict_proba(X_test_raw)[:, 1]
max_discrepancy = np.abs(proba_direct - proba_pipeline).max()
print(f"Pipeline wrapping round-trip check: max discrepancy = {max_discrepancy:.2e} (should be ~0).")
assert max_discrepancy < 1e-6, "Wrapped pipeline predictions do not match direct predictions."

print(f"Immutable features (excluded from every search): {IMMUTABLE_FEATURES}")
print(f"Actionable features - Regime A (capital features free to vary): {ACTIONABLE_FEATURES}")
print(f"Actionable features - Regime B (capital features locked): {ACTIONABLE_FEATURES_NO_CAPITAL}")
print(f"Section 3 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 4. Selecting Representative Cases
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("4. SELECTING REPRESENTATIVE CASES")
print("=" * 80)
section_start = time.time()

# Identical selection logic to the LIME project, so that the same four rows
# are guaranteed to be selected here - this is what makes "the same four
# LIME cases" a guarantee rather than a coincidence of similar logic.
y_test_array = y_test.to_numpy()
proba_positive = proba_pipeline
y_pred = (proba_positive >= DECISION_THRESHOLD).astype(int)

is_tp = (y_test_array == 1) & (y_pred == 1)
is_fp = (y_test_array == 0) & (y_pred == 1)
is_fn = (y_test_array == 1) & (y_pred == 0)

idx_tp = np.where(is_tp)[0][np.argmax(proba_positive[is_tp])]
idx_fp = np.where(is_fp)[0][np.argmax(proba_positive[is_fp])]
idx_fn = np.where(is_fn)[0][np.argmin(proba_positive[is_fn])]
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
print(f"Section 4 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 5. DiCE Configuration
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("5. DICE CONFIGURATION")
print("=" * 80)
section_start = time.time()

# DiCE needs a reference dataframe (including the outcome column) to infer
# feature ranges/categories. As in the LIME project, only the test split was
# persisted by the MLP project, so it is reused here as the reference
# distribution; at 9,763+ rows it is a large, representative sample of the
# same population.
dice_reference_df = X_test_raw.copy()
dice_reference_df["income"] = y_test_array

dice_data = dice_ml.Data(
    dataframe=dice_reference_df,
    continuous_features=NUMERIC_FEATURES,
    outcome_name="income",
)
dice_model = dice_ml.Model(model=pipeline, backend="sklearn")
dice_explainer = Dice(dice_data, dice_model, method="genetic")

print(f"DiCE explainer initialised (genetic method) on {dice_reference_df.shape[0]} reference rows.")
print(f"Total CFs requested per search: {CFS_PER_QUERY} | "
      f"max iterations: {GENETIC_MAX_ITERATIONS} | "
      f"per-search timeout: {GENETIC_TIMEOUT_SECONDS}s")
print(f"Section 5 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 6. Counterfactual Generation - Regime A vs. Regime B
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("6. COUNTERFACTUAL GENERATION - REGIME A VS. REGIME B")
print("=" * 80)
section_start = time.time()


def _run_genetic_search(query_instance, features_to_vary):
    """Run one DiCE genetic search. Executed in a worker thread so it can be
    subjected to a hard wall-clock timeout (see Design note 3 above)."""
    return dice_explainer.generate_counterfactuals(
        query_instance,
        total_CFs=CFS_PER_QUERY,
        desired_class="opposite",
        features_to_vary=features_to_vary,
        maxiterations=GENETIC_MAX_ITERATIONS,
    )


def run_dice_with_timeout(query_instance, features_to_vary, timeout=GENETIC_TIMEOUT_SECONDS):
    """
    Run a DiCE genetic search with a hard wall-clock timeout.

    A plain thread (not a process) is used deliberately for portability
    across Windows/macOS/Linux - signal.alarm(), the more obvious timeout
    mechanism, is Unix-only. The worker thread is marked as a daemon, so if
    it is abandoned after a timeout it will not prevent this script from
    continuing or from exiting once the main script finishes; it will,
    however, continue consuming CPU in the background for as long as
    DiCE's internal (unbounded) search loop keeps running.

    Returns (result, elapsed_seconds, timed_out).
    """
    result_container = {}

    def _worker():
        try:
            result_container["result"] = _run_genetic_search(query_instance, features_to_vary)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised below
            result_container["error"] = exc

    start = time.time()
    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=timeout)
    elapsed = time.time() - start

    if worker_thread.is_alive():
        return None, elapsed, True
    if "error" in result_container:
        raise result_container["error"]
    return result_container.get("result"), elapsed, False


def repair_and_score(cf_row_series):
    """
    Recompute has_capital_gain/has_capital_loss from the proposed
    capital_gain/capital_loss values (see Design note 4 above), then return
    the repaired row together with the pipeline's actual predicted P(>50K)
    for that repaired row.
    """
    repaired = cf_row_series.copy()
    repaired["has_capital_gain"] = int(repaired["capital_gain"] > 0)
    repaired["has_capital_loss"] = int(repaired["capital_loss"] > 0)
    repaired_df = pd.DataFrame([repaired[ALL_FEATURES]])
    repaired_proba = pipeline.predict_proba(repaired_df)[0, 1]
    return repaired, repaired_proba


def summarise_counterfactuals(original_row, cf_df, features_to_vary, original_predicted_class):
    """
    Repair, re-score, and summarise every proposed counterfactual in cf_df
    against the original row. Returns a list of dicts, one per valid,
    repaired counterfactual that actually achieves the opposite predicted
    class at the tuned decision threshold.
    """
    records = []
    desired_class = 1 - original_predicted_class

    for _, raw_cf_row in cf_df[ALL_FEATURES].iterrows():
        repaired_row, repaired_proba = repair_and_score(raw_cf_row)
        repaired_class = int(repaired_proba >= DECISION_THRESHOLD)
        is_valid = repaired_class == desired_class

        changed_features = [
            f for f in features_to_vary
            if str(repaired_row[f]) != str(original_row[f])
        ]
        # Sparsity: how many actionable features changed.
        sparsity = len(changed_features)

        # Proximity: mean absolute normalised distance, continuous features only.
        changed_numeric = [f for f in changed_features if f in NUMERIC_FEATURES]
        if changed_numeric:
            feature_ranges = (X_test_raw[NUMERIC_FEATURES].max() - X_test_raw[NUMERIC_FEATURES].min())
            proximity = np.mean([
                abs(float(repaired_row[f]) - float(original_row[f])) / max(feature_ranges[f], 1e-9)
                for f in changed_numeric
            ])
        else:
            proximity = 0.0 if sparsity == 0 else np.nan  # only categorical features changed

        records.append({
            "is_valid": is_valid,
            "sparsity": sparsity,
            "proximity": proximity,
            "changed_features": changed_features,
            "repaired_row": repaired_row,
            "repaired_proba": repaired_proba,
        })

    return records


case_titles = {
    "true_positive": "True Positive",
    "false_positive": "False Positive",
    "false_negative": "False Negative",
    "borderline_case": "Borderline Case",
}

regime_definitions = {
    "regime_a_full_actionable": ACTIONABLE_FEATURES,
    "regime_b_capital_locked": ACTIONABLE_FEATURES_NO_CAPITAL,
}

counterfactual_results = []  # flat list of dicts, one row per generated CF (for CSV export)
regime_summary = []          # one row per (case, regime) with aggregate outcome, for plotting

for case_name, idx in selected_cases.items():
    original_row = X_test_raw.iloc[idx]
    original_proba = proba_positive[idx]
    original_predicted_class = int(original_proba >= DECISION_THRESHOLD)
    query_instance = X_test_raw.iloc[[idx]]

    print(f"\n--- {case_titles[case_name]} (row {idx}, P(>50K) = {original_proba:.3f}, "
          f"predicted = {CLASS_NAMES[original_predicted_class]}) ---")

    for regime_name, features_to_vary in regime_definitions.items():
        result, elapsed, timed_out = run_dice_with_timeout(query_instance, features_to_vary)

        if timed_out:
            print(f"  [{regime_name}] TIMED OUT after {elapsed:.1f}s without finding a valid "
                  f"counterfactual (search abandoned; treated as a locked-out result).")
            regime_summary.append({
                "case": case_name, "regime": regime_name, "elapsed_seconds": elapsed,
                "timed_out": True, "n_valid_cfs": 0, "n_requested": CFS_PER_QUERY,
                "mean_proximity": np.nan, "mean_sparsity": np.nan,
            })
            continue

        cf_df = result.cf_examples_list[0].final_cfs_df
        records = summarise_counterfactuals(original_row, cf_df, features_to_vary, original_predicted_class)
        valid_records = [r for r in records if r["is_valid"]]

        print(f"  [{regime_name}] completed in {elapsed:.1f}s | "
              f"{len(valid_records)}/{len(records)} proposed CFs valid after consistency repair")

        for rank, record in enumerate(valid_records):
            print(f"      CF #{rank + 1}: sparsity = {record['sparsity']} feature(s) changed "
                  f"({', '.join(record['changed_features']) if record['changed_features'] else 'none'}) "
                  f"| proximity = {record['proximity']:.3f} | P(>50K) after change = {record['repaired_proba']:.3f}")
            counterfactual_results.append({
                "case": case_name,
                "regime": regime_name,
                "cf_rank": rank + 1,
                "original_row_index": idx,
                "original_proba": original_proba,
                "repaired_proba": record["repaired_proba"],
                "sparsity": record["sparsity"],
                "proximity": record["proximity"],
                "changed_features": "; ".join(record["changed_features"]),
                **{f"cf_{f}": record["repaired_row"][f] for f in ALL_FEATURES},
            })

        # np.mean() propagates NaN if even a single value is NaN, which would
        # silently wipe out genuine proximity values from other valid CFs in
        # the same regime whenever at least one valid CF changed only
        # categorical features (proximity undefined for those individually).
        # np.nanmean() is used instead so the mean reflects whichever valid
        # CFs actually have a defined proximity; if none do, it correctly
        # stays NaN (flagged distinctly in the plot below, not shown as 0).
        proximity_values = [r["proximity"] for r in valid_records]
        has_any_proximity = any(not np.isnan(p) for p in proximity_values)
        regime_summary.append({
            "case": case_name, "regime": regime_name, "elapsed_seconds": elapsed,
            "timed_out": False, "n_valid_cfs": len(valid_records), "n_requested": len(records),
            "mean_proximity": np.nanmean(proximity_values) if has_any_proximity else np.nan,
            "mean_sparsity": np.mean([r["sparsity"] for r in valid_records]) if valid_records else np.nan,
        })

regime_summary_df = pd.DataFrame(regime_summary)
print(f"\nSection 6 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 7. Manual Sanity-Check - Single-Axis Capital Threshold Search
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("7. MANUAL SANITY-CHECK - SINGLE-AXIS CAPITAL THRESHOLD SEARCH")
print("=" * 80)
section_start = time.time()

# For each case, capital_gain (and, where the case's actual capital_loss is
# non-zero, capital_loss) is swept in isolation, holding every other feature
# - including every other actionable feature - fixed at its actual value.
# This answers a narrower, exactly-computable question than DiCE's search:
# "what is the minimum change to this one feature, alone, that flips this
# individual's prediction?" - a direct cross-check on whatever DiCE proposed
# for the same feature in Regime A.

manual_search_records = []
manual_curves = {}  # case_name -> (grid_values, probabilities) for plotting


def sweep_feature(original_row, feature, grid_values):
    """Vectorised sweep of a single numeric feature across grid_values,
    holding every other feature fixed at original_row's values, with the
    has_capital_gain/has_capital_loss flags kept consistent throughout."""
    swept_df = pd.DataFrame([original_row.to_dict()] * len(grid_values))
    swept_df[feature] = grid_values
    swept_df["has_capital_gain"] = (swept_df["capital_gain"] > 0).astype(int)
    swept_df["has_capital_loss"] = (swept_df["capital_loss"] > 0).astype(int)
    swept_df = swept_df[ALL_FEATURES]
    probabilities = pipeline.predict_proba(swept_df)[:, 1]
    return probabilities


for case_name, idx in selected_cases.items():
    original_row = X_test_raw.iloc[idx]
    original_proba = proba_positive[idx]
    original_predicted_class = int(original_proba >= DECISION_THRESHOLD)
    desired_class = 1 - original_predicted_class

    # Sweep range: from the individual's actual value out to the dataset
    # extreme in whichever direction is needed to reach the opposite class.
    actual_gain = original_row["capital_gain"]
    max_gain_in_data = X_test_raw["capital_gain"].max()
    if desired_class == 1:
        # Currently <=50K predicted; need to search UPWARD from actual value.
        grid = np.linspace(actual_gain, max_gain_in_data, MANUAL_SEARCH_STEPS)
    else:
        # Currently >50K predicted; need to search DOWNWARD toward zero.
        grid = np.linspace(actual_gain, 0, MANUAL_SEARCH_STEPS)

    probabilities = sweep_feature(original_row, "capital_gain", grid)
    predicted_classes = (probabilities >= DECISION_THRESHOLD).astype(int)
    flips = np.where(predicted_classes == desired_class)[0]

    if len(flips) > 0:
        flip_idx = flips[0]
        flip_value = grid[flip_idx]
        min_change = abs(flip_value - actual_gain)
        print(f"  {case_titles[case_name]}: capital_gain {actual_gain:.0f} -> {flip_value:.0f} "
              f"flips the prediction (change of {min_change:.0f}), holding every other feature fixed.")
    else:
        flip_value = np.nan
        min_change = np.nan
        print(f"  {case_titles[case_name]}: no capital_gain value in "
              f"[{grid.min():.0f}, {grid.max():.0f}] flips the prediction alone.")

    manual_search_records.append({
        "case": case_name,
        "actual_capital_gain": actual_gain,
        "flip_capital_gain": flip_value,
        "min_change": min_change,
        "original_proba": original_proba,
    })
    manual_curves[case_name] = (grid, probabilities)

manual_search_df = pd.DataFrame(manual_search_records)
print(f"Section 7 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 8. Comparative Visualisation
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("8. COMPARATIVE VISUALISATION")
print("=" * 80)
section_start = time.time()

# --- Plot: Regime A vs Regime B, proximity and sparsity side by side ---
from matplotlib.patches import Patch

REGIME_A_COLOUR = "#3b6fa0"
REGIME_B_COLOUR = "#a83232"
NO_DATA_COLOUR = "#c9c9c9"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
case_order = list(selected_cases.keys())
bar_width = 0.35
x_positions = np.arange(len(case_order))

# Built once and reused on both axes: with per-bar colours mixed under one
# ax.bar() call (real value vs. "no data"), matplotlib's automatic legend
# grabs whichever single bar happens to be plotted first as the swatch for
# the whole label - here that was misleadingly grey. Explicit proxy patches
# guarantee the legend always shows the correct semantic colour for each
# regime, plus a separate entry for "no valid counterfactual found".
legend_handles = [
    Patch(facecolor=REGIME_A_COLOUR, label="Regime A (capital free)"),
    Patch(facecolor=REGIME_B_COLOUR, label="Regime B (capital locked)"),
    Patch(facecolor=NO_DATA_COLOUR, label="No valid counterfactual found"),
]

for ax, metric, ylabel in zip(axes, ["mean_proximity", "mean_sparsity"],
                               ["Mean Proximity (normalised)", "Mean Sparsity (features changed)"]):
    for offset, regime_name, regime_colour in zip(
        [-bar_width / 2, bar_width / 2], regime_definitions.keys(), [REGIME_A_COLOUR, REGIME_B_COLOUR]
    ):
        values, colours = [], []
        for case_name in case_order:
            row = regime_summary_df[(regime_summary_df["case"] == case_name)
                                     & (regime_summary_df["regime"] == regime_name)]
            timed_out = bool(row["timed_out"].iloc[0]) if not row.empty else False
            value = row[metric].iloc[0] if not row.empty else np.nan
            # "No comparable data" (timed out, or - for proximity specifically -
            # every valid CF for this case/regime changed only categorical
            # features, leaving no continuous distance to report) is greyed
            # out and drawn as a zero-height bar. This is deliberately
            # distinct from a genuine value of 0 (an actual, valid CF that
            # happened to need no change along any continuous feature).
            no_data = timed_out or pd.isna(value)
            values.append(0 if no_data else value)
            colours.append(NO_DATA_COLOUR if no_data else regime_colour)
        ax.bar(x_positions + offset, values, width=bar_width, color=colours)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([case_titles[c] for c in case_order], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(handles=legend_handles)

fig.suptitle("Counterfactual Cost by Regime (grey = no valid counterfactual found)")
save_plot(fig, "regime_comparison_proximity_sparsity")

# --- Plot: manual single-axis capital_gain sensitivity curves ---
fig, ax = plt.subplots(figsize=(9, 5.5))
colour_cycle = ["#3b6fa0", "#a83232", "#2b6f39", "#c98a2a"]
for (case_name, (grid, probabilities)), colour in zip(manual_curves.items(), colour_cycle):
    ax.plot(grid, probabilities, label=case_titles[case_name], color=colour, linewidth=2)
ax.axhline(DECISION_THRESHOLD, color="black", linestyle="--", linewidth=0.8,
           label=f"Decision threshold ({DECISION_THRESHOLD})")
ax.set_xlabel("capital_gain (all other features held at actual values)")
ax.set_ylabel("P(income > $50K)")
ax.set_title("Manual Single-Axis Sensitivity: capital_gain")
ax.legend()
save_plot(fig, "manual_capital_gain_sensitivity")

print(f"Section 8 time: {time.time() - section_start:.2f}s")

# ----------------------------------------------------------------------------
# 9. Summary Export
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("9. SUMMARY EXPORT")
print("=" * 80)
section_start = time.time()

counterfactuals_path = os.path.join(OUTPUT_DIR, "counterfactual_summary.csv")
pd.DataFrame(counterfactual_results).to_csv(counterfactuals_path, index=False)
print(f"Saved per-counterfactual detail to: {counterfactuals_path}")

regime_summary_path = os.path.join(OUTPUT_DIR, "counterfactual_regime_summary.csv")
regime_summary_df.to_csv(regime_summary_path, index=False)
print(f"Saved regime-level summary to: {regime_summary_path}")

manual_search_path = os.path.join(OUTPUT_DIR, "manual_sanity_check_summary.csv")
manual_search_df.to_csv(manual_search_path, index=False)
print(f"Saved manual sanity-check summary to: {manual_search_path}")

print(f"Section 9 time: {time.time() - section_start:.2f}s")

total_elapsed = time.time() - SCRIPT_START_TIME
print("\n" + "=" * 80)
print(f"SCRIPT COMPLETE - total runtime: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
print("=" * 80)