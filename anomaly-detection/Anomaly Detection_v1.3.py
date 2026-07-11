"""
Anomaly Detection (Isolation Forest) — AI4I 2020 Predictive Maintenance Dataset
================================================================================
Portfolio project: marcgrover-datascience.github.io

Business context
-----------------
A manufacturer wants to monitor a fleet of milling machines for signs of
impending failure. No reliable historical failure labels are assumed to be
available at deployment time, so an unsupervised anomaly detector — Isolation
Forest — is trained purely on sensor readings. The dataset's true failure
labels are withheld from training and used only afterwards, to validate how
well the unsupervised model's flags align with genuine equipment failures.

Two error types carry asymmetric cost in this domain:
    - A missed anomaly (false negative) may allow an undetected fault to
      progress towards unplanned failure and production downtime.
    - A false alert (false positive) triggers an unnecessary inspection,
      consuming maintenance resource and eroding operator trust.

The contamination parameter is therefore tuned against an illustrative cost
function reflecting this asymmetry, rather than left at a default value.

Design decisions agreed before coding
--------------------------------------
    - Full dataset is used for both fitting and scoring (no train/test split).
      With an overall failure rate of 3.4% split across five failure modes
      (some occurring in fewer than 20 rows), a held-out test set would be
      too thin to support per-failure-mode evaluation. This is a deliberate
      trade-off between deployment realism and statistical power on rare
      classes, not a claim that generalisation is unimportant for
      unsupervised models — this limitation is revisited in the Conclusions.
    - No feature scaling is applied. Isolation Forest partitions on raw
      feature values via random splits, so — unlike the portfolio's
      distance-based projects (KNN, SVM) — scaling confers no methodological
      benefit here.
    - The cost ratio (missed failure : false alert) is swept across several
      illustrative values rather than fixed up front, since the appropriate
      ratio is a judgement call best made after seeing how sensitive the
      results are to it.
    - Both a binary ("any failure") view and the five individual failure
      modes (TWF, HDF, PWF, OSF, RNF) are evaluated, since AI4I's rare and
      overlapping failure modes may not support robust per-mode conclusions.

Author: Marc Grover
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42
rng = np.random.RandomState(RANDOM_STATE)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_MODE_NAMES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}


# ---------------------------------------------------------------------------
# Section 1: Dataset and Business Context
# ---------------------------------------------------------------------------
def load_data():
    """Fetch the AI4I 2020 Predictive Maintenance dataset from the UCI ML
    Repository. Features and targets are returned as separate dataframes,
    consistent with how the dataset is structured on UCI (6 sensor/process
    features; 6 targets comprising the overall failure flag and five
    individual failure modes)."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=601)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    print("Dataset loaded: AI4I 2020 Predictive Maintenance Dataset")
    print(f"  Observations: {X.shape[0]}")
    print(f"  Features: {list(X.columns)}")
    print(f"  Targets: {list(y.columns)}")
    print(f"  Overall failure rate: {y['Machine failure'].mean():.2%}")
    print("  Individual failure mode counts:")
    for mode in FAILURE_MODES:
        print(f"    {mode} ({FAILURE_MODE_NAMES[mode]}): {y[mode].sum()}")
    print()

    return X, y


# ---------------------------------------------------------------------------
# Section 2: Exploratory Data Analysis
# ---------------------------------------------------------------------------
def run_eda(X, y):
    # Column names are read dynamically rather than hardcoded with units,
    # since the exact labels returned by ucimlrepo (e.g. with or without
    # "[K]"/"[rpm]" suffixes) can vary between package versions.
    numeric_cols = [col for col in X.columns if col != "Type"]

    # 01: Feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(X[col], kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
    axes.flat[-1].axis("off")
    fig.suptitle("Distribution of Process and Sensor Readings", fontsize=14)
    fig.tight_layout()
    fig.savefig("01_feature_distributions.png")
    plt.close(fig)

    # 02: Correlation heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = X[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Between Numeric Features")
    fig.tight_layout()
    fig.savefig("02_correlation_heatmap.png")
    plt.close(fig)

    # 03: Feature boxplots split by overall failure flag
    plot_df = X[numeric_cols].copy()
    plot_df["Machine failure"] = y["Machine failure"].map({0: "No failure", 1: "Failure"})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, numeric_cols):
        sns.boxplot(
            data=plot_df, x="Machine failure", y=col, hue="Machine failure",
            ax=ax, palette="Set2", legend=False,
        )
        ax.set_title(col)
        ax.set_xlabel("")
    axes.flat[-1].axis("off")
    fig.suptitle("Feature Distributions by Failure Status", fontsize=14)
    fig.tight_layout()
    fig.savefig("03_feature_boxplots_by_failure.png")
    plt.close(fig)

    # 04: Failure mode counts
    mode_counts = y[FAILURE_MODES].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=mode_counts.index, y=mode_counts.values, ax=ax, color="firebrick")
    ax.set_title("Failure Mode Frequency (Overlapping, Not Mutually Exclusive)")
    ax.set_ylabel("Number of Occurrences")
    for i, v in enumerate(mode_counts.values):
        ax.text(i, v + 1, str(v), ha="center")
    fig.tight_layout()
    fig.savefig("04_failure_mode_counts.png")
    plt.close(fig)

    print("EDA complete. Saved plots 01-04.")
    print(
        "Note: individual feature boxplots show some separation for torque "
        "and rotational speed (higher/lower medians under failure), but "
        "substantial overlap remains, particularly for the two temperature "
        "features. The strong negative correlation between rotational speed "
        "and torque (see correlation heatmap) suggests failures may be "
        "better characterised by feature interactions than by any single "
        "reading, motivating a multivariate isolation-based approach.\n"
    )


# ---------------------------------------------------------------------------
# Section 3: Preprocessing and Feature Preparation
# ---------------------------------------------------------------------------
def _find_column(X, keyword):
    """Locate a column by case-insensitive keyword match, so engineered
    features work regardless of whether unit suffixes (e.g. '[K]', '[rpm]')
    are present in the column names returned by ucimlrepo."""
    matches = [col for col in X.columns if keyword.lower() in col.lower()]
    if not matches:
        raise KeyError(f"No column found matching keyword '{keyword}'")
    return matches[0]


def preprocess(X):
    """One-hot encode the categorical Type feature and add engineered
    features that reflect the physical mechanisms behind AI4I's failure
    modes, rather than relying on raw sensor readings alone:
        - power: torque x angular velocity (rad/s). Rotational speed and
          torque are strongly anti-correlated (~-0.88) under normal
          operation, consistent with power being held roughly constant.
          Power failure (PWF) is defined by this quantity falling outside a
          band, so it should be far more isolable as an explicit feature
          than as an interaction Isolation Forest must discover on its own.
        - temp_diff: process temperature minus air temperature, relevant to
          heat dissipation failure (HDF).
        - overstrain: tool wear x torque, relevant to overstrain failure
          (OSF).
        - overstrain_ratio: overstrain divided by the product-Type-specific
          OSF threshold (L=11,000, M=12,000, H=13,000 minNm, per AI4I's
          generating logic). Overstrain failure is only defined relative to
          this Type-dependent threshold, so normalising by it puts all three
          product types on a comparable scale and lets Isolation Forest
          isolate genuine overstrain excursions rather than differences that
          simply reflect a machine's Type.

    No scaling is applied, since Isolation Forest partitions on raw feature
    values rather than distances."""
    torque_col = _find_column(X, "torque")
    speed_col = _find_column(X, "rotational speed")
    wear_col = _find_column(X, "tool wear")
    air_temp_col = _find_column(X, "air temperature")
    process_temp_col = _find_column(X, "process temperature")

    osf_threshold_by_type = {"L": 11000, "M": 12000, "H": 13000}

    X_encoded = pd.get_dummies(X, columns=["Type"], prefix="Type")
    X_encoded["power"] = X[torque_col] * X[speed_col] * (2 * np.pi / 60)
    X_encoded["temp_diff"] = X[process_temp_col] - X[air_temp_col]
    X_encoded["overstrain"] = X[wear_col] * X[torque_col]
    osf_thresholds = X["Type"].map(osf_threshold_by_type)
    X_encoded["overstrain_ratio"] = X_encoded["overstrain"] / osf_thresholds

    print(f"Preprocessing complete. Feature matrix shape: {X_encoded.shape}")
    print(f"Encoded columns: {list(X_encoded.columns)}\n")
    return X_encoded


# ---------------------------------------------------------------------------
# Section 5: Contamination Tuning Against a Cost Function
# ---------------------------------------------------------------------------
def tune_contamination(X_encoded, y_true, cost_ratios, contamination_grid):
    """For each candidate cost ratio (missed failure : false alert), sweep
    contamination values, fit Isolation Forest, and compute a weighted cost
    of the resulting false negatives and false positives against the true
    Machine failure label. Returns a results dataframe and the optimal
    contamination per cost ratio."""
    results = []

    for contamination in contamination_grid:
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_STATE,
        )
        preds = model.fit_predict(X_encoded)
        predicted_anomaly = (preds == -1).astype(int)

        false_negatives = int(((predicted_anomaly == 0) & (y_true == 1)).sum())
        false_positives = int(((predicted_anomaly == 1) & (y_true == 0)).sum())
        true_positives = int(((predicted_anomaly == 1) & (y_true == 1)).sum())

        row = {
            "contamination": contamination,
            "false_negatives": false_negatives,
            "false_positives": false_positives,
            "true_positives": true_positives,
        }
        for ratio in cost_ratios:
            row[f"cost_ratio_{ratio}"] = ratio * false_negatives + false_positives
        results.append(row)

    results_df = pd.DataFrame(results)

    optimal_contamination = {}
    for ratio in cost_ratios:
        col = f"cost_ratio_{ratio}"
        best_idx = results_df[col].idxmin()
        optimal_contamination[ratio] = results_df.loc[best_idx, "contamination"]

    # 05: Cost curves across contamination values, one line per cost ratio
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for ratio in cost_ratios:
        ax.plot(
            results_df["contamination"],
            results_df[f"cost_ratio_{ratio}"],
            marker="o",
            markersize=3,
            label=f"Cost ratio {ratio}:1",
        )
        best_c = optimal_contamination[ratio]
        best_cost = results_df.loc[
            results_df["contamination"] == best_c, f"cost_ratio_{ratio}"
        ].values[0]
        ax.scatter([best_c], [best_cost], zorder=5, s=60, edgecolor="black")
    ax.set_xlabel("Contamination")
    ax.set_ylabel("Weighted cost (ratio \u00d7 missed failures + false alerts)")
    ax.set_title("Contamination Tuning Against Illustrative Cost Ratios")
    ax.legend()
    fig.tight_layout()
    fig.savefig("05_cost_curve_contamination_sweep.png")
    plt.close(fig)

    print("Contamination tuning complete. Saved plot 05.")
    for ratio in cost_ratios:
        print(
            f"  Cost ratio {ratio}:1 -> optimal contamination = "
            f"{optimal_contamination[ratio]:.3f}"
        )
    print()

    return results_df, optimal_contamination


# ---------------------------------------------------------------------------
# Section 6: Model Training (final model at chosen contamination)
# ---------------------------------------------------------------------------
def train_final_model(X_encoded, contamination):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    model.fit(X_encoded)
    print(f"Final Isolation Forest trained with contamination = {contamination:.3f}\n")
    return model


# ---------------------------------------------------------------------------
# Results: evaluation of the final model against true failure labels
# ---------------------------------------------------------------------------
def evaluate_model(model, X_encoded, y):
    scores = model.decision_function(X_encoded)  # higher = more normal
    preds = model.predict(X_encoded)  # -1 = anomaly, 1 = normal
    predicted_anomaly = (preds == -1).astype(int)
    y_true = y["Machine failure"].values

    # 06: Anomaly score distribution by true failure status
    plot_df = pd.DataFrame({
        "score": scores,
        "status": np.where(y_true == 1, "Failure", "No failure"),
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=plot_df, x="score", hue="status", kde=True, ax=ax,
        palette={"No failure": "steelblue", "Failure": "firebrick"},
        stat="density", common_norm=False,
    )
    ax.set_title("Anomaly Score Distribution by True Failure Status")
    ax.set_xlabel("Isolation Forest decision function (lower = more anomalous)")
    fig.tight_layout()
    fig.savefig("06_anomaly_score_distribution.png")
    plt.close(fig)

    # 07: Confusion matrix (predicted anomaly vs true failure)
    tp = int(((predicted_anomaly == 1) & (y_true == 1)).sum())
    fp = int(((predicted_anomaly == 1) & (y_true == 0)).sum())
    fn = int(((predicted_anomaly == 0) & (y_true == 1)).sum())
    tn = int(((predicted_anomaly == 0) & (y_true == 0)).sum())
    conf_matrix = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(
        conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Predicted normal", "Predicted anomaly"],
        yticklabels=["Actual normal", "Actual failure"],
    )
    ax.set_title("Confusion Matrix: Predicted Anomaly vs Actual Failure")
    fig.tight_layout()
    fig.savefig("07_confusion_matrix.png")
    plt.close(fig)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("Evaluation against true Machine failure label:")
    print(f"  True positives:  {tp}")
    print(f"  False positives: {fp}")
    print(f"  False negatives: {fn}")
    print(f"  True negatives:  {tn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 score:  {f1:.3f}\n")

    # 08: Per-failure-mode detection rate
    detection_rates = {}
    mode_counts = {}
    for mode in FAILURE_MODES:
        mode_mask = y[mode] == 1
        mode_counts[mode] = int(mode_mask.sum())
        if mode_mask.sum() > 0:
            detection_rates[mode] = float(predicted_anomaly[mode_mask].mean())
        else:
            detection_rates[mode] = np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    modes = list(detection_rates.keys())
    rates = [detection_rates[m] * 100 for m in modes]
    bars = ax.bar(modes, rates, color="darkorange")
    for bar, mode in zip(bars, modes):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"n={mode_counts[mode]}", ha="center", fontsize=9,
        )
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Detection Rate by Individual Failure Mode")
    fig.tight_layout()
    fig.savefig("08_failure_mode_detection_rates.png")
    plt.close(fig)

    print("Per-failure-mode detection rates (sample sizes are small for some modes):")
    for mode in FAILURE_MODES:
        print(
            f"  {mode} ({FAILURE_MODE_NAMES[mode]}, n={mode_counts[mode]}): "
            f"{detection_rates[mode]:.1%}"
        )
    print("\nSaved plots 06-08.")

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "detection_rates": detection_rates, "mode_counts": mode_counts,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    X, y = load_data()
    run_eda(X, y)
    X_encoded = preprocess(X)

    cost_ratios = [5, 10, 20]
    contamination_grid = np.round(np.arange(0.01, 0.16, 0.005), 3)

    results_df, optimal_contamination = tune_contamination(
        X_encoded, y["Machine failure"].values, cost_ratios, contamination_grid
    )

    # Primary cost ratio for the main write-up: 10:1, treated as an
    # illustrative assumption (missed failures assumed substantially more
    # costly than false alerts, consistent with typical unplanned-downtime
    # economics). All three ratios are reported above for transparency.
    chosen_ratio = 10
    chosen_contamination = optimal_contamination[chosen_ratio]

    model = train_final_model(X_encoded, chosen_contamination)
    evaluate_model(model, X_encoded, y)


if __name__ == "__main__":
    main()