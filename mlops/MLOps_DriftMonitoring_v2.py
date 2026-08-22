# MLOps: Drift Monitoring, Detection & Automated Retraining - Breast Cancer Dataset
#
# Business scenario: an XGBoost classifier (reusing the tuned model from the
# Gradient Boosted Trees project) has been deployed as a clinical decision
# support tool, flagging fine needle aspirate (FNA) biopsy measurements for
# priority pathology review. This script does not ask "which model performs
# best?" - that question was answered in the Gradient Boosted Trees project.
# It asks: how do we know when a deployed model is no longer trustworthy,
# and what do we do about it?
#
# Two synthetic drift scenarios are simulated against the same reference
# model and reference distribution:
#   Scenario A - Gradual drift  : feature means shift a little further with
#                                  every incoming batch
#   Scenario B - Sudden drift   : a fixed, one-off shift is applied from a
#                                  single batch onward
# For each scenario the narrative arc is: stable -> drift introduced ->
# drift detected (PSI / KS) -> performance degradation quantified ->
# retrain triggered -> performance recovery demonstrated.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
import time
import warnings
warnings.filterwarnings('ignore')

# Set visualisation style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Reproducibility
RNG = np.random.default_rng(42)

# Start timer
t0 = time.time()

print("MLOPS: DRIFT MONITORING, DETECTION & AUTOMATED RETRAINING")
print("BREAST CANCER DIAGNOSTIC DEPLOYMENT - CLINICAL DECISION SUPPORT")

# ============================================================================
# 1. LOAD DATA & RECONSTRUCT THE REFERENCE (BASELINE) MODEL
# ============================================================================
print("\n1. Loading Dataset and Reconstructing the Reference XGBoost Model")

data = load_breast_cancer()
X = data.data
y = data.target
feature_names = list(data.feature_names)
target_names = data.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Same 80/20 stratified split used throughout the supervised ML series,
# so the reference model and reference distribution below are identical
# to those established in the Gradient Boosted Trees project.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df['target'] = y_train

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# The reference model is reconstructed using the exact optimal
# hyperparameters identified via the four-phase tuning process in the
# Gradient Boosted Trees project. This is deliberate: this project is
# about the deployment and monitoring layer, not the modelling layer, so
# the model itself is treated as a fixed, already-validated artefact.
REFERENCE_MODEL_PARAMS = dict(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.6,
    colsample_bytree=0.6,
    gamma=0.0,
    reg_lambda=0.5,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

reference_model = XGBClassifier(**REFERENCE_MODEL_PARAMS)
reference_model.fit(X_train, y_train)

ref_train_acc = reference_model.score(X_train, y_train)
ref_test_acc = reference_model.score(X_test, y_test)
ref_test_auc = roc_auc_score(y_test, reference_model.predict_proba(X_test)[:, 1])

print(f"\nReference model reconstructed using optimal hyperparameters:")
for k, v in REFERENCE_MODEL_PARAMS.items():
    if k not in ('random_state', 'eval_metric', 'n_jobs'):
        print(f"  {k}: {v}")
print(f"\nReference model training accuracy: {ref_train_acc:.4f}")
print(f"Reference model testing accuracy:  {ref_test_acc:.4f}")
print(f"Reference model testing ROC-AUC:   {ref_test_auc:.4f}")
print("(Confirms parity with the Gradient Boosted Trees project benchmark)")

# ============================================================================
# 2. DRIFT-TARGET FEATURES
# ============================================================================
print("\n2. Selecting Drift-Target Features")

# The three features drifted in this simulation are deliberately the same
# cluster identified as the dominant predictive signal in both the
# Gradient Boosted Trees project (top 3 by gain importance) and the SHAP
# project (top 3 by mean absolute SHAP value): worst area, worst perimeter,
# worst concave points. This is a realistic choice as well as a narratively
# coherent one - a change in imaging equipment or measurement protocol
# would plausibly affect precisely these cell-nucleus size and shape
# measurements, and drifting the features the model relies on most heavily
# produces a meaningful, detectable performance impact rather than an
# arbitrary one that a production model might shrug off.

DRIFT_FEATURES = ['worst area', 'worst perimeter', 'worst concave points']
print(f"Drift-target features: {DRIFT_FEATURES}")

# Reference (training) distribution statistics, used both as the PSI/KS
# comparison baseline and to express drift shifts in standard-deviation
# units relative to genuine feature scale.
ref_stats = train_df[DRIFT_FEATURES].agg(['mean', 'std']).T
print("\nReference (training) distribution statistics for drift-target features:")
print(ref_stats)

# ============================================================================
# 3. SYNTHETIC BATCH GENERATION
# ============================================================================
print("\n3. Defining Synthetic Batch Generation")

BATCH_SIZE = 100
N_BATCHES = 15
POST_RETRAIN_BATCHES = 5  # additional batches simulated after retraining

print(f"Batch size: {BATCH_SIZE} observations")
print(f"Batches per scenario: {N_BATCHES} (plus {POST_RETRAIN_BATCHES} post-retrain batches)")
print("Batches 1-5 are stable (no shift) in both scenarios.")


def generate_batch(shift_sds, batch_size=BATCH_SIZE, rng=RNG):
    """
    Generate one synthetic incoming batch by resampling (with replacement,
    class-stratified) from the full 569-observation Wisconsin dataset, then
    applying a mean shift to the drift-target features only.

    The true diagnosis label travels with the resampled observation
    unchanged: a shift in *measured* feature values (e.g. an imaging or
    measurement protocol change) does not change the underlying biology of
    the tumour. This mirrors a realistic clinical setting in which ground
    truth (confirmed via pathology) eventually becomes available for a
    batch, allowing performance to be monitored and a retraining decision
    made - while the features observed at prediction time have drifted
    away from what the model was trained on.

    shift_sds: dict mapping each drift-target feature to a shift expressed
    in standard-deviation units of the *reference training* distribution
    for that feature. A shift of 0.0 for all features produces a batch
    statistically indistinguishable from the reference distribution.
    """
    # Class-stratified resampling with replacement preserves the original
    # ~63/37 benign/malignant balance in every batch.
    idx_benign = rng.choice(df.index[df['target'] == 1], size=int(batch_size * 0.63), replace=True)
    idx_malignant = rng.choice(df.index[df['target'] == 0], size=batch_size - len(idx_benign), replace=True)
    batch = df.loc[np.concatenate([idx_benign, idx_malignant])].copy().reset_index(drop=True)

    # Apply the mean shift to drift-target features only, in raw feature
    # units (shift_sds * reference std), leaving all 27 non-drifted
    # features and the label untouched.
    for feature, sd_shift in shift_sds.items():
        batch[feature] = batch[feature] + sd_shift * ref_stats.loc[feature, 'std']

    return batch


def zero_shift():
    return {f: 0.0 for f in DRIFT_FEATURES}


# Quick sanity check: a zero-shift batch should closely resemble the
# reference distribution.
sanity_batch = generate_batch(zero_shift())
print("\nSanity check - zero-shift batch feature means vs reference training means:")
print(pd.DataFrame({
    'batch_mean': sanity_batch[DRIFT_FEATURES].mean(),
    'reference_mean': ref_stats['mean']
}))

# ============================================================================
# 4. DRIFT DETECTION METHODOLOGY: PSI AND KS TEST
# ============================================================================
print("\n4. Defining Drift Detection Methodology (PSI and KS Test)")
print("Two complementary statistical drift measures are used and compared:")
print("  - Population Stability Index (PSI): a binned distributional")
print("    distance measure, widely used in industry (esp. credit risk) as")
print("    an interpretable single-number drift score.")
print("  - Kolmogorov-Smirnov (KS) test: a non-parametric hypothesis test")
print("    comparing two empirical distributions directly, without binning.")
print("PSI and KS answer subtly different questions - PSI quantifies *how")
print("much* a distribution has shifted (a continuous magnitude), while KS")
print("tests *whether* it has shifted at all (a statistical significance")
print("decision). Reporting both, rather than either alone, is the basis")
print("for the scenario comparison later in this project.")

# Conventional PSI interpretation thresholds (industry standard, most
# commonly seen in credit-risk model monitoring):
#   PSI < 0.10           : no significant population change
#   0.10 <= PSI < 0.20    : moderate population change - investigate
#   PSI >= 0.20           : significant population change - action required
PSI_ALERT_THRESHOLD = 0.20
KS_ALERT_PVALUE = 0.05


def calculate_psi(reference, current, n_bins=10):
    """
    Population Stability Index between a reference distribution and a
    current (batch) distribution for a single continuous feature.

    Bin edges are defined on the reference distribution using quantiles,
    so that each reference bin holds approximately 1/n_bins of the
    reference observations by construction. The same edges are then
    applied to the current batch - if the batch distribution has not
    shifted, it should populate the bins similarly; if it has shifted,
    mass moves into different bins and PSI rises accordingly.
    """
    # Quantile-based bin edges from the reference distribution, with the
    # outer edges opened to -inf/+inf so that any out-of-range batch
    # values (a batch that has drifted beyond the reference's observed
    # range) still fall into a bin rather than being silently dropped.
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.unique(np.quantile(reference, quantiles))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions, applying a small epsilon floor to avoid
    # division-by-zero / log(0) for bins that happen to be empty in
    # either distribution.
    eps = 1e-4
    ref_props = np.maximum(ref_counts / ref_counts.sum(), eps)
    cur_props = np.maximum(cur_counts / cur_counts.sum(), eps)

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return psi


def calculate_ks(reference, current):
    """Two-sample Kolmogorov-Smirnov test statistic and p-value."""
    statistic, p_value = ks_2samp(reference, current)
    return statistic, p_value


def assess_batch_drift(batch):
    """
    Compute per-feature PSI and KS statistics for a batch against the
    reference training distribution, for each drift-target feature, plus
    an aggregate (mean) PSI across the drift-target features as a single
    per-batch drift score.
    """
    results = []
    for feature in DRIFT_FEATURES:
        ref_values = train_df[feature].values
        cur_values = batch[feature].values

        psi = calculate_psi(ref_values, cur_values)
        ks_stat, ks_p = calculate_ks(ref_values, cur_values)

        results.append({
            'feature': feature,
            'psi': psi,
            'ks_statistic': ks_stat,
            'ks_p_value': ks_p,
            'psi_alert': psi >= PSI_ALERT_THRESHOLD,
            'ks_alert': ks_p < KS_ALERT_PVALUE
        })

    results_df = pd.DataFrame(results)
    aggregate_psi = results_df['psi'].mean()
    aggregate_drift_alert = (results_df['psi_alert'].any()) or (results_df['ks_alert'].any())

    return results_df, aggregate_psi, aggregate_drift_alert


# Sanity check: a zero-shift batch should show negligible PSI/KS drift.
sanity_results, sanity_psi, sanity_alert = assess_batch_drift(sanity_batch)
print("\nSanity check - drift statistics on a zero-shift batch (expect low PSI, high KS p-values):")
print(sanity_results.to_string(index=False))
print(f"Aggregate PSI: {sanity_psi:.4f} | Drift alert: {sanity_alert}")

# ============================================================================
# 5. MONITORING PIPELINE
# ============================================================================
print("\n5. Defining the Monitoring Pipeline")
print("For each incoming batch: predict with the current model, compute")
print("PSI/KS drift statistics, and log accuracy/ROC-AUC using the batch's")
print("true labels - standing in for a delayed pathology-confirmed ground")
print("truth feed. A drift alert fires when PSI or KS breaches threshold;")
print("two consecutive alerts flag the retrain decision point. Monitoring")
print("continues for the full batch sequence regardless of the flag, so")
print("that the counterfactual - what would have happened if the alert")
print("had been ignored - is visible in the same run. Retraining itself is")
print("then performed as a separate, explicit step, not fired automatically")
print("inside the loop, reflecting a deliberate human-in-the-loop decision.")

RETRAIN_CONSECUTIVE_ALERTS = 2


def run_monitoring_batches(model, shift_schedule, rng):
    """
    Run a model across a sequence of synthetic batches, logging per-batch
    performance and drift statistics. Unlike an operational system, this
    does NOT stop at the retrain trigger point - it runs the full schedule
    so that the cost of inaction is visible on the same chart as the
    detection point. Returns (log_list, generated_batches_list).
    """
    log = []
    batches = []
    consecutive_alerts = 0

    for i, shift in enumerate(shift_schedule):
        batch_number = i + 1
        batch = generate_batch(shift, rng=rng)
        batches.append(batch)

        X_batch = batch[feature_names].values
        y_batch = batch['target'].values

        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)[:, 1]
        acc = accuracy_score(y_batch, preds)
        auc = roc_auc_score(y_batch, probs) if len(np.unique(y_batch)) > 1 else np.nan

        drift_results, agg_psi, drift_alert = assess_batch_drift(batch)
        max_ks_stat = drift_results['ks_statistic'].max()
        min_ks_p = drift_results['ks_p_value'].min()

        consecutive_alerts = consecutive_alerts + 1 if drift_alert else 0
        retrain_flag = consecutive_alerts >= RETRAIN_CONSECUTIVE_ALERTS

        log.append({
            'batch_number': batch_number,
            'accuracy': acc,
            'roc_auc': auc,
            'aggregate_psi': agg_psi,
            'max_ks_statistic': max_ks_stat,
            'min_ks_p_value': min_ks_p,
            'drift_alert': drift_alert,
            'consecutive_alerts': consecutive_alerts,
            'retrain_flag': retrain_flag,
        })

        alert_str = " <-- DRIFT ALERT" if drift_alert else ""
        retrain_str = " *** RETRAIN TRIGGER ***" if retrain_flag and consecutive_alerts == RETRAIN_CONSECUTIVE_ALERTS else ""
        print(f"  Batch {batch_number:2d}: accuracy={acc:.4f}  ROC-AUC={auc:.4f}  "
              f"agg_PSI={agg_psi:.4f}  max_KS={max_ks_stat:.4f} (p={min_ks_p:.4f})"
              f"{alert_str}{retrain_str}")

    return log, batches


def first_retrain_trigger(log):
    """First batch number at which the retrain condition was met."""
    return next((entry['batch_number'] for entry in log if entry['retrain_flag']), None)


def retrain_model(trigger_window_batches):
    """
    Retrain the model on the original training set combined with the
    labelled observations from the drifted batches collected since the
    retrain trigger fired. Combining original and drifted data - rather
    than drifted data alone - guards against the retrained model
    overfitting to a small recent window and losing the general patterns
    learned from the full original training distribution.
    """
    drift_X = pd.concat([b[feature_names] for b in trigger_window_batches], axis=0).values
    drift_y = pd.concat([b['target'] for b in trigger_window_batches], axis=0).values

    X_retrain = np.vstack([X_train, drift_X])
    y_retrain = np.concatenate([y_train, drift_y])

    new_model = XGBClassifier(**REFERENCE_MODEL_PARAMS)
    new_model.fit(X_retrain, y_retrain)
    return new_model


def evaluate_model_on_batches(model, batches, start_batch_number):
    """Evaluate a model across a list of already-generated batches."""
    rows = []
    for i, batch in enumerate(batches):
        X_batch = batch[feature_names].values
        y_batch = batch['target'].values
        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)[:, 1]
        acc = accuracy_score(y_batch, preds)
        auc = roc_auc_score(y_batch, probs) if len(np.unique(y_batch)) > 1 else np.nan
        rows.append({'batch_number': start_batch_number + i, 'accuracy': acc, 'roc_auc': auc})
    return pd.DataFrame(rows)


def pooled_classification_report(model, batches):
    """
    Pool all observations across a list of batches and compute a full
    classification report (precision, recall, F1, specificity) plus a
    confusion matrix. Used to compare the original vs. retrained model
    on identical, larger-sample post-retrain data - a single accuracy
    figure is too coarse to assess the clinically critical false
    negative rate, which is why precision/recall/specificity are
    reported explicitly here rather than relying on accuracy alone.
    """
    X_pooled = pd.concat([b[feature_names] for b in batches], axis=0).values
    y_pooled = pd.concat([b['target'] for b in batches], axis=0).values
    preds = model.predict(X_pooled)

    cm = confusion_matrix(y_pooled, preds)
    tn, fp, fn, tp = cm.ravel()
    report = {
        'n_observations': len(y_pooled),
        'accuracy': accuracy_score(y_pooled, preds),
        'precision': precision_score(y_pooled, preds),
        'recall_sensitivity': recall_score(y_pooled, preds),
        'specificity': tn / (tn + fp),
        'f1_score': f1_score(y_pooled, preds),
        'false_negatives': int(fn),
        'false_positives': int(fp),
    }
    return report, cm


def run_scenario(scenario_name, shift_schedule, post_shift_schedule, seed):
    """
    Run one full scenario end-to-end:
      1. Monitor the reference model across shift_schedule (pre-retrain).
      2. Identify the retrain trigger batch and retrain using data from
         the trigger batch onward.
      3. Generate post_shift_schedule batches (continuing the drift
         pattern) and evaluate BOTH the original reference model and the
         newly retrained model on the identical batches - an apples-to-
         apples recovery comparison.
    Returns a dict of all logs/models needed for reporting and plotting.
    """
    print("\n" + "=" * 76)
    print(scenario_name)
    print("=" * 76)

    rng = np.random.default_rng(seed)
    log, batches = run_monitoring_batches(reference_model, shift_schedule, rng)
    trigger_batch = first_retrain_trigger(log)
    print(f"\nRetrain trigger fired at batch: {trigger_batch}")

    trigger_drift_detail, _, _ = assess_batch_drift(batches[trigger_batch - 1])
    print(f"Per-feature drift statistics at trigger batch {trigger_batch}:")
    print(trigger_drift_detail.to_string(index=False))

    # Retrain using labelled data from the trigger batch through to the
    # end of the pre-retrain sequence (the most recent drifted evidence
    # available at the point the decision is made).
    trigger_window = batches[trigger_batch - 1:]
    print(f"Retraining on original training set + {len(trigger_window)} drifted batch(es) "
          f"(batches {trigger_batch}-{len(batches)}), {len(trigger_window) * BATCH_SIZE} additional labelled observations")
    new_model = retrain_model(trigger_window)
    new_test_acc = new_model.score(X_test, y_test)
    print(f"Retrained model accuracy on original (undrifted) held-out test set: {new_test_acc:.4f}")

    # Post-retrain batches: continue generating batches under the SAME
    # drift pattern (the underlying cause of drift has not been fixed -
    # only the model has been updated to account for it), and evaluate
    # both models on identical data for a fair recovery comparison.
    print(f"\nGenerating {len(post_shift_schedule)} further batches under continued drift, "
          f"comparing original vs retrained model:")
    post_batches = [generate_batch(shift, rng=rng) for shift in post_shift_schedule]

    old_model_post = evaluate_model_on_batches(reference_model, post_batches, len(batches) + 1)
    new_model_post = evaluate_model_on_batches(new_model, post_batches, len(batches) + 1)

    comparison = old_model_post.merge(
        new_model_post, on='batch_number', suffixes=('_original_model', '_retrained_model')
    )
    print(comparison.to_string(index=False))

    # Pooled classification report (precision/recall/specificity/F1) across
    # all post-retrain batches combined, for both models on identical data -
    # accuracy alone does not surface the clinically critical false
    # negative rate.
    old_report, old_cm = pooled_classification_report(reference_model, post_batches)
    new_report, new_cm = pooled_classification_report(new_model, post_batches)
    print(f"\nPooled classification report across {len(post_batches)} post-retrain batches "
          f"({old_report['n_observations']} observations):")
    print(pd.DataFrame([old_report, new_report], index=['original_model', 'retrained_model']).T)

    return {
        'name': scenario_name,
        'log': pd.DataFrame(log),
        'trigger_batch': trigger_batch,
        'trigger_drift_detail': trigger_drift_detail,
        'new_model': new_model,
        'new_model_test_accuracy': new_test_acc,
        'post_comparison': comparison,
        'old_report': old_report,
        'new_report': new_report,
        'old_cm': old_cm,
        'new_cm': new_cm,
    }


# ----------------------------------------------------------------------
# SCENARIO A: GRADUAL DRIFT
# ----------------------------------------------------------------------
# Batches 1-5 stable; from batch 6 the mean shift ramps up by +0.05 SD per
# batch (batch 6 = +0.05 SD ... batch 15 = +0.50 SD), simulating a slow
# measurement or equipment drift. Post-retrain batches continue the same
# ramp (+0.55 SD ... +0.75 SD) so the comparison is made under conditions
# at least as severe as those that triggered the retrain.

def gradual_shift_schedule(batch_numbers, ramp_start_batch=6, sd_per_batch=0.05):
    schedule = []
    for b in batch_numbers:
        if b < ramp_start_batch:
            schedule.append(zero_shift())
        else:
            magnitude = sd_per_batch * (b - ramp_start_batch + 1)
            schedule.append({f: magnitude for f in DRIFT_FEATURES})
    return schedule


scenario_a_pre = gradual_shift_schedule(range(1, N_BATCHES + 1))
scenario_a_post = gradual_shift_schedule(range(N_BATCHES + 1, N_BATCHES + POST_RETRAIN_BATCHES + 1))
scenario_a = run_scenario("SCENARIO A: GRADUAL DRIFT", scenario_a_pre, scenario_a_post, seed=101)

# ----------------------------------------------------------------------
# SCENARIO B: SUDDEN (STEP-CHANGE) DRIFT
# ----------------------------------------------------------------------
# Batches 1-5 stable; from batch 6 a fixed +1.5 SD shift is applied
# instantly and held constant, simulating an abrupt equipment change
# (e.g. a scanner recalibration or replacement) rather than a slow drift.

SUDDEN_SHIFT_MAGNITUDE = 1.5
SUDDEN_SHIFT_START_BATCH = 6


def sudden_shift_schedule(batch_numbers, shift_start_batch=SUDDEN_SHIFT_START_BATCH,
                           magnitude=SUDDEN_SHIFT_MAGNITUDE):
    schedule = []
    for b in batch_numbers:
        if b < shift_start_batch:
            schedule.append(zero_shift())
        else:
            schedule.append({f: magnitude for f in DRIFT_FEATURES})
    return schedule


scenario_b_pre = sudden_shift_schedule(range(1, N_BATCHES + 1))
scenario_b_post = sudden_shift_schedule(range(N_BATCHES + 1, N_BATCHES + POST_RETRAIN_BATCHES + 1))
scenario_b = run_scenario("SCENARIO B: SUDDEN (STEP-CHANGE) DRIFT", scenario_b_pre, scenario_b_post, seed=202)

# ============================================================================
# 6. VISUALISATION - PER-SCENARIO MONITORING TIMELINE
# ============================================================================
print("\n6. Producing Monitoring Timeline Visualisations")


def plot_monitoring_timeline(scenario_result, filename, title):
    """
    Dual-axis time-series plot: accuracy and ROC-AUC (left axis) against
    aggregate PSI (right axis) across all monitored batches, with a
    vertical marker at the retrain trigger batch and a shaded region
    showing where drift alerts were active.
    """
    log = scenario_result['log']
    trigger = scenario_result['trigger_batch']

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(log['batch_number'], log['accuracy'], marker='o', color='#1f77b4',
              linewidth=2, label='Accuracy')
    ax1.plot(log['batch_number'], log['roc_auc'], marker='s', color='#2ca02c',
              linewidth=2, linestyle='--', label='ROC-AUC')
    ax1.set_xlabel('Batch Number', fontsize=12)
    ax1.set_ylabel('Model Performance', fontsize=12)
    ax1.set_ylim(0.5, 1.02)
    ax1.set_xticks(log['batch_number'])

    ax2 = ax1.twinx()
    ax2.plot(log['batch_number'], log['aggregate_psi'], marker='^', color='#d62728',
              linewidth=2, linestyle=':', label='Aggregate PSI')
    ax2.axhline(y=PSI_ALERT_THRESHOLD, color='#d62728', linewidth=1, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Aggregate PSI', fontsize=12, color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    ax1.axvline(x=trigger, color='black', linewidth=1.5, linestyle='-.', alpha=0.7)
    ax1.text(trigger + 0.15, 0.53, f'Retrain\ntrigger\n(batch {trigger})', fontsize=9,
             va='bottom')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


plot_monitoring_timeline(
    scenario_a, 'plot_01_scenario_a_monitoring_timeline.png',
    'Scenario A (Gradual Drift): Performance vs. Drift Statistics'
)
plot_monitoring_timeline(
    scenario_b, 'plot_02_scenario_b_monitoring_timeline.png',
    'Scenario B (Sudden Drift): Performance vs. Drift Statistics'
)

# ============================================================================
# 7. VISUALISATION - GRADUAL VS SUDDEN DRIFT COMPARISON
# ============================================================================
print("\n7. Producing Gradual vs. Sudden Drift Comparison")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(scenario_a['log']['batch_number'], scenario_a['log']['accuracy'],
         marker='o', linewidth=2, label='Scenario A (Gradual)', color='#1f77b4')
ax1.plot(scenario_b['log']['batch_number'], scenario_b['log']['accuracy'],
         marker='s', linewidth=2, label='Scenario B (Sudden)', color='#ff7f0e')
ax1.axvline(x=6, color='grey', linewidth=1, linestyle='--', alpha=0.6)
ax1.text(6.1, 0.72, 'drift\nonset', fontsize=9)
ax1.set_xlabel('Batch Number', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy: Gradual vs. Sudden Drift', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_xticks(scenario_a['log']['batch_number'])

ax2.plot(scenario_a['log']['batch_number'], scenario_a['log']['aggregate_psi'],
         marker='o', linewidth=2, label='Scenario A (Gradual)', color='#1f77b4')
ax2.plot(scenario_b['log']['batch_number'], scenario_b['log']['aggregate_psi'],
         marker='s', linewidth=2, label='Scenario B (Sudden)', color='#ff7f0e')
ax2.axhline(y=PSI_ALERT_THRESHOLD, color='#d62728', linewidth=1, linestyle='--',
            alpha=0.6, label='PSI alert threshold')
ax2.axvline(x=6, color='grey', linewidth=1, linestyle='--', alpha=0.6)
ax2.set_yscale('log')
ax2.set_xlabel('Batch Number', fontsize=12)
ax2.set_ylabel('Aggregate PSI (log scale)', fontsize=12)
ax2.set_title('Drift Magnitude: Gradual vs. Sudden Drift', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_xticks(scenario_a['log']['batch_number'])

plt.tight_layout()
plt.savefig('plot_03_gradual_vs_sudden_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. VISUALISATION - RETRAINING RECOVERY
# ============================================================================
print("\n8. Producing Retraining Recovery Visualisation")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, scenario, label in zip(axes, [scenario_a, scenario_b], ['Scenario A (Gradual)', 'Scenario B (Sudden)']):
    comp = scenario['post_comparison']
    x = np.arange(len(comp))
    width = 0.35
    ax.bar(x - width/2, comp['accuracy_original_model'], width,
           label='Original model (no retrain)', color='#d62728', alpha=0.8)
    ax.bar(x + width/2, comp['accuracy_retrained_model'], width,
           label='Retrained model', color='#2ca02c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comp['batch_number'])
    ax.set_ylim(0.5, 1.05)
    ax.set_xlabel('Batch Number (post-retrain, continued drift)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{label}: Recovery After Retraining', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(y=ref_test_acc, color='grey', linewidth=1, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.savefig('plot_04_retraining_recovery.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. VISUALISATION - CONFUSION MATRIX COMPARISON (ORIGINAL VS RETRAINED)
# ============================================================================
print("\n9. Producing Confusion Matrix Comparison")

fig, axes = plt.subplots(2, 2, figsize=(11, 10))

for row, (scenario, label) in enumerate(zip([scenario_a, scenario_b],
                                             ['Scenario A (Gradual)', 'Scenario B (Sudden)'])):
    for col, (cm, model_label) in enumerate(zip([scenario['old_cm'], scenario['new_cm']],
                                                  ['Original Model (No Retrain)', 'Retrained Model'])):
        ax = axes[row, col]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{label}\n{model_label}', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_05_confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFalse negatives (malignant misclassified as benign) on pooled post-retrain batches:")
print(f"  Scenario A - Original model: {scenario_a['old_report']['false_negatives']}, "
      f"Retrained model: {scenario_a['new_report']['false_negatives']}")
print(f"  Scenario B - Original model: {scenario_b['old_report']['false_negatives']}, "
      f"Retrained model: {scenario_b['new_report']['false_negatives']}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\nMLOPS PROJECT SUMMARY")
print(f"Reference model: XGBoost, {ref_test_acc:.4f} test accuracy, {ref_test_auc:.4f} ROC-AUC")
print(f"Drift-target features: {DRIFT_FEATURES}")
print(f"\nScenario A (Gradual Drift):")
print(f"  Retrain triggered at batch {scenario_a['trigger_batch']}")
print(f"  Accuracy at trigger batch: {scenario_a['log'].loc[scenario_a['log']['batch_number'] == scenario_a['trigger_batch'], 'accuracy'].values[0]:.4f}")
print(f"  Retrained model test-set accuracy: {scenario_a['new_model_test_accuracy']:.4f}")
print(f"  Mean post-retrain accuracy - original model: {scenario_a['post_comparison']['accuracy_original_model'].mean():.4f}")
print(f"  Mean post-retrain accuracy - retrained model: {scenario_a['post_comparison']['accuracy_retrained_model'].mean():.4f}")
print(f"\nScenario B (Sudden Drift):")
print(f"  Retrain triggered at batch {scenario_b['trigger_batch']}")
print(f"  Accuracy at trigger batch: {scenario_b['log'].loc[scenario_b['log']['batch_number'] == scenario_b['trigger_batch'], 'accuracy'].values[0]:.4f}")
print(f"  Retrained model test-set accuracy: {scenario_b['new_model_test_accuracy']:.4f}")
print(f"  Mean post-retrain accuracy - original model: {scenario_b['post_comparison']['accuracy_original_model'].mean():.4f}")
print(f"  Mean post-retrain accuracy - retrained model: {scenario_b['post_comparison']['accuracy_retrained_model'].mean():.4f}")

t1 = time.time()
print(f"\nTime Taken: {t1 - t0:.4f} seconds")