# Random Forest Classification - Breast Cancer Dataset
# Proof-of-Concept for Categorical Prediction

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings
import time

warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Start timer
t0 = time.time()  # Add at start of process

print("RANDOM FOREST CLASSIFICATION - BREAST CANCER DATASET")

# ============================================================================
# 1. LOAD THE BREAST CANCER DATASET
# ============================================================================
print("\n1. Loading Breast Cancer Dataset")
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Target classes: {target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Create DataFrame for easier manipulation
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("\nDataset sample:")
print(df)

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n2. Performing Data Preprocessing")

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"Missing values: {missing_values}")

# Check for any preprocessing needs
print(f"Data types:\n{df.dtypes.value_counts()}")

print("No missing values found, all features are numeric - no encoding required")

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n3. Performing Exploratory Data Analysis")

# Class distribution visualization
plt.figure(figsize=(10, 6))
class_counts = pd.Series(y).value_counts()
bars = sns.barplot(x=[target_names[0], target_names[1]], y=class_counts.values, palette='viridis')

# Add value labels on bars
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    bars.text(bar.get_x() + bar.get_width() / 2., height,
              f'{int(height)}',
              ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Class Distribution in Breast Cancer Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('rf_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(12, 10))
top_features = df[list(feature_names)].corr()
sns.heatmap(top_features, annot=True, fmt='.1f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('rf_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. TRAIN-TEST SPLIT (80-20)
# ============================================================================
print("\n4. Splitting Data into Training and Testing Sets (80-20)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Testing set class distribution: {np.bincount(y_test)}")

# ============================================================================
# 5. MODEL TUNING - OPTIMAL PARAMETERS
# ============================================================================
print("\n5. Fine-tuning Random Forest Model")

# Test different number of trees
n_estimators_range = [10, 25, 50, 75, 100, 150, 200]
max_depths = [3, 5, 7, 10, 15, 20, None]

print("\n5a. Optimising Number of Trees")
train_scores_trees = []
test_scores_trees = []
cv_scores_trees = []

for n_trees in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_scores_trees.append(rf.score(X_train, y_train))
    test_scores_trees.append(rf.score(X_test, y_test))

    cv_score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_scores_trees.append(cv_score.mean())

    print(f"  Trees: {n_trees:3d} | Train: {train_scores_trees[-1]:.4f} | "
          f"Test: {test_scores_trees[-1]:.4f} | CV: {cv_scores_trees[-1]:.4f}")

optimal_n_trees = n_estimators_range[np.argmax(cv_scores_trees)]
print(f"\nOptimal number of trees: {optimal_n_trees}")

# Visualisation of number of trees analysis
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, train_scores_trees, label='Training Accuracy', marker='o', linewidth=2)
plt.plot(n_estimators_range, test_scores_trees, label='Testing Accuracy', marker='s', linewidth=2)
plt.plot(n_estimators_range, cv_scores_trees, label='CV Accuracy (5-fold)', marker='^', linewidth=2)
plt.axvline(x=optimal_n_trees, color='red', linestyle='--', label=f'Optimal Trees = {optimal_n_trees}')
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance vs Number of Trees', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_trees_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n5b. Optimizing Tree Depth")
train_scores_depth = []
test_scores_depth = []
cv_scores_depth = []

for depth in max_depths:
    rf = RandomForestClassifier(n_estimators=optimal_n_trees, max_depth=depth,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_scores_depth.append(rf.score(X_train, y_train))
    test_scores_depth.append(rf.score(X_test, y_test))

    cv_score = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    cv_scores_depth.append(cv_score.mean())

    depth_str = str(depth) if depth is not None else "None"
    print(f"  Depth: {depth_str:4s} | Train: {train_scores_depth[-1]:.4f} | "
          f"Test: {test_scores_depth[-1]:.4f} | CV: {cv_scores_depth[-1]:.4f}")

optimal_depth = max_depths[np.argmax(cv_scores_depth)]
print(f"\nOptimal tree depth: {optimal_depth}")

# Visualisation of depth analysis
plt.figure(figsize=(10, 6))
depth_labels = [str(d) if d is not None else "None" for d in max_depths]
x_pos = np.arange(len(depth_labels))

plt.plot(x_pos, train_scores_depth, label='Training Accuracy', marker='o', linewidth=2)
plt.plot(x_pos, test_scores_depth, label='Testing Accuracy', marker='s', linewidth=2)
plt.plot(x_pos, cv_scores_depth, label='CV Accuracy (5-fold)', marker='^', linewidth=2)
optimal_idx = max_depths.index(optimal_depth)
plt.axvline(x=optimal_idx, color='red', linestyle='--',
            label=f'Optimal Depth = {optimal_depth}')
plt.xticks(x_pos, depth_labels)
plt.xlabel('Maximum Tree Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance vs Tree Depth', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_depth_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. TRAIN OPTIMAL RANDOM FOREST MODEL
# ============================================================================
print("\n6. Training Optimal Random Forest Model")

rf_optimal = RandomForestClassifier(
    n_estimators=optimal_n_trees,
    max_depth=optimal_depth,
    random_state=42,
    n_jobs=-1
)
rf_optimal.fit(X_train, y_train)

# print(f"Optimal Random Forest model trained with n_estimators={optimal_n_trees}, max_depth={optimal_depth}")

# Forest characteristics
print(f"\nForest Characteristics:")
print(f"Total number of trees: {len(rf_optimal.estimators_)}")
print(f"Maximum tree depth: {optimal_depth}")
print(f"Number of features: {rf_optimal.n_features_in_}")
print(f"Number of classes: {rf_optimal.n_classes_}")

# ============================================================================
# 7. VISUALIZE ONE TREE FROM THE FOREST
# ============================================================================
print("\n7. Visualizing a Single Tree from the Forest")

# Extract one tree (the first tree in the forest)
single_tree = rf_optimal.estimators_[0]

plt.figure(figsize=(20, 10))
plot_tree(single_tree,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=9)
plt.title(f'Single Decision Tree from Random Forest (Tree #1 of {optimal_n_trees})',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('rf_single_tree_structure.png', dpi=300, bbox_inches='tight')
plt.show()

# Tree statistics
tree_depths = [estimator.get_depth() for estimator in rf_optimal.estimators_]
tree_n_nodes = [estimator.tree_.node_count for estimator in rf_optimal.estimators_]

print(f"\nForest Tree Statistics:")
print(f"Average tree depth: {np.mean(tree_depths):.2f}")
print(f"Min/Max tree depth: {np.min(tree_depths)}/{np.max(tree_depths)}")
print(f"Average nodes per tree: {np.mean(tree_n_nodes):.2f}")
print(f"Total nodes in forest: {np.sum(tree_n_nodes)}")

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n8. Performing Feature Importance Analysis...")
print("\nRANDOM FOREST FEATURE IMPORTANCE:")
print("Feature importance in Random Forest is calculated by:")
print("• Measuring how much each feature decreases impurity (Gini/entropy)")
print("• Averaging importance across ALL trees in the forest")
print("• More robust than single tree due to ensemble averaging")
print("• Accounts for feature interactions across different trees")

# Get feature importances
feature_importance = rf_optimal.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# Calculate cumulative importance
importance_df['Cumulative'] = importance_df['Importance'].cumsum()
features_90_pct = (importance_df['Cumulative'] <= 0.90).sum()
print(f"\nFeatures accounting for 90% of importance: {features_90_pct}")

# Visualise feature importance (top 10)
plt.figure(figsize=(10, 8))
top_n = 10
top_features = importance_df.head(top_n)
bars = sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')

# Add value labels on bars
for i, bar in enumerate(bars.patches):
    width = bar.get_width()
    bars.text(width, bar.get_y() + bar.get_height() / 2.,
              f'{width:.4f}',
              ha='left', va='center', fontsize=10, fontweight='bold',
              color= 'black')

plt.title(f'Top {top_n} Feature Importances - Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Cumulative importance plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(importance_df) + 1), importance_df['Cumulative'].values,
         marker='o', linewidth=2, markersize=4)
plt.axhline(y=0.90, color='red', linestyle='--', label='90% Threshold')
plt.axvline(x=features_90_pct, color='red', linestyle='--',
            label=f'{features_90_pct} Features')
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Cumulative Importance', fontsize=12)
plt.title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rf_cumulative_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 9. MODEL EVALUATION METRICS
# ============================================================================
print("\n9. Evaluating Random Forest Performance")

# Make predictions
y_pred_train = rf_optimal.predict(X_train)
y_pred_test = rf_optimal.predict(X_test)

# Get prediction probabilities for additional analysis - returns probabilities for each of the 114 X_test observations
y_pred_proba = rf_optimal.predict_proba(X_test)

# print(y_pred_proba)

# Calculate metrics for training set
train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train)
train_recall = recall_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train)

# Calculate metrics for testing set
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print("\nTRAINING SET METRICS:")
print(f"  Accuracy:  {train_accuracy:.4f}")
print(f"  Precision: {train_precision:.4f}")
print(f"  Recall:    {train_recall:.4f}")
print(f"  F1-Score:  {train_f1:.4f}")

print("\nTESTING SET METRICS:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {test_f1:.4f}")

# Out-of-bag score (if bootstrap=True, which is default)
if hasattr(rf_optimal, 'oob_score_'):
    print(f"\nOut-of-Bag Score: {rf_optimal.oob_score_:.4f}")

print("\nDetailed Classification Report (Testing Set):")
print(classification_report(y_test, y_pred_test, target_names=target_names))

# Prediction confidence analysis
confidence_scores = np.max(y_pred_proba, axis=1)
print(f"\nPrediction Confidence Statistics:")
print(f"  Mean confidence: {np.mean(confidence_scores):.4f}")
print(f"  Min confidence:  {np.min(confidence_scores):.4f}")
print(f"  Max confidence:  {np.max(confidence_scores):.4f}")
print(f"  Standard Deviation of confidence:  {np.std(confidence_scores):.4f}")

# Metrics comparison visualisation (training performance compared to testing performance)
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Training': [train_accuracy, train_precision, train_recall, train_f1],
    'Testing': [test_accuracy, test_precision, test_recall, test_f1]
})

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics_df['Metric']))
width = 0.35
plt.bar(x - width / 2, metrics_df['Training'], width, label='Training', alpha=0.8)
plt.bar(x + width / 2, metrics_df['Testing'], width, label='Testing', alpha=0.8)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Random Forest Performance Metrics Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, metrics_df['Metric'])
plt.legend(fontsize=10)
plt.ylim([0.85, 1.0])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('rf_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Prediction confidence distribution
plt.figure(figsize=(10, 6))
sns.histplot(confidence_scores, bins=30, kde=True, color='steelblue')
plt.xlabel('Prediction Confidence', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Confidence Scores', fontsize=14, fontweight='bold')
plt.axvline(x=np.mean(confidence_scores), color='red', linestyle='--',
            label=f'Mean: {np.mean(confidence_scores):.3f}')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('rf_confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. CONFUSION MATRIX
# ============================================================================
print("\n10. Generating Confusion Matrix")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
plt.title('Confusion Matrix - Random Forest Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"\nAdditional Metrics:")
print(f"  True Positives:  {tp}")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  Sensitivity (TPR): {sensitivity:.4f}")
print(f"  Specificity (TNR): {specificity:.4f}")

# ============================================================================
# SUMMARY & COMPARISON
# ============================================================================
print("\n11. Random Forest Model Summary:")
print(f"Optimal Number of Trees: {optimal_n_trees}")
print(f"Optimal Tree Depth: {optimal_depth}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Most Important Feature: {importance_df.iloc[0]['Feature']}")
print(f"\nKey Insights:")
print(f"  • Random Forest achieved {test_accuracy:.1%} accuracy on test set")
print(f"  • Model shows high confidence (avg {np.mean(confidence_scores):.3f}) in predictions")
print(f"  • Low false negative rate ({fn}) - critical for cancer detection")

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")