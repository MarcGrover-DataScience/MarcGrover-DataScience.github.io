# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
# # Load
# X, y = load_breast_cancer(return_X_y=True)
# print(X)
#
# # Train/test split (stratified since binary)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42, stratify=y
# )
#
# # Model (limit depth to reduce overfitting)
# clf = DecisionTreeClassifier(max_depth=5, random_state=42)
# clf.fit(X_train, y_train)
#
# # Evaluation
# y_pred = clf.predict(X_test)
# print("Test Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
# # Cross-validation (robustness)
# cv_scores = cross_val_score(clf, X, y, cv=5)
# print("CV Accuracy (mean ± std):", cv_scores.mean(), cv_scores.std())



# Decision Tree Classification - Breast Cancer Dataset
# Proof-of-Concept for Categorical Prediction


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler
import warnings

# Configure dataframe printing
desired_width = 320                                                 # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps


warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("DECISION TREE CLASSIFICATION - BREAST CANCER DATASET")

# ============================================================================
# 1. LOAD THE BREAST CANCER DATASET
# ============================================================================
print("\n1. Load Breast Cancer Dataset")
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
print(f"\nDataset description:\n{data.DESCR[:600]}...")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n2. Performing Data Preprocessing...")

# Check for missing values
missing_values = df.isnull().sum().sum()
print(f"Missing values: {missing_values}")

# Check for any preprocessing needs
print(f"\nData types:\n{df.dtypes.value_counts()}")

# No null values in this dataset, no categorical encoding needed

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n3. Performing Exploratory Data Analysis")
print("Generating distribution and correlation plots")

# Class distribution visualization
plt.figure(figsize=(10, 6))
# class_counts = pd.Series(y).value_counts()
# sns.barplot(x=[target_names[0], target_names[1]], y=class_counts.values, palette='viridis')
# plt.title('Class Distribution in Breast Cancer Dataset', fontsize=14, fontweight='bold')
# plt.xlabel('Diagnosis', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# plt.tight_layout()
# plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
# plt.show()



class_counts = pd.Series(y).value_counts()
bars = sns.barplot(x=[target_names[0], target_names[1]], y=class_counts.values, palette='viridis')

# Add value labels on bars
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    bars.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Class Distribution in Breast Cancer Dataset', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# Feature correlation heatmap
plt.figure(figsize=(12, 10))
# top_features = df[list(feature_names[:10])].corr()    # To include only first 10 features
top_features = df[list(feature_names)].corr()
sns.heatmap(top_features, annot=True, fmt='.1f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. TRAIN-TEST SPLIT (80-20)
# ============================================================================
print("\n4. Splitting Data into Training and Testing Sets (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Testing set class distribution: {np.bincount(y_test)}")

# ============================================================================
# 5. MODEL TUNING - OPTIMAL TREE DEPTH
# ============================================================================
print("\n5. Fine-tuning Model - Determining Optimal Tree Depth")

# Test different tree depths
max_depths = range(1, 13)
train_scores = []
test_scores = []
cv_scores = []

for depth in max_depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

    # Cross-validation score
    cv_score = cross_val_score(dt, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(cv_score.mean())

# Find optimal depth
optimal_depth = max_depths[np.argmax(test_scores)]
print(f"Optimal tree depth: {optimal_depth}")
print(f"Best test accuracy: {max(test_scores):.4f}")

# Visualisation of depth analysis
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores, label='Training Accuracy', marker='o', linewidth=2)
plt.plot(max_depths, test_scores, label='Testing Accuracy', marker='s', linewidth=2)
plt.plot(max_depths, cv_scores, label='CV Accuracy (5-fold)', marker='^', linewidth=2)
plt.axvline(x=optimal_depth, color='red', linestyle='--', label=f'Optimal Depth = {optimal_depth}')
plt.xlabel('Tree Depth', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance vs Tree Depth', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('depth_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. TRAIN OPTIMAL MODEL
# ============================================================================
print("\n6. Training Optimal Decision Tree Model")

# Train with optimal depth
dt_optimal = DecisionTreeClassifier(max_depth=optimal_depth, random_state=42)
dt_optimal.fit(X_train, y_train)
dt_optimal_accuracy = dt_optimal.score(X_test, y_test)

print(f"Model trained with max_depth={optimal_depth}")
print(f"Accuracy: {dt_optimal_accuracy:.4f}")

# ============================================================================
# 7. DECISION TREE VISUALIZATION
# ============================================================================
print("\n7. Visualizing Decision Tree Structure")

plt.figure(figsize=(12, 6))
plot_tree(dt_optimal,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title(f'Decision Tree Visualisation (Depth = {optimal_depth})',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('decision_tree_structure.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n8. Performing Feature Importance Analysis")

# Get feature importances
feature_importance = dt_optimal.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# # Visualize feature importance (top 6)
# plt.figure(figsize=(10, 8))
# top_n = 6
# top_features = importance_df.head(top_n)
# sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
# plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
# plt.xlabel('Importance Score', fontsize=12)
# plt.ylabel('Feature', fontsize=12)
# plt.tight_layout()
# plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
# plt.show()


# Visualize feature importance (top 6)
plt.figure(figsize=(10, 8))
top_n = 6
top_features = importance_df.head(top_n)
bars = sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')

for i, bar in enumerate(bars.patches):
    width = bar.get_width()
    bars.text(width, bar.get_y() + bar.get_height()/2.,
             f'{width:.3f}',
             ha='left', va='center', fontsize=11, fontweight='bold',
             color='black')

plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# 9. MODEL EVALUATION METRICS
# ============================================================================
print("\n9. Evaluating Model Performance")

# Make predictions
y_pred_train = dt_optimal.predict(X_train)
y_pred_test = dt_optimal.predict(X_test)

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
print(f"Accuracy:  {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall:    {train_recall:.4f}")
print(f"F1-Score:  {train_f1:.4f}")

print("\nTESTING SET METRICS:")
print(f"Accuracy:  {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

print("\nDetailed Classification Report (Testing Set):")
print(classification_report(y_test, y_pred_test, target_names=target_names))

# Metrics comparison visualization
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Training': [train_accuracy, train_precision, train_recall, train_f1],
    'Testing': [test_accuracy, test_precision, test_recall, test_f1]
})

plt.figure(figsize=(10, 6))
x = np.arange(len(metrics_df['Metric']))
width = 0.35
plt.bar(x - width / 2, metrics_df['Training'], width, label='Training', alpha=0.8, )
plt.bar(x + width / 2, metrics_df['Testing'], width, label='Testing', alpha=0.8)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, metrics_df['Metric'])
plt.legend(fontsize=10)
plt.ylim([0.85, 1.0])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 10. CONFUSION MATRIX
# ============================================================================
print("\n[10] Generating Confusion Matrix")

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
plt.title('Confusion Matrix - Decision Tree Classifier', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"\nAdditional Metrics:")
print(f"True Positives:  {tp}")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"Sensitivity (TPR): {sensitivity:.4f}")
print(f"Specificity (TNR): {specificity:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\nSUMMARY")
print(f"Dataset: Breast Cancer (569 samples, 30 features)")
print(f"Training/Testing Split: 80/20")
print(f"Optimal Tree Depth: {optimal_depth}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print(f"Most Important Feature: {importance_df.iloc[0]['Feature']}")
