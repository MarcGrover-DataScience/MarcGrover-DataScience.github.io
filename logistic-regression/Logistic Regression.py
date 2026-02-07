# Bank Customer Churn Prediction using Logistic Regression
# Binary classification using logistic regression to predict customer churn in the banking sector.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)
from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')
import time

# Configure dataframe printing
desired_width = 320                                                 # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps

# Start timer
t0 = time.time()  # Add at start of process

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# 1. DATA LOADING

df = pd.read_csv('Bank Customer Churn Prediction.csv')
print(f"\nSample records:\n{df}")
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

# 2. EXPLORATORY DATA ANALYSIS

# Check target variable distribution
print(f"\nTarget variable distribution:")
print(df['churn'].value_counts())
print(f"\nChurn rate: {df['churn'].mean():.2%}")

churn_ratio = df['churn'].value_counts()[1] / df['churn'].value_counts()[0]
print(f"Class imbalance ratio (churn/no-churn): {churn_ratio:.2f}")
if churn_ratio < 0.5:
    print("Class imbalance detected - will address with class weights")

# 3. FEATURE ENGINEERING

df_engineered = df.copy()

# Create balance per product ratio value
df_engineered['balance_per_product'] = df_engineered['balance'] / (df_engineered['products_number'] + 0.1)

# Create customer engagement score (combination of tenure and products)
df_engineered['engagement_score'] = df_engineered['tenure'] * df_engineered['products_number']

# Age group categories
df_engineered['age_group'] = pd.cut(df_engineered['age'],
                                    bins=[0, 30, 40, 50, 60, 100],
                                    labels=['<30', '30-40', '40-50', '50-60', '60+'])

# Balance category
df_engineered['balance_category'] = pd.cut(df_engineered['balance'],
                                           bins=[0, 1, 50000, 100000, 250000],
                                           labels=['Zero', 'Low', 'Medium', 'High'])

# Salary to balance ratio
df_engineered['salary_balance_ratio'] = df_engineered['estimated_salary'] / (df_engineered['balance'] + 1)

# High value customer flag (multiple products + high balance)
df_engineered['high_value_customer'] = ((df_engineered['products_number'] >= 2) &
                                        (df_engineered['balance'] > 100000)).astype(int)

# Credit score categories
df_engineered['credit_category'] = pd.cut(df_engineered['credit_score'],
                                          bins=[0, 580, 670, 740, 800, 850],
                                          labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

print("\nEngineered features created:")
print("- balance_per_product: Balance divided by number of products")
print("- engagement_score: Tenure × Products (customer engagement proxy)")
print("- age_group: Categorical age groups")
print("- balance_category: Categorical balance groups")
print("- salary_balance_ratio: Income to balance ratio")
print("- high_value_customer: Binary flag for valuable customers")
print("- credit_category: Categorical credit score groups")
print(f"\nSample records (engineered features):\n{df_engineered}")

# 4. DATA PREPROCESSING

df_processed = df_engineered.copy()

# Remove customer_id as it's not predictive
df_processed = df_processed.drop('customer_id', axis=1)

# Binary encoding for gender (Female=1, Male=0)
df_processed['gender'] = (df_processed['gender'] == 'Female').astype(int)

# One-hot encoding for country
country_dummies = pd.get_dummies(df_processed['country'], prefix='country', drop_first=True)
df_processed = pd.concat([df_processed.drop('country', axis=1), country_dummies], axis=1)

#     categorical_features = ['age_group', 'balance_category', 'credit_category']
# One-hot encoding for age_group
age_dummies = pd.get_dummies(df_processed['age_group'], prefix='age_group', drop_first=True)
df_processed = pd.concat([df_processed.drop('age_group', axis=1), age_dummies], axis=1)

# One-hot encoding for balance_category
balance_category_dummies = pd.get_dummies(df_processed['balance_category'], prefix='balance_category', drop_first=True)
df_processed = pd.concat([df_processed.drop('balance_category', axis=1), balance_category_dummies], axis=1)

# One-hot encoding for credit_category
credit_category_dummies = pd.get_dummies(df_processed['credit_category'], prefix='credit_category', drop_first=True)
df_processed = pd.concat([df_processed.drop('credit_category', axis=1), credit_category_dummies], axis=1)

print(df_processed)

# 5. FEATURE-TARGET SPLIT, TRAIN-TEST SPLIT AND SCALING
print("\nSPLITTING AND SCALING")

# Separate features and target
X = df_processed.drop('churn', axis=1)
y = df_processed['churn']

print(f"\nFinal feature set: {X.shape[1]} features")
print(f"Feature names: {X.columns.tolist()}")

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples ({(1 - 0.2) * 100:.0f}%)")
print(f"\nTesting set size: {X_test.shape[0]} samples ({0.2 * 100:.0f}%)")
print(f"\nTraining set churn rate: {y_train.mean():.2%}")
print(f"\nTesting set churn rate: {y_test.mean():.2%}")

# Apply StandardScaler to numerical features
# Identify numerical columns (exclude binary/one-hot encoded columns)
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# Don't scale binary features (those with only 0 and 1)
binary_features = [col for col in numerical_features
                   if X_train[col].nunique() == 2 and set(X_train[col].unique()).issubset({0, 1})]

features_to_scale = [col for col in numerical_features if col not in binary_features]

print(f"\nScaling {len(features_to_scale)} numerical features")
print(f"Not scaling {len(binary_features)} binary features")

# Fit scaler on training data only
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

print("\nScaling completed (fitted on training data only)")

# 6. MODEL TRAINING
print("\nMODEL TRAINING")

# Train the model
model = LogisticRegression(
    class_weight='balanced',
    C=1.0,
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)

model.fit(X_train_scaled, y_train)
print("\nModel training completed")

# 7. MODEL EVALUATION
print("\nMODEL EVALUATION")

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics for both train and test
print("TRAINING SET PERFORMANCE")
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_train, y_train_proba):.4f}")
print(f"\nClassification Report:\n{classification_report(y_train, y_train_pred)}")

print("TESTING SET PERFORMANCE")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_test_pred)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")

# Store metrics for visualization
metrics = {
    'train_accuracy': accuracy_score(y_train, y_train_pred),
    'test_accuracy': accuracy_score(y_test, y_test_pred),
    'train_roc_auc': roc_auc_score(y_train, y_train_proba),
    'test_roc_auc': roc_auc_score(y_test, y_test_proba),
    'test_f1': f1_score(y_test, y_test_pred),
    'confusion_matrix': cm,
    'y_test': y_test,
    'y_test_pred': y_test_pred,
    'y_test_proba': y_test_proba,
    'y_train_proba': y_train_proba
}

test_roc_auc = roc_auc_score(y_test, y_test_proba)

# 8. FEATURE IMPORTANCE ANALYSIS

coefficients = model.coef_[0]
top_n = 10

# Create DataFrame for analysis
feature_importance = pd.DataFrame({
    'feature': X.columns.tolist(), # feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop {top_n} Most Important Features:")
print("-" * 80)
for idx, row in feature_importance.head(top_n).iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"{row['feature']:30s} | Coef: {row['coefficient']:8.4f} | {direction} churn probability")


# 9. VISUALIZATIONS

# 9.1. Target Variable Distribution
plt.figure(figsize=(10, 6))
churn_counts = df['churn'].value_counts()
ax = sns.barplot(x=['No Churn', 'Churn'], y=churn_counts.values, palette='Greens', hue=['No Churn', 'Churn'], legend=False)
plt.title('Target Variable Distribution', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Churn Status', fontsize=12)
for i, v in enumerate(churn_counts.values):
    plt.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold', fontsize=12)
def thousands_formatter(x, pos):
    return f'{int(x):,}'
ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.tight_layout()
plt.savefig('plot_1_target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            annot_kws={'size': 14, 'weight': 'bold'},
            linewidths=2, linecolor='white')
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('plot_2_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.3. ROC Curve
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.plot(fpr, tpr, linewidth=3, label=f"ROC AUC = {test_roc_auc:.3f}", color='#3498db')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_3_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.4. Precision-Recall Curve
plt.figure(figsize=(10, 8))
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
plt.plot(recall, precision, linewidth=3, color='#9b59b6')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_4_precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.5. Feature Importance (Top 10)
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10).copy()
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features['coefficient']]
ax = sns.barplot(y=top_features['feature'], x=top_features['coefficient'],
                 palette='Greens_r', orient='h', hue=top_features['feature'], legend=False)
for i, (idx, row) in enumerate(top_features.iterrows()):
    x_pos = row['coefficient']
    plt.text(x_pos, i, f'  {x_pos:.4f}',
             va='center', ha='left' if x_pos > 0 else 'right',
             fontweight='bold', fontsize=10)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 10 Feature Importance', fontsize=16, fontweight='bold', pad=20)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
plt.tight_layout()
plt.savefig('plot_5_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 9.6. Predicted Probability Distribution
plt.figure(figsize=(12, 8))
# Create data for plotting
prob_data = pd.DataFrame({
    'Probability': y_test_proba,
    'Actual': ['Churn' if x == 1 else 'No Churn' for x in y_test]
})
sns.histplot(data=prob_data, x='Probability', hue='Actual', bins=50,
             palette={'No Churn': '#2ecc71', 'Churn': '#e74c3c'},
             alpha=0.6, kde=True, legend=True)
plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Predicted Probability Distribution by Actual Class', fontsize=16, fontweight='bold', pad=20)
# plt.legend(fontsize=11)
# plt.legend(title='Actual Class', fontsize=11, title_fontsize=12)
# Manually create a clear legend with colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.6, label='No Churn'),
    Patch(facecolor='#e74c3c', alpha=0.6, label='Churn'),
    plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Threshold (0.5)')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=11, title='Legend')
plt.tight_layout()
plt.savefig('plot_6_probability_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# # 7. Model Performance Metrics Comparison
# print("\nCreating Plot 7: Model Performance Metrics...")
# plt.figure(figsize=(10, 7))
# metrics_comparison = {
#     'Accuracy': test_accuracy,
#     'ROC-AUC': test_roc_auc,
#     'F1-Score': test_f1
# }
# metrics_df = pd.DataFrame(list(metrics_comparison.items()), columns=['Metric', 'Score'])
# ax = sns.barplot(x='Metric', y='Score', data=metrics_df,
#                  palette=['#3498db', '#9b59b6', '#e67e22'])
# plt.ylabel('Score', fontsize=12)
# plt.xlabel('Metric', fontsize=12)
# plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
# plt.ylim([0, 1])
# for i, (metric, score) in enumerate(metrics_comparison.items()):
#     plt.text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold', fontsize=12)
# plt.tight_layout()
# plt.savefig('plot_7_performance_metrics.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("✓ Saved: plot_7_performance_metrics.png")
#
# # 8. Churn Rate by Age
# print("\nCreating Plot 8: Churn Rate by Age...")
# plt.figure(figsize=(14, 7))
# age_churn = df.groupby('age')['churn'].mean()
# sns.lineplot(x=age_churn.index, y=age_churn.values, linewidth=3, color='#e74c3c', marker='o')
# plt.xlabel('Age', fontsize=12)
# plt.ylabel('Churn Rate', fontsize=12)
# plt.title('Churn Rate by Age', fontsize=16, fontweight='bold', pad=20)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot_8_churn_by_age.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("✓ Saved: plot_8_churn_by_age.png")
#
# # 9. Churn Rate by Number of Products (with dual axis)
# print("\nCreating Plot 9: Churn Rate by Number of Products...")
# fig, ax1 = plt.subplots(figsize=(10, 7))
# product_churn = df.groupby('products_number')['churn'].agg(['mean', 'count'])
#
# # Bar plot for churn rate
# color1 = '#3498db'
# ax1.bar(product_churn.index, product_churn['mean'], color=color1, alpha=0.7, label='Churn Rate')
# ax1.set_xlabel('Number of Products', fontsize=12)
# ax1.set_ylabel('Churn Rate', fontsize=12, color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.set_ylim([0, max(product_churn['mean']) * 1.2])
#
# # Line plot for customer count on secondary y-axis
# ax2 = ax1.twinx()
# color2 = '#e74c3c'
# ax2.plot(product_churn.index, product_churn['count'], 'o-', linewidth=3,
#          markersize=10, label='Customer Count', color=color2)
# ax2.set_ylabel('Customer Count', fontsize=12, color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)
#
# plt.title('Churn Rate and Customer Count by Number of Products',
#           fontsize=16, fontweight='bold', pad=20)
# # Combine legends
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)
# plt.tight_layout()
# plt.savefig('plot_9_churn_by_products.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("✓ Saved: plot_9_churn_by_products.png")
#
# # 10. BONUS: Churn Rate by Country (if country data exists)
# if 'country' in df.columns:
#     print("\nCreating Plot 10: Churn Rate by Country...")
#     plt.figure(figsize=(10, 6))
#     country_churn = df.groupby('country')['churn'].agg(['mean', 'count']).reset_index()
#     ax = sns.barplot(x='country', y='mean', data=country_churn, palette='viridis')
#     plt.ylabel('Churn Rate', fontsize=12)
#     plt.xlabel('Country', fontsize=12)
#     plt.title('Churn Rate by Country', fontsize=16, fontweight='bold', pad=20)
#     # Add value labels
#     for i, row in country_churn.iterrows():
#         plt.text(i, row['mean'] + 0.01, f"{row['mean']:.2%}\n(n={row['count']})",
#                 ha='center', fontsize=10, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig('plot_10_churn_by_country.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print("✓ Saved: plot_10_churn_by_country.png")
#
# # 11. BONUS: Churn Rate by Age Group (if engineered)
# if 'age_group' in df_engineered.columns:
#     print("\nCreating Plot 11: Churn Rate by Age Group...")
#     plt.figure(figsize=(10, 6))
#     age_group_churn = df_engineered.groupby('age_group')['churn'].agg(['mean', 'count']).reset_index()
#     ax = sns.barplot(x='age_group', y='mean', data=age_group_churn, palette='rocket')
#     plt.ylabel('Churn Rate', fontsize=12)
#     plt.xlabel('Age Group', fontsize=12)
#     plt.title('Churn Rate by Age Group', fontsize=16, fontweight='bold', pad=20)
#     plt.xticks(rotation=0)
#     # Add value labels
#     for i, row in age_group_churn.iterrows():
#         plt.text(i, row['mean'] + 0.01, f"{row['mean']:.2%}\n(n={row['count']})",
#                 ha='center', fontsize=10, fontweight='bold')
#     plt.tight_layout()
#     plt.savefig('plot_11_churn_by_age_group.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print("✓ Saved: plot_11_churn_by_age_group.png")
#
# # 12. BONUS: Correlation Heatmap of Key Features
# print("\nCreating Plot 12: Feature Correlation Heatmap...")
# plt.figure(figsize=(14, 10))
# # Select numerical features for correlation
# numerical_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number',
#                  'credit_card', 'active_member', 'estimated_salary', 'churn']
# correlation_matrix = df[numerical_cols].corr()
# sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
#             center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
# plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
# plt.tight_layout()
# plt.savefig('plot_12_correlation_heatmap.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("✓ Saved: plot_12_correlation_heatmap.png")
#
# # 13. BONUS: Box Plot - Age Distribution by Churn Status
# print("\nCreating Plot 13: Age Distribution by Churn Status...")
# plt.figure(figsize=(10, 7))
# churn_labels = {0: 'No Churn', 1: 'Churn'}
# df_plot = df.copy()
# df_plot['churn_label'] = df_plot['churn'].map(churn_labels)
# sns.boxplot(x='churn_label', y='age', data=df_plot, palette=['#2ecc71', '#e74c3c'])
# sns.swarmplot(x='churn_label', y='age', data=df_plot, color='black', alpha=0.3, size=2)
# plt.ylabel('Age', fontsize=12)
# plt.xlabel('Churn Status', fontsize=12)
# plt.title('Age Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
# plt.tight_layout()
# plt.savefig('plot_13_age_boxplot.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("✓ Saved: plot_13_age_boxplot.png")
#
# # 14. BONUS: Violin Plot - Balance Distribution by Churn Status
# print("\nCreating Plot 14: Balance Distribution by Churn Status...")
# plt.figure(figsize=(10, 7))
# sns.violinplot(x='churn_label', y='balance', data=df_plot, palette=['#2ecc71', '#e74c3c'])
# plt.ylabel('Account Balance', fontsize=12)
# plt.xlabel('Churn Status', fontsize=12)
# plt.title('Balance Distribution by Churn Status', fontsize=16, fontweight='bold', pad=20)
# plt.tight_layout()
# plt.savefig('plot_14_balance_violin.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("✓ Saved: plot_14_balance_violin.png")
#
# # 15. BONUS: Count Plot - Gender and Churn
# print("\nCreating Plot 15: Churn by Gender...")
# plt.figure(figsize=(10, 6))
# gender_labels = {0: 'Male', 1: 'Female'}
# df_plot['gender_label'] = df_plot['gender'].map(gender_labels)
# sns.countplot(x='gender_label', hue='churn_label', data=df_plot, palette=['#2ecc71', '#e74c3c'])
# plt.ylabel('Count', fontsize=12)
# plt.xlabel('Gender', fontsize=12)
# plt.title('Customer Churn by Gender', fontsize=16, fontweight='bold', pad=20)
# plt.legend(title='Status', fontsize=11)
# plt.tight_layout()
# plt.savefig('plot_15_churn_by_gender.png', dpi=300, bbox_inches='tight')
# plt.show()
# print("✓ Saved: plot_15_churn_by_gender.png")
#






# ##  OLD SECTION 9
#
# def create_visualizations(df, metrics, feature_importance, model, X_test):
#     """Create comprehensive visualizations"""
#     print("\n" + "=" * 80)
#     print("CREATING VISUALIZATIONS")
#     print("=" * 80)
#
#     # Create figure with subplots
#     fig = plt.figure(figsize=(20, 15))
#
#     # 1. Target Distribution
#     ax1 = plt.subplot(3, 3, 1)
#     churn_counts = df['churn'].value_counts()
#     ax1.bar(['No Churn', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
#     ax1.set_title('Target Variable Distribution', fontsize=12, fontweight='bold')
#     ax1.set_ylabel('Count')
#     for i, v in enumerate(churn_counts.values):
#         ax1.text(i, v + 500, str(v), ha='center', fontweight='bold')
#
#     # 2. Confusion Matrix Heatmap
#     ax2 = plt.subplot(3, 3, 2)
#     sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No Churn', 'Churn'],
#                 yticklabels=['No Churn', 'Churn'], ax=ax2)
#     ax2.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
#     ax2.set_ylabel('Actual')
#     ax2.set_xlabel('Predicted')
#
#     # 3. ROC Curve
#     ax3 = plt.subplot(3, 3, 3)
#     fpr, tpr, _ = roc_curve(metrics['y_test'], metrics['y_test_proba'])
#     ax3.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {metrics['test_roc_auc']:.3f}")
#     ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
#     ax3.set_xlabel('False Positive Rate')
#     ax3.set_ylabel('True Positive Rate')
#     ax3.set_title('ROC Curve', fontsize=12, fontweight='bold')
#     ax3.legend()
#     ax3.grid(True, alpha=0.3)
#
#     # 4. Precision-Recall Curve
#     ax4 = plt.subplot(3, 3, 4)
#     precision, recall, _ = precision_recall_curve(metrics['y_test'], metrics['y_test_proba'])
#     ax4.plot(recall, precision, linewidth=2, color='#9b59b6')
#     ax4.set_xlabel('Recall')
#     ax4.set_ylabel('Precision')
#     ax4.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
#     ax4.grid(True, alpha=0.3)
#
#     # 5. Feature Importance (Top 10)
#     ax5 = plt.subplot(3, 3, 5)
#     top_features = feature_importance.head(10)
#     colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_features['coefficient']]
#     ax5.barh(range(len(top_features)), top_features['coefficient'], color=colors)
#     ax5.set_yticks(range(len(top_features)))
#     ax5.set_yticklabels(top_features['feature'])
#     ax5.set_xlabel('Coefficient Value')
#     ax5.set_title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
#     ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
#     ax5.invert_yaxis()
#
#     # 6. Predicted Probability Distribution
#     ax6 = plt.subplot(3, 3, 6)
#     ax6.hist(metrics['y_test_proba'][metrics['y_test'] == 0], bins=50, alpha=0.6,
#              label='No Churn', color='#2ecc71')
#     ax6.hist(metrics['y_test_proba'][metrics['y_test'] == 1], bins=50, alpha=0.6,
#              label='Churn', color='#e74c3c')
#     ax6.set_xlabel('Predicted Probability')
#     ax6.set_ylabel('Frequency')
#     ax6.set_title('Predicted Probability Distribution', fontsize=12, fontweight='bold')
#     ax6.legend()
#     ax6.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Threshold')
#
#     # 7. Model Performance Metrics Comparison
#     ax7 = plt.subplot(3, 3, 7)
#     metrics_comparison = {
#         'Accuracy': metrics['test_accuracy'],
#         'ROC-AUC': metrics['test_roc_auc'],
#         'F1-Score': metrics['test_f1']
#     }
#     bars = ax7.bar(metrics_comparison.keys(), metrics_comparison.values(),
#                    color=['#3498db', '#9b59b6', '#e67e22'])
#     ax7.set_ylabel('Score')
#     ax7.set_title('Model Performance Metrics', fontsize=12, fontweight='bold')
#     ax7.set_ylim([0, 1])
#     for bar in bars:
#         height = bar.get_height()
#         ax7.text(bar.get_x() + bar.get_width() / 2., height,
#                  f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
#
#     # 8. Churn by Age (if age exists)
#     if 'age' in df.columns:
#         ax8 = plt.subplot(3, 3, 8)
#         age_churn = df.groupby('age')['churn'].mean()
#         ax8.plot(age_churn.index, age_churn.values, linewidth=2, color='#e74c3c')
#         ax8.set_xlabel('Age')
#         ax8.set_ylabel('Churn Rate')
#         ax8.set_title('Churn Rate by Age', fontsize=12, fontweight='bold')
#         ax8.grid(True, alpha=0.3)
#
#     # 9. Churn by Number of Products (if exists)
#     if 'products_number' in df.columns:
#         ax9 = plt.subplot(3, 3, 9)
#         product_churn = df.groupby('products_number')['churn'].agg(['mean', 'count'])
#         ax9.bar(product_churn.index, product_churn['mean'], color='#3498db', alpha=0.7)
#         ax9.set_xlabel('Number of Products')
#         ax9.set_ylabel('Churn Rate')
#         ax9.set_title('Churn Rate by Number of Products', fontsize=12, fontweight='bold')
#         ax9_twin = ax9.twinx()
#         ax9_twin.plot(product_churn.index, product_churn['count'], 'ro-', linewidth=2,
#                       markersize=8, label='Count')
#         ax9_twin.set_ylabel('Customer Count', color='r')
#         ax9_twin.tick_params(axis='y', labelcolor='r')
#
#     plt.tight_layout()
#     plt.savefig('/home/claude/churn_analysis_visualizations.png', dpi=300, bbox_inches='tight')
#     print("✓ Visualizations saved as 'churn_analysis_visualizations.png'")
#
#     return fig
#
#
# # ============================================================================
# # 10. COMPARATIVE ANALYSIS (WITH/WITHOUT FEATURE ENGINEERING)
# # ============================================================================
#
# def compare_models(df):
#     """Compare model performance with and without engineered features"""
#     print("\n" + "=" * 80)
#     print("COMPARATIVE ANALYSIS: FEATURE ENGINEERING IMPACT")
#     print("=" * 80)
#
#     results = {}
#
#     # Model 1: Without engineered features
#     print("\n[1/2] Training baseline model (without engineered features)...")
#     X_base, y_base = preprocess_data(df, include_engineered_features=False)
#     X_train_base, X_test_base, y_train_base, y_test_base, _ = split_and_scale_data(X_base, y_base)
#     model_base = train_logistic_regression(X_train_base, y_train_base)
#
#     y_pred_base = model_base.predict(X_test_base)
#     y_proba_base = model_base.predict_proba(X_test_base)[:, 1]
#
#     results['baseline'] = {
#         'accuracy': accuracy_score(y_test_base, y_pred_base),
#         'roc_auc': roc_auc_score(y_test_base, y_proba_base),
#         'f1_score': f1_score(y_test_base, y_pred_base)
#     }
#
#     # Model 2: With engineered features
#     print("\n[2/2] Training enhanced model (with engineered features)...")
#     df_engineered = create_engineered_features(df)
#     X_eng, y_eng = preprocess_data(df_engineered, include_engineered_features=True)
#     X_train_eng, X_test_eng, y_train_eng, y_test_eng, _ = split_and_scale_data(X_eng, y_eng)
#     model_eng = train_logistic_regression(X_train_eng, y_train_eng)
#
#     y_pred_eng = model_eng.predict(X_test_eng)
#     y_proba_eng = model_eng.predict_proba(X_test_eng)[:, 1]
#
#     results['engineered'] = {
#         'accuracy': accuracy_score(y_test_eng, y_pred_eng),
#         'roc_auc': roc_auc_score(y_test_eng, y_proba_eng),
#         'f1_score': f1_score(y_test_eng, y_pred_eng)
#     }
#
#     # Display comparison
#     print("\n" + "-" * 80)
#     print("PERFORMANCE COMPARISON")
#     print("-" * 80)
#     comparison_df = pd.DataFrame(results).T
#     comparison_df['improvement'] = (comparison_df.iloc[1] - comparison_df.iloc[0]) / comparison_df.iloc[0] * 100
#
#     print("\n", comparison_df.to_string())
#
#     print("\n" + "-" * 80)
#     print("INSIGHTS:")
#     print("-" * 80)
#     if results['engineered']['roc_auc'] > results['baseline']['roc_auc']:
#         improvement = (results['engineered']['roc_auc'] - results['baseline']['roc_auc']) / results['baseline'][
#             'roc_auc'] * 100
#         print(f"✓ Feature engineering improved ROC-AUC by {improvement:.2f}%")
#     else:
#         print("⚠ Feature engineering did not improve performance significantly")
#
#     return results, model_eng, X_test_eng, y_test_eng
#
#
# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================
#
# def main():
#     """Main execution function"""
#
#     print("\n" + "=" * 80)
#     print("BANK CUSTOMER CHURN PREDICTION - LOGISTIC REGRESSION")
#     print("=" * 80)
#
#     # Update this path to your CSV file location
#     filepath = 'Bank Customer Churn Prediction.csv'  # Update this path
#
#     try:
#         # 1. Load data
#         df = load_data(filepath)
#
#         # 2. EDA
#         churn_ratio = perform_eda(df)
#
#         # 3. Feature Engineering
#         df_engineered = create_engineered_features(df)
#
#         # 4. Preprocess data
#         X, y = preprocess_data(df_engineered, include_engineered_features=True)
#
#         # 5. Split and scale
#         X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
#
#         # 6. Train model with class weights to handle imbalance
#         model = train_logistic_regression(X_train, y_train, use_class_weights=True)
#
#         # 7. Evaluate model
#         metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
#
#         # 8. Feature importance
#         feature_importance = analyze_feature_importance(model, X.columns.tolist())
#
#         # 9. Visualizations
#         fig = create_visualizations(df_engineered, metrics, feature_importance, model, X_test)
#
#         # 10. Comparative analysis
#         comparison_results, best_model, X_test_final, y_test_final = compare_models(df)
#
#         print("\n" + "=" * 80)
#         print("ANALYSIS COMPLETE")
#         print("=" * 80)
#         print("\nKey Deliverables:")
#         print("  ✓ Trained logistic regression model")
#         print("  ✓ Comprehensive performance metrics")
#         print("  ✓ Feature importance analysis")
#         print("  ✓ Visualization dashboard saved")
#         print("  ✓ Feature engineering impact assessment")
#
#         print("\nNext Steps:")
#         print("  1. Review visualizations in 'churn_analysis_visualizations.png'")
#         print("  2. Analyze feature importance for business insights")
#         print("  3. Consider threshold optimization for business objectives")
#         print("  4. Deploy model or integrate into portfolio documentation")
#
#         return model, X_test, y_test, metrics, feature_importance
#
#     except FileNotFoundError:
#         print(f"\n❌ Error: Could not find file '{filepath}'")
#         print("Please update the 'filepath' variable with the correct path to your CSV file.")
#         return None
#
#     except Exception as e:
#         print(f"\n❌ Error occurred: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None
#
#
# if __name__ == "__main__":
#     # Execute the analysis
#     results = main()
#
#     # Keep plots displayed
#     plt.show()

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")