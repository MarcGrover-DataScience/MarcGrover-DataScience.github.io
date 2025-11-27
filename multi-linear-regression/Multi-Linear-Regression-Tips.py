# Simple Multiple Linear Regression Analysis
# Dataset: Restaurant Tips Dataset from Seaborn
# Predicting tip amount based on bill, party size, and time of day

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')

# Dataframe presentation configuration
desired_width = 320                                                 # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("MULTIPLE LINEAR REGRESSION ANALYSIS")
print("Dataset: Restaurant Tips")
 

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================

print("\n1. DATA LOADING AND EXPLORATION")

# Load the tips dataset
df = sns.load_dataset('tips')

print("\nDataset Shape:", df.shape)
print(f"Number of observations: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")

print("\nFirst few rows:")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nNumerical Variables Summary:")
print(df.describe())

print("\nCategorical Variables:")
print(f"Sex: {df['sex'].unique()}")
print(f"Smoker: {df['smoker'].unique()}")
print(f"Day: {df['day'].unique()}")
print(f"Time: {df['time'].unique()}")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nRESEARCH QUESTION")
print("Can we predict the tip amount based on:")
print("  - Total bill amount")
print("  - Party size")
print("  - Time of day (Lunch vs Dinner)")

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

print("\n2. DATA PREPARATION")
 

# Create a copy for modeling
df_model = df.copy()

# Convert time to binary (0 = Lunch, 1 = Dinner)
df_model['time_dinner'] = (df_model['time'] == 'Dinner').astype(int)

# Select features for our model
features = ['total_bill', 'size', 'time_dinner']
X = df_model[features]
y = df_model['tip']

print("\nFeature Statistics (independent variables):")
print(X.describe())

# ============================================================================
# 3. CORRELATION ANALYSIS
# ============================================================================

print("\n3. CORRELATION ANALYSIS")
 

# Create dataframe with features and target
df_corr = X.copy()
df_corr['tip'] = y

correlation_matrix = df_corr.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

print("\nCorrelation with Tip Amount:")
print(correlation_matrix['tip'].sort_values(ascending=False))

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.3f', square=True, linewidths=2, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("\nInterpretation:")
print("- Strong positive correlation means as one variable increases, the other increases")
print("- Values close to 1 or -1 indicate strong relationships")
print("- Values close to 0 indicate weak relationships")

# ============================================================================
# 4. SCATTER PLOTS - EXPLORING RELATIONSHIPS
# ============================================================================

print("\n4. SCATTER PLOTS - EXPLORING RELATIONSHIPS")
 

# Total Bill vs Tip
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_model, x='total_bill', y='tip', alpha=0.6, s=80)
plt.title('Total Bill vs Tip Amount', fontsize=14, fontweight='bold')
plt.xlabel('Total Bill ($)', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Party Size vs Tip
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_model, x='size', y='tip', alpha=0.6, s=80)
plt.title('Party Size vs Tip Amount', fontsize=14, fontweight='bold')
plt.xlabel('Party Size (number of people)', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Time of Day vs Tip (Box plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='time', y='tip', palette='Set2')
plt.title('Tip Amount by Time of Day', fontsize=14, fontweight='bold')
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Tip ($)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ============================================================================
# 5. MULTICOLLINEARITY CHECK (VIF)
# ============================================================================

print("\n5. MULTICOLLINEARITY CHECK")
 

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("\nVariance Inflation Factor (VIF):")
print(vif_data)

print("\nInterpretation Guide:")
print("  VIF = 1: No correlation with other features")
print("  VIF < 5: Low multicollinearity (Good!)")
print("  VIF 5-10: Moderate multicollinearity")
print("  VIF > 10: High multicollinearity (Problem!)")

plt.figure(figsize=(10, 6))
colors = ['green' if vif < 5 else 'orange' if vif < 10 else 'red' for vif in vif_data['VIF']]
sns.barplot(data=vif_data, x='VIF', y='Feature', palette=colors)
plt.title('Variance Inflation Factor (VIF) - Checking Multicollinearity', 
          fontsize=14, fontweight='bold')
plt.xlabel('VIF Value', fontsize=12)
plt.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='Moderate threshold')
plt.axvline(x=10, color='red', linestyle='--', linewidth=2, label='High threshold')
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================================
# 6. SPLIT DATA INTO TRAINING AND TEST SETS
# ============================================================================

print("\n6. SPLITTING DATA")
 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTotal observations: {len(X)}")
print(f"Training set: {len(X_train)} observations (80%)")
print(f"Test set: {len(X_test)} observations (20%)")

print("\nWhy do we split the data?")
print("  - Training set: Used to build/train the model")
print("  - Test set: Used to evaluate how well the model works on new data")
print("  - This helps us avoid overfitting!")

# ============================================================================
# 7. FEATURE SCALING (STANDARDIZATION)
# ============================================================================

print("\n7. FEATURE SCALING")
 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\nBefore scaling:")
print(X_train.describe().loc[['mean', 'std']])

print("\nAfter scaling (Standardization):")
print(X_train_scaled.describe().loc[['mean', 'std']])

print("\nWhy scale features?")
print("  - Different features have different ranges (e.g., bill: $3-50, size: 1-6)")
print("  - Scaling puts all features on the same scale")
print("  - Makes coefficients directly comparable")
print("  - Improves model training and interpretation")

# ============================================================================
# 8. BUILD THE LINEAR REGRESSION MODEL
# ============================================================================

print("\n8. BUILDING THE LINEAR REGRESSION MODEL")
 

# Create and train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n✓ Model trained successfully!")

print(f"\nModel Equation:")
print(f"Tip = {model.intercept_:.3f}", end="")
for feature, coef in zip(X.columns, model.coef_):
    sign = "+" if coef >= 0 else ""
    print(f" {sign} {coef:.3f}×{feature}", end="")
print()

# Display coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nModel Coefficients (Standardized):")
print(coefficients)

print("\nInterpretation of Coefficients:")
print("  - Positive coefficient: Feature increases → Tip increases")
print("  - Negative coefficient: Feature increases → Tip decreases")
print("  - Larger absolute value: Stronger effect on tip amount")

# Visualize coefficients
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coefficients['Coefficient']]
sns.barplot(data=coefficients, x='Coefficient', y='Feature', palette=colors)
plt.title('Model Coefficients (Feature Importance)', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient Value (Standardized)', fontsize=12)
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.tight_layout()
plt.show()

# ============================================================================
# 9. MAKE PREDICTIONS
# ============================================================================

print("\n9. MAKING PREDICTIONS")
 

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\nExample predictions on test set:")
comparison = pd.DataFrame({
    'Actual Tip': y_test.values[:10],
    'Predicted Tip': y_test_pred[:10],
    'Difference': y_test.values[:10] - y_test_pred[:10]
})
comparison.index = range(1, 11)
print(comparison)

# ============================================================================
# 10. MODEL EVALUATION
# ============================================================================

print("\n10. MODEL EVALUATION")
 

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nTRAINING SET PERFORMANCE:")
print(f"  R² Score: {train_r2:.4f} ({train_r2*100:.1f}% of variance explained)")
print(f"  RMSE: ${train_rmse:.2f}")
print(f"  MAE: ${train_mae:.2f}")

print("\nTEST SET PERFORMANCE:")
print(f"  R² Score: {test_r2:.4f} ({test_r2*100:.1f}% of variance explained)")
print(f"  RMSE: ${test_rmse:.2f}")
print(f"  MAE: ${test_mae:.2f}")

print("\nWhat do these metrics mean?")
print(f"  R² Score: Our model explains {test_r2*100:.1f}% of the variation in tips")
print(f"  RMSE: On average, predictions are off by about ${test_rmse:.2f}")
print(f"  MAE: On average, absolute error is ${test_mae:.2f}")

# ============================================================================
# 11. ACTUAL VS PREDICTED PLOTS
# ============================================================================

print("\n11. ACTUAL VS PREDICTED VALUES")
 

# Training set
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6, s=80)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=3, label='Perfect Prediction Line')
plt.xlabel('Actual Tip ($)', fontsize=12)
plt.ylabel('Predicted Tip ($)', fontsize=12)
plt.title(f'Training Set: Actual vs Predicted Tips (R² = {train_r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Test set
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, s=80, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=3, label='Perfect Prediction Line')
plt.xlabel('Actual Tip ($)', fontsize=12)
plt.ylabel('Predicted Tip ($)', fontsize=12)
plt.title(f'Test Set: Actual vs Predicted Tips (R² = {test_r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nHow to read these plots:")
print("  - Points close to the red line = Good predictions")
print("  - Points far from the red line = Prediction errors")
print("  - Random scatter around the line = Good model fit")

# ============================================================================
# 12. RESIDUAL ANALYSIS
# ============================================================================

print("\n12. RESIDUAL ANALYSIS")
 

# Calculate residuals
train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

print("\nWhat are residuals?")
print("  Residual = Actual Value - Predicted Value")
print("  Residuals tell us how far off our predictions are")

print(f"\nTraining Residuals:")
print(f"  Mean: ${train_residuals.mean():.4f} (should be close to 0)")
print(f"  Std Dev: ${train_residuals.std():.2f}")

print(f"\nTest Residuals:")
print(f"  Mean: ${test_residuals.mean():.4f} (should be close to 0)")
print(f"  Std Dev: ${test_residuals.std():.2f}")

# Residuals vs Predicted Values (Training)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred, y=train_residuals, alpha=0.6, s=80)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.xlabel('Predicted Tip ($)', fontsize=12)
plt.ylabel('Residuals ($)', fontsize=12)
plt.title('Training Set: Residual Plot', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Residuals vs Predicted Values (Test)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_pred, y=test_residuals, alpha=0.6, s=80, color='green')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.xlabel('Predicted Tip ($)', fontsize=12)
plt.ylabel('Residuals ($)', fontsize=12)
plt.title('Test Set: Residual Plot', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nWhat to look for in residual plots:")
print("  ✓ Points randomly scattered around zero = Good!")
print("  ✗ Pattern or curve = Model missing something")
print("  ✗ Funnel shape = Heteroscedasticity (variance not constant)")

# Distribution of Residuals (Training)
plt.figure(figsize=(10, 6))
sns.histplot(train_residuals, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Residuals ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Training Set: Distribution of Residuals', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Distribution of Residuals (Test)
plt.figure(figsize=(10, 6))
sns.histplot(test_residuals, kde=True, bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Residuals ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Test Set: Distribution of Residuals', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Q-Q Plot (Training)
plt.figure(figsize=(10, 6))
stats.probplot(train_residuals, dist="norm", plot=plt)
plt.title('Training Set: Q-Q Plot (Testing Normality)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Q-Q Plot (Test)
plt.figure(figsize=(10, 6))
stats.probplot(test_residuals, dist="norm", plot=plt)
plt.title('Test Set: Q-Q Plot (Testing Normality)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nHow to read Q-Q plots:")
print("  - Points should fall along the diagonal red line")
print("  - Deviations from the line indicate non-normality")
print("  - This tests if residuals follow a normal distribution")

# ============================================================================
# 13. TESTING ASSUMPTIONS
# ============================================================================

print("\n13. TESTING LINEAR REGRESSION ASSUMPTIONS")
 

print("\nLinear regression requires these assumptions:")
print("  1. Linear relationship between X and Y")
print("  2. Residuals are normally distributed")
print("  3. Homoscedasticity (constant variance of residuals)")
print("  4. Independence of observations")
print("  5. No multicollinearity among features")

# Test 1: Normality of Residuals
print("\n--- ASSUMPTION 1: NORMALITY OF RESIDUALS ---")
print("Test: Shapiro-Wilk Test")
print("  H₀ (null hypothesis): Residuals are normally distributed")
print("  H₁ (alternative): Residuals are NOT normally distributed")

shapiro_stat, shapiro_p = shapiro(test_residuals)
print(f"\nShapiro-Wilk Test Results:")
print(f"  Test Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4f}")

if shapiro_p > 0.05:
    print("  ✓ Result: Residuals are normally distributed (p > 0.05)")
    print("  Conclusion: Assumption satisfied!")
else:
    print("  ⚠ Result: Residuals may not be perfectly normal (p ≤ 0.05)")
    print("  Note: Linear regression is robust to small deviations from normality")

# Test 2: Homoscedasticity
print("\n--- ASSUMPTION 2: HOMOSCEDASTICITY (CONSTANT VARIANCE) ---")
from scipy.stats import spearmanr

abs_residuals = np.abs(test_residuals)
spearman_corr, spearman_p = spearmanr(y_test_pred, abs_residuals)

print("Test: Spearman Correlation between predictions and absolute residuals")
print(f"\nSpearman Correlation: {spearman_corr:.4f}")
print(f"P-value: {spearman_p:.4f}")

if spearman_p > 0.05:
    print("  ✓ Result: Constant variance assumption satisfied (p > 0.05)")
    print("  Conclusion: Homoscedasticity confirmed!")
else:
    print("  ⚠ Result: Evidence of heteroscedasticity (p ≤ 0.05)")
    print("  Recommendation: Consider transforming the target variable")

# Test 3: Multicollinearity (already done with VIF)
print("\n--- ASSUMPTION 3: NO MULTICOLLINEARITY ---")
print(f"We already tested this with VIF (see section 5)")
if vif_data['VIF'].max() < 5:
    print("  ✓ All VIF values < 5: No multicollinearity issues!")
else:
    print("  ⚠ Some VIF values > 5: Moderate multicollinearity present")

# ============================================================================
# 14. FEATURE IMPORTANCE SUMMARY
# ============================================================================

print("\n14. FEATURE IMPORTANCE RANKING")
 

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'Abs_Importance': np.abs(model.coef_)
}).sort_values('Abs_Importance', ascending=False)

print("\nFeature Ranking (by importance):")
for idx, row in feature_importance.iterrows():
    print(f"  {idx+1}. {row['Feature']}: {row['Coefficient']:.3f}")

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, y='Feature', x='Abs_Importance', palette='viridis')
plt.title('Feature Importance Ranking', fontsize=14, fontweight='bold')
plt.xlabel('Importance (Absolute Coefficient Value)', fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================================
# 15. FINAL CONCLUSIONS
# ============================================================================

print("\n15. CONCLUSIONS AND KEY TAKEAWAYS")
 

print(f"""

RESEARCH QUESTION:
  Can we predict restaurant tips based on bill amount, party size, and time?

MODEL PERFORMANCE:
  • R² Score: {test_r2:.3f} ({test_r2*100:.1f}% of variance explained)
  • Average Prediction Error: ${test_mae:.2f}
  • The model works reasonably well for a simple dataset

MOST IMPORTANT PREDICTORS:
  1. {feature_importance.iloc[0]['Feature']}: {feature_importance.iloc[0]['Coefficient']:.3f}
  2. {feature_importance.iloc[1]['Feature']}: {feature_importance.iloc[1]['Coefficient']:.3f}
  3. {feature_importance.iloc[2]['Feature']}: {feature_importance.iloc[2]['Coefficient']:.3f}

KEY FINDINGS:
  • Total bill amount is the strongest predictor of tip size
  • Larger parties tend to leave {'larger' if model.coef_[1] > 0 else 'smaller'} tips
  • {'Dinner' if model.coef_[2] > 0 else 'Lunch'} time {'increases' if model.coef_[2] > 0 else 'decreases'} expected tips

ASSUMPTION TESTING:
  • Normality: {'✓ Satisfied' if shapiro_p > 0.05 else '⚠ Slight deviation'}
  • Constant Variance: {'✓ Satisfied' if spearman_p > 0.05 else '⚠ Some heteroscedasticity'}
  • Multicollinearity: {'✓ No issues' if vif_data['VIF'].max() < 5 else '⚠ Moderate'}

PRACTICAL INTERPRETATION:
  For every $1 increase in the total bill, we expect the tip to increase by
  approximately ${model.coef_[0] * (df['tip'].std() / df['total_bill'].std()):.2f}, 
  holding other factors constant.

LIMITATIONS:
  • Model doesn't account for service quality or customer satisfaction
  • R² of {test_r2:.3f} means {(1-test_r2)*100:.1f}% of variation is unexplained
  • Limited to the patterns in this restaurant's data

RECOMMENDATIONS FOR IMPROVEMENT:
  • Collect additional features (e.g., day of week, server ID, meal type)
  • Increase sample size for better generalization
  • Consider non-linear relationships (polynomial features)
  • Explore interaction effects between features

STUDENT LEARNING OUTCOMES:
  ✓ Understood the multiple linear regression process
  ✓ Learned to check assumptions and interpret diagnostics
  ✓ Practiced making predictions and evaluating model performance
  ✓ Gained experience with real-world data analysis """)