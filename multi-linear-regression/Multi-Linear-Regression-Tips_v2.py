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
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from scipy.stats import shapiro
import warnings
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
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

print(f"Number of observations: {df.shape[0]}")
print(f"Number of variables: {df.shape[1]}")

print("\nExample data:")
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
print('There are no null values in this dataset, so no observations need to be removed.')

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
plt.savefig("mlr_corr_mat.png", dpi=150)
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
plt.xlabel('Total Bill', fontsize=12)
plt.ylabel('Tip', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_scat_bill.png", dpi=150)
plt.show()

# Party Size vs Tip
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_model, x='size', y='tip', alpha=0.6, s=80)
plt.title('Party Size vs Tip Amount', fontsize=14, fontweight='bold')
plt.xlabel('Party Size (number of people)', fontsize=12)
plt.ylabel('Tip', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_scat_size.png", dpi=150)
plt.show()

# Time of Day vs Tip (Box plot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='time', y='tip', palette='Set2')
plt.title('Tip Amount by Time of Day', fontsize=14, fontweight='bold')
plt.xlabel('Time of Day', fontsize=12)
plt.ylabel('Tip', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("mlr_boxplot_time.png", dpi=150)
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
plt.savefig("mlr_vif.png", dpi=150)
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
print("  - Different features have different ranges (e.g., bill: 3-50, size: 1-6)")
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
plt.savefig("mlr_model_coef.png", dpi=150)
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
print(f"  RMSE: {train_rmse:.2f}")
print(f"  MAE: {train_mae:.2f}")

print("\nTEST SET PERFORMANCE:")
print(f"  R² Score: {test_r2:.4f} ({test_r2*100:.1f}% of variance explained)")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  MAE: {test_mae:.2f}")

print("\nWhat do these metrics mean?")
print(f"  R² Score: Our model explains {test_r2*100:.1f}% of the variation in tips")
print(f"  RMSE: On average, predictions are off by about {test_rmse:.2f}")
print(f"  MAE: On average, absolute error is {test_mae:.2f}")

# ============================================================================
# 11. ACTUAL VS PREDICTED PLOTS
# ============================================================================

print("\n11. ACTUAL VS PREDICTED VALUES")
 

# Training set
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6, s=80)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=3, label='Perfect Prediction Line')
plt.xlabel('Actual Tip', fontsize=12)
plt.ylabel('Predicted Tip', fontsize=12)
plt.title(f'Training Set: Actual vs Predicted Tips (R² = {train_r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_scatter_pred_act_training.png", dpi=150)
plt.show()

# Test set
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, s=80, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=3, label='Perfect Prediction Line')
plt.xlabel('Actual Tip', fontsize=12)
plt.ylabel('Predicted Tip', fontsize=12)
plt.title(f'Test Set: Actual vs Predicted Tips (R² = {test_r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_scatter_pred_act.png", dpi=150)
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
print(f"  Mean: {train_residuals.mean():.4f} (should be close to 0)")
print(f"  Std Dev: {train_residuals.std():.2f}")

print(f"\nTest Residuals:")
print(f"  Mean: {test_residuals.mean():.4f} (should be close to 0)")
print(f"  Std Dev: {test_residuals.std():.2f}")

# Residuals vs Predicted Values (Training)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred, y=train_residuals, alpha=0.6, s=80)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.xlabel('Predicted Tip', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Training Set: Residual Plot', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_scatter_res_training.png", dpi=150)
plt.show()

# Residuals vs Predicted Values (Test)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_pred, y=test_residuals, alpha=0.6, s=80, color='green')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.xlabel('Predicted Tip', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Test Set: Residual Plot', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_scatter_res.png", dpi=150)
plt.show()

print("\nWhat to look for in residual plots:")
print(" Points randomly scattered around zero = Good!")
print(" Pattern or curve = Model missing something")
print(" Funnel shape = Heteroscedasticity (variance not constant)")

# Distribution of Residuals (Training)
plt.figure(figsize=(10, 6))
sns.histplot(train_residuals, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Training Set: Distribution of Residuals', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("mlr_hist_res_training.png", dpi=150)
plt.show()

# Distribution of Residuals (Test)
plt.figure(figsize=(10, 6))
sns.histplot(test_residuals, kde=True, bins=30, color='lightgreen', edgecolor='black')
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Test Set: Distribution of Residuals', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("mlr_hist_res.png", dpi=150)
plt.show()

# Q-Q Plot (Training)
plt.figure(figsize=(10, 6))
stats.probplot(train_residuals, dist="norm", plot=plt)
plt.title('Training Set: Q-Q Plot (Testing Normality)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_qq_training.png", dpi=150)
plt.show()

# Q-Q Plot (Test)
plt.figure(figsize=(10, 6))
stats.probplot(test_residuals, dist="norm", plot=plt)
plt.title('Test Set: Q-Q Plot (Testing Normality)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_qq.png", dpi=150)
plt.show()

print("\nHow to read Q-Q plots:")
print("- Points should fall along the diagonal red line")
print("- Deviations from the line indicate non-normality")
print("- This tests if residuals follow a normal distribution")

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
    print("  Result: Residuals are normally distributed (p > 0.05)")
    print("  Conclusion: Assumption satisfied!")
else:
    print("  Result: Residuals may not be perfectly normal (p ≤ 0.05)")
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
    print("  Result: Constant variance assumption satisfied (p > 0.05)")
    print("  Conclusion: Homoscedasticity confirmed!")
else:
    print("  Result: Evidence of heteroscedasticity (p ≤ 0.05)")
    print("  Recommendation: Consider transforming the target variable")

# Test 3: Multicollinearity (already done with VIF)
print("\n--- ASSUMPTION 3: NO MULTICOLLINEARITY ---")
print(f"We already tested this with VIF (see section 5)")
if vif_data['VIF'].max() < 5:
    print(" All VIF values < 5: No multicollinearity issues!")
else:
    print(" Some VIF values > 5: Moderate multicollinearity present")

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

for rank, (_, row) in enumerate(feature_importance.iterrows(), start=1):
    print(f"  {rank}. {row['Feature']}: {row['Coefficient']:.3f}")

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, y='Feature', x='Abs_Importance', palette='viridis')
plt.title('Feature Importance Ranking', fontsize=14, fontweight='bold')
plt.xlabel('Importance (Absolute Coefficient Value)', fontsize=12)
plt.tight_layout()
plt.savefig("mlr_feat_imp.png", dpi=150)
plt.show()

# ============================================================================
# 14A: STATSMODELS OLS - STATISTICAL SIGNIFICANCE OF COEFFICIENTS
# ============================================================================

print("\n14A. STATSMODELS OLS - COEFFICIENT SIGNIFICANCE TESTING")

# Fit OLS on UNSCALED features (statsmodels works in original units, giving directly interpretable coefficient magnitudes)
X_ols = sm.add_constant(X)   # adds intercept column
ols_model = sm.OLS(y, X_ols).fit()

print("\nOLS Regression Summary:")
print(ols_model.summary())

# Extract and display the key elements cleanly
print("\nCoefficient Significance Summary:")
print(f"{'Feature':<15} {'Coef':>8} {'Std Err':>9} {'t-stat':>8} {'p-value':>9} {'Sig':>5}")
print("-" * 60)

param_names = ['const', 'total_bill', 'size', 'time_dinner']

for name in param_names:
    coef   = ols_model.params[name]
    se     = ols_model.bse[name]
    tstat  = ols_model.tvalues[name]
    pval   = ols_model.pvalues[name]
    sig    = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'n.s.'
    display_name = 'Intercept' if name == 'const' else name
    print(f"{display_name:<15} {coef:>8.4f} {se:>9.4f} {tstat:>8.3f} {pval:>9.4f} {sig:>5}")

print("\nSignificance codes: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")

print(f"\nModel-level statistics:")
print(f"  R²:          {ols_model.rsquared:.4f}")
print(f"  Adjusted R²: {ols_model.rsquared_adj:.4f}")
print(f"  F-statistic: {ols_model.fvalue:.3f}  (p = {ols_model.f_pvalue:.4e})")
print(f"  AIC:         {ols_model.aic:.2f}")

print("\nInterpretation (in original units, per unit increase in each variable):")
tb_coef = ols_model.params['total_bill']
sz_coef = ols_model.params['size']
td_coef = ols_model.params['time_dinner']
print(f"  total_bill:  +{tb_coef:.4f} per $1 increase in bill (holding others constant)")
print(f"  size:        +{sz_coef:.4f} per additional person in party")
print(f"  time_dinner:  {td_coef:+.4f} for dinner vs lunch sitting")

# Coefficient confidence interval plot
ci = ols_model.conf_int()
ci.columns = ['lower', 'upper']
ci['coef'] = ols_model.params
ci_plot = ci.drop('const').copy()

fig, ax = plt.subplots(figsize=(10, 5))
features_plot = ci_plot.index.tolist()
y_pos = range(len(features_plot))

for i, feat in enumerate(features_plot):
    low = ci_plot.loc[feat, 'lower']
    high = ci_plot.loc[feat, 'upper']
    coef = ci_plot.loc[feat, 'coef']
    color = 'steelblue' if coef > 0 else 'tomato'
    ax.plot([low, high], [i, i], color=color, linewidth=2.5, solid_capstyle='round')
    ax.plot(coef, i, 'o', color=color, markersize=8, zorder=5)

ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.6, label='No effect (zero)')
ax.set_yticks(list(y_pos))
ax.set_yticklabels(features_plot, fontsize=12)
ax.set_xlabel('Coefficient value (original units) with 95% Confidence Interval', fontsize=12)
ax.set_title('OLS Coefficient Estimates and 95% Confidence Intervals', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("mlr_coef_ci.png", dpi=150)
plt.show()

print("\nHow to read this chart:")
print("  - Each dot is the estimated coefficient; bars show the 95% confidence interval")
print("  - If the interval crosses zero, the predictor is NOT statistically significant")
print("  - Wider intervals indicate more uncertainty in the estimate")

# ============================================================================
# 14B: CROSS-VALIDATION
# ============================================================================

# PRIORITY: Significant. With only 244 observations, a single 80/20 split produces unreliable performance estimates (test set = 48 observations).
# K-fold cross-validation gives a more robust, less split-dependent estimate.

print("\n14B. CROSS-VALIDATION")

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Run CV on the unscaled pipeline (using Pipeline to apply scaling within each fold)
from sklearn.pipeline import Pipeline

cv_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

cv_r2   = cross_val_score(cv_pipeline, X, y, cv=kf, scoring='r2')
cv_mae  = cross_val_score(cv_pipeline, X, y, cv=kf, scoring='neg_mean_absolute_error')
cv_rmse = cross_val_score(cv_pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')

print(f"\n10-Fold Cross-Validation Results:")
print(f"{'Metric':<10} {'Mean':>8} {'Std Dev':>9} {'Min':>8} {'Max':>8}")
print("-" * 47)
print(f"{'R²':<10} {cv_r2.mean():>8.4f} {cv_r2.std():>9.4f} {cv_r2.min():>8.4f} {cv_r2.max():>8.4f}")
print(f"{'MAE':<10} {(-cv_mae).mean():>8.4f} {(-cv_mae).std():>9.4f} {(-cv_mae).min():>8.4f} {(-cv_mae).max():>8.4f}")
print(f"{'RMSE':<10} {(-cv_rmse).mean():>8.4f} {(-cv_rmse).std():>9.4f} {(-cv_rmse).min():>8.4f} {(-cv_rmse).max():>8.4f}")

print(f"\nComparison with single train/test split:")
print(f"  Single split R²:  {test_r2:.4f}")
print(f"  10-fold CV R²:    {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# Visualise cross-validation R² distribution
plt.figure(figsize=(10, 5))
fold_labels = [f"Fold {i+1}" for i in range(len(cv_r2))]
colors = ['steelblue' if r > cv_r2.mean() else 'tomato' for r in cv_r2]
bars = plt.bar(fold_labels, cv_r2, color=colors, edgecolor='black', linewidth=0.7, alpha=0.85)
plt.axhline(y=cv_r2.mean(), color='navy', linestyle='--', linewidth=2,
            label=f'Mean R² = {cv_r2.mean():.4f}')
plt.axhline(y=test_r2, color='green', linestyle=':', linewidth=2,
            label=f'Single split R² = {test_r2:.4f}')
plt.xlabel('Fold', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('10-Fold Cross-Validation: R² Score per Fold', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.ylim(0, max(cv_r2.max(), test_r2) * 1.15)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("mlr_cv_r2.png", dpi=150)
plt.show()

# ============================================================================
# 14C: OUTLIER DETECTION - COOK'S DISTANCE
# ============================================================================

print("\n14C. OUTLIER DETECTION - COOK'S DISTANCE")

influence = OLSInfluence(ols_model)
cooks_d, _ = influence.cooks_distance

n_obs = len(y)
cooks_threshold = 4 / n_obs     # conventional threshold
influential_idx = np.where(cooks_d > cooks_threshold)[0]

print(f"\nCook's Distance Analysis:")
print(f"  Threshold (4/n = 4/{n_obs}): {cooks_threshold:.4f}")
print(f"  Observations above threshold: {len(influential_idx)}")

if len(influential_idx) > 0:
    print(f"\n  Influential observations:")
    for i in influential_idx:
        print(f"    Index {i}: Cook's D = {cooks_d[i]:.4f}  | "
              f"total_bill = {X.iloc[i]['total_bill']:.2f}, "
              f"tip = {y.iloc[i]:.2f}, "
              f"tip% = {y.iloc[i]/X.iloc[i]['total_bill']*100:.1f}%")

# Cook's Distance plot
plt.figure(figsize=(12, 5))
plt.stem(range(n_obs), cooks_d, markerfmt='o', linefmt='grey', basefmt=' ')
plt.axhline(y=cooks_threshold, color='red', linestyle='--', linewidth=2,
            label=f"Threshold (4/n = {cooks_threshold:.3f})")

for i in influential_idx:
    plt.annotate(f"Obs {i}", xy=(i, cooks_d[i]), xytext=(i+3, cooks_d[i]+0.005),
                 fontsize=9, color='red')

plt.xlabel('Observation Index', fontsize=12)
plt.ylabel("Cook's Distance", fontsize=12)
plt.title("Cook's Distance — Influential Observation Detection", fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("mlr_cooks_distance.png", dpi=150)
plt.show()

print("\nWhat Cook's Distance measures:")
print("  - Combines leverage (unusual X values) and residual (poor prediction)")
print("  - High Cook's D = removing this observation would substantially change the model")
print(f"  - {len(influential_idx)} observation(s) exceed the 4/n threshold")

# ============================================================================
# 14D: TIP PERCENTAGE ANALYSIS
# ============================================================================

print("\n14D. TIP PERCENTAGE ANALYSIS")

df_model['tip_pct'] = (df_model['tip'] / df_model['total_bill']) * 100

print(f"\nTip Percentage Descriptive Statistics:")
print(df_model['tip_pct'].describe().round(3))
print(f"\nMedian tip percentage: {df_model['tip_pct'].median():.2f}%")

# Distribution of tip percentage
plt.figure(figsize=(10, 6))
sns.histplot(df_model['tip_pct'], kde=True, bins=30, color='steelblue', edgecolor='black')
plt.axvline(df_model['tip_pct'].mean(), color='red', linestyle='--', linewidth=2,
            label=f"Mean: {df_model['tip_pct'].mean():.1f}%")
plt.axvline(df_model['tip_pct'].median(), color='green', linestyle=':', linewidth=2,
            label=f"Median: {df_model['tip_pct'].median():.1f}%")
plt.xlabel('Tip Percentage (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Tip as a Percentage of Total Bill', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("mlr_tip_pct_hist.png", dpi=150)
plt.show()

# Tip percentage vs total bill — this reveals the heteroscedasticity visually
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_model, x='total_bill', y='tip_pct', alpha=0.6, s=80)
plt.axhline(y=df_model['tip_pct'].mean(), color='red', linestyle='--', linewidth=1.5,
            label=f"Mean tip %: {df_model['tip_pct'].mean():.1f}%")
plt.xlabel('Total Bill ($)', fontsize=12)
plt.ylabel('Tip Percentage (%)', fontsize=12)
plt.title('Tip Percentage vs Total Bill\n(Wide spread at low bills explains heteroscedasticity)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_tip_pct_vs_bill.png", dpi=150)
plt.show()

print("\nKey insight:")
print(f"  Tip % ranges from {df_model['tip_pct'].min():.1f}% to {df_model['tip_pct'].max():.1f}%")
print("  The wide spread of tip % at low bill values is the visual signature")
print("  of heteroscedasticity — variance is NOT constant across bill values.")
print("  This motivates the square-root transformation of the target variable.")

# ============================================================================
# 14E: SQUARE ROOT TRANSFORMATION — COMPARISON MODEL
# ============================================================================

print("\n14E. SQUARE ROOT TRANSFORMATION — COMPARISON MODEL")

print("\nMotivation: The Spearman test confirmed heteroscedasticity (non-constant variance).")
print("A square-root transformation of the target variable can stabilise variance.")
print("We now refit the model using sqrt(tip) as the target and compare performance.\n")

y_sqrt = np.sqrt(y)

X_train_sq, X_test_sq, y_train_sq, y_test_sq = train_test_split(
    X, y_sqrt, test_size=0.2, random_state=42
)

scaler_sq = StandardScaler()
X_train_sq_sc = scaler_sq.fit_transform(X_train_sq)
X_test_sq_sc  = scaler_sq.transform(X_test_sq)

model_sqrt = LinearRegression()
model_sqrt.fit(X_train_sq_sc, y_train_sq)

y_train_sq_pred = model_sqrt.predict(X_train_sq_sc)
y_test_sq_pred  = model_sqrt.predict(X_test_sq_sc)

# Back-transform predictions to original tip scale for comparable MAE/RMSE
y_test_sq_pred_orig  = y_test_sq_pred ** 2
y_train_sq_pred_orig = y_train_sq_pred ** 2

train_sq_r2   = r2_score(y_train_sq, y_train_sq_pred)
test_sq_r2    = r2_score(y_test_sq,  y_test_sq_pred)
test_sq_mae   = mean_absolute_error(y_test.values, y_test_sq_pred_orig)
test_sq_rmse  = np.sqrt(mean_squared_error(y_test.values, y_test_sq_pred_orig))

# Re-run Spearman test on sqrt model residuals
sq_test_residuals = y_test_sq - y_test_sq_pred
sq_spearman_corr, sq_spearman_p = stats.spearmanr(y_test_sq_pred, np.abs(sq_test_residuals))

print("Model comparison (test set):")
print(f"{'Metric':<35} {'Original':>10} {'Sqrt Transform':>15}")
print("-" * 62)
print(f"{'R² (on respective target scale)':<35} {test_r2:>10.4f} {test_sq_r2:>15.4f}")
print(f"{'MAE (original tip scale)':<35} {test_mae:>10.4f} {test_sq_mae:>15.4f}")
print(f"{'RMSE (original tip scale)':<35} {test_rmse:>10.4f} {test_sq_rmse:>15.4f}")
print(f"{'Spearman corr (heteroscedasticity)':<35} {spearman_corr:>10.4f} {sq_spearman_corr:>15.4f}")
print(f"{'Spearman p-value':<35} {spearman_p:>10.4f} {sq_spearman_p:>15.4f}")

if sq_spearman_p > 0.05:
    print("\n  Result: Square-root transformation RESOLVED heteroscedasticity (p > 0.05)")
else:
    print(f"\n  Result: Heteroscedasticity partially reduced but still present (p = {sq_spearman_p:.4f})")

# Side-by-side residual scatter comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_test_pred, test_residuals, alpha=0.6, s=70, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Tip', fontsize=11)
axes[0].set_ylabel('Residuals', fontsize=11)
axes[0].set_title(f'Original model\nSpearman p = {spearman_p:.4f}', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_sq_pred, sq_test_residuals, alpha=0.6, s=70, color='darkorange')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted √Tip', fontsize=11)
axes[1].set_ylabel('Residuals', fontsize=11)
axes[1].set_title(f'Square-root model\nSpearman p = {sq_spearman_p:.4f}', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Residual Plots: Original vs Square-Root Transformation',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("mlr_sqrt_residual_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 14F: IMPROVED PARTY SIZE SCATTER (JITTER)
# ============================================================================

print("\n14F. PARTY SIZE SCATTER WITH JITTER")

np.random.seed(42)
jitter_amount = 0.12
jittered_size = df_model['size'] + np.random.uniform(-jitter_amount, jitter_amount,
                                                       size=len(df_model))

plt.figure(figsize=(10, 6))
plt.scatter(jittered_size, df_model['tip'], alpha=0.5, s=70, color='steelblue', edgecolors='none')
plt.xticks(ticks=sorted(df_model['size'].unique()),
           labels=sorted(df_model['size'].unique()))
plt.xlabel('Party Size (number of people)', fontsize=12)
plt.ylabel('Tip', fontsize=12)
plt.title('Party Size vs Tip Amount (with jitter to reduce overplotting)',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlr_scat_size_jitter.png", dpi=150)
plt.show()

print("\nNote: Small horizontal jitter applied to reveal the true density of observations at each party size value. The integer axis ticks are preserved.")

# ============================================================================
# 15. FINAL CONCLUSIONS
# ============================================================================

print("\n15. CONCLUSIONS AND KEY TAKEAWAYS")
 

print(f"""

RESEARCH QUESTION:
  Can we predict restaurant tips based on bill amount, party size, and time?

MODEL PERFORMANCE:
  • R² Score: {test_r2:.3f} ({test_r2*100:.1f}% of variance explained)
  • Average Prediction Error: {test_mae:.2f}
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
  • Normality: {'Satisfied' if shapiro_p > 0.05 else 'Slight deviation'}
  • Constant Variance: {'Satisfied' if spearman_p > 0.05 else 'Some heteroscedasticity'}
  • Multicollinearity: {'No issues' if vif_data['VIF'].max() < 5 else 'Moderate'}

PRACTICAL INTERPRETATION:
  For every $ increase in the total bill, we expect the tip to increase by
  approximately {model.coef_[0] / df['total_bill'].std():.2f}, 
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
""")