"""
Regularised Regression (Ridge, Lasso, Elastic Net) on the Ames Housing Dataset
================================================================================

Business scenario
------------------
A residential valuation / investment analytics team wants a house price model
that remains reliable when the feature set is large and features are
correlated with one another (e.g. multiple size measures, multiple quality
ratings). This script compares an Ordinary Least Squares (OLS) baseline
against Ridge, Lasso and Elastic Net regression to determine whether
regularisation produces more stable, generalisable price predictions than
unpenalised OLS, and to reveal which property characteristics carry robust,
independent pricing signal versus which are redundant or noisy.

Dataset
-------
Ames Housing (De Cock, 2011) - 2,930 residential property sales in Ames,
Iowa, 2006-2010, with 82 recorded fields per property.
Original source: https://jse.amstat.org/v19n3/decock/AmesHousing.txt
CSV mirror used here: https://github.com/sabbeehh/Encoding (AmesHousing.csv,
an unmodified copy of the original De Cock file).

Author: Marc Grover
"""

# ============================================================================
# SECTION 0: IMPORTS & CONFIGURATION
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Fixed random state used everywhere a random process occurs (train/test
# split, CV fold assignment) so the analysis is fully reproducible.
RANDOM_STATE = 42

# Consistent chart styling across the portfolio.
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# # All numbered chart outputs are written to this folder.
# OUTPUT_DIR = "charts"
# import os
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# SECTION 1: LOAD DATA & INITIAL EDA
# ============================================================================

print("=" * 70)
print("SECTION 1: Load data & initial EDA")
print("=" * 70)

df = pd.read_csv("AmesHousing.csv")
print(f"Raw shape: {df.shape[0]} observations, {df.shape[1]} columns")

# --- Chart 01: SalePrice distribution - raw vs log-transformed ---
# SalePrice is right-skewed (a small number of very expensive properties
# stretch the upper tail), which violates the OLS assumption of normally
# distributed residuals and gives high-value properties disproportionate
# leverage over the fitted coefficients. A log transform compresses that
# tail and is the standard treatment for house-price targets.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df["SalePrice"], kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("SalePrice (raw)")
axes[0].set_xlabel("Sale Price ($)")

log_price = np.log(df["SalePrice"])
sns.histplot(log_price, kde=True, ax=axes[1], color="darkorange")
axes[1].set_title("log(SalePrice)")
axes[1].set_xlabel("log(Sale Price)")

plt.tight_layout()
plt.savefig("01_saleprice_distribution.png")
plt.close()

print(f"SalePrice skew (raw):     {df['SalePrice'].skew():.3f}")
print(f"SalePrice skew (log):     {log_price.skew():.3f}")
# The log transform is adopted for the reason demonstrated above: it visibly
# reduces skew, giving OLS-family models a target distribution much closer
# to the normal-residual assumption they rely on.

# --- Chart 02: Correlation heatmap of the strongest numeric predictors ---
# A full 38-numeric-column heatmap is unreadable, so this chart is
# restricted to the features with the strongest linear correlation to
# SalePrice, purely to orient the reader before the full-feature modelling
# in later sections (all numeric and encoded categorical features are used
# in the models themselves - this chart is diagnostic, not a feature list).
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["Order", "PID"])
top_corr_features = (
    numeric_df.corr()["SalePrice"].abs().sort_values(ascending=False).head(15).index
)
plt.figure(figsize=(9, 7))

sns.heatmap(
    numeric_df[top_corr_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8}
)
plt.title("Correlation matrix - top 15 numeric predictors of SalePrice")
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png")
plt.close()

print("\nTop 10 numeric correlations with SalePrice:")
print(numeric_df.corr()["SalePrice"].abs().sort_values(ascending=False).head(11)[1:])


# ============================================================================
# SECTION 2: FEATURE DECISION & PREPROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Feature decision & preprocessing")
print("=" * 70)

# --- 2a. Drop identifier / leakage-adjacent / redundant columns ---
# Order and PID are row identifiers with no predictive content, and their
# inclusion would let the model "memorise" rows rather than learn genuine
# relationships. Garage Yr Blt is dropped for two reasons: it duplicates
# information already carried by Year Built (properties are rarely built
# without a garage of the same vintage) and its 159 missing values (garages
# that do not exist) are already captured by the Garage Type/Finish/Qual
# categorical fields, so keeping it would add redundant, partially-missing
# signal to an already broad feature set - exactly the kind of duplication
# that inflates VIF and motivates regularisation in the first place.
df = df.drop(columns=["Order", "PID", "Garage Yr Blt"])

# --- 2b. Structural missingness: NaN means "feature absent", not "unknown" ---
# For a large group of categorical columns, a missing value has a specific,
# documented meaning in the Ames data dictionary: the property simply does
# not have that feature (no pool, no alley access, no basement, no garage,
# no fireplace, no fence, no masonry veneer). Encoding these as "None" -
# rather than imputing a mode or dropping the column - preserves this
# genuine signal (a property with no basement is meaningfully different
# from one with an unrecorded basement quality).
none_fill_cols = [
    "Pool QC", "Misc Feature", "Alley", "Fence", "Fireplace Qu",
    "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond",
    "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2",
    "Mas Vnr Type",
]
for col in none_fill_cols:
    df[col] = df[col].fillna("None")

# Numeric counterparts of the same "feature absent" logic: where there is no
# basement/garage/masonry veneer, the associated area or count is genuinely
# zero, not unknown.
zero_fill_cols = [
    "Mas Vnr Area", "BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
    "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath",
    "Garage Cars", "Garage Area",
]
for col in zero_fill_cols:
    df[col] = df[col].fillna(0)

# --- 2c. Genuine missing data (not structural) ---
# Lot Frontage (16.7% missing) has no "feature absent" interpretation -
# every property has some length of street frontage. Missingness here more
# plausibly reflects incomplete records. Imputing with the neighbourhood
# median (rather than the global median) respects the fact that frontage is
# strongly driven by local plot layout conventions, which vary considerably
# across Ames' 28 neighbourhoods.
#
# Two neighbourhoods (GrnHill, Landmrk) have no non-missing Lot Frontage
# values at all, so pandas computes a median over an all-NaN group for those
# rows - numerically a no-op (the result is NaN, caught by the global-median
# fallback immediately below) but NumPy raises a RuntimeWarning ("Mean of
# empty slice") while doing so. That warning is expected and harmless here,
# so it is suppressed for the duration of this one call rather than left to
# clutter the console output, in the same spirit as the VIF warning handling
# in Section 3.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    df["Lot Frontage"] = df.groupby("Neighborhood")["Lot Frontage"].transform(
        lambda s: s.fillna(s.median())
    )
# A small number of neighbourhoods (GrnHill, Landmrk) have no non-missing
# Lot Frontage values at all, leaving the group median itself undefined for
# those rows. These are filled with the dataset-wide median as a fallback,
# since a neighbourhood-level estimate is simply unavailable for them.
df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())

# Electrical has a single missing value with no structural interpretation;
# the modal category is a reasonable, low-impact fill for one row out of
# 2,930.
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

remaining_na = df.isnull().sum().sum()
print(f"Remaining missing values after imputation: {remaining_na}")
assert remaining_na == 0, "Unhandled missing values remain."

# --- 2d. Target transformation ---
df["log_SalePrice"] = np.log(df["SalePrice"])
y = df["log_SalePrice"]
X_raw = df.drop(columns=["SalePrice", "log_SalePrice"])

# --- 2e. One-hot encode categorical features ---
# drop_first=True avoids the dummy variable trap (perfect multicollinearity
# between a full set of one-hot columns and the intercept), which would
# otherwise make the design matrix singular for OLS and artificially inflate
# VIF for reasons that have nothing to do with the genuine multicollinearity
# this project is investigating.
categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

print(f"Feature count before encoding: {X_raw.shape[1]}")
print(f"Feature count after encoding:  {X_encoded.shape[1]}")
# Feature-set decision (per the agreed analytical plan): the full column set
# is retained rather than pre-selecting a curated subset. A broad,
# high-dimensional, correlated feature set of this kind is precisely the
# setting in which regularisation earns its place over OLS, and pre-pruning
# features by hand would undercut the comparison this project exists to
# make - Lasso and Elastic Net are used here to perform that selection
# systematically instead.

# --- 2f. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train set: {X_train.shape[0]} observations, {X_train.shape[1]} features")
print(f"Test set:  {X_test.shape[0]} observations")

# --- 2g. Feature scaling ---
# Ridge and Lasso penalise the magnitude of each coefficient directly, so on
# unscaled data a feature measured in the hundreds (e.g. Gr Liv Area, square
# feet) would need a proportionally tiny coefficient to fit the data and
# would therefore be penalised far less than a 0/1 dummy variable achieving
# the same practical effect on price. Standardising every feature to zero
# mean and unit variance ensures the penalty is applied evenhandedly across
# the feature set. The scaler is fit on the training data only and applied
# to both sets, preventing any information about the test set's distribution
# from leaking into preprocessing.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)


# ============================================================================
# SECTION 3: MULTICOLLINEARITY DIAGNOSTICS (VIF)
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Multicollinearity diagnostics")
print("=" * 70)

# VIF on the full one-hot-encoded, scaled feature set is computationally
# heavy and, for sparse dummy columns, frequently ill-conditioned. VIF is
# therefore calculated on the numeric (non-dummy) predictors only, which is
# where genuine, continuous multicollinearity (competing size and quality
# measures) is concentrated and interpretable - this mirrors the scope of
# the VIF analysis on the Multiple Linear Regression project, which assessed
# its three continuous/binary predictors rather than a full dummy matrix.
numeric_feature_cols = [
    c for c in X_train.columns
    if c in X_raw.select_dtypes(include=[np.number]).columns
]
vif_data = X_train_scaled[numeric_feature_cols].copy()

with warnings.catch_warnings():
    # statsmodels raises a SingularMatrixWarning per affected column below -
    # expected and explained immediately after, so suppressed here to keep
    # script output readable rather than 34 repeated warning blocks.
    warnings.simplefilter("ignore")
    vif_results = pd.DataFrame({
        "Feature": numeric_feature_cols,
        "VIF": [
            variance_inflation_factor(vif_data.values, i)
            for i in range(vif_data.shape[1])
        ],
    }).sort_values("VIF", ascending=False)

# A subset of these features are not merely correlated but exactly
# collinear: Total Bsmt SF = BsmtFin SF 1 + BsmtFin SF 2 + Bsmt Unf SF, and
# Gr Liv Area = 1st Flr SF + 2nd Flr SF + Low Qual Fin SF, hold as exact
# arithmetic identities across all 2,930 rows in this dataset (verified
# separately - the maximum absolute discrepancy is 0.0 in both cases).
# This makes the design matrix exactly rank-deficient for those columns,
# which is why their VIF values below are not merely "high" but numerically
# unbounded (statsmodels reports them near its floating-point ceiling) -
# this is a more extreme form of the multicollinearity problem than the
# Multiple Linear Regression project encountered, where the elevated VIFs
# (total_bill: 9.216, size: 9.271) reflected strong but non-exact
# correlation. OLS does not fail outright on near-singular data of this
# kind, but its coefficient estimates for the implicated columns become
# arbitrarily unstable; Ridge, Lasso and Elastic Net remain well-defined
# regardless, which is itself a direct, concrete demonstration of why
# regularisation is the appropriate tool here.
EXACT_COLLINEARITY_THRESHOLD = 1e6
exact_collinear = vif_results[vif_results["VIF"] > EXACT_COLLINEARITY_THRESHOLD]
elevated_only = vif_results[
    (vif_results["VIF"] <= EXACT_COLLINEARITY_THRESHOLD) & (vif_results["VIF"] > 10)
]
print(f"\nFeatures with exact structural collinearity (VIF numerically unbounded): "
      f"{len(exact_collinear)}")
print(exact_collinear.to_string(index=False))
print(f"\nFeatures with elevated but finite VIF > 10: {len(elevated_only)}")
print(elevated_only.to_string(index=False))
print(vif_results[(vif_results["VIF"] <= 10)].head(10).to_string(index=False))

# --- Chart 03: VIF bar chart (log scale), with the Multiple Linear
# Regression project's own values plotted as a reference line ---
# The MLR project (Restaurant Tips dataset) found VIF values of 9.216
# (total_bill) and 9.271 (size) - just under the conventional high-
# multicollinearity threshold of 10 - and treated that as a moderate,
# tolerable level of correlation for a 3-predictor OLS model. Plotting that
# threshold here against Ames' ~35 numeric predictors, on a log axis to
# accommodate the exactly-collinear features' near-infinite values alongside
# the merely-elevated ones, makes explicit how much further beyond
# "moderate" this dataset sits.
MLR_VIF_REFERENCE = 10.0  # High-multicollinearity threshold used on the MLR project
top_vif = vif_results.head(20).copy()
# Values are capped for display only (not for the printed/interpreted
# results above) so the log-scale chart renders sensibly rather than being
# dominated by floating-point-ceiling artefacts.
DISPLAY_CAP = 1e6
top_vif["VIF_display"] = top_vif["VIF"].clip(upper=DISPLAY_CAP)

plt.figure(figsize=(9, 7))
colors = ["firebrick" if v > MLR_VIF_REFERENCE else "steelblue" for v in top_vif["VIF"]]
sns.barplot(data=top_vif, x="VIF_display", y="Feature", hue="Feature", palette=colors, legend=False)
plt.axvline(MLR_VIF_REFERENCE, color="black", linestyle="--", linewidth=1.2,
            label=f"MLR project high-multicollinearity threshold (VIF={MLR_VIF_REFERENCE:.0f})")
plt.xscale("log")
plt.legend(loc="lower right")
plt.title("Variance Inflation Factor - top 20 numeric predictors (log scale)\n"
          "Bars at the right edge indicate exact structural collinearity, not merely high correlation")
plt.xlabel("VIF (log scale; capped at 1e6 for exactly-collinear features)")
plt.tight_layout()
plt.savefig("03_vif_multicollinearity.png")
plt.close()


# ============================================================================
# SECTION 4: BASELINE OLS
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Baseline OLS")
print("=" * 70)

# scikit-learn's LinearRegression is used for the OLS baseline rather than
# statsmodels. Unlike the Multiple Linear Regression project - where
# statsmodels' inferential output (p-values, confidence intervals) was the
# point - this project's comparison is predictive and structural (does
# penalisation improve generalisation and produce sparser, more stable
# coefficients?). Fitting all four models through the same scikit-learn
# interface, on the identical scaled feature matrix, keeps that comparison
# on equal footing and avoids conflating a modelling-library difference with
# a genuine methodological one.
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

ols_train_pred = ols.predict(X_train_scaled)
ols_test_pred = ols.predict(X_test_scaled)

def evaluate(y_true_log, y_pred_log, label):
    """Evaluate a model's predictions in both log and original SalePrice
    ($) space. Reporting RMSE/MAE back-transformed to dollars makes model
    error interpretable to a non-technical decision-maker, whereas the log
    scale alone would not."""
    r2 = r2_score(y_true_log, y_pred_log)
    y_true_dollars = np.exp(y_true_log)
    y_pred_dollars = np.exp(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))
    mae = mean_absolute_error(y_true_dollars, y_pred_dollars)
    print(f"{label:22s}  R²={r2:.4f}   RMSE=${rmse:,.0f}   MAE=${mae:,.0f}")
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

print("\nOLS baseline performance:")
ols_train_metrics = evaluate(y_train, ols_train_pred, "Train")
ols_test_metrics = evaluate(y_test, ols_test_pred, "Test")
print(f"\nTrain/test R² gap: {ols_train_metrics['R2'] - ols_test_metrics['R2']:.4f}")
# A large train/test R² gap is the empirical signature of overfitting under
# high dimensionality: the unpenalised model fits training-set noise that
# does not generalise. This gap is the baseline against which Ridge, Lasso
# and Elastic Net are compared in Section 6 - narrowing it is the central
# claim regularisation needs to substantiate.


# ============================================================================
# SECTION 5: RIDGE, LASSO, ELASTIC NET
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Ridge, Lasso, Elastic Net")
print("=" * 70)

# A shared set of candidate alpha values and a shared CV fold split are used
# across all three penalised models, so that differences in fitted alpha
# reflect genuine differences between the penalty types rather than
# differences in the search grid or fold assignment.
alphas = np.logspace(-3, 2, 100)
cv_folds = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

# --- Ridge ---
ridge = RidgeCV(alphas=alphas, cv=cv_folds)
ridge.fit(X_train_scaled, y_train)
print(f"\nRidge selected alpha:        {ridge.alpha_:.5f}")

# --- Lasso ---
# max_iter raised above the scikit-learn default because Lasso's coordinate
# descent solver can require more iterations to converge on a feature set
# this wide (~270 columns after encoding) without triggering a convergence
# warning.
lasso = LassoCV(alphas=alphas, cv=cv_folds, max_iter=20000, random_state=RANDOM_STATE)
lasso.fit(X_train_scaled, y_train)
print(f"Lasso selected alpha:        {lasso.alpha_:.5f}")

# --- Elastic Net ---
# l1_ratio is searched alongside alpha, spanning from Ridge-like (0.1) to
# Lasso-like (0.9) behaviour, so ElasticNetCV can locate whichever blend of
# the two penalties best suits this feature set rather than assuming a
# 50/50 split a priori.
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
elastic_net = ElasticNetCV(
    alphas=alphas, l1_ratio=l1_ratios, cv=cv_folds,
    max_iter=20000, random_state=RANDOM_STATE,
)
elastic_net.fit(X_train_scaled, y_train)
print(f"Elastic Net selected alpha:  {elastic_net.alpha_:.5f}")
print(f"Elastic Net selected l1_ratio: {elastic_net.l1_ratio_:.2f}")

models = {
    "OLS": ols,
    "Ridge": ridge,
    "Lasso": lasso,
    "Elastic Net": elastic_net,
}

# --- Sparsity check: how many coefficients did Lasso / Elastic Net zero out? ---
lasso_zero = (lasso.coef_ == 0).sum()
en_zero = (elastic_net.coef_ == 0).sum()
ridge_zero = (ridge.coef_ == 0).sum()
print(f"\nCoefficients driven to exactly zero:")
print(f"  Ridge:       {ridge_zero} / {len(ridge.coef_)}  (Ridge shrinks but never zeroes)")
print(f"  Lasso:       {lasso_zero} / {len(lasso.coef_)}")
print(f"  Elastic Net: {en_zero} / {len(elastic_net.coef_)}")

zeroed_by_lasso = X_train_scaled.columns[lasso.coef_ == 0].tolist()
print(f"\nExample features Lasso zeroed out (first 15 of {len(zeroed_by_lasso)}):")
print(zeroed_by_lasso[:15])

# --- Chart 04: Coefficient comparison across all four models ---
# Restricted to the 20 features with the largest absolute OLS coefficient,
# for legibility - the full ~270-feature comparison is available in the
# underlying coefficient table but is not readable as a single chart.
coef_df = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "OLS": ols.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_,
    "Elastic Net": elastic_net.coef_,
})
top_ols_features = coef_df.reindex(
    coef_df["OLS"].abs().sort_values(ascending=False).index
).head(20)
coef_melted = top_ols_features.melt(id_vars="Feature", var_name="Model", value_name="Coefficient")

plt.figure(figsize=(10, 8))
sns.barplot(data=coef_melted, y="Feature", x="Coefficient", hue="Model")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Coefficient comparison: OLS vs Ridge vs Lasso vs Elastic Net\n(20 largest OLS coefficients, standardised scale)")
plt.tight_layout()
plt.savefig("04_coefficient_comparison.png")
plt.close()

# --- Chart 05: Lasso regularisation path ---
# Shows how each coefficient's magnitude shrinks toward, and in most cases
# reaches, zero as alpha increases - the visual signature of Lasso's
# feature-selection property, tracked here for the 15 features with the
# largest coefficient magnitude at the smallest (least-penalised) alpha, so
# the chart reads as "which features survive longest under increasing
# penalty" rather than an unreadable 270-line plot.
path_alphas = np.logspace(-3, 1, 60)
path_coefs = []
for a in path_alphas:
    m = Lasso(alpha=a, max_iter=20000)
    m.fit(X_train_scaled, y_train)
    path_coefs.append(m.coef_)
path_coefs = np.array(path_coefs)

# Rank features by their coefficient magnitude at the smallest alpha (the
# least-regularised end of the path).
initial_magnitudes = np.abs(path_coefs[0])
top_path_idx = np.argsort(initial_magnitudes)[::-1][:15]

plt.figure(figsize=(10, 7))
for idx in top_path_idx:
    plt.plot(path_alphas, path_coefs[:, idx], label=X_train_scaled.columns[idx])
plt.xscale("log")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("alpha (log scale)")
plt.ylabel("Coefficient value")
plt.title("Lasso regularisation path - 15 largest-magnitude features")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("05_lasso_regularisation_path.png")
plt.close()


# ============================================================================
# SECTION 6: MODEL COMPARISON & RESULTS CONSOLIDATION
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Model comparison")
print("=" * 70)

results = []
for name, model in models.items():
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(test_pred)))
    test_mae = mean_absolute_error(np.exp(y_test), np.exp(test_pred))
    results.append({
        "Model": name, "Train R2": train_r2, "Test R2": test_r2,
        "Test RMSE ($)": test_rmse, "Test MAE ($)": test_mae,
        "R2 Gap": train_r2 - test_r2,
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

# --- Chart 06: Test RMSE by model ---
plt.figure(figsize=(7, 5))
sns.barplot(data=results_df, x="Model", y="Test RMSE ($)", hue="Model",
            palette="viridis", legend=False)
plt.title("Test set RMSE by model (back-transformed to $)")
plt.ylabel("RMSE ($)")
for i, v in enumerate(results_df["Test RMSE ($)"]):
    plt.text(i, v + 300, f"${v:,.0f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("06_test_rmse_comparison.png")
plt.close()

# --- Chart 07: Cross-validation stability (R² distribution per model) ---
# 10-fold CV on the training set, using the same fold split used for
# hyperparameter search above, directly addresses the "stability" half of
# the business question: a model whose CV fold scores are tightly clustered
# is one whose performance a decision-maker can rely on regardless of which
# properties happen to fall into a given data split, whereas wide dispersion
# signals a model sensitive to the particular sample it was fitted on.
cv_scores = {}
for name, model in models.items():
    # A fresh, unfitted estimator of the same type/hyperparameters is used
    # for cross_val_score, since passing an already-fitted CV estimator
    # would silently refit its own internal search inside every fold.
    if name == "OLS":
        est = LinearRegression()
    elif name == "Ridge":
        est = RidgeCV(alphas=[ridge.alpha_], cv=None)  # fixed at selected alpha
    elif name == "Lasso":
        from sklearn.linear_model import Lasso as LassoFixed
        est = LassoFixed(alpha=lasso.alpha_, max_iter=20000, random_state=RANDOM_STATE)
    else:
        from sklearn.linear_model import ElasticNet as ElasticNetFixed
        est = ElasticNetFixed(alpha=elastic_net.alpha_, l1_ratio=elastic_net.l1_ratio_,
                               max_iter=20000, random_state=RANDOM_STATE)
    scores = cross_val_score(est, X_train_scaled, y_train, cv=cv_folds, scoring="r2")
    cv_scores[name] = scores
    print(f"{name:12s} CV R²: mean={scores.mean():.4f}  std={scores.std():.4f}")

cv_df = pd.DataFrame(cv_scores).melt(var_name="Model", value_name="CV R2")
plt.figure(figsize=(8, 5.5))
sns.boxplot(data=cv_df, x="Model", y="CV R2", hue="Model", palette="Set2", legend=False)
sns.stripplot(data=cv_df, x="Model", y="CV R2", color="black", size=4, alpha=0.6)
plt.title("10-fold cross-validation R² distribution by model")
plt.tight_layout()
plt.savefig("07_cv_stability_boxplot.png")
plt.close()


# ============================================================================
# SECTION 7: INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Interpretation")
print("=" * 70)

# Features with a non-trivial coefficient (above a small magnitude
# threshold) in every one of the four models - robust, model-agnostic
# pricing signal.
THRESHOLD = 0.01
robust_mask = (
    (coef_df["OLS"].abs() > THRESHOLD)
    & (coef_df["Ridge"].abs() > THRESHOLD)
    & (coef_df["Lasso"].abs() > THRESHOLD)
    & (coef_df["Elastic Net"].abs() > THRESHOLD)
)
robust_features = coef_df.loc[robust_mask].reindex(
    coef_df.loc[robust_mask, "OLS"].abs().sort_values(ascending=False).index
)
print(f"\nFeatures with non-trivial signal across all four models ({robust_mask.sum()} total):")
print(robust_features.head(15)[["Feature", "OLS", "Lasso"]].to_string(index=False))

# Features OLS treats as non-trivial but Lasso zeroes out entirely -
# fragile, likely collinearity-driven signal.
fragile_mask = (coef_df["OLS"].abs() > THRESHOLD) & (coef_df["Lasso"] == 0)
fragile_features = coef_df.loc[fragile_mask].reindex(
    coef_df.loc[fragile_mask, "OLS"].abs().sort_values(ascending=False).index
)
print(f"\nFeatures OLS treats as meaningful but Lasso zeroes out entirely "
      f"({fragile_mask.sum()} total, first 10):")
print(fragile_features.head(10)[["Feature", "OLS", "Ridge"]].to_string(index=False))

print("\nScript complete. Charts created")
