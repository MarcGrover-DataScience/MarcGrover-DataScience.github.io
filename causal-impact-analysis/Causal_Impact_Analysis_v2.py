# Causal Impact Analysis — Rossmann Store Sales
# Portfolio demonstration of Causal Impact Analysis applied to real-world retail sales data to measure the causal effect
# of a promotional campaign on store revenue.
# Dataset : train.csv + store.csv (Kaggle)   https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales

# 1. LIBRARY IMPORTS & CONFIGURATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# from datetime import datetime
from scipy import stats
# from sklearn.preprocessing import StandardScaler
# from causalimpact import CausalImpactAnalysis
from causalimpact import fit_causalimpact
from causalimpact import summary
import time
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import logging
logging.getLogger("pytensor").setLevel(logging.ERROR)

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

# Seaborn global theme
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOT_FIGSIZE = (12, 6)
OUTPUT_DIR   = "charts_causal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths — update if your CSVs live elsewhere
TRAIN_PATH = "cia_train.csv"
STORE_PATH = "cia_store.csv"

# Analysis parameters
# The Rossmann dataset covers 2013-01-01 to 2015-07-31.
# We treat a sustained promotional period in Store 30 as our intervention.
# Promo2 (continuous loyalty promotion) for Store 30 begins 2014-02-03 — we use this as the intervention.
# StoreType a, Assortment a
TREATED_STORE     = 30
INTERVENTION_DATE = "2014-03-03"
PRE_PERIOD_START  = "2013-01-01"
PRE_PERIOD_END    = "2014-03-02"
POST_PERIOD_START = "2014-03-03"
POST_PERIOD_END   = "2014-12-31"
N_CONTROL_STORES   = 5

# 2. DATA LOADING & INITIAL EXPLORATION


print("SECTION 2 — DATA LOADING & INITIAL EXPLORATION")

train = pd.read_csv(TRAIN_PATH, low_memory=False)
store = pd.read_csv(STORE_PATH, low_memory=False)

print(f"\ntrain.csv shape : {train.shape[0]:,} rows × {train.shape[1]} columns")
print(f"store.csv shape : {store.shape[0]:,} rows × {store.shape[1]} columns")

print("\ntrain.csv columns  :", train.columns.tolist())
print("store.csv columns  :", store.columns.tolist())

print("\ntrain.csv — first 5 rows:\n", train.head())
print("\nstore.csv — first 5 rows:\n", store.head())

print("\ntrain.csv — data types:\n", train.dtypes)
print("\nstore.csv — data types:\n", store.dtypes)

print("\ntrain.csv — missing values:")
print(train.isnull().sum())
print("\nstore.csv — missing values:")
print(store.isnull().sum())

print("\ntrain.csv — descriptive statistics:\n", train.describe())


# 3. DATA VALIDATION & PRE-PROCESSING

print("\nSECTION 3 — DATA VALIDATION & PRE-PROCESSING")

df = train.copy()

# Step 1: Parse and validate dates
df["Date"] = pd.to_datetime(df["Date"])
print(f"\n[Step 1] Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Step 2: Remove closed store days
# Days where Open == 0 contribute zero sales by definition, distorting the time series baseline and post-period mean.
n_closed = (df["Open"] == 0).sum()
df = df[df["Open"] == 1].copy()
print(f"[Step 2] Closed store days removed        : {n_closed:,} rows")

# Step 3: Remove rows with zero or negative sales
# Zero sales on open days indicate data quality issues.
n_zero_sales = (df["Sales"] <= 0).sum()
df = df[df["Sales"] > 0].copy()
print(f"[Step 3] Zero/negative sales removed      : {n_zero_sales:,} rows")

# Step 4: Remove rows with zero customers
# Open stores with zero customers are likely erroneous records.
n_zero_cust = (df["Customers"] == 0).sum()
df = df[df["Customers"] > 0].copy()
print(f"[Step 4] Zero customer rows removed       : {n_zero_cust:,} rows")

# Step 5: Merge store metadata
store_clean = store.copy()

# Fill missing CompetitionDistance with median (reasonable imputation)
store_clean["CompetitionDistance"] = store_clean["CompetitionDistance"].fillna(
    store_clean["CompetitionDistance"].median()
)

# Fill Promo2 participation fields
store_clean["Promo2SinceWeek"]  = store_clean["Promo2SinceWeek"].fillna(0)
store_clean["Promo2SinceYear"]  = store_clean["Promo2SinceYear"].fillna(0)
store_clean["PromoInterval"]    = store_clean["PromoInterval"].fillna("None")

df = df.merge(store_clean, on="Store", how="left")
print(f"[Step 5] Store metadata merged. Shape: {df.shape}")

# Step 6: Scope analysis to 2013-2014
df = df[(df["Date"] >= PRE_PERIOD_START) & (df["Date"] <= POST_PERIOD_END)].copy()
print(f"[Step 6] Scoped to 2013-2014. Rows remaining: {df.shape[0]:,}")

# Step 7: Feature engineering
df["Month"]     = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["WeekOfYear"]= df["Date"].dt.isocalendar().week.astype(int)
df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

print(f"\nCleaned dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Unique stores        : {df['Store'].nunique():,}")
print(f"Date range           : {df['Date'].min().date()} to {df['Date'].max().date()}")


# 4. EXPLORATORY DATA ANALYSIS

print("\nSECTION 4 — EXPLORATORY DATA ANALYSIS")

# Plot 1: Sales distribution across all stores
fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
sns.histplot(df["Sales"], bins=60, kde=True, color="Seagreen", ax=ax)
ax.set_title("Distribution of Daily Sales Across All Stores (2013–2014)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Daily Sales (€)")
ax.set_ylabel("Frequency")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
plt.tight_layout()
plt.savefig("01_sales_distribution.png", dpi=150)
plt.show()

# Plot 2: Average daily sales by store type
store_type_sales = (df.groupby("StoreType")["Sales"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Sales": "Avg Daily Sales"}))

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=store_type_sales, x="StoreType", y="Avg Daily Sales",
            palette="Greens_r", ax=ax, errorbar=None)
ax.set_title("Average Daily Sales by Store Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Store Type")
ax.set_ylabel("Average Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f"€{x:,.0f}", padding=5, fontsize=10)
plt.tight_layout()
plt.savefig("02_sales_by_store_type.png", dpi=150)
plt.show()

# Plot 3: Monthly average sales — treated store vs all stores - across both years
treated_monthly = (df[df["Store"] == TREATED_STORE]
                   .groupby("Month")["Sales"].mean()
                   .reset_index()
                   .rename(columns={"Sales": "Treated Store"}))

all_monthly = (df.groupby("Month")["Sales"]
               .mean()
               .reset_index()
               .rename(columns={"Sales": "All Stores Avg"}))

monthly_comp = treated_monthly.merge(all_monthly, on="Month")
monthly_comp_melt = monthly_comp.melt(id_vars="Month",
                                       value_vars=["Treated Store", "All Stores Avg"],
                                       var_name="Series", value_name="Avg Sales")

fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
sns.lineplot(data=monthly_comp_melt, x="Month", y="Avg Sales",
             hue="Series", marker="o", ax=ax, palette={"Treated Store": "darkgreen", "All Stores Avg": "limegreen"})
ax.set_title(f"Monthly Average Sales — Store {TREATED_STORE} vs All Stores",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Average Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.set_xticks(range(1, 13))
plt.tight_layout()
plt.savefig("03_monthly_sales_comparison.png", dpi=150)
plt.show()

# Plot 4: Effect of promotion on daily sales (all stores)
promo_sales = (df.groupby("Promo")["Sales"]
               .mean()
               .reset_index()
               .rename(columns={"Sales": "Avg Daily Sales"}))
promo_sales["Promo"] = promo_sales["Promo"].map({0: "No Promotion", 1: "Promotion Active"})

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=promo_sales, x="Promo", y="Avg Daily Sales",
            palette="Greens", ax=ax, errorbar=None)
ax.set_title("Average Daily Sales — Promotion vs No Promotion (All Stores)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Average Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f"€{x:,.0f}", padding=5, fontsize=10)
plt.tight_layout()
plt.savefig("4_promo_vs_no_promo.png", dpi=150)
plt.show()

# Plot 5: Treated store full sales time series
treated_ts = (df[df["Store"] == TREATED_STORE]
              .groupby("Date")["Sales"]
              .sum()
              .reset_index())

fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
ax.plot(treated_ts["Date"], treated_ts["Sales"], color="mediumseagreen",
        linewidth=0.9, alpha=0.85)
ax.axvline(pd.to_datetime(INTERVENTION_DATE), color="crimson",
           linestyle="--", linewidth=1.8, label=f"Intervention: {INTERVENTION_DATE}")
ax.set_title(f"Store {TREATED_STORE} — Daily Sales Time Series with Intervention Marker",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("05_treated_store_time_series.png", dpi=150)
plt.show()

# Plot 6: Sales by day of week — treated store
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
dow_sales = (df[df["Store"] == TREATED_STORE]
             .groupby("DayOfWeek")["Sales"]
             .mean()
             .reset_index())
dow_sales["Day"] = dow_sales["DayOfWeek"].map(dict(enumerate(dow_labels)))

fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
sns.barplot(data=dow_sales, x="Day", y="Sales", palette="Greens_r",
            ax=ax, order=dow_labels, errorbar=None)
ax.set_title(f"Store {TREATED_STORE} — Average Sales by Day of Week",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Day of Week")
ax.set_ylabel("Average Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
plt.tight_layout()
plt.savefig("06_sales_by_day_of_week.png", dpi=150)
plt.show()


# 5. TREATED & CONTROL STORE SELECTION

print("\nSECTION 5 — TREATED & CONTROL STORE SELECTION")

# Retrieve treated store metadata
treated_meta = store_clean[store_clean["Store"] == TREATED_STORE].iloc[0]
treated_store_type   = treated_meta["StoreType"]
treated_assortment   = treated_meta["Assortment"]

print(f"\nTreated Store : {TREATED_STORE}")
print(f"  Store Type  : {treated_store_type}")
print(f"  Assortment  : {treated_assortment}")
print(f"  Promo2      : {int(treated_meta['Promo2'])}")

# Candidate control stores
# Control stores must:
#   1. Be the same store type as the treated store
#   2. Have the same assortment level
#   3. NOT be the treated store itself
#   4. NOT have Promo2 active (to avoid contamination)
#   5. Have complete daily data across the full analysis window
candidate_stores = store_clean[
    (store_clean["StoreType"]  == treated_store_type) &
    (store_clean["Assortment"] == treated_assortment) &
    (store_clean["Store"]      != TREATED_STORE) &
    (store_clean["Promo2"]     == 0)
]["Store"].tolist()

print(f"\nCandidate control stores (same type/assortment, no Promo2): {len(candidate_stores)}")

# Build daily sales pivot for pre-period
pre_df = df[
    (df["Date"] >= PRE_PERIOD_START) &
    (df["Date"] <= PRE_PERIOD_END)
].copy()

pivot = (pre_df.groupby(["Date", "Store"])["Sales"]
         .sum()
         .unstack("Store")
         .fillna(0))

print('\nDaily sales by store in pre-period:')
print(pivot)

# Retain only candidates with complete pre-period data - taken to be sales on both 2013-01-12 and 2014-03-01

complete_candidates = [
    s for s in candidate_stores
    if s in pivot.columns
    and pd.Timestamp("2013-01-12") in pivot.index
    and pivot.loc[pd.Timestamp("2013-01-12"), s] > 0
    and pd.Timestamp("2014-03-01") in pivot.index
    and pivot.loc[pd.Timestamp("2014-03-01"), s] > 0]

print('\nSet of candidate stores without promo2:')
print(complete_candidates)

print(f"\nCandidates with complete pre-period data: {len(complete_candidates)}")

# Select control stores by Pearson correlation
# High pre-period correlation with the treated store ensures the control series would have tracked the treated store
# closely in the absence of the intervention — a key validity requirement.
treated_pre = pivot[TREATED_STORE]

correlations = {}
for s in complete_candidates:
    corr, _ = stats.pearsonr(treated_pre, pivot[s])
    correlations[s] = corr

corr_series = pd.Series(correlations).sort_values(ascending=False)
control_stores = corr_series.head(N_CONTROL_STORES).index.tolist()

print(f"\nTop {N_CONTROL_STORES} control stores selected by pre-period correlation:")
for s in control_stores:
    print(f"  Store {s:>4d}  |  Correlation = {corr_series[s]:.4f}")


# 6. PRE-INTERVENTION CORRELATION VALIDATION

print("\nSECTION 6 — PRE-INTERVENTION CORRELATION VALIDATION")

# Plot 7: Correlation heatmap — treated vs control (pre)
pre_stores = [TREATED_STORE] + control_stores
pre_corr_df = pivot[pre_stores].copy()
pre_corr_df.columns = [f"Store {s}" for s in pre_stores]
corr_matrix = pre_corr_df.corr()

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            cbar_kws={"label": "Pearson Correlation"})
ax.set_title("Pre-Intervention Correlation — Treated vs Control Stores",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("07_pre_intervention_correlation.png", dpi=150)
plt.show()

# Plot 8: Pre-period time series overlay
fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
ax.plot(pivot.index, pivot[TREATED_STORE], color="seagreen",
        linewidth=0.7, label=f"Store {TREATED_STORE} (Treated)", zorder=5)

for s in control_stores:
    ax.plot(pivot.index, pivot[s], linewidth=0.7, alpha=0.5,
            label=f"Store {s} (Control)")

ax.set_title("Pre-Intervention Period — Treated vs Control Store Sales",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(fontsize=9)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("08_pre_period_overlay.png", dpi=150)
plt.show()

# Statistical parallel trends check
# We compare the mean weekly growth rate of the treated store vs the average of control stores in the pre-period.
# Similar growth rates support the parallel trends assumption.
weekly_pivot = pre_corr_df.resample("W").mean()
treated_growth = weekly_pivot[f"Store {TREATED_STORE}"].pct_change().mean()
control_growth = weekly_pivot[[f"Store {s}" for s in control_stores]].pct_change().mean().mean()

control_growth2 = weekly_pivot[[f"Store {s}" for s in control_stores]].pct_change().mean()


print(f"\nParallel Trends Check (mean weekly growth rate, pre-period):")
print(f"  Treated store  : {treated_growth:.4f} ({treated_growth*100:.2f}%)")
print(f"  Control stores : {control_growth:.4f} ({control_growth*100:.2f}%)")
print(f"  Difference     : {abs(treated_growth - control_growth):.4f}")
if abs(treated_growth - control_growth) < 0.02:
    print(" Parallel trends assumption supported (difference < 2%)")
else:
    print(" Parallel trends assumption may be weak — interpret with caution")

print("\nMean weekly growth rate per control store (pre-period):")
print(control_growth2.apply(lambda x: f"{x:.4f}"))


# 7. CAUSAL IMPACT MODELLING

print("\nSECTION 7 — CAUSAL IMPACT MODELLING")

# Build full analysis time series
full_df = df[
    (df["Date"] >= PRE_PERIOD_START) &
    (df["Date"] <= POST_PERIOD_END)
].copy()

full_pivot = (full_df.groupby(["Date", "Store"])["Sales"]
              .sum()
              .unstack("Store")
              .fillna(0))
# print(full_pivot)

# Construct the input dataframe: treated store + control stores
analysis_stores = [TREATED_STORE] + control_stores
ci_data = full_pivot[analysis_stores].copy()
ci_data.columns = [f"Store_{s}" for s in analysis_stores]
ci_data.index = pd.to_datetime(ci_data.index)
ci_data = ci_data.sort_index()
ci_data = ci_data.dropna()
# print(ci_data)

print(f"\nCausal Impact input data shape: {ci_data.shape}")
print(f"Columns: {ci_data.columns.tolist()}")

# Define pre and post periods
pre_period  = (PRE_PERIOD_START,  PRE_PERIOD_END)
post_period = (POST_PERIOD_START, POST_PERIOD_END)

print(pre_period)
print(post_period)

print(f"\nPre-period  : {pre_period[0]} to {pre_period[1]}")
print(f"Post-period : {post_period[0]} to {post_period[1]}")

# Fit Causal Impact model
# CausalImpact fits a Bayesian structural time series model.
# It uses the control store series as covariates to construct a counterfactual
# This is what the treated store's sales would have looked like without the intervention.
#

ci = fit_causalimpact(data=ci_data, pre_period=pre_period, post_period=post_period)

print("\nCausal Impact Summary:")
print(summary(ci))

print("\nCausal Impact Detailed Report:")
print(summary(ci, output_format='report'))


# 8. MODEL VALIDATION & DIAGNOSTICS

print("\nSECTION 8 — MODEL VALIDATION & DIAGNOSTICS")

# Extract model series
inferences   = ci.series.copy()

# Validation 1: Pre-period model fit
# In the pre-period the model should closely track actuals.
# We compute MAPE and R² to assess counterfactual fit quality.
pre_mask = inferences.index < pd.to_datetime(INTERVENTION_DATE)

# Filter to trading days only - for MAPE calculation, and other validation
pre_trading = inferences.loc[pre_mask].copy()
pre_trading = pre_trading[pre_trading["observed"] > 0]

pre_actual    = pre_trading["observed"]
pre_predicted = pre_trading["posterior_mean"]
# pre_actual    = inferences.loc[pre_mask, "observed"]
# pre_predicted = inferences.loc[pre_mask, "posterior_mean"]

mape = (np.abs((pre_actual - pre_predicted) / pre_actual)).mean() * 100
ss_res = np.sum((pre_actual - pre_predicted) ** 2)
ss_tot = np.sum((pre_actual - pre_actual.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"\n[V1] Pre-period model fit:")
print(f" MAPE : {mape:.2f}%  (lower is better; <10% indicates strong fit)")
print(f" R²   : {r2:.4f}  (closer to 1.0 indicates strong fit)")

# Validation 2: Residual normality (Shapiro-Wilk)
pre_residuals = (pre_actual - pre_predicted).dropna()
stat, p_value = stats.shapiro(pre_residuals[:50])  # Shapiro works best on <50 obs
print(f"\n[V2] Residual normality (Shapiro-Wilk, pre-period):")
print(f"     Statistic = {stat:.4f}  |  p-value = {p_value:.4f}")
if p_value > 0.05:
    print("Residuals are approximately normal (p > 0.05)")
else:
    print("Residuals may not be normal — interpret CIs with caution")

# Validation 3: Posterior tail probability (p-value)
# CausalImpact provides a Bayesian one-sided p-value.
# p < 0.05 means the observed effect is unlikely under the null hypothesis of no intervention effect.
summary_data = ci.summary
post_prob = summary_data.loc["average", "p_value"]
print(f"\n[V3] Bayesian posterior tail probability:")
print(f"     p = {post_prob:.4f}")
if post_prob < 0.05:
    print(" Effect is statistically significant (p < 0.05)")
elif post_prob < 0.10:
    print(" Effect is marginal (0.05 ≤ p < 0.10)")
else:
    print(" Effect is not statistically significant (p ≥ 0.10)")

# Validation 4: Credible interval excludes zero

avg_effect_lower = summary_data.loc["average", "abs_effect_lower"]
avg_effect_upper = summary_data.loc["average", "abs_effect_upper"]
print(f"\n[V4] Average daily effect — 95% credible interval:")
print(f"     Lower : €{avg_effect_lower:,.2f}")
print(f"     Upper : €{avg_effect_upper:,.2f}")
if avg_effect_lower > 0:
    print(" Credible interval is entirely positive — effect is genuine uplift")
elif avg_effect_upper < 0:
    print(" Credible interval is entirely negative — effect is genuine decline")
else:
    print(" Credible interval spans zero — effect direction is uncertain")

# Validation 5: Cumulative effect plausibility
cum_effect       = summary_data.loc["cumulative", "actual"] - summary_data.loc["cumulative", "predicted"]
cum_effect_lower = summary_data.loc["cumulative", "abs_effect_lower"]
cum_effect_upper = summary_data.loc["cumulative", "abs_effect_upper"]
print(f"\n[V5] Cumulative causal effect over post-period:")
print(f" Point estimate : €{cum_effect:,.2f}")
print(f" 95% CI         : €{cum_effect_lower:,.2f} to €{cum_effect_upper:,.2f}")

# Validation 6: Relative effect
rel_effect       = summary_data.loc["average", "rel_effect"]
rel_effect_lower = summary_data.loc["average", "rel_effect_lower"]
rel_effect_upper = summary_data.loc["average", "rel_effect_upper"]
print(f"\n[V6] Relative effect of intervention:")
print(f" Point estimate : {rel_effect*100:.2f}%")
print(f" 95% CI         : {rel_effect_lower*100:.2f}% to {rel_effect_upper*100:.2f}%")

# Plot 9: Pre-period residuals
fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
sns.histplot(pre_residuals, bins=35, kde=True, color="seagreen", ax=ax)
ax.axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="Zero residual")
ax.set_title("Pre-Period Model Residuals Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Residual (Actual − Predicted) (€)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("09_residuals_distribution.png", dpi=150)
plt.show()

# Plot 10: Actual vs predicted — pre-period
fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
ax.plot(pre_trading.index, pre_actual,
        color="Seagreen", linewidth=1.2, label="Actual Sales")
ax.plot(pre_trading.index, pre_predicted,
        color="Darkslateblue", linewidth=1.2, linestyle="--", label="Model Prediction")
# ax.plot(inferences.loc[pre_mask].index, pre_actual,
#         color="Seagreen", linewidth=1.2, label="Actual Sales")
# ax.plot(inferences.loc[pre_mask].index, pre_predicted,
#         color="Darkslateblue", linewidth=1.2, linestyle="--", label="Model Prediction")
ax.set_title("Pre-Period: Actual vs Model Predicted Sales (Counterfactual Fit)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("10_pre_period_actual_vs_predicted.png", dpi=150)
plt.show()


# 9. BUSINESS INSIGHT EXTRACTION & VISUALISATION

print("\nSECTION 9 — BUSINESS INSIGHT EXTRACTION & VISUALISATION")

post_mask = inferences.index >= pd.to_datetime(INTERVENTION_DATE)

post_actual    = inferences.loc[post_mask, "observed"]
post_predicted = inferences.loc[post_mask, "posterior_mean"]
post_lower     = inferences.loc[post_mask, "posterior_lower"]
post_upper     = inferences.loc[post_mask, "posterior_upper"]
point_effect   = inferences.loc[post_mask, "point_effects_mean"]
effect_lower   = inferences.loc[post_mask, "point_effects_lower"]
effect_upper   = inferences.loc[post_mask, "point_effects_upper"]
cum_effect_ts  = inferences.loc[post_mask, "cumulative_effects_mean"]
cum_lower_ts   = inferences.loc[post_mask, "cumulative_effects_lower"]
cum_upper_ts   = inferences.loc[post_mask, "cumulative_effects_upper"]


# Plot 11: Full time series — actual vs counterfactual
fig, ax = plt.subplots(figsize=(14, 6))

# Pre-period
ax.plot(pre_trading.index, pre_actual,
        color="steelblue", linewidth=1.0, label="Actual Sales (Pre)")
ax.plot(pre_trading.index, pre_predicted,
        color="grey", linewidth=1.0, linestyle="--", alpha=0.7)

# Post-period actual
ax.plot(post_actual.index, post_actual,
        color="steelblue", linewidth=1.2, label="Actual Sales (Post)")

# Counterfactual with credible interval
ax.plot(post_predicted.index, post_predicted,
        color="darkorange", linewidth=1.5, linestyle="--", label="Counterfactual")
ax.fill_between(post_predicted.index, post_lower, post_upper,
                color="darkorange", alpha=0.15, label="95% Credible Interval")

ax.axvline(pd.to_datetime(INTERVENTION_DATE), color="crimson",
           linestyle="--", linewidth=1.8, label=f"Intervention: {INTERVENTION_DATE}")

ax.set_title(f"Store {TREATED_STORE} — Actual Sales vs Counterfactual with Credible Interval",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(fontsize=9)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("11_actual_vs_counterfactual.png", dpi=150)
plt.show()

# Plot 12: Daily causal effect (point effect)
fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
ax.plot(point_effect.index, point_effect,
        color="steelblue", linewidth=1.0, label="Daily Causal Effect")
ax.fill_between(point_effect.index, effect_lower, effect_upper,
                color="steelblue", alpha=0.15, label="95% Credible Interval")
ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5, label="Zero Effect Line")
ax.set_title("Daily Causal Effect of Intervention on Sales",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Estimated Effect (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("12_daily_causal_effect.png", dpi=150)
plt.show()

# Plot 13: Cumulative causal effect

fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
ax.plot(cum_effect_ts.index, cum_effect_ts,
        color="steelblue", linewidth=1.5, label="Cumulative Causal Effect")
ax.fill_between(cum_effect_ts.index, cum_lower_ts, cum_upper_ts,
                color="steelblue", alpha=0.15, label="95% Credible Interval")
ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5, label="Zero Effect Line")
ax.set_title("Cumulative Causal Effect of Intervention on Sales",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Effect (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("13_cumulative_causal_effect.png", dpi=150)
plt.show()

# Plot 14: Monthly average actual vs counterfactual (post)
post_comparison = pd.DataFrame({
    "Actual"        : post_actual,
    "Counterfactual": post_predicted
})
post_comparison["Month"] = post_comparison.index.month
monthly_post = post_comparison.groupby("Month")[["Actual", "Counterfactual"]].mean().reset_index()
monthly_post_melt = monthly_post.melt(id_vars="Month",
                                       value_vars=["Actual", "Counterfactual"],
                                       var_name="Series", value_name="Avg Daily Sales")

fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
sns.barplot(data=monthly_post_melt, x="Month", y="Avg Daily Sales",
            hue="Series", palette=["steelblue", "darkorange"],
            ax=ax, errorbar=None)
ax.set_title("Post-Intervention: Monthly Avg Actual vs Counterfactual Sales",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Average Daily Sales (€)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
ax.legend(title="Series")
plt.tight_layout()
plt.savefig("14_monthly_actual_vs_counterfactual.png", dpi=150)
plt.show()

# Plot 15: Effect distribution across post-period days
fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
sns.histplot(point_effect, bins=40, kde=True, color="seagreen", ax=ax)
ax.axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="Zero Effect")
ax.axvline(point_effect.mean(), color="steelblue", linestyle="-",
           linewidth=1.5, label=f"Mean Effect: €{point_effect.mean():,.0f}")
ax.set_title("Distribution of Daily Causal Effect Estimates (Post-Period)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Daily Effect (€)")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("15_effect_distribution.png", dpi=150)
plt.show()

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")