"""
Exploratory Data Analysis (EDA)
================================
Dataset : UCI Adult Income ("Census Income") Dataset
Source  : UCI Machine Learning Repository, id=2
          https://archive.ics.uci.edu/dataset/2/adult

Goal    : Demonstrate a structured, end-to-end Exploratory Data Analysis
          workflow - data loading, validation, missingness diagnosis and
          treatment, univariate and bivariate analysis, and an automated
          profiling report - using a real-world dataset with a genuine
          data quality issue (disguised missing values) and a meaningful
          business target (income classification).

Author  : Marc Grover
Portfolio: https://marcgrover-datascience.github.io/

Environment note: this script was developed using fg-data-profiling
(the maintained successor to ydata-profiling, renamed April 2026). The
import below uses the new package name; if running against an older
environment with only ydata-profiling installed, replace
`from data_profiling import ProfileReport`
with
`from ydata_profiling import ProfileReport`
The API is otherwise identical.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from data_profiling import ProfileReport

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DPI = 150
RANDOM_STATE = 42

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = PLOT_DPI
plt.rcParams["savefig.bbox"] = "tight"

plot_counter = 1


def save_plot(fig, name):
    """Save a figure as a sequentially numbered PNG and close it."""
    global plot_counter
    filename = f"plot_{plot_counter:02d}_{name}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=PLOT_DPI)
    plt.close(fig)
    print(f"Saved: {filename}")
    plot_counter += 1


# ----------------------------------------------------------------------------
# 1. Data Loading
# ----------------------------------------------------------------------------

print("=" * 80)
print("1. DATA LOADING")
print("=" * 80)

adult = fetch_ucirepo(id=2)

X = adult.data.features
y = adult.data.targets

df = pd.concat([X, y], axis=1)

# Standardise column naming (the UCI export uses hyphens; underscores are
# easier to work with downstream and avoid attribute-access ambiguity).
df.columns = [col.replace("-", "_").strip() for col in df.columns]

print(f"Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ----------------------------------------------------------------------------
# 2. Data Validation
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("2. DATA VALIDATION")
print("=" * 80)

print(f"\nData types:\n{df.dtypes}")

# --- 2.1 Exact duplicate records ---
n_duplicates = df.duplicated().sum()
print(f"\nExact duplicate rows: {n_duplicates}")
if n_duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed. New shape: {df.shape}")

# --- 2.2 Target variable consistency ---
# The income target is known to carry a trailing full stop on some UCI
# export variants (e.g. ">50K." vs ">50K"). Strip and standardise.
target_col = "income"
df[target_col] = df[target_col].astype(str).str.strip().str.rstrip(".")
print(f"\nTarget classes after standardisation: {df[target_col].unique().tolist()}")

# --- 2.3 Disguised missing values ---
# This dataset does not use NaN for missingness. Missing values are encoded
# as the literal string "?" with a leading space in several categorical
# columns. A naive .isnull().sum() call would report zero missing values,
# masking a real data quality issue - so this check is performed explicitly.
print("\nChecking for disguised missing values (literal '?' placeholders)...")

categorical_cols = df.select_dtypes(include="object").columns.tolist()
categorical_cols = [c for c in categorical_cols if c != target_col]

disguised_missing_counts = {}
for col in categorical_cols:
    # Note: do NOT call .astype(str) here. ucimlrepo loads this dataset via
    # pandas.read_csv() internally, and pandas' default na_values list
    # already converts some "?" placeholders to genuine NaN on load before
    # this script ever sees the data (the remainder survive as literal "?"
    # strings, which is why both forms must be handled). Calling
    # .astype(str) on a column that already contains real NaN silently
    # converts those NaN values into the literal string "nan", which is
    # no longer recognised as missing by pandas - it would then survive
    # imputation untouched and appear as its own spurious category in
    # any downstream plot. .str.strip() works directly on object-dtype
    # columns without this side effect, so it is used alone here.
    df[col] = df[col].str.strip()
    n_missing = (df[col] == "?").sum()
    if n_missing > 0:
        disguised_missing_counts[col] = n_missing

if disguised_missing_counts:
    print("Columns containing '?' placeholder values:")
    for col, count in disguised_missing_counts.items():
        pct = 100 * count / len(df)
        print(f"  {col:<15s}: {count:>6d} missing  ({pct:.2f}%)")
else:
    print("No disguised missing values found.")

# Standard NaN-based missing value check, run for completeness alongside
# the disguised-missingness check above.
standard_nulls = df.isnull().sum()
print(f"\nStandard (NaN) missing values per column:\n{standard_nulls[standard_nulls > 0]}")

# Replace the literal "?" placeholder with proper NaN so pandas' native
# missing-value tooling (isnull, dropna, fillna) can be used consistently
# from this point forward.
df[categorical_cols] = df[categorical_cols].replace("?", np.nan)

# Safety-net check: confirm no stray string-typed placeholder values
# (e.g. the literal string "nan", "none", "null") have survived as
# ordinary category values. These would not be caught by .isnull() and
# would silently bypass imputation, appearing as a spurious category in
# downstream plots. None are expected at this point; this assertion
# exists to catch any future regression in the cleaning logic above.
stray_placeholder_mask = df[categorical_cols].isin(["nan", "none", "null", "NaN", "None", "NULL"])
n_stray = stray_placeholder_mask.sum().sum()
if n_stray > 0:
    stray_cols = stray_placeholder_mask.sum()
    raise ValueError(
        f"Found {n_stray} stray string-typed placeholder value(s) that bypassed "
        f"missing-value handling: {stray_cols[stray_cols > 0].to_dict()}. "
        f"Check for unintended .astype(str) calls on columns containing NaN."
    )

total_missing = df.isnull().sum().sum()
total_cells = df.shape[0] * df.shape[1]
print(
    f"\nTotal missing values after standardisation: {total_missing} "
    f"({100 * total_missing / total_cells:.2f}% of all cells)"
)

# --- 2.4 Missingness pattern ---
# Before deciding how to treat missing values, the pattern of missingness
# is examined: are missing values isolated to single columns, or do they
# co-occur across columns within the same records? This distinction
# matters because co-occurring missingness often points to a shared root
# cause (e.g. an individual who has never worked has no recorded workclass
# AND no recorded occupation), which should inform the imputation strategy.
missing_cols = [col for col in categorical_cols if df[col].isnull().sum() > 0]
print(f"\nColumns with missing values: {missing_cols}")

if len(missing_cols) >= 2:
    co_occurrence = df[missing_cols].isnull().corr()
    print(f"\nCo-occurrence of missingness (correlation of missing indicators):\n{co_occurrence}")

    workclass_and_occupation_missing = (
        df["workclass"].isnull() & df["occupation"].isnull()
    ).sum()
    occupation_missing_total = df["occupation"].isnull().sum()
    print(
        f"\nRecords missing both workclass and occupation: "
        f"{workclass_and_occupation_missing} "
        f"({100 * workclass_and_occupation_missing / occupation_missing_total:.1f}% "
        f"of all occupation-missing records)"
    )

# --- 2.5 Numeric range sanity checks ---
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"\nNumeric column summary:\n{df[numeric_cols].describe()}")

# Sanity check: age should fall within a plausible adult census range.
implausible_age = df[(df["age"] < 16) | (df["age"] > 100)]
print(f"\nRecords with implausible age values (<16 or >100): {len(implausible_age)}")

# ----------------------------------------------------------------------------
# 3. Missing Value Treatment
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("3. MISSING VALUE TREATMENT")
print("=" * 80)

# Design decision: rather than imputing the mode (which would silently
# overwrite a meaningful absence with an artificial "most common" value,
# and risks distorting the categorical distribution of an already
# imbalanced feature), missing values in workclass, occupation, and
# native_country are imputed with an explicit "Unknown" category.
#
# This is preferred here for three reasons:
#   1. It is honest: the model and any downstream reader can see that a
#      value was genuinely unrecorded, rather than inferring a false
#      certainty from a mode-imputed label.
#   2. It is informative: "Unknown" workclass/occupation is not missing
#      at random - it is associated with respondents who are not
#      currently employed in a conventional sense, which is itself a
#      potentially predictive signal for income.
#   3. It avoids inflating the already-dominant categories (e.g.
#      "Private" workclass) with imputed records that did not originate
#      from that class, which would distort the true class balance shown
#      in the EDA visuals that follow.
for col in missing_cols:
    df[col] = df[col].fillna("Unknown")

print(f"Imputed missing values in {missing_cols} with explicit 'Unknown' category.")
print(f"Remaining missing values after imputation: {df.isnull().sum().sum()}")

print(f"\nFinal cleaned dataset shape: {df.shape}")

# ----------------------------------------------------------------------------
# 4. Exploratory Data Analysis - Univariate
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("4. EXPLORATORY DATA ANALYSIS - UNIVARIATE")
print("=" * 80)

# --- Plot 1: Target class distribution ---
income_counts = df[target_col].value_counts()
print(f"\nIncome class distribution:\n{income_counts}")
print(f"\nIncome class proportions:\n{(100 * income_counts / len(df)).round(2)}")

fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=df, x=target_col, order=income_counts.index, ax=ax)
ax.set_title("Distribution of Income Classification")
ax.set_xlabel("Income Band")
ax.set_ylabel("Number of Individuals")
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height()):,}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center", va="bottom", fontsize=10,
    )
save_plot(fig, "income_class_distribution")

# --- Plot 2: Age distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=40, kde=True, ax=ax)
ax.set_title("Distribution of Age")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Count")
save_plot(fig, "age_distribution")

# --- Plot 3: Hours per week distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=df, x="hours_per_week", bins=40, kde=True, ax=ax)
ax.set_title("Distribution of Hours Worked Per Week")
ax.set_xlabel("Hours Per Week")
ax.set_ylabel("Count")
save_plot(fig, "hours_per_week_distribution")

# --- Plot 4: Education level counts ---
education_order = df["education"].value_counts().index
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, y="education", order=education_order, ax=ax)
ax.set_title("Distribution of Education Level")
ax.set_xlabel("Number of Individuals")
ax.set_ylabel("Education Level")
save_plot(fig, "education_distribution")

# --- Plot 5: Workclass counts (including imputed Unknown category) ---
workclass_order = df["workclass"].value_counts().index
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df, y="workclass", order=workclass_order, ax=ax)
ax.set_title("Distribution of Workclass (including imputed 'Unknown')")
ax.set_xlabel("Number of Individuals")
ax.set_ylabel("Workclass")
save_plot(fig, "workclass_distribution")

# ----------------------------------------------------------------------------
# 5. Exploratory Data Analysis - Bivariate (relationship with income)
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("5. EXPLORATORY DATA ANALYSIS - BIVARIATE")
print("=" * 80)

# --- Plot 6: Age by income band ---
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=df, x=target_col, y="age", order=income_counts.index, ax=ax)
ax.set_title("Age Distribution by Income Band")
ax.set_xlabel("Income Band")
ax.set_ylabel("Age (years)")
save_plot(fig, "boxplot_age_by_income")

# --- Plot 7: Hours per week by income band ---
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=df, x=target_col, y="hours_per_week", order=income_counts.index, ax=ax)
ax.set_title("Hours Worked Per Week by Income Band")
ax.set_xlabel("Income Band")
ax.set_ylabel("Hours Per Week")
save_plot(fig, "boxplot_hours_by_income")

# --- Plot 8: Education level by income band (proportion within education) ---
education_income = (
    df.groupby("education")[target_col]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)
education_income_high = (
    education_income[education_income[target_col] == ">50K"]
    .sort_values("proportion", ascending=False)
)
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(
    data=education_income_high,
    y="education",
    x="proportion",
    order=education_income_high["education"],
    ax=ax,
)
ax.set_title("Proportion Earning >50K by Education Level")
ax.set_xlabel("Proportion Earning >50K")
ax.set_ylabel("Education Level")
ax.set_xlim(0, 1)
save_plot(fig, "income_proportion_by_education")

# --- Plot 9: Marital status by income band (proportion within group) ---
marital_income = (
    df.groupby("marital_status")[target_col]
    .value_counts(normalize=True)
    .rename("proportion")
    .reset_index()
)
marital_income_high = (
    marital_income[marital_income[target_col] == ">50K"]
    .sort_values("proportion", ascending=False)
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=marital_income_high,
    y="marital_status",
    x="proportion",
    order=marital_income_high["marital_status"],
    ax=ax,
)
ax.set_title("Proportion Earning >50K by Marital Status")
ax.set_xlabel("Proportion Earning >50K")
ax.set_ylabel("Marital Status")
ax.set_xlim(0, 1)
save_plot(fig, "income_proportion_by_marital_status")

# --- Plot 10: Hours per week vs age, coloured by income ---
# A sample is used for the scatter plot to keep the chart legible and the
# file size reasonable, since the full dataset contains tens of thousands
# of points that would otherwise overplot heavily.
sample_df = df.sample(n=min(3000, len(df)), random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(
    data=sample_df, x="age", y="hours_per_week", hue=target_col,
    alpha=0.4, s=25, ax=ax,
)
ax.set_title("Age vs Hours Worked Per Week, by Income Band (sampled n=3,000)")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Hours Per Week")
ax.legend(title="Income Band")
save_plot(fig, "scatter_age_hours_by_income")

# ----------------------------------------------------------------------------
# 6. Correlation Analysis
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("6. CORRELATION ANALYSIS")
print("=" * 80)

corr_df = df[numeric_cols].copy()
correlation_matrix = corr_df.corr()
print(f"\nCorrelation matrix:\n{correlation_matrix}")

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(
    correlation_matrix, annot=True, fmt=".2f", cmap="viridis",
    square=True, linewidths=0.5, ax=ax,
)
ax.set_title("Correlation Heatmap of Numeric Features")
save_plot(fig, "correlation_heatmap")

# ----------------------------------------------------------------------------
# 7. Automated Profiling Report
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("7. AUTOMATED PROFILING REPORT")
print("=" * 80)

print("Generating automated profiling report (fg-data-profiling)...")
profile = ProfileReport(
    df,
    title="UCI Adult Income Dataset - Automated Profiling Report",
    explorative=True,
)
report_path = os.path.join(OUTPUT_DIR, "adult_income_profiling_report.html")
profile.to_file(report_path)
print(f"Profiling report saved to: {report_path}")

# ----------------------------------------------------------------------------
# 8. Cleaned Data Export
# ----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("8. CLEANED DATA EXPORT")
print("=" * 80)

cleaned_path = os.path.join(OUTPUT_DIR, "adult_income_cleaned.csv")
df.to_csv(cleaned_path, index=False)
print(f"Cleaned dataset exported to: {cleaned_path}")

print("\n" + "=" * 80)
print("SCRIPT COMPLETE")
print("=" * 80)