# Demonstrates Great Expectations (v1.18) using the Seaborn Titanic dataset.
#
# The project builds a single ingestion-stage expectation suite and validates it against TWO batches:
#   1. "Clean" batch  - the unmodified Titanic dataset, used to confirm the suite is correctly specified against known-good data.
#   2. "Corrupted" batch - the same dataset with eight deliberately injected, realistic data quality issues (nulls, out-of-range values,
#                        bad categorical codes, type drift, duplicate rows, a dropped column), used to confirm the SAME suite reliably
#                        detects ingestion-stage corruption.
#
# Both runs use the identical suite - nothing is loosened or tightened between runs - so every failure on the corrupted
# batch is a genuine drift-detection result rather than an artefact of different validation criteria.
#
# Outputs:
#   - Great Expectations Data Docs (static HTML), covering both validation runs
#   - PNG charts in ./outputs/ (EDA charts that justify threshold choices, plus a clean-vs-corrupted pass/fail summary chart)
#   - A markdown failure-detail table (per failed expectation, observed value, threshold, and unexpected percentage) printed to console and saved to disk

# NOTE that the html output is saved to:  \DataProcessing\gx\uncommitted\data_docs\local_site

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import great_expectations as gx
from great_expectations import expectations as gxe
from great_expectations.checkpoint import UpdateDataDocsAction

# -----------------------------
# 0) Settings and utilities
# -----------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

# ---------------------------------------------------------
# 1) Load the clean Titanic dataset
# ---------------------------------------------------------
print_header("1) Loading the Seaborn Titanic dataset (clean batch)")

titanic_clean = sns.load_dataset("titanic")

# NOTE on dtypes: depending on the pandas version installed, columns such as "sex" and "embarked" may be reported
# internally as either the legacy "object" dtype or the newer pandas "str" dtype (PDEP-14, default in pandas # 3.x).
# Great Expectations' ExpectColumnValuesToBeOfType check is sensitive to this distinction. To keep the suite's behaviour
# identical regardless of the pandas version running it, string columns are explicitly cast to "object" # immediately after loading.

STRING_COLUMNS = ["sex", "embarked", "who", "embark_town", "alive"]
titanic_clean = titanic_clean.astype({c: "object" for c in STRING_COLUMNS})

print("Shape:", titanic_clean.shape)
print("Columns:", list(titanic_clean.columns))
print("\nDtypes:")
print(titanic_clean.dtypes)

# -----------------------------------------------------------------
# 2) EDA: characterise the clean data to justify validation thresholds
# -----------------------------------------------------------------
# Every threshold used in the expectation suite below (Section 4) is derived
# from one of the two charts produced here, rather than chosen arbitrarily.
print_header("2) Exploratory analysis - establishing validation thresholds from data")

# 2a. Missing values by column - this directly informs the null-tolerance
# thresholds used for "age" and "deck" in the suite. Without this chart, a
# threshold like "deck must be at least 85% non-null" would be an unfounded
# guess; deck is in fact only ~23% populated in the real data, so the
# threshold must reflect that, not an arbitrary round number.
missing_pct = titanic_clean.isna().mean().sort_values(ascending=False).reset_index()
missing_pct.columns = ["column", "missing_fraction"]
print("Missing value fractions by column:")
print(missing_pct.to_string(index=False))

plt.figure(figsize=(10, 4))
sns.barplot(data=missing_pct, x="column", y="missing_fraction", color="#4C72B0")
plt.title("Titanic (Clean Batch): Fraction of Missing Values by Column")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Missing Fraction")
plt.xlabel("Column")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_01_missing_fraction_by_column.png"), dpi=150)
plt.close()

# 2b. Age and fare distributions - these directly inform the range bounds
# used for ExpectColumnValuesToBeBetween on "age" and "fare" in the suite.
age_min, age_max = titanic_clean["age"].min(), titanic_clean["age"].max()
fare_min, fare_max = titanic_clean["fare"].min(), titanic_clean["fare"].max()
print(f"\nObserved age range: {age_min:.1f} to {age_max:.1f}")
print(f"Observed fare range: {fare_min:.2f} to {fare_max:.2f}")

plt.figure(figsize=(6, 4))
sns.histplot(data=titanic_clean, x="age", kde=True, color="#55A868", bins=30)
plt.title("Titanic (Clean Batch): Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_02_age_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(data=titanic_clean, x="fare", kde=True, color="#C44E52", bins=30)
plt.title("Titanic (Clean Batch): Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_03_fare_distribution.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 3) Build the corrupted batch: eight deliberate, realistic data flaws
# ---------------------------------------------------------------------
# Each corruption below is designed to trip ONE specific expectation in the
# suite defined in Section 4, so that every failure on the corrupted run has
# a clear, attributable cause. Corruptions are applied only to rows that do
# not already hold a null/missing value for that column, so that the
# resulting failure counts cleanly reflect the injected corruption rather
# than being mixed in with pre-existing missingness.
print_header("3) Constructing the corrupted batch (8 deliberate injected flaws)")

rng = np.random.default_rng(RANDOM_STATE)
titanic_corrupt = titanic_clean.copy()

# Flaw 1: invalid ages (negative values) on ~8% of non-null age rows.
# Targets -> ExpectColumnValuesToBeBetween(age, 0, 80, mostly=0.99)
age_idx = titanic_corrupt[titanic_corrupt["age"].notna()].sample(
    frac=0.08, random_state=RANDOM_STATE
).index
titanic_corrupt.loc[age_idx, "age"] = -rng.uniform(1, 10, size=len(age_idx))

# Flaw 2: extreme fare outliers (e.g. a pricing/decimal error) on a small
# number of rows. Targets -> ExpectColumnValuesToBeBetween(fare, 0, 600, mostly=0.99)
fare_idx = titanic_corrupt.sample(frac=0.02, random_state=RANDOM_STATE + 1).index
titanic_corrupt.loc[fare_idx, "fare"] = rng.uniform(5000, 50000, size=len(fare_idx))

# Flaw 3: non-standard categorical codes for sex ("M"/"F" instead of
# "male"/"female"), simulating a source system sending an unexpected encoding.
# Targets -> ExpectColumnValuesToBeInSet(sex, ["male", "female"])
sex_idx = titanic_corrupt.sample(frac=0.05, random_state=RANDOM_STATE + 2).index
titanic_corrupt.loc[sex_idx, "sex"] = titanic_corrupt.loc[sex_idx, "sex"].map(
    {"male": "M", "female": "F"}
)

# Flaw 4: additional nulls injected into "embarked", simulating a partial
# upstream extraction failure. Targets ->
# ExpectColumnValuesToBeInSet(embarked, ["C","Q","S"], mostly=0.95)
embarked_idx = titanic_corrupt[titanic_corrupt["embarked"].notna()].sample(
    frac=0.15, random_state=RANDOM_STATE + 3
).index
titanic_corrupt.loc[embarked_idx, "embarked"] = None

# Flaw 5: "pclass" delivered as text rather than integer, simulating a schema
# drift at the source (e.g. a CSV column re-typed upstream).
# Targets -> ExpectColumnValuesToBeOfType(pclass, "int64")
titanic_corrupt["pclass"] = titanic_corrupt["pclass"].astype("object")
pclass_idx = titanic_corrupt.sample(frac=0.3, random_state=RANDOM_STATE + 4).index
titanic_corrupt.loc[pclass_idx, "pclass"] = titanic_corrupt.loc[pclass_idx, "pclass"].astype(str)

# Flaw 6: nulls injected into "survived", the core target/label column.
# Targets -> ExpectColumnValuesToNotBeNull(survived)
survived_idx = titanic_corrupt.sample(frac=0.1, random_state=RANDOM_STATE + 5).index
titanic_corrupt.loc[survived_idx, "survived"] = None

# Flaw 7: duplicate rows appended, simulating a retry or replay in the
# ingestion pipeline that re-sends already-loaded records.
# Targets -> ExpectTableRowCountToBeBetween(min_value=800, max_value=1000)
duplicate_rows = titanic_corrupt.sample(n=150, random_state=RANDOM_STATE + 6)
titanic_corrupt = pd.concat([titanic_corrupt, duplicate_rows], ignore_index=True)

# Flaw 8: an entire column dropped, simulating a source system that silently
# stopped sending a field. Targets ->
# ExpectTableColumnCountToEqual(value=15) and ExpectTableColumnsToMatchSet(...)
titanic_corrupt = titanic_corrupt.drop(columns=["deck"])

print("Clean batch shape:    ", titanic_clean.shape)
print("Corrupted batch shape:", titanic_corrupt.shape)
print("\nCorruptions applied:")
print(f"  1. {len(age_idx)} rows with negative age values")
print(f"  2. {len(fare_idx)} rows with extreme fare outliers")
print(f"  3. {len(sex_idx)} rows with non-standard sex codes ('M'/'F')")
print(f"  4. {len(embarked_idx)} additional nulls injected into 'embarked'")
print(f"  5. {len(pclass_idx)} rows with 'pclass' values cast to string")
print(f"  6. {len(survived_idx)} nulls injected into 'survived'")
print(f"  7. 150 duplicate rows appended")
print(f"  8. 'deck' column dropped entirely")

# ---------------------------------------------------
# 4) Initialise Great Expectations Data Context
# ---------------------------------------------------
print_header("4) Initialising Great Expectations context")

context = gx.get_context(mode="file")  # creates ./gx if not already present

try:
    pandas_ds = context.data_sources.get("titanic_pandas_ds")
except Exception:
    pandas_ds = context.data_sources.add_pandas(name="titanic_pandas_ds")

# Two SEPARATE named assets - one per batch - so that both the clean and
# corrupted validation runs appear as distinct, clearly labelled entries in
# Data Docs, rather than overwriting one another as sequential batches of a
# single asset would.
def get_or_add_asset(datasource, name):
    try:
        return datasource.get_asset(name)
    except Exception:
        return datasource.add_dataframe_asset(name=name)

def get_or_add_batch_definition(asset, name):
    try:
        return asset.get_batch_definition(name)
    except Exception:
        return asset.add_batch_definition_whole_dataframe(name)

clean_asset = get_or_add_asset(pandas_ds, "titanic_ingestion_clean_df")
corrupt_asset = get_or_add_asset(pandas_ds, "titanic_ingestion_corrupted_df")

clean_bd = get_or_add_batch_definition(clean_asset, "clean_whole_df")
corrupt_bd = get_or_add_batch_definition(corrupt_asset, "corrupted_whole_df")

# -----------------------------------------------------------------
# 5) Define the ingestion-stage expectation suite (used for BOTH batches)
# -----------------------------------------------------------------
print_header("5) Defining the ingestion-stage expectation suite")

suite_name = "titanic_ingestion_suite"
try:
    suite = context.suites.get(suite_name)
except Exception:
    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))

# -------------------------
# Structural checks
# -------------------------
suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=800, max_value=1000))
suite.add_expectation(gxe.ExpectTableColumnCountToEqual(value=15))
suite.add_expectation(
    gxe.ExpectTableColumnsToMatchSet(
        column_set=[
            "survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked",
            "class", "who", "adult_male", "deck", "embark_town", "alive", "alone",
        ],
        exact_match=True,
    )
)

# -------------------------
# Column types
# -------------------------
suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="survived", type_="int64"))
suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="pclass", type_="int64"))
suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="sex", type_="object"))
suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="age", type_="float64"))
suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="fare", type_="float64"))

# -------------------------
# Content and domain checks
# -------------------------
suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="survived", value_set=[0, 1]))
suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="pclass", value_set=[1, 2, 3]))
suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="sex", value_set=["male", "female"]))
suite.add_expectation(
    gxe.ExpectColumnValuesToBeInSet(column="embarked", value_set=["C", "Q", "S"], mostly=0.95)
)
# Bounds set from the observed clean-batch range (Section 2b), not arbitrary
# round numbers: max observed age was ~80, max observed fare was ~512.
suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=80, mostly=0.99))
suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="fare", min_value=0, max_value=600, mostly=0.99))

# -------------------------
# Missingness tolerances
# -------------------------
suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="survived"))
suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="pclass"))
suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="sex"))
# "age" is ~80% populated in the clean data (Section 2a), so the tolerance is
# set just below that observed level, not at an arbitrary 0.85+ figure that
# was never checked against the real missingness rate.
suite.add_expectation(gxe.ExpectColumnProportionOfNonNullValuesToBeBetween(column="age", min_value=0.75))
# "deck" is only ~23% populated in the clean data - the ORIGINAL suite set a
# minimum of 0.85 here, which would have failed even on perfectly clean data.
# The threshold below is set from the actual observed completeness rate.
suite.add_expectation(gxe.ExpectColumnProportionOfNonNullValuesToBeBetween(column="deck", min_value=0.15))

context.suites.add_or_update(suite)
print(f"Suite '{suite_name}' defined with {len(suite.expectations)} expectations.")

# -------------------------------------------------------------
# 6) Bind the suite to both batches and register checkpoints (GX 1.x)
# -------------------------------------------------------------
print_header("6) Binding the suite to both batches")

vd_clean = gx.ValidationDefinition(name="vd_titanic_clean", data=clean_bd, suite=suite)
vd_corrupt = gx.ValidationDefinition(name="vd_titanic_corrupted", data=corrupt_bd, suite=suite)

vd_clean = context.validation_definitions.add_or_update(vd_clean)
vd_corrupt = context.validation_definitions.add_or_update(vd_corrupt)

# Data Docs site - created before the checkpoints run so results render
# immediately after each run.
site_name = "local_site"
site_dir = os.path.abspath(os.path.join(OUTPUT_DIR, "data_docs"))
site_config = {
    "class_name": "SiteBuilder",
    "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
    "store_backend": {
        "class_name": "TupleFilesystemStoreBackend",
        "base_directory": site_dir,
    },
}
try:
    context.add_data_docs_site(site_name=site_name, site_config=site_config)
except Exception:
    pass  # site already exists

checkpoint_clean = gx.Checkpoint(
    name="titanic_clean_checkpoint",
    validation_definitions=[vd_clean],
    actions=[UpdateDataDocsAction(name="update_docs", site_names=[site_name])],
)
checkpoint_corrupt = gx.Checkpoint(
    name="titanic_corrupted_checkpoint",
    validation_definitions=[vd_corrupt],
    actions=[UpdateDataDocsAction(name="update_docs", site_names=[site_name])],
)

checkpoint_clean = context.checkpoints.add_or_update(checkpoint_clean)
checkpoint_corrupt = context.checkpoints.add_or_update(checkpoint_corrupt)

# ---------------------------------------------------
# 7) Run validation on both batches
# ---------------------------------------------------
print_header("7) Running validation: clean batch")
results_clean = checkpoint_clean.run(batch_parameters={"dataframe": titanic_clean})
print("Clean batch overall success:", results_clean.success)

print_header("7b) Running validation: corrupted batch")
results_corrupt = checkpoint_corrupt.run(batch_parameters={"dataframe": titanic_corrupt})
print("Corrupted batch overall success:", results_corrupt.success)

# ----------------------------------------------------------------------
# 8) Extract structured pass/fail data from both result objects
# ----------------------------------------------------------------------
print_header("8) Extracting structured results for comparison")

def extract_expectation_rows(checkpoint_result, batch_label: str) -> pd.DataFrame:
    """
    Flattens a CheckpointResult's per-expectation outcomes into a tidy
    DataFrame, handling both column-level expectations (which report
    unexpected_count / unexpected_percent) and table-level expectations
    (which report only an observed_value).
    """
    rows = []
    for validation_result in checkpoint_result.run_results.values():
        for expectation_result in validation_result.results:
            d = expectation_result.to_json_dict()
            config = d["expectation_config"]
            result = d.get("result", {}) or {}
            exception_info = d.get("exception_info", {}) or {}
            # Some expectations on a dropped column don't fail cleanly - they
            # raise an exception (e.g. "column does not exist") instead of
            # returning a result dict. Surface that message rather than
            # reporting an empty/uninformative result.
            exception_message = None
            if isinstance(exception_info, dict) and exception_info.get("raised_exception"):
                exception_message = exception_info.get("exception_message")
            elif isinstance(exception_info, dict):
                for v in exception_info.values():
                    if isinstance(v, dict) and v.get("raised_exception"):
                        exception_message = v.get("exception_message")
                        break
            rows.append(
                {
                    "batch": batch_label,
                    "expectation_type": config["type"],
                    "column": config["kwargs"].get("column", "(table-level)"),
                    "success": d["success"],
                    "observed_value": result.get("observed_value"),
                    "unexpected_count": result.get("unexpected_count"),
                    "unexpected_percent": result.get("unexpected_percent"),
                    "unexpected_percent_total": result.get("unexpected_percent_total"),
                    "partial_unexpected_list": result.get("partial_unexpected_list"),
                    "exception_message": exception_message,
                }
            )
    return pd.DataFrame(rows)

df_clean = extract_expectation_rows(results_clean, "Clean")
df_corrupt = extract_expectation_rows(results_corrupt, "Corrupted")
df_all = pd.concat([df_clean, df_corrupt], ignore_index=True)

print(f"Clean batch:     {df_clean['success'].sum()} / {len(df_clean)} expectations passed")
print(f"Corrupted batch: {df_corrupt['success'].sum()} / {len(df_corrupt)} expectations passed")

# Category grouping, used for the comparison chart in Section 9. Each
# expectation type is assigned to one of the four validation dimensions used
# throughout this project (Structural, Type, Content, Missingness).
CATEGORY_MAP = {
    "expect_table_row_count_to_be_between": "Structural",
    "expect_table_column_count_to_equal": "Structural",
    "expect_table_columns_to_match_set": "Structural",
    "expect_column_values_to_be_of_type": "Type",
    "expect_column_values_to_be_in_set": "Content",
    "expect_column_values_to_be_between": "Content",
    "expect_column_values_to_not_be_null": "Missingness",
    "expect_column_proportion_of_non_null_values_to_be_between": "Missingness",
}
df_all["category"] = df_all["expectation_type"].map(CATEGORY_MAP).fillna("Other")

# ---------------------------------------------------------------
# 9) Comparison chart: pass/fail rate by category, clean vs corrupted
# ---------------------------------------------------------------
print_header("9) Building the clean-vs-corrupted comparison chart")

summary = (
    df_all.groupby(["category", "batch"])["success"]
    .mean()
    .mul(100)
    .reset_index()
    .rename(columns={"success": "pass_rate_pct"})
)
print(summary.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(
    data=summary,
    x="category",
    y="pass_rate_pct",
    hue="batch",
    palette={"Clean": "#55A868", "Corrupted": "#C44E52"},
)
plt.title("Expectation Pass Rate by Category: Clean vs Corrupted Batch")
plt.xlabel("Validation Category")
plt.ylabel("Pass Rate (%)")
plt.ylim(0, 110)
plt.legend(title="Batch")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "plot_04_pass_rate_by_category.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 10) Failure-detail table for the corrupted batch
# ---------------------------------------------------------------
print_header("10) Failure-detail table (corrupted batch)")

failures = df_corrupt[~df_corrupt["success"]].copy()

def format_observed(row):
    exc = row.get("exception_message")
    if isinstance(exc, str) and exc:
        return f"error: {exc}"
    pct = row["unexpected_percent"]
    cnt = row["unexpected_count"]
    if pd.notna(pct) and pd.notna(cnt):
        return f"{pct:.1f}% unexpected (n={int(cnt)})"
    obs = row["observed_value"]
    if isinstance(obs, list):
        return f"observed columns = {obs}"
    if obs is not None and not (isinstance(obs, float) and pd.isna(obs)):
        return f"observed = {obs}"
    return "n/a"

failures["observed_summary"] = failures.apply(format_observed, axis=1)
failure_table = failures[["expectation_type", "column", "observed_summary"]].rename(
    columns={"expectation_type": "Expectation", "column": "Column", "observed_summary": "Observed Result"}
)

print(failure_table.to_string(index=False))

# Save as a markdown table for direct embedding in the portfolio page.
md_path = os.path.join(OUTPUT_DIR, "failure_detail_table.md")
with open(md_path, "w") as f:
    f.write("| Expectation | Column | Observed Result |\n")
    f.write("|---|---|---|\n")
    for _, row in failure_table.iterrows():
        f.write(f"| {row['Expectation']} | {row['Column']} | {row['Observed Result']} |\n")
print(f"\nMarkdown failure table saved to {md_path}")

# ---------------------------------------------------
# 11) Build Data Docs (both runs are now reflected in the site)
# ---------------------------------------------------
print_header("11) Building Data Docs")

docs_index = context.build_data_docs(site_names=[site_name])
print(docs_index)

print_header("Done")
print(f"Clean batch success:     {results_clean.success}")
print(f"Corrupted batch success: {results_corrupt.success}")
print(f"PNG outputs and failure table written to ./{OUTPUT_DIR}/")