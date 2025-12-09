
# Demonstrates Great Expectations (v1.9.2) with the Seaborn Titanic dataset.
# It builds an ingestion-stage expectation suite, runs validations via Checkpoints
# Generates static HTML Data Docs, and produces Seaborn charts (no sub-plots).
#
# Outputs:
#   - Great Expectations Data Docs (static HTML)
#   - PNG charts in ./outputs/

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import great_expectations as gx
from great_expectations import expectations as gxe
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import UpdateDataDocsAction

# -----------------------------
# 0) Settings and utilities
# -----------------------------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

# -----------------------------
# 1) Load freely-available data
# -----------------------------
print_header("1) Loading the Seaborn Titanic dataset")
titanic = sns.load_dataset("titanic")  # 891 rows, 15 columns (classic demo dataset)
print("Shape:", titanic.shape)
print("Columns:", list(titanic.columns))
print('\nExample dataset:')
print(titanic)

# ---------------------------------------
# 2) Exploratory visuals (no sub-plots)
# ---------------------------------------
print_header("2) Creating Seaborn visuals (no sub-plots)")
# 2a. % Missing values by column (bar chart)
missing_pct = titanic.isna().mean().sort_values(ascending=False).reset_index()
missing_pct.columns = ["column", "missing_fraction"]

plt.figure(figsize=(10, 4))
sns.barplot(data=missing_pct, x="column", y="missing_fraction", color="#4C72B0")
plt.title("Titanic: Fraction of Missing Values by Column")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Missing Fraction")
plt.xlabel("Column")
plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "missing_fraction_by_column.png"), dpi=150)

# 2b. Survival by sex (count plot)
plt.figure(figsize=(6, 4))
sns.countplot(data=titanic, x="sex", hue="survived", palette="Set2")
plt.title("Survival by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "survival_by_sex.png"), dpi=150)

# 2c. Age distribution (histogram + KDE)
plt.figure(figsize=(6, 4))
sns.histplot(data=titanic, x="age", kde=True, color="#55A868", bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
# plt.savefig(os.path.join(OUTPUT_DIR, "age_distribution.png"), dpi=150)
# plt.show()

# ---------------------------------------------------
# 3) Initialize Great Expectations Data Context
# ---------------------------------------------------
print_header("3) Initializing Great Expectations context")


# 1) Create/resolve a File Data Context (persists configs & Data Docs)
context = gx.get_context(mode="file")  # creates ./great_expectations if not present

# 2) Add a pandas Fluent Data Source
try:
    pandas_ds = context.data_sources.get("titanic_runtime")  # reuse name if you like
except Exception:
    pandas_ds = context.data_sources.add_pandas(name="titanic_runtime")

# 3) Register a DataFrame Data Asset for ingestion-stage data
try:
    ingestion_asset = pandas_ds.get_asset("titanic_ingestion_df")
except Exception:
    ingestion_asset = pandas_ds.add_dataframe_asset(name="titanic_ingestion_df")

# 4) Define how batches are created for this asset
try:
    ingestion_bd = ingestion_asset.get_batch_definition("ingestion_whole_df")
except Exception:
    ingestion_bd = ingestion_asset.add_batch_definition_whole_dataframe("ingestion_whole_df")

# 5) Materialize a Batch by passing the in-memory DataFrame at runtime
ingestion_batch = ingestion_bd.get_batch(batch_parameters={"dataframe": titanic})


# -----------------------------------------------------------------
# 4) Define ingestion-stage expectation suite (structure + content)
# -----------------------------------------------------------------
print_header("4) Defining ingestion-stage expectations")

ingestion_suite_name = "titanic_ingestion_suite"

# 1) Create (or update) the suite using the 1.x Suites API
try:
    ingestion_suite = context.suites.get(ingestion_suite_name)
except Exception:
    ingestion_suite = context.suites.add(gx.ExpectationSuite(name=ingestion_suite_name))

# -------------------------
# Structural checks
# -------------------------
ingestion_suite.add_expectation(
    gxe.ExpectTableRowCountToBeBetween(min_value=800, max_value=1000)
)
ingestion_suite.add_expectation(
    gxe.ExpectTableColumnCountToEqual(value=15)
)
ingestion_suite.add_expectation(
    gxe.ExpectTableColumnsToMatchSet(
        column_set=[
            "survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked",
            "class", "who", "adult_male", "deck", "embark_town", "alive", "alone"
        ],
        exact_match=True,
    )
)

# -------------------------
# Column types
# (Note: strict dtype checking on CSV/DFs can vary across environments;
# if you hit false negatives, consider relaxing or pre-casting types.)
# -------------------------
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="survived", type_="int64"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="pclass",   type_="int64"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="sex",      type_="object"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="age",      type_="float64"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="sibsp",    type_="int64"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="parch",    type_="int64"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="fare",     type_="float64"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="embarked", type_="object"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="adult_male", type_="bool"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="alone",      type_="bool"))

# -------------------------
# Content & domain checks
# -------------------------
ingestion_suite.add_expectation(
    gxe.ExpectColumnValuesToBeInSet(column="survived", value_set=[0, 1])
)
ingestion_suite.add_expectation(
    gxe.ExpectColumnValuesToBeInSet(column="pclass", value_set=[1, 2, 3])
)
ingestion_suite.add_expectation(
    gxe.ExpectColumnValuesToBeInSet(column="sex", value_set=["male", "female"])
)
ingestion_suite.add_expectation(
    gxe.ExpectColumnValuesToBeInSet(column="embarked", value_set=["C", "Q", "S"], mostly=0.95)
)
ingestion_suite.add_expectation(
    gxe.ExpectColumnValuesToBeBetween(column="age", min_value=0, max_value=80, mostly=0.99)
)
ingestion_suite.add_expectation(
    gxe.ExpectColumnValuesToBeBetween(column="fare", min_value=0, max_value=600, mostly=0.99)
)

# -------------------------
# Missingness tolerances
# -------------------------
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="survived"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="pclass"))
ingestion_suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="sex"))
ingestion_suite.add_expectation(gxe.ExpectColumnProportionOfNonNullValuesToBeBetween(column="age", min_value=0.85))
ingestion_suite.add_expectation(
    gxe.ExpectColumnProportionOfNonNullValuesToBeBetween(column="deck", min_value=0.85)
)

# Persist the suite (idempotent)
context.suites.add_or_update(ingestion_suite)

# -------------------------
# Bind suite to data & run (GX 1.x)
# -------------------------
vd_ingestion = gx.ValidationDefinition(
    name="vd_titanic_ingestion",
    data=ingestion_bd,          # <-- Batch Definition for your raw titanic DataFrame
    suite=ingestion_suite,
)


# --- Ensure the Data Docs site exists BEFORE the checkpoint runs ---
site_name = "local_site"
site_config = {
    "class_name": "SiteBuilder",
    "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
    "store_backend": {
        "class_name": "TupleFilesystemStoreBackend",
        "base_directory": "data_docs/local_site/",
    },
}
try:
    context.add_data_docs_site(site_name=site_name, site_config=site_config)
except Exception:
    # Already exists; that's fine
    pass


from great_expectations.checkpoint import UpdateDataDocsAction
checkpoint = gx.Checkpoint(
    name="titanic_ingestion_checkpoint",
    validation_definitions=[vd_ingestion],
    actions=[UpdateDataDocsAction(name="update_docs", site_names=["local_site"])],
)


# Register the ValidationDefinition and the Checkpoint on the Data Context (GX 1.x)
vd_ingestion  = context.validation_definitions.add_or_update(vd_ingestion)     # or add_or_update(...)
checkpoint    = context.checkpoints.add_or_update(checkpoint)                  # or add_or_update(...)

results = checkpoint.run(batch_parameters={"dataframe": titanic})

# (Optional) build Data Docs explicitly and get index path(s)
docs_index = context.build_data_docs(site_names=["local_site"])
print(docs_index)
