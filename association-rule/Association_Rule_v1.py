# Association Rule Learning
# Portfolio demonstration of Association Rule Learning using the Apriori/FP-Growth algorithm on real transactional data.
# UCI Online Retail II (2010-2011) Data Set - https://www.kaggle.com/datasets/jillwang87/online-retail-ii


# 1. LIBRARY IMPORTS & CONFIGURATION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import re
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
# warnings.filterwarnings("ignore")

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

# Set Seaborn global theme
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PLOT_FIGSIZE = (12, 6)
OUTPUT_DIR   = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 2. DATA LOADING & INITIAL EXPLORATION

print("SECTION 2 — DATA LOADING & INITIAL EXPLORATION")

df_raw = pd.read_csv("online_retail_10_11.csv", encoding="latin-1")

print(f"\nRaw dataset shape : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print("\nColumn names      :", df_raw.columns.tolist())
print("\nData types:\n", df_raw.dtypes)
print("\nExample data:\n", df_raw)
print("\nBasic statistics:\n", df_raw.describe(include="all"))

# Missing value summary (note the NaN considered to be a value)
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print("\nMissing values:\n", missing_df[missing_df["Missing Count"] > 0])


# 3. DATA VALIDATION & PRE-PROCESSING

print("\nSECTION 3 — DATA VALIDATION & PRE-PROCESSING")

# Create new df and rename column relating to Invoice Number
df = df_raw.copy()
df = df.rename(columns={'ï»¿InvoiceNo': 'Invoice'})

# Step 1: Standardise column names
df.columns = df.columns.str.strip()

# Step 2: Remove cancelled orders (Invoice starts with 'C')
mask_cancelled = df["Invoice"].astype(str).str.startswith("C")
n_cancelled = mask_cancelled.sum()
df = df[~mask_cancelled]
print(f"\nCancelled orders removed       : {n_cancelled:,} rows")

# Step 3: Remove rows with missing CustomerID
n_missing_cust = df["CustomerID"].isnull().sum()
df = df.dropna(subset=["CustomerID"])
print(f"Missing CustomerID removed     : {n_missing_cust:,} rows")

# Step 4: Remove rows with negative or zero Quantity
n_neg_qty = (df["Quantity"] <= 0).sum()
df = df[df["Quantity"] > 0]
print(f"Negative/zero Quantity removed : {n_neg_qty:,} rows")

# Step 5: Remove rows with missing or zero Price
n_bad_price = (df["UnitPrice"].isnull() | (df["UnitPrice"] <= 0)).sum()
df = df[df["UnitPrice"].notnull() & (df["UnitPrice"] > 0)]
print(f"Bad Price rows removed         : {n_bad_price:,} rows")

# Step 6: Clean Description field
df["Description"] = df["Description"].astype(str).str.strip().str.upper()

# Step 7: Remove non-product stock codes
#    These codes represent postage, bank charges, samples etc.
non_product_codes = ["POST", "DOT", "AMAZONFEE", "M", "BANK CHARGES",
                     "CRUK", "PADS", "D", "C2", "S"]
n_non_product = df["StockCode"].astype(str).str.strip().str.upper().isin(non_product_codes).sum()
df = df[~df["StockCode"].astype(str).str.strip().str.upper().isin(non_product_codes)]
print(f"Non-product stock codes removed: {n_non_product:,} rows")

# Step 8: Keep only UK transactions for a focused analysis
n_non_uk = (df["Country"] != "United Kingdom").sum()
df_uk = df[df["Country"] == "United Kingdom"].copy()
print(f"Non-UK transactions excluded   : {n_non_uk:,} rows")

# Step 9: Remove single-item invoices
invoice_sizes = df_uk.groupby("Invoice")["StockCode"].nunique()
multi_item_invoices = invoice_sizes[invoice_sizes > 1].index
n_single = (~df_uk["Invoice"].isin(multi_item_invoices)).sum()
df_uk = df_uk[df_uk["Invoice"].isin(multi_item_invoices)]
print(f"[Step 8] Single-item invoices removed   : {n_single:,} rows")

print(f"\nCleaned dataset shape: {df_uk.shape[0]:,} rows × {df_uk.shape[1]} columns")
print(f"Unique invoices      : {df_uk['Invoice'].nunique():,}")
print(f"Unique products      : {df_uk['Description'].nunique():,}")
print(f"Unique customers     : {df_uk['CustomerID'].nunique():,}")


# # 4. EXPLORATORY DATA ANALYSIS

print("SECTION 4 — EXPLORATORY DATA ANALYSIS")

# Parse InvoiceDate and generate Revenue Column
df_uk["InvoiceDate"] = pd.to_datetime(df_uk["InvoiceDate"])
df_uk["Month"]       = df_uk["InvoiceDate"].dt.to_period("M").astype(str)
df_uk["DayOfWeek"]   = df_uk["InvoiceDate"].dt.day_name()
df_uk["Hour"]        = df_uk["InvoiceDate"].dt.hour
df_uk["Revenue"]     = df_uk["Quantity"] * df_uk["UnitPrice"]

print(df_uk)

# Plot 1: Top 20 Best-Selling Products
top_products = (df_uk.groupby("Description")["Quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index())

fig, ax = plt.subplots(figsize=(12, 7))
sns.barplot(data=top_products, y="Description", x="Quantity", ax=ax, palette="Greens_r", hue='Description')
ax.set_title("Top 20 Best-Selling Products by Quantity", fontsize=14, fontweight="bold")
ax.set_xlabel("Total Units Sold")
ax.set_ylabel("")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
for container in ax.containers:
    ax.bar_label(container, fmt=lambda x: f"{int(x):,}", padding=5, fontsize=9)
plt.tight_layout()
plt.savefig("01_top20_products.png", dpi=150)
plt.show()

# Plot 2: Monthly Transaction Volume
monthly = (df_uk.groupby("Month")["Invoice"]
           .nunique()
           .reset_index()
           .rename(columns={"Invoice": "Transactions"}))

fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
sns.lineplot(data=monthly, x="Month", y="Transactions", marker="o", color="Seagreen", ax=ax)
ax.set_title("Monthly Transaction Volume (UK)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Number of Invoices")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("02_monthly_transactions.png", dpi=150)
plt.show()

# # ── Plot 3: Transactions by Day of Week ──────────────────────
# dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# dow = (df_uk.groupby("DayOfWeek")["Invoice"]
#        .nunique()
#        .reindex(dow_order)
#        .reset_index()
#        .rename(columns={"Invoice": "Transactions"}))
#
# fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
# sns.barplot(data=dow, x="DayOfWeek", y="Transactions", palette="muted", ax=ax)
# ax.set_title("Transactions by Day of Week (UK)", fontsize=14, fontweight="bold")
# ax.set_xlabel("Day of Week")
# ax.set_ylabel("Number of Invoices")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/03_transactions_by_day.png", dpi=150)
# plt.show()
# print("[Chart 3 saved] Transactions by Day of Week")
#
# # ── Plot 4: Revenue by Month ──────────────────────────────────
# monthly_rev = (df_uk.groupby("Month")["Revenue"]
#                .sum()
#                .reset_index()
#                .rename(columns={"Revenue": "Total Revenue"}))
#
# fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
# sns.barplot(data=monthly_rev, x="Month", y="Total Revenue", palette="Blues_d", ax=ax)
# ax.set_title("Monthly Revenue (UK)", fontsize=14, fontweight="bold")
# ax.set_xlabel("Month")
# ax.set_ylabel("Revenue (£)")
# ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/04_monthly_revenue.png", dpi=150)
# plt.show()
# print("[Chart 4 saved] Monthly Revenue")
#
# # ── Plot 5: Basket Size Distribution ─────────────────────────
# basket_sizes = df_uk.groupby("Invoice")["StockCode"].nunique()
#
# fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
# sns.histplot(basket_sizes[basket_sizes <= 30], bins=29, kde=True, color="steelblue", ax=ax)
# ax.set_title("Distribution of Basket Size (up to 30 items, UK)", fontsize=14, fontweight="bold")
# ax.set_xlabel("Number of Unique Products per Invoice")
# ax.set_ylabel("Frequency")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/05_basket_size_distribution.png", dpi=150)
# plt.show()
# print("[Chart 5 saved] Basket Size Distribution")
# print(f"\nBasket size stats:\n{basket_sizes.describe().round(2)}")
#
#
# # =============================================================
# # 5. BASKET CONSTRUCTION
# # =============================================================
#
# print("\n" + "=" * 65)
# print("SECTION 5 — BASKET CONSTRUCTION")
# print("=" * 65)
#
# # Build one-hot encoded basket matrix:
# # Rows = Invoices, Columns = Products, Values = 1/0
# basket = (df_uk.groupby(["Invoice", "Description"])["Quantity"]
#           .sum()
#           .unstack(fill_value=0))
#
# # Binarise: any quantity > 0 becomes True
# basket_encoded = basket.applymap(lambda x: True if x > 0 else False)
#
# print(f"\nBasket matrix shape : {basket_encoded.shape[0]:,} invoices × {basket_encoded.shape[1]:,} products")
# print(f"Matrix density      : {basket_encoded.values.mean():.4f} (proportion of True values)")
#
#
# # =============================================================
# # 6. FREQUENT ITEMSET MINING (FP-GROWTH)
# # =============================================================
#
# print("\n" + "=" * 65)
# print("SECTION 6 — FREQUENT ITEMSET MINING")
# print("=" * 65)
#
# # ── Min support selection ─────────────────────────────────────
# # Support = proportion of transactions containing the itemset.
# # Too high → too few rules; too low → millions of trivial rules.
# # 0.02 (2%) is a pragmatic starting point for this dataset.
# MIN_SUPPORT    = 0.02
# MIN_CONFIDENCE = 0.20
# MIN_LIFT       = 1.5
#
# print(f"\nParameters:")
# print(f"  Minimum Support    : {MIN_SUPPORT}  ({MIN_SUPPORT*100:.0f}% of transactions)")
# print(f"  Minimum Confidence : {MIN_CONFIDENCE}")
# print(f"  Minimum Lift       : {MIN_LIFT}")
#
# frequent_itemsets = fpgrowth(basket_encoded,
#                              min_support=MIN_SUPPORT,
#                              use_colnames=True)
#
# frequent_itemsets["itemset_size"] = frequent_itemsets["itemsets"].apply(len)
#
# print(f"\nFrequent itemsets found : {len(frequent_itemsets):,}")
# print("\nItemset size distribution:")
# print(frequent_itemsets["itemset_size"].value_counts().sort_index())
# print("\nTop 10 most frequent itemsets:")
# print(frequent_itemsets.sort_values("support", ascending=False)
#       .head(10)
#       .to_string(index=False))
#
# # ── Plot 6: Support Distribution of Frequent Itemsets ────────
# fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
# sns.histplot(frequent_itemsets["support"], bins=40, kde=True, color="teal", ax=ax)
# ax.set_title("Distribution of Support Across Frequent Itemsets", fontsize=14, fontweight="bold")
# ax.set_xlabel("Support")
# ax.set_ylabel("Count")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/06_support_distribution.png", dpi=150)
# plt.show()
# print("[Chart 6 saved] Support Distribution")
#
#
# # =============================================================
# # 7. ASSOCIATION RULE GENERATION
# # =============================================================
#
# print("\n" + "=" * 65)
# print("SECTION 7 — ASSOCIATION RULE GENERATION")
# print("=" * 65)
#
# rules = association_rules(frequent_itemsets,
#                           metric="lift",
#                           min_threshold=MIN_LIFT)
#
# # Apply confidence filter
# rules = rules[rules["confidence"] >= MIN_CONFIDENCE].copy()
#
# # Convert frozensets to readable strings for display
# rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
# rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
#
# print(f"\nTotal rules generated : {len(rules):,}")
# print(f"\nRules metrics summary:")
# print(rules[["support", "confidence", "lift", "leverage", "conviction"]].describe().round(4))
#
# print("\nTop 15 rules by Lift:")
# print(rules.sort_values("lift", ascending=False)
#       [["antecedents_str", "consequents_str", "support", "confidence", "lift"]]
#       .head(15)
#       .to_string(index=False))
#
#
# # =============================================================
# # 8. MODEL VALIDATION
# # =============================================================
#
# print("\n" + "=" * 65)
# print("SECTION 8 — MODEL VALIDATION")
# print("=" * 65)
#
# # ── Validation 1: Trivial rule check ─────────────────────────
# # A rule is considered trivial if lift ≈ 1, meaning no real
# # association beyond chance.  We already filtered lift >= 1.5
# # so this confirms quality.
# low_lift = (rules["lift"] < 1.5).sum()
# print(f"\n[V1] Rules with lift < 1.5 (trivial)      : {low_lift}")
# print(f"     Rules with lift >= 1.5 (meaningful)   : {(rules['lift'] >= 1.5).sum()}")
#
# # ── Validation 2: Leverage check ─────────────────────────────
# # Leverage > 0 confirms co-occurrence is above what would be
# # expected if items were independent.
# neg_leverage = (rules["leverage"] <= 0).sum()
# print(f"\n[V2] Rules with negative leverage (spurious): {neg_leverage}")
# print(f"     Rules with positive leverage (genuine) : {(rules['leverage'] > 0).sum()}")
#
# # ── Validation 3: Conviction ──────────────────────────────────
# # Conviction > 1 means the rule is directionally meaningful.
# # Conviction of infinity means the rule is perfectly reliable.
# low_conviction = (rules["conviction"] < 1.0).sum()
# print(f"\n[V3] Rules with conviction < 1.0 (weak)   : {low_conviction}")
# print(f"     Rules with conviction >= 1.0 (strong) : {(rules['conviction'] >= 1.0).sum()}")
#
# # ── Validation 4: Rule stability via support range ────────────
# # Rules with very low support may not be stable across time.
# # We flag rules where support < 2.5% as potentially unstable.
# low_support = (rules["support"] < 0.025).sum()
# print(f"\n[V4] Rules with support < 2.5% (less stable): {low_support}")
# print(f"     Rules with support >= 2.5% (stable)    : {(rules['support'] >= 0.025).sum()}")
#
# # ── Validation 5: Reciprocal rule check ──────────────────────
# # Strong associations often appear in both directions.
# # We check how many rules have a corresponding reverse rule,
# # confirming bidirectional co-purchase behaviour.
# rule_pairs = set()
# reciprocal_count = 0
# for _, row in rules.iterrows():
#     pair = (row["antecedents_str"], row["consequents_str"])
#     reverse = (row["consequents_str"], row["antecedents_str"])
#     if reverse in rule_pairs:
#         reciprocal_count += 1
#     rule_pairs.add(pair)
#
# print(f"\n[V5] Reciprocal rule pairs found           : {reciprocal_count}")
# print("     (Bidirectional rules confirm robust product associations)")
#
# # ── Validation 6: Parameter sensitivity analysis ─────────────
# # Test what happens to rule count across a range of min_support
# print("\n[V6] Rule count sensitivity to min_support threshold:")
# print(f"     {'Min Support':>12} | {'Itemsets':>10} | {'Rules (lift>1.5, conf>0.2)':>26}")
# print("     " + "-" * 55)
# for ms in [0.01, 0.015, 0.02, 0.025, 0.03]:
#     fi_temp = fpgrowth(basket_encoded, min_support=ms, use_colnames=True)
#     r_temp  = association_rules(fi_temp, metric="lift", min_threshold=1.5)
#     r_temp  = r_temp[r_temp["confidence"] >= 0.20]
#     print(f"     {ms:>12.3f} | {len(fi_temp):>10,} | {len(r_temp):>26,}")
#
# # ── Plot 7: Lift vs Confidence scatter ────────────────────────
# fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
# scatter = ax.scatter(rules["confidence"], rules["lift"],
#                      c=rules["support"], cmap="YlOrRd",
#                      alpha=0.7, edgecolors="grey", linewidth=0.3, s=60)
# cbar = plt.colorbar(scatter, ax=ax)
# cbar.set_label("Support")
# ax.set_title("Lift vs Confidence (colour = Support)", fontsize=14, fontweight="bold")
# ax.set_xlabel("Confidence")
# ax.set_ylabel("Lift")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/07_lift_vs_confidence.png", dpi=150)
# plt.show()
# print("\n[Chart 7 saved] Lift vs Confidence Scatter")
#
# # ── Plot 8: Support vs Confidence scatter ────────────────────
# fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
# scatter2 = ax.scatter(rules["support"], rules["confidence"],
#                       c=rules["lift"], cmap="coolwarm",
#                       alpha=0.7, edgecolors="grey", linewidth=0.3, s=60)
# cbar2 = plt.colorbar(scatter2, ax=ax)
# cbar2.set_label("Lift")
# ax.set_title("Support vs Confidence (colour = Lift)", fontsize=14, fontweight="bold")
# ax.set_xlabel("Support")
# ax.set_ylabel("Confidence")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/08_support_vs_confidence.png", dpi=150)
# plt.show()
# print("[Chart 8 saved] Support vs Confidence Scatter")
#
#
# # =============================================================
# # 9. BUSINESS INSIGHT EXTRACTION & VISUALISATION
# # =============================================================
#
# print("\n" + "=" * 65)
# print("SECTION 9 — BUSINESS INSIGHT EXTRACTION")
# print("=" * 65)
#
# # ── Top rules by Lift ─────────────────────────────────────────
# top_lift = (rules.sort_values("lift", ascending=False)
#             .drop_duplicates(subset=["antecedents_str", "consequents_str"])
#             .head(20))
#
# # ── Plot 9: Top 20 Rules by Lift ──────────────────────────────
# top_lift["rule_label"] = (top_lift["antecedents_str"].str[:30]
#                           + "  →  "
#                           + top_lift["consequents_str"].str[:30])
#
# fig, ax = plt.subplots(figsize=(13, 8))
# sns.barplot(data=top_lift, y="rule_label", x="lift", palette="Reds_d", ax=ax)
# ax.set_title("Top 20 Association Rules by Lift", fontsize=14, fontweight="bold")
# ax.set_xlabel("Lift")
# ax.set_ylabel("")
# ax.axvline(x=1, color="grey", linestyle="--", linewidth=1, label="Lift = 1 (random)")
# ax.legend()
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/09_top20_rules_by_lift.png", dpi=150)
# plt.show()
# print("[Chart 9 saved] Top 20 Rules by Lift")
#
# # ── Top rules by Confidence ───────────────────────────────────
# top_conf = (rules.sort_values("confidence", ascending=False)
#             .drop_duplicates(subset=["antecedents_str", "consequents_str"])
#             .head(20))
#
# top_conf["rule_label"] = (top_conf["antecedents_str"].str[:30]
#                           + "  →  "
#                           + top_conf["consequents_str"].str[:30])
#
# # ── Plot 10: Top 20 Rules by Confidence ──────────────────────
# fig, ax = plt.subplots(figsize=(13, 8))
# sns.barplot(data=top_conf, y="rule_label", x="confidence", palette="Blues_d", ax=ax)
# ax.set_title("Top 20 Association Rules by Confidence", fontsize=14, fontweight="bold")
# ax.set_xlabel("Confidence")
# ax.set_ylabel("")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/10_top20_rules_by_confidence.png", dpi=150)
# plt.show()
# print("[Chart 10 saved] Top 20 Rules by Confidence")
#
# # ── Plot 11: Lift Distribution ────────────────────────────────
# fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
# sns.histplot(rules["lift"], bins=40, kde=True, color="coral", ax=ax)
# ax.set_title("Distribution of Lift Across All Rules", fontsize=14, fontweight="bold")
# ax.set_xlabel("Lift")
# ax.set_ylabel("Count")
# ax.axvline(x=1, color="grey", linestyle="--", linewidth=1.2, label="Lift = 1 (baseline)")
# ax.legend()
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/11_lift_distribution.png", dpi=150)
# plt.show()
# print("[Chart 11 saved] Lift Distribution")
#
# # ── Plot 12: Top 15 most frequently appearing antecedents ─────
# antecedent_freq = (rules["antecedents_str"]
#                    .value_counts()
#                    .head(15)
#                    .reset_index()
#                    .rename(columns={"index": "Antecedent", "antecedents_str": "Count"}))
# # Handle pandas version differences in value_counts column naming
# antecedent_freq.columns = ["Antecedent", "Count"]
#
# fig, ax = plt.subplots(figsize=(12, 7))
# sns.barplot(data=antecedent_freq, y="Antecedent", x="Count", palette="Greens_d", ax=ax)
# ax.set_title("Top 15 Most Frequent Antecedents in Rules", fontsize=14, fontweight="bold")
# ax.set_xlabel("Number of Rules as Antecedent")
# ax.set_ylabel("")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/12_top_antecedents.png", dpi=150)
# plt.show()
# print("[Chart 12 saved] Top Antecedents")
#
# # ── Plot 13: Heatmap — top products co-occurrence ─────────────
# # Select the top 15 products by appearance in rules
# top_items = pd.Series(
#     [item for itemset in frequent_itemsets["itemsets"] for item in itemset]
# ).value_counts().head(15).index.tolist()
#
# # Build co-occurrence matrix
# co_matrix = pd.DataFrame(0, index=top_items, columns=top_items)
# for _, row in rules.iterrows():
#     ants = list(row["antecedents"])
#     cons = list(row["consequents"])
#     for a in ants:
#         for c in cons:
#             if a in top_items and c in top_items:
#                 co_matrix.loc[a, c] += 1
#
# # Shorten labels for readability
# short_labels = [x[:28] for x in top_items]
# co_matrix.index   = short_labels
# co_matrix.columns = short_labels
#
# fig, ax = plt.subplots(figsize=(13, 10))
# sns.heatmap(co_matrix, annot=True, fmt="d", cmap="YlOrRd",
#             linewidths=0.4, ax=ax, cbar_kws={"label": "Rule Count"})
# ax.set_title("Product Co-occurrence Heatmap (Top 15 Products)", fontsize=14, fontweight="bold")
# plt.xticks(rotation=45, ha="right")
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/13_cooccurrence_heatmap.png", dpi=150)
# plt.show()
# print("[Chart 13 saved] Co-occurrence Heatmap")
#
#
# # =============================================================
# # 10. SUMMARY REPORT
# # =============================================================
#
# print("\n" + "=" * 65)
# print("SECTION 10 — SUMMARY REPORT & BUSINESS INSIGHTS")
# print("=" * 65)
#
# print(f"""
# ╔══════════════════════════════════════════════════════════════╗
# ║          ASSOCIATION RULE ANALYSIS — SUMMARY REPORT         ║
# ╚══════════════════════════════════════════════════════════════╝
#
# DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────
#   Source       : Online Retail II — UK Transactions 2010-2011
#   Clean rows   : {df_uk.shape[0]:,}
#   Invoices     : {df_uk['Invoice'].nunique():,}
#   Products     : {df_uk['Description'].nunique():,}
#   Customers    : {df_uk['CustomerID'].nunique():,}
#
# PRE-PROCESSING STEPS APPLIED
# ─────────────────────────────────────────────────────────────
#   ✔  Cancelled orders (invoice prefix 'C') removed
#   ✔  Rows with missing CustomerID removed
#   ✔  Rows with negative or zero Quantity removed
#   ✔  Rows with missing or zero Price removed
#   ✔  Non-product stock codes removed (POST, DOT, etc.)
#   ✔  Non-UK transactions excluded for focused analysis
#   ✔  Single-item invoices removed (no basket to analyse)
#
# MODELLING PARAMETERS
# ─────────────────────────────────────────────────────────────
#   Algorithm        : FP-Growth
#   Min Support      : {MIN_SUPPORT} ({MIN_SUPPORT*100:.0f}%)
#   Min Confidence   : {MIN_CONFIDENCE}
#   Min Lift         : {MIN_LIFT}
#
# RESULTS
# ─────────────────────────────────────────────────────────────
#   Frequent itemsets : {len(frequent_itemsets):,}
#   Rules generated   : {len(rules):,}
#   Avg Lift          : {rules['lift'].mean():.2f}
#   Max Lift          : {rules['lift'].max():.2f}
#   Avg Confidence    : {rules['confidence'].mean():.2f}
#   Max Confidence    : {rules['confidence'].max():.2f}
#
# TOP 5 RULES BY LIFT
# ─────────────────────────────────────────────────────────────""")
#
# for i, row in rules.sort_values("lift", ascending=False).head(5).iterrows():
#     print(f"  IF   [{row['antecedents_str'][:55]}]")
#     print(f"  THEN [{row['consequents_str'][:55]}]")
#     print(f"       Support={row['support']:.3f}  Confidence={row['confidence']:.3f}  Lift={row['lift']:.2f}")
#     print()
#
# print("""
# BUSINESS INSIGHT INTERPRETATION
# ─────────────────────────────────────────────────────────────
#
#   LIFT
#   ────
#   Lift measures how much more likely two products are to be
#   purchased together compared to by chance.  A lift of 4.0
#   means customers are 4x more likely to buy both items than
#   if purchase decisions were independent.  Rules with high
#   lift are the strongest candidates for:
#     • Cross-sell recommendations at checkout
#     • 'Frequently bought together' website features
#     • Bundle promotions and targeted discount offers
#
#   CONFIDENCE
#   ──────────
#   Confidence is the conditional probability: given a customer
#   bought the antecedent, how often did they also buy the
#   consequent?  A confidence of 0.75 means 75% of baskets
#   containing the antecedent also contained the consequent.
#   High-confidence rules are actionable for:
#     • Next-best-product recommendation engines
#     • Email campaigns triggered by recent purchases
#     • Staff training on upsell opportunities
#
#   SUPPORT
#   ───────
#   Support measures how frequently a rule fires across all
#   transactions.  Rules with both high lift AND reasonable
#   support are the most commercially valuable — the
#   association is strong AND affects a meaningful share of
#   customers.  Rules with very low support, even with high
#   lift, may reflect niche behaviour not worth operationalising
#   at scale.
#
#   LEVERAGE & CONVICTION
#   ─────────────────────
#   Leverage confirms the co-occurrence is above chance.
#   Conviction measures the degree to which the antecedent
#   implies the consequent directionally.  Both metrics were
#   used in validation to confirm rule quality beyond lift alone.
#
#   PRACTICAL APPLICATIONS
#   ──────────────────────
#   1. Website product recommendations — surface consequent
#      products on the basket/checkout page when antecedent
#      items are detected.
#   2. Promotional bundling — create bundle offers based on
#      top high-lift, high-confidence pairs to increase
#      average order value.
#   3. Store layout / category management — place strongly
#      associated product categories in proximity.
#   4. Inventory planning — stock correlated products at
#      similar levels to avoid one driving demand for the
#      other when out of stock.
#   5. Targeted email marketing — send personalised product
#      suggestions based on a customer's last purchase.
#
# CHARTS SAVED TO: ./charts/
# ─────────────────────────────────────────────────────────────
#   01_top20_products.png
#   02_monthly_transactions.png
#   03_transactions_by_day.png
#   04_monthly_revenue.png
#   05_basket_size_distribution.png
#   06_support_distribution.png
#   07_lift_vs_confidence.png
#   08_support_vs_confidence.png
#   09_top20_rules_by_lift.png
#   10_top20_rules_by_confidence.png
#   11_lift_distribution.png
#   12_top_antecedents.png
#   13_cooccurrence_heatmap.png
# ══════════════════════════════════════════════════════════════
# """)

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")