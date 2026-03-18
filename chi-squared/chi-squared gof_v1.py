import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# 0. GLOBAL SETTINGS
RANDOM_SEED = 42
N_DEFECTS = 1200
SIGNIFICANCE_LEVEL = 0.05

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "axes.titlesize": 13,
                     "axes.labelsize": 11})

# 1. SIMULATE BIASED PRODUCTION STATION DEFECTS
# True (hidden) probability distribution:
#   Station 6 is biased upward;  stations 1–5 are slightly suppressed.
#   Probabilities sum exactly to 1.0.
TRUE_PROBS = np.array([0.155, 0.158, 0.160, 0.162, 0.165, 0.200])
STATIONS = np.arange(1, 7)

assert np.isclose(TRUE_PROBS.sum(), 1.0), "Probabilities must sum to 1."

rng = np.random.default_rng(RANDOM_SEED)
defects = rng.choice(STATIONS, size=N_DEFECTS, p=TRUE_PROBS)

df = pd.DataFrame({"station": defects})

print("=" * 60)
print("CHI-SQUARED GOODNESS-OF-FIT: PRODUCTION LINE DEFECT ANALYSIS")
print("=" * 60)

# 2. DATA VALIDATION
print("\nDATA VALIDATION")

# 2a. Shape and type
print(f"  Total defects recorded  : {len(df):,}")
print(f"  Column dtype          : {df['station'].dtype}")

# 2b. Check for missing values
n_missing = df['station'].isna().sum()
print(f"  Missing values        : {n_missing}")
if n_missing > 0:
    raise ValueError("Missing values detected — review data generation.")

# 2c. Check all stations are valid integers in [1, 6]
invalid = df[~df['station'].isin(STATIONS)]
print(f"  Invalid station values   : {len(invalid)}")
if len(invalid) > 0:
    raise ValueError(f"Invalid station values found: {invalid['station'].unique()}")

# 2d. Check all six stations are represented
observed_stations = set(df['station'].unique())
expected_stations = set(STATIONS)
missing_stations = expected_stations - observed_stations
print(f"  Stations represented     : {sorted(observed_stations)}")
if missing_stations:
    raise ValueError(f"Stations not represented in data: {missing_stations}")

# 2e. Minimum expected frequency check (chi-sq assumption: each cell >= 5)
expected_counts = np.full(6, N_DEFECTS / 6)
min_expected = expected_counts.min()
print(f"  Min expected frequency: {min_expected:.1f}  "
      f"({'✓ OK' if min_expected >= 5 else '✗ FAIL — chi-sq assumption violated'})")

print("  All validation checks passed.\n")

# 3. DESCRIPTIVE STATISTICS
print("\nDESCRIPTIVE STATISTICS")

observed_counts = df['station'].value_counts().sort_index()
observed_props  = observed_counts / N_DEFECTS
expected_props  = pd.Series(np.full(6, 1 / 6), index=STATIONS)

desc = pd.DataFrame({
    "Observed Count" : observed_counts,
    "Observed %"     : (observed_props * 100).round(2),
    "Expected %"     : (expected_props  * 100).round(2),
    "Deviation (pp)" : ((observed_props - expected_props) * 100).round(2)
})
print(desc.to_string())
print(f"\n  Mean station value (uniform = 3.5): {df['station'].mean():.4f}")
print(f"  Std dev                         : {df['station'].std():.4f}")
print(f"  Skewness                        : {df['station'].skew():.4f}")

# 4. VISUALISATION — CHART 1: Observed vs Expected Counts (Bar Chart)

obs_vals = observed_counts.values.astype(float)
exp_vals = expected_counts

bar_df = pd.DataFrame({
    "Station"   : np.tile(STATIONS, 2),
    "Count"  : np.concatenate([obs_vals, exp_vals]),
    "Type"   : ["Observed"] * 6 + ["Expected (Uniform Distribution)"] * 6
})

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=bar_df, x="Station", y="Count", hue="Type",
    palette={"Observed": "seagreen", "Expected (Uniform Distribution)": "steelblue"},
    errorbar=None, ax=ax
)
ax.axhline(N_DEFECTS / 6, color="#E84855", linewidth=1.2,
           linestyle="--", alpha=0.6)
ax.set_title("Observed vs Expected Defect Counts — Production Line", pad=12)
ax.set_xlabel("Production Station")
ax.set_ylabel("Count")
ax.legend(title="Distribution", frameon=True, loc="lower right")
for bar in ax.patches:
    h = bar.get_height()
    ax.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color="dimgray")
plt.tight_layout()
plt.savefig("chart1_observed_vs_expected.png", dpi=150)
plt.show()

# 5. VISUALISATION — CHART 2: Percentage Deviation from Expected

dev_df = pd.DataFrame({
    "Station"          : STATIONS,
    "Deviation (pp)": ((observed_props.values - 1/6) * 100)
})

dev_df["Colour"] = dev_df["Deviation (pp)"].apply(
    lambda d: "Above Expected" if d > 0 else "Below Expected"
)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=dev_df, x="Station", y="Deviation (pp)", hue="Colour",
    palette={"Above Expected": "seagreen", "Below Expected": "darkred"},
    errorbar=None, legend=False, ax=ax
)
for bar in ax.patches:
    h = bar.get_height()
    va = "bottom" if h >= 0 else "top"
    offset = 4 if h >= 0 else -4
    ax.annotate(f"{h:+.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, offset), textcoords="offset points",
                ha="center", va=va, fontsize=8, color="dimgray")
ax.axhline(0, color="black", linewidth=1.0, linestyle="-")
ax.set_title("Deviation from Expected Proportion per Station (percentage points)",
             pad=12)
ax.set_xlabel("Production Station")
ax.set_ylabel("Deviation from Expected (pp)")
plt.tight_layout()
plt.savefig("chart2_deviation_from_expected.png", dpi=150)
plt.show()

# 6. VISUALISATION — CHART 3: Cumulative Proportion by Station

cumulative_obs = observed_props.cumsum().values
cumulative_exp = expected_props.cumsum().values

cum_df = pd.DataFrame({
    "Station"       : np.tile(STATIONS, 2),
    "Cumulative %" : np.concatenate([cumulative_obs * 100, cumulative_exp * 100]),
    "Type"       : ["Observed"] * 6 + ["Expected (Uniform Distribution)"] * 6
})

fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(
    data=cum_df, x="Station", y="Cumulative %", hue="Type",
    palette={"Observed": "seagreen", "Expected (Uniform Distribution)": "steelblue"},
    marker="o", linewidth=2, ax=ax
)
ax.set_xticks(STATIONS)
ax.set_title("Cumulative Proportion of Defects by Station", pad=12)
ax.set_xlabel("Production Station")
ax.set_ylabel("Cumulative Proportion (%)")
ax.legend(title="Distribution", frameon=True)
plt.tight_layout()
plt.savefig("chart3_cumulative_proportion.png", dpi=150)
plt.show()

# 7. CHI-SQUARED GOODNESS-OF-FIT TEST

print("\nCHI-SQUARED GOODNESS-OF-FIT TEST ")

chi2_stat, p_value = chisquare(f_obs=obs_vals, f_exp=exp_vals)

degrees_of_freedom = len(STATIONS) - 1
critical_value_95  = 11.070   # chi-sq critical value, df=5, alpha=0.05

# Per-cell contributions to the test statistic
cell_contributions = ((obs_vals - exp_vals) ** 2) / exp_vals

contrib_df = pd.DataFrame({
    "Station"             : STATIONS,
    "Observed"         : obs_vals.astype(int),
    "Expected"         : exp_vals.astype(int),
    "(O-E)²/E"         : cell_contributions.round(4)
})
print(contrib_df.to_string(index=False))

print(f"\n  Chi-squared statistic : {chi2_stat:.4f}")
print(f"  Degrees of freedom    : {degrees_of_freedom}")
print(f"  Critical value (α=0.05, df=5): {critical_value_95}")
print(f"  p-value               : {p_value:.6f}")
print(f"  Significance level    : {SIGNIFICANCE_LEVEL}")

if p_value < SIGNIFICANCE_LEVEL:
    verdict = "REJECT H₀"
    interpretation = (
        "The observed distribution is significantly different from a uniform distribution. The defect distribution is non-uniform — a station fault is indicated."
    )
else:
    verdict = "FAIL TO REJECT H₀"
    interpretation = (
        "There is insufficient evidence to conclude the The defect distribution is non-uniform — a station fault is not indicated."
    )

print(f"\n  Verdict : {verdict}")
print(f"  {interpretation}")

# 8. VISUALISATION — CHART 4: Per-Cell Chi-Squared Contributions

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=contrib_df, x="Station", y="(O-E)²/E",
    color="seagreen", errorbar=None, ax=ax
)
ax.set_title("Per-Station Contribution to Chi-Squared Statistic  (O−E)²/E",
             pad=12)
ax.set_xlabel("Production Station")
ax.set_ylabel("(O − E)² / E")
for bar in ax.patches:
    h = bar.get_height()
    ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color="dimgray")
plt.tight_layout()
plt.savefig("chart4_chi2_contributions.png", dpi=150)
plt.show()

# 9. RESULTS SUMMARY

print("\nRESULTS SUMMARY:")
print(f"  Sample size           : {N_DEFECTS:,} defects")
print(f"  Chi-squared statistic : {chi2_stat:.4f}")
print(f"  p-value               : {p_value:.6f}")
print(f"  Verdict               : {verdict}")
print(f"  Largest contributor   : Station {contrib_df.loc[contrib_df['(O-E)²/E'].idxmax(), 'Station']} "
      f"({contrib_df['(O-E)²/E'].max():.4f})")
print(f"  % of χ² from Station 6  : "
      f"{cell_contributions[5] / chi2_stat * 100:.1f}%")
print("\n  Script complete.")