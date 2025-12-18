# ANOVA – Two-way with replication

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns

# Set Dataframe printing options
desired_width=320                                                   # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load the tips dataset from seaborn (a classic example for 2-way ANOVA)
# This dataset contains information about penguins
df = sns.load_dataset('penguins')

print("2-WAY ANOVA WITH REPLICATION")
print("\nDATASET: Palmer Penguin Data - from Seaborn")
print("\nResearch Question: Does flipper length (mm) differ by species and sex?")
print("  Test for interaction: species × sex")

# Display basic information about the dataset
print("\n1. DATA OVERVIEW - BASIC EDA")
print(f"Total observations: {len(df)}")
print(f"\nDataset shape: {df.shape}\n")

# Clean dataset
print(f"Missing values before cleaning:")
print(df[['flipper_length_mm', 'species', 'sex']].isnull().sum())

# Remove rows with missing values in relevant columns
df_clean = df[['flipper_length_mm', 'species', 'sex']].dropna()


print(f"\nObservations after cleaning: {len(df_clean)}\n")
print(df_clean)

print(f"\n2. Variables for analysis:")
print("Dependent Variable: Flipper Length (mm)")
print(f"Species: {df_clean['species'].unique()}")
print(f"Sex: {df_clean['sex'].unique()}")

# Check replication
print("\n3. CHECKING FOR REPLICATION:")
replication = df_clean.groupby(['species', 'sex'], observed=False).size()
print("\nObservations per group combination:")
print(replication)
print(f"\nAll groups have replication (n > 1): {(replication > 1).all()}")

# Descriptive statistics
print("\n4. DESCRIPTIVE STATISTICS")

# Overall statistics
print(f"\nOverall flipper length statistics:")
print(df_clean['flipper_length_mm'].describe().round(2))

# By species and sex
print(f"\nFlipper length by Species and Sex:")
desc_stats = df_clean.groupby(['species', 'sex'], observed=False)['flipper_length_mm'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(2)
print(desc_stats)

# Marginal means
print("\nBy Species:")
print(df_clean.groupby('species', observed=False)['flipper_length_mm'].agg(['count', 'mean', 'std']).round(2))

print("\nBy Sex:")
print(df_clean.groupby('sex', observed=False)['flipper_length_mm'].agg(['count', 'mean', 'std']).round(2))

# VISUALIZATIONS
print("\n5. VISUALIZATIONS")

# Histogram of all dependent variable values
plt.figure(figsize=(8, 6))
sns.histplot(data=df_clean['flipper_length_mm'], bins=10, kde=True, color='skyblue',
                 edgecolor='black', alpha=0.6)
plt.title('Flipper Length Distribution with KDE', fontsize=12, fontweight='bold')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Frequency')
plt.show()

# Box plot by sex
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='sex', y='flipper_length_mm', hue='sex', palette='Blues')
plt.title('Flipper Length Distribution by Sex', fontsize=12, fontweight='bold')
plt.xlabel('Sex')
plt.ylabel('Flipper Length (mm)')
plt.show()

# Box plot by species
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='species', y='flipper_length_mm', hue='species', palette='Greens')
plt.title('Flipper Length Distribution by Species', fontsize=12, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.show()

# Box plot for all combinations
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='species', y='flipper_length_mm',
            hue='sex', palette='Purples')
plt.title('Flipper Length by Species and Sex',
                     fontsize=12, fontweight='bold')
plt.ylabel('Flipper Length (mm)')
plt.xlabel('Species')
plt.legend(title='Sex', fontsize=9)
plt.show()

# Interaction plot - Critical for understanding interaction effects
plt.figure(figsize=(8, 6))
interaction_data = df_clean.groupby(['species', 'sex'], observed=False)['flipper_length_mm'].mean().reset_index()
sns.lineplot(
    data=interaction_data,
    x='sex',
    y='flipper_length_mm',       # This is the mean tip you already calculated
    hue='species',     # 'hue' automatically creates the separate lines
    marker='o',
    linewidth=2,
    markersize=8)
plt.title('Interaction Plot: Species × Sex', fontsize=12, fontweight='bold')
plt.xlabel('Penguin sex')
plt.ylabel('Flipper Length (mm)')
plt.legend(title='Species', fontsize=9)
plt.show()


# Check ANOVA assumptions
print("\n6. CHECKING ANOVA ASSUMPTIONS\n")

# 1. Normality test
print("\n6.1. NORMALITY TEST (Shapiro-Wilk) for each group:")
print("H0: Data is normally distributed")
print("If p > 0.05, data is approximately normal\n")

for (species, sex), group in df_clean.groupby(['species', 'sex']):
    if len(group) >= 3:  # Need at least 3 observations
        stat, p_value = stats.shapiro(group['flipper_length_mm'])
        status = 'Normal' if p_value > 0.05 else 'Check'
        print(f"{species}, {sex}: W={stat:.4f}, p={p_value:.4f} {status}")

# 2. Homogeneity of variance
print("\n6.2. HOMOGENEITY OF VARIANCE (Levene's Test):")
print("H0: Variances are equal across groups")
print("If p > 0.05, equal variances assumption is satisfied\n")

groups = [group['flipper_length_mm'].values
          for name, group in df_clean.groupby(['species', 'sex'])]
levene_stat, levene_p = stats.levene(*groups)

print(f"Levene's Test: F={levene_stat:.4f}, p={levene_p:.4f}")
if levene_p > 0.05:
    print("Equal variances assumption is satisfied")
else:
    print("Equal variances assumption may be violated")


# Perform 2-Way ANOVA
print("\n7. TWO-WAY ANOVA TEST")

# Fit the model
model = ols('flipper_length_mm ~ C(species) + C(sex) + C(species):C(sex)',
            data=df_clean).fit()
anova_table = anova_lm(model, typ=2)

print("\nFitted model summary:")
print(f"  Number of observations: {model.nobs:.0f}")
print(f"  R-squared: {model.rsquared:.4f}")
print(f"  Degrees of freedom (residuals): {model.df_resid:.0f}")

print("\nANOVA Table (Type II Sum of Squares):\n")
print(anova_table)
print('\nsum_sq: Measures the variation explained by each effect')
print('F (F-statistic): Test statistic for hypothesis testing')
print('PR(>F) (p-value): Probability of observing this F-statistic if null hypothesis is true')
print("The \'Residual\' row represents the UNEXPLAINED variation - the error term")

# Effect sizes (Eta-squared)
print("\n8. EFFECT SIZES (η² - Eta Squared)")
print('\nEta-Squared measures the proportion of total variance in the DV explained by an IV.')
print('\nThis quantifies the effect (p-value determines if it is real)')
print("\nInterpretation (Cohen's variance): Small (0.01), Medium (0.06), Large (0.14)")


# Calculate total sum of squares
SS_total = anova_table['sum_sq'].sum()

# Add eta-squared column
anova_table['eta_squared'] = anova_table['sum_sq'] / SS_total

print("\nANOVA Table with Eta-Squared:")
print(anova_table)

SS_total = anova_table['sum_sq'].sum()
print(f"\nTotal Sum of Squares (SS_total) = {SS_total:.4f}")

print("\nEFFECT SIZE INTERPRETATION:")
print("Cohen's Guidelines: Small (0.01), Medium (0.06), Large (0.14)")
print()


def interpret_eta(eta_sq):
    if eta_sq < 0.01:
        return "Negligible"
    elif eta_sq < 0.06:
        return "Small"
    elif eta_sq < 0.14:
        return "Medium"
    else:
        return "Large"


for effect in ['C(species)', 'C(sex)', 'C(species):C(sex)']:
    eta_sq = anova_table.loc[effect, 'eta_squared']
    p_val = anova_table.loc[effect, 'PR(>F)']
    f_stat = anova_table.loc[effect, 'F']

    effect_name = effect.replace('C(', '').replace(')', '').replace(':', ' × ')

    print(f"{effect_name}:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(
        f"  p-value: {p_val:.6f}")
    print(f"  η² = {eta_sq:.4f} ({eta_sq * 100:.2f}% of variance)")
    print(f"  Interpretation: {interpret_eta(eta_sq)}")
    print()


# Visualise the effect (eta-squared) of each factor
print("\nVISUALIZING EFFECT SIZES (ETA-SQUARED):")

# Prepare data for plotting
plot_df = anova_table.drop('Residual').copy()
plot_df['Effect'] = plot_df.index
plot_df['Effect'] = plot_df['Effect'].str.replace('C(', '').str.replace(')', '').str.replace(':', ' × ')

fig, ax = plt.subplots(figsize=(10, 8))

# Create bar plot using Seaborn
sns.barplot(
    data=plot_df,
    x='Effect',
    y='eta_squared',
    palette='Blues',
    hue='Effect',
    edgecolor='black',
    linewidth=2,
    ax=ax
)

# Add Cohen's guidelines
ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, linewidth=2,
           label='Small (0.01)')
ax.axhline(y=0.06, color='orange', linestyle='--', alpha=0.5, linewidth=2,
           label='Medium (0.06)')
ax.axhline(y=0.14, color='red', linestyle='--', alpha=0.5, linewidth=2,
           label='Large (0.14)')

# Add value labels
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    eta_val = plot_df.iloc[i]['eta_squared']
    p_val = plot_df.iloc[i]['PR(>F)']

    sig_stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    ax.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 0.01,
        f'η² = {eta_val:.4f}\n({eta_val * 100:.1f}%)\n{sig_stars}\n[{interpret_eta(eta_val)}]',
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold'
    )

ax.set_xlabel('Effect', fontsize=14, fontweight='bold')
ax.set_ylabel('Eta-Squared (η²)', fontsize=14, fontweight='bold')
ax.set_title('Effect Sizes: Palmer Penguins - Flipper Length Analysis',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9, title="Cohen's Guidelines")
ax.set_ylim(0, max(plot_df['eta_squared']) * 1.3)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# STATISTICAL INTERPRETATION

print("\n9. STATISTICAL INTERPRETATION")

alpha = 0.05

species_p = anova_table.loc['C(species)', 'PR(>F)']
sex_p = anova_table.loc['C(sex)', 'PR(>F)']
interaction_p = anova_table.loc['C(species):C(sex)', 'PR(>F)']

species_eta = anova_table.loc['C(species)', 'eta_squared']
sex_eta = anova_table.loc['C(sex)', 'eta_squared']
interaction_eta = anova_table.loc['C(species):C(sex)', 'eta_squared']

print(f"\nINTERACTION EFFECT: Species × Sex")
print(f"p-value: {interaction_p:.6f}")
print(f"η² = {interaction_eta:.4f} ({interaction_eta * 100:.2f}% of variance)")
print(f"Effect size: {interpret_eta(interaction_eta)}")

if interaction_p < alpha:
    print(f"\nSIGNIFICANT INTERACTION (p < {alpha})")
    print("The effect of species on flipper length DEPENDS ON sex")
    print("(or vice versa: the effect of sex depends on species)")
else:
    print(f"\nNO SIGNIFICANT INTERACTION (p ≥ {alpha})")
    print("Species and sex have independent effects on flipper length")

print(f"\nMAIN EFFECT: Species")

print(f"p-value: {species_p:.6f}")
print(f"η² = {species_eta:.4f} ({species_eta * 100:.2f}% of variance)")
print(f"Effect size: {interpret_eta(species_eta)}")

if species_p < alpha:
    print(f"SIGNIFICANT (p < {alpha}) - Different species have different flipper lengths")
else:
    print(f"NOT SIGNIFICANT (p ≥ {alpha})")

print(f"\nMAIN EFFECT: Sex")
print(f"p-value: {sex_p:.6f}")
print(f"η² = {sex_eta:.4f} ({sex_eta * 100:.2f}% of variance)")
print(f"Effect size: {interpret_eta(sex_eta)}")

if sex_p < alpha:
    print(f"SIGNIFICANT (p < {alpha}) - Males and females have different flipper lengths")
else:
    print(f"NOT SIGNIFICANT (p ≥ {alpha})")

# PRACTICAL INTERPRETATION

print("\n10. PRACTICAL INTERPRETATION")

print("\nMEAN FLIPPER LENGTHS BY GROUP:")
group_means = df_clean.groupby(['species', 'sex'])['flipper_length_mm'].mean().round(2)
for (species, sex), mean in group_means.items():
    print(f"  {species}, {sex}: {mean} mm")

print("\nKEY FINDINGS:")

# Species differences
print("\n1. SPECIES EFFECT:")
species_means = df_clean.groupby('species')['flipper_length_mm'].mean().sort_values(ascending=False)
print(f"Largest flippers: {species_means.index[0]} ({species_means.iloc[0]:.1f} mm)")
print(f"Smallest flippers: {species_means.index[-1]} ({species_means.iloc[-1]:.1f} mm)")
print(f"Difference: {species_means.iloc[0] - species_means.iloc[-1]:.1f} mm")

# Sex differences
print("\n2. SEX EFFECT:")
sex_means = df_clean.groupby('sex')['flipper_length_mm'].mean()
print(sex_means)
male_mean = sex_means['Male']
female_mean = sex_means['Female']
print(f"Male mean: {male_mean:.1f} mm")
print(f"Female mean: {female_mean:.1f} mm")
print(f"Difference: {abs(male_mean - female_mean):.1f} mm")
print(f"Males have {'longer' if male_mean > female_mean else 'shorter'} flippers")

# Interaction interpretation
print("\n3. INTERACTION:")
if interaction_p < alpha:
    print("The sex difference varies across species")
    print("Sexual dimorphism is not consistent across all penguin species")
else:
    print("No significant interaction detected")
    print("Sexual dimorphism is relatively consistent across species")