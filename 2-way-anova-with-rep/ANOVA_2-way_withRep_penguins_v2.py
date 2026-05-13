# ANOVA – Two-way with replication

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import probplot
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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

# Load the Palmer Penguins dataset from seaborn (a classic example for 2-way ANOVA)
df = sns.load_dataset('penguins')

print("\n2-WAY ANOVA WITH REPLICATION")
print("─" * 50)

print("\nDATASET: Palmer Penguin Data - from Seaborn")
print("\nResearch Question: Does flipper length (mm) differ by species and sex?")
print("  Test for interaction: species × sex")

# Display basic information about the dataset
print("\n1. DATA OVERVIEW - BASIC EDA")
print("─" * 50)

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
print("─" * 50)

print("Dependent Variable: Flipper Length (mm)")
print(f"Species: {df_clean['species'].unique()}")
print(f"Sex: {df_clean['sex'].unique()}")

# Check replication
print("\n3. CHECKING FOR REPLICATION:")
print("─" * 50)

replication = df_clean.groupby(['species', 'sex'], observed=True).size()
print("\nObservations per group combination:")
print(replication)
print(f"\nAll groups have replication (n > 1): {(replication > 1).all()}")

# Descriptive statistics
print("\n4. DESCRIPTIVE STATISTICS")
print("─" * 50)

# Overall statistics
print(f"\nOverall flipper length statistics:")
print(df_clean['flipper_length_mm'].describe().round(2))

# By species and sex
print(f"\nFlipper length by Species and Sex:")
desc_stats = df_clean.groupby(['species', 'sex'], observed=True)['flipper_length_mm'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(2)
print(desc_stats)

# Marginal means
print("\nBy Species:")
print(df_clean.groupby('species', observed=True)['flipper_length_mm'].agg(['count', 'mean', 'std']).round(2))

print("\nBy Sex:")
print(df_clean.groupby('sex', observed=True)['flipper_length_mm'].agg(['count', 'mean', 'std']).round(2))

# OUTLIER INVESTIGATION
print("\n4b. OUTLIER INVESTIGATION (IQR Method)")
print("─" * 50)

outlier_summary = []
for (species, sex), group in df_clean.groupby(['species', 'sex'], observed=True):
    Q1 = group['flipper_length_mm'].quantile(0.25)
    Q3 = group['flipper_length_mm'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = group[(group['flipper_length_mm'] < lower) |
                     (group['flipper_length_mm'] > upper)]
    outlier_summary.append({
        'species': species, 'sex': sex,
        'n_outliers': len(outliers),
        'lower_bound': round(lower, 2),
        'upper_bound': round(upper, 2),
        'outlier_values': outliers['flipper_length_mm'].tolist()
    })
    print(f"{species}, {sex}: {len(outliers)} outlier(s) "
          f"[bounds: {lower:.1f}–{upper:.1f}mm]"
          + (f" → values: {outliers['flipper_length_mm'].tolist()}" if len(outliers) > 0 else ""))

total_outliers = sum(d['n_outliers'] for d in outlier_summary)
print(f"\nTotal outliers identified: {total_outliers}")
print("Note: Outliers represent biologically plausible measurements.")
print("      No observations will be removed from the analysis.")


# VISUALIZATIONS
print("\n5. VISUALIZATIONS")
print("─" * 50)

# Histogram of all dependent variable values
plt.figure(figsize=(8, 6))
sns.histplot(data=df_clean['flipper_length_mm'], bins=10, kde=True, color='seagreen',
                 edgecolor='black', alpha=0.6)
plt.title('Flipper Length Distribution with KDE', fontsize=12, fontweight='bold')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Frequency')
plt.savefig('2way_anova_with_dist.png', dpi=150, bbox_inches='tight')
plt.show()

# Box plot by sex
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='sex', y='flipper_length_mm', hue='sex', palette='Blues')
plt.title('Flipper Length Distribution by Sex', fontsize=12, fontweight='bold')
plt.xlabel('Sex')
plt.ylabel('Flipper Length (mm)')
plt.savefig('2way_anova_with_box_species_gender.png', dpi=150, bbox_inches='tight')
plt.show()

# Box plot by species
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='species', y='flipper_length_mm', hue='species', palette='Greens')
plt.title('Flipper Length Distribution by Species', fontsize=12, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Flipper Length (mm)')
plt.savefig('2way_anova_with_box_species.png', dpi=150, bbox_inches='tight')
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
plt.savefig('2way_anova_with_box.png', dpi=150, bbox_inches='tight')
plt.show()

# Standard Deviation by Species and Sex Group
# This chart directly supports the interpretation of the Levene's Test result by showing which groups carry the highest within-group variance

std_by_group = df_clean.groupby(['species', 'sex'], observed=True)['flipper_length_mm'].std().reset_index()
std_by_group.columns = ['species', 'sex', 'std']
std_by_group['group'] = std_by_group['species'] + '\n(' + std_by_group['sex'] + ')'

plt.figure(figsize=(10, 6))
ax = sns.barplot(data=std_by_group, x='group', y='std',
                 hue='species', dodge=False,
                 palette='Greens', edgecolor='black', linewidth=1.5)
plt.title('Standard Deviation of Flipper Length by Species and Sex',
          fontsize=12, fontweight='bold')
plt.xlabel('Species and Sex Group')
plt.ylabel('Standard Deviation of Flipper Length (mm)')
plt.axhline(y=std_by_group['std'].mean(), color='red', linestyle='--',
            alpha=0.6, linewidth=1.5, label=f"Overall mean SD: {std_by_group['std'].mean():.2f}mm")
# Annotate each bar with its SD value
for bar in ax.patches:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                f'{height:.2f}mm', ha='center', va='bottom', fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig('2way_anova_std_by_group.png', dpi=150, bbox_inches='tight')
plt.show()


# Interaction plot - Critical for understanding interaction effects
plt.figure(figsize=(8, 6))
interaction_data = df_clean.groupby(['species', 'sex'], observed=True)['flipper_length_mm'].mean().reset_index()
sns.lineplot(
    data=interaction_data,
    x='sex',
    y='flipper_length_mm',       # This is the mean flipper length already calculated
    hue='species',     # 'hue' automatically creates the separate lines
    marker='o',
    linewidth=2,
    markersize=8)
plt.title('Interaction Plot: Species × Sex', fontsize=12, fontweight='bold')
plt.xlabel('Penguin sex')
plt.ylabel('Mean Flipper Length (mm)')
plt.legend(title='Species', fontsize=9)
plt.savefig('2way_anova_with_interaction.png', dpi=150, bbox_inches='tight')
plt.show()


# Check ANOVA assumptions
print("\n6. CHECKING ANOVA ASSUMPTIONS")
print("─" * 50)

# 1. Normality test
print("\n6.1. NORMALITY TEST (Shapiro-Wilk) for each group:")
print("H0: Data is normally distributed")
print("If p > 0.05, data is approximately normal\n")

for (species, sex), group in df_clean.groupby(['species', 'sex']):
    if len(group) >= 3:  # Need at least 3 observations
        stat, p_value = stats.shapiro(group['flipper_length_mm'])
        status = 'Normal' if p_value > 0.05 else 'Check'
        print(f"{species}, {sex}: W={stat:.4f}, p={p_value:.4f} {status}")

# Q-Q Plots per Group (Normality Visual Check)
groups_list = list(df_clean.groupby(['species', 'sex'], observed=True))

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
fig.suptitle('Q-Q Plots by Species and Sex (Normality Check)',
             fontsize=14, fontweight='bold')

for idx, ((species, sex), group_data) in enumerate(groups_list):
    ax = axes[idx]
    probplot(group_data['flipper_length_mm'], dist='norm', plot=ax)
    sw_stat, sw_p = stats.shapiro(group_data['flipper_length_mm'])
    ax.set_title(f'{species}, {sex}\nShapiro-Wilk p={sw_p:.4f}',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=9)
    ax.set_ylabel('Sample Quantiles', fontsize=9)

plt.tight_layout()
plt.savefig('2way_anova_qq_plots.png', dpi=150, bbox_inches='tight')
plt.show()


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
print("─" * 50)

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
print("─" * 50)

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
plt.savefig('2way_anova_with_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# GENDER GAP PER SPECIES (Quantifying the Interaction)
print("\n8a. GENDER EFFECT PER SPECIES (Sexual Dimorphism by Species)")
print("─" * 50)

gender_gap = df_clean.groupby(['species', 'sex'], observed=True)['flipper_length_mm'].mean().unstack()
gender_gap['difference_mm'] = gender_gap['Male'] - gender_gap['Female']
gender_gap.columns.name = None
print("\nMean flipper length and male-female difference by species:")
print(gender_gap.round(2))

# PLOT: Gender gap per species
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=gender_gap.index, y=gender_gap['difference_mm'],
                 palette='Greens', edgecolor='black', linewidth=1.5, hue=gender_gap.index)
plt.title('Male–Female Flipper Length Difference by Species\n(Interaction Effect Quantified)',
          fontsize=12, fontweight='bold')
plt.xlabel('Species')
plt.ylabel('Mean Difference: Male − Female (mm)')
plt.axhline(y=0, color='black', linewidth=1)
for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
            f'+{height:.1f}mm', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('2way_anova_gender_gap.png', dpi=150, bbox_inches='tight')
plt.show()


# POST-HOC ANALYSIS: TUKEY'S HSD
print("\n8b. POST-HOC ANALYSIS: TUKEY'S HSD (Species)")
print("─" * 50)
print("Applied to species main effect (3 levels → 3 pairwise comparisons)")
print("Family-wise error rate controlled at α = 0.05\n")

tukey_result = pairwise_tukeyhsd(endog=df_clean['flipper_length_mm'],
                                  groups=df_clean['species'],
                                  alpha=0.05)
print(tukey_result)

# PLOT: Tukey HSD Results
tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                         columns=tukey_result._results_table.data[0])
tukey_df['meandiff'] = tukey_df['meandiff'].astype(float)
tukey_df['lower'] = tukey_df['lower'].astype(float)
tukey_df['upper'] = tukey_df['upper'].astype(float)
tukey_df['comparison'] = tukey_df['group1'] + ' vs\n' + tukey_df['group2']
tukey_df['significant'] = tukey_df['reject'].map({True: 'Significant', False: 'Not Significant'})

plt.figure(figsize=(9, 6))
colors = ['mediumseagreen' if sig == 'Significant' else 'red'
          for sig in tukey_df['significant']]
ax = plt.gca()

for pos, (_, row) in enumerate(tukey_df.iterrows()):
    mean_diff = float(row['meandiff'])
    lower     = float(row['lower'])
    upper     = float(row['upper'])
    p_adj     = float(row['p-adj'])

    ax.barh(row['comparison'], mean_diff,
            xerr=[[mean_diff - lower], [upper - mean_diff]],
            color=colors[pos], edgecolor='black', linewidth=1.2,
            capsize=6, height=0.5)

    p_label = 'p < 0.001' if p_adj < 0.001 else f'p = {p_adj:.4f}'
    ax.text(upper + 0.3, pos, p_label, va='center', fontsize=10)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Mean Difference in Flipper Length (mm)', fontsize=12)
ax.set_title("Tukey's HSD: Pairwise Species Comparisons",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('2way_anova_tukey_hsd.png', dpi=150, bbox_inches='tight')
plt.show()


# PLOT: Model Residual Analysis
print("\n8c. Model Residual Analysis")
print("─" * 50)

residuals = model.resid

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Model Residual Diagnostics', fontsize=14, fontweight='bold')

# Q-Q Plot
probplot(residuals, dist='norm', plot=axes[0])
axes[0].set_title('Residual Q-Q Plot', fontsize=12, fontweight='bold')

# Residual Histogram
axes[1].hist(residuals, bins=20, color='seagreen', edgecolor='black',
             alpha=0.7, density=True)
axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Density')
# Overlay normal curve
import numpy as np
xmin, xmax = axes[1].get_xlim()
x = np.linspace(xmin, xmax, 100)
from scipy.stats import norm
axes[1].plot(x, norm.pdf(x, residuals.mean(), residuals.std()),
             'r-', linewidth=2, label='Normal fit')
axes[1].legend()

plt.tight_layout()
plt.savefig('2way_anova_residuals.png', dpi=150, bbox_inches='tight')
plt.show()

# PLOT: Residuals vs Fitted Values
# Visual check of homoscedasticity in the model residuals — distinct from Levene's Test, which tests raw group variances before the model is fitted.

fitted_values = model.fittedvalues
residuals     = model.resid

plt.figure(figsize=(9, 6))
ax = plt.gca()

# Scatter: colour by species to aid interpretation of any patterns
species_colours = {'Adelie': '#4C9BE8', 'Chinstrap': '#F4A261', 'Gentoo': '#2A9D8F'}
for species, colour in species_colours.items():
    mask = df_clean['species'] == species
    ax.scatter(fitted_values[mask], residuals[mask],
               color=colour, alpha=0.55, edgecolors='none',
               s=40, label=species)

# Reference line at zero
ax.axhline(y=0, color='black', linewidth=1.2, linestyle='--')

# Lowess smoothed trend line to reveal any systematic pattern
from statsmodels.nonparametric.smoothers_lowess import lowess
smooth = lowess(residuals, fitted_values, frac=0.6)
ax.plot(smooth[:, 0], smooth[:, 1], color='red', linewidth=2,
        linestyle='-', label='Lowess trend')

ax.set_xlabel('Fitted Values (mm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Residuals (mm)', fontsize=12, fontweight='bold')
ax.set_title('Residuals vs Fitted Values',
             fontsize=13, fontweight='bold')
ax.legend(title='Species', fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('2way_anova_residuals_vs_fitted.png', dpi=150, bbox_inches='tight')
plt.show()


# Print residual normality test
sw_stat, sw_p = stats.shapiro(residuals)
print(f"\nShapiro-Wilk on model residuals: W={sw_stat:.4f}, p={sw_p:.4f}")
print("Residuals are normally distributed" if sw_p > 0.05 else "Residual normality assumption may be violated")


# STATISTICAL INTERPRETATION
print("\n9. STATISTICAL INTERPRETATION")
print("─" * 50)

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
print("─" * 50)

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