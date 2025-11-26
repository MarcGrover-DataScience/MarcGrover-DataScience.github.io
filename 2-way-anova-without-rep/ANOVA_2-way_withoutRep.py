# ANOVA – Two-way without replication

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.datasets import fetch_openml
# import warnings
# warnings.filterwarnings('ignore')

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
plt.rcParams['figure.figsize'] = (10, 6)

print("2-WAY ANOVA WITHOUT REPLICATION ANALYSIS")

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n1. DATA PREPARATION")

# Load the wine quality dataset from sklearn
print("\nSample input data - unprocessed:")
wine = fetch_openml('wine-quality-red', version=1, as_frame=True, parser='auto')
wine_df = wine.frame
# Rename column 'class' to 'quality'
wine_df = wine_df.rename(columns={'class':'quality'})
print(wine_df)
print('\nColumn types:')
print(wine_df.dtypes)
print(f"\nAvailable columns: {wine_df.columns.tolist()}")

# Create a structured dataset for 2-way ANOVA without replication
# We'll analyze how alcohol content varies by quality rating and pH level categories
# Select relevant columns
data = wine_df[['alcohol', 'quality', 'pH']].copy()

# Categorize pH into levels (Low, Medium, High)
data['pH_category'] = pd.cut(data['pH'], bins=3, labels=['Low', 'Medium', 'High'])

# Filter to get quality ratings 5, 6, 7 for balanced design
data = data[data['quality'].isin(['5', '6', '7'])]

data['quality'] = data['quality'].astype(int).astype(str)


# Create one observation per combination (no replication)
# Group by both factors and take mean
anova_data = data.groupby(['quality', 'pH_category'], observed=False)['alcohol'].mean().reset_index()
anova_data.columns = ['Quality', 'pH_Level', 'Alcohol']

print("\nDataset Info:")
print(f"- Total observations: {len(anova_data)}")
print(f"- Factor 1 (Quality): {anova_data['Quality'].unique()}")
print(f"- Factor 2 (pH Level): {anova_data['pH_Level'].unique()}")
print(f"\nSample rows for analysis:")
print(anova_data)

# ============================================================================
# STEP 2: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n2. DESCRIPTIVE STATISTICS")

print("\nOverall Statistics:")
print(anova_data['Alcohol'].describe())

print("\n\nMean Alcohol by Quality:")
print(anova_data.groupby('Quality', observed=False)['Alcohol'].agg(['mean', 'std', 'count']))

print("\n\nMean Alcohol by pH Level:")
print(anova_data.groupby('pH_Level', observed=False)['Alcohol'].agg(['mean', 'std', 'count']))

print("\n\nMean Alcohol by Quality and pH Level (Data to be analysed):")
pivot_table = anova_data.pivot_table(values='Alcohol',
                                       index='Quality',
                                       columns='pH_Level',
                                       aggfunc='mean',
                                       observed=False)
print(pivot_table)

# ============================================================================
# STEP 3: VISUALIZATIONS
# ============================================================================
print("\n3. DATA VISUALIZATIONS")

# Plot 1: Distribution of Alcohol Content
print("\nGenerating Plot 1: Distribution of Alcohol Content...")
plt.figure(figsize=(10, 6))
sns.histplot(data=anova_data, x='Alcohol', kde=True, bins=15, color='steelblue')
plt.title('Distribution of Alcohol Content', fontsize=16, fontweight='bold')
plt.xlabel('Alcohol Content (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
# plt.savefig('plot1_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Box plot by Quality
print("\nGenerating Plot 2: Alcohol Content by Quality...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=anova_data, x='Quality', y='Alcohol', palette='Blues', hue='Quality')
plt.title('Alcohol Content by Wine Quality', fontsize=16, fontweight='bold')
plt.xlabel('Wine Quality Rating', fontsize=12)
plt.ylabel('Alcohol Content (%)', fontsize=12)
plt.tight_layout()
# plt.savefig('plot2_quality_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Box plot by pH Level
print("\nGenerating Plot 3: Alcohol Content by pH Level...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=anova_data, x='pH_Level', y='Alcohol', palette='Greens', hue='pH_Level')
plt.title('Alcohol Content by pH Level', fontsize=16, fontweight='bold')
plt.xlabel('pH Level Category', fontsize=12)
plt.ylabel('Alcohol Content (%)', fontsize=12)
plt.tight_layout()
# plt.savefig('plot3_ph_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Interaction plot
print("\nGenerating Plot 4: Interaction Plot...")
plt.figure(figsize=(10, 6))
for ph_level in anova_data['pH_Level'].unique():
    subset = anova_data[anova_data['pH_Level'] == ph_level]
    plt.plot(subset['Quality'], subset['Alcohol'], marker='o',
             label=f'pH: {ph_level}', linewidth=2, markersize=8)
plt.title('Interaction Plot: Quality × pH Level', fontsize=16, fontweight='bold')
plt.xlabel('Wine Quality Rating', fontsize=12)
plt.ylabel('Alcohol Content (%)', fontsize=12)
plt.legend(title='pH Level', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('plot4_interaction.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 5: Heatmap
print("\nGenerating Plot 5: Heatmap of Alcohol Content...")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='YlOrRd',
            cbar_kws={'label': 'Alcohol Content (%)'}, linewidths=0.5)
plt.title('Alcohol Content by Quality and pH Level',
          fontsize=16, fontweight='bold')
plt.xlabel('pH Level', fontsize=12)
plt.ylabel('Wine Quality Rating', fontsize=12)
plt.tight_layout()
# plt.savefig('plot5_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 4: TEST ASSUMPTIONS
# ============================================================================
print("\n4. TESTING ANOVA ASSUMPTIONS")

# Assumption 1: Normality Test (Shapiro-Wilk)
print("\nAssumption 1: Normality Test")
print("H0: Data is normally distributed")
print("H1: Data is not normally distributed")
statistic, p_value = stats.shapiro(anova_data['Alcohol'])
print(f"\nShapiro-Wilk Test:")
print(f"Test Statistic: {statistic:.6f}")
print(f"P-value: {p_value:.6f}")
if p_value > 0.05:
    print(f"Result: PASS (p > 0.05) - Data appears normally distributed")
else:
    print(f"Result: FAIL (p ≤ 0.05) - Data may not be normally distributed")
    print(f"Note: ANOVA is robust to moderate violations of normality")

# Q-Q Plot for normality
print("\nGenerating Q-Q Plot for normality assessment")
plt.figure(figsize=(10, 6))
stats.probplot(anova_data['Alcohol'], dist="norm", plot=plt)
plt.title('Q-Q Plot: Normality Assessment', fontsize=16, fontweight='bold')
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Sample Quantiles', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('plot6_qq_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Assumption 2: Homogeneity of Variances (Levene's Test)
print("\nAssumption 2: Homogeneity of Variances (Levene's Test)")
print("H0: Variances are equal across groups")
print("H1: Variances are not equal across groups")

# Test for Quality groups
groups_quality = [group['Alcohol'].values for name, group in anova_data.groupby('Quality')]
stat_q, p_q = stats.levene(*groups_quality)
print(f"\nLevene's Test (by Quality):")
print(f"Test Statistic: {stat_q:.6f}")
print(f"P-value: {p_q:.6f}")
if p_q > 0.05:
    print(f"Result: PASS (p > 0.05) - Equal variances assumed")
else:
    print(f"Result: FAIL (p ≤ 0.05) - Variances may not be equal")

# Test for pH Level groups
groups_ph = [group['Alcohol'].values for name, group in anova_data.groupby('pH_Level', observed=False)]
stat_p, p_p = stats.levene(*groups_ph)
print(f"\nLevene's Test (by pH Level):")
print(f"Test Statistic: {stat_p:.6f}")
print(f"P-value: {p_p:.6f}")
if p_p > 0.05:
    print(f"Result: PASS (p > 0.05) - Equal variances assumed")
else:
    print(f"Result: FAIL (p ≤ 0.05) - Variances may not be equal")


# ============================================================================
# STEP 5: PERFORM 2-WAY ANOVA WITHOUT REPLICATION
# ============================================================================
print("\n5. 2-WAY ANOVA WITHOUT REPLICATION")

# Fit the model
model = ols('Alcohol ~ C(Quality) + C(pH_Level)', data=anova_data).fit()
anova_table = anova_lm(model, typ=2)

print("ANOVA TABLE:")
print(anova_table)

# ============================================================================
# STEP 6: STATISTICAL FINDINGS AND INTERPRETATION
# ============================================================================
print("\n6. STATISTICAL FINDINGS AND INTERPRETATION")

# Extract key statistics
alpha = 0.05
print(f"\nSignificance level (α): {alpha}")

print("\nMAIN EFFECT: QUALITY")
quality_f = anova_table.loc['C(Quality)', 'F']
quality_p = anova_table.loc['C(Quality)', 'PR(>F)']
quality_df = anova_table.loc['C(Quality)', 'df']
print(f"F-statistic: {quality_f:.4f}")
print(f"P-value: {quality_p:.6f}")
print(f"Degrees of freedom: {quality_df:.0f}")

if quality_p < alpha:
    print(f"\nSIGNIFICANT EFFECT (p < {alpha})")
    print(f"Wine quality has a statistically significant effect on alcohol content.")
    print(f"We reject the null hypothesis that quality levels have equal mean alcohol.")
else:
    print(f"\nNOT SIGNIFICANT (p ≥ {alpha})")
    print(f"Wine quality does NOT have a statistically significant effect on alcohol.")
    print(f"We fail to reject the null hypothesis.")

print("\nMAIN EFFECT: pH LEVEL")
ph_f = anova_table.loc['C(pH_Level)', 'F']
ph_p = anova_table.loc['C(pH_Level)', 'PR(>F)']
ph_df = anova_table.loc['C(pH_Level)', 'df']
print(f"F-statistic: {ph_f:.4f}")
print(f"P-value: {ph_p:.6f}")
print(f"Degrees of freedom: {ph_df:.0f}")

if ph_p < alpha:
    print(f"\nSIGNIFICANT EFFECT (p < {alpha})")
    print(f"pH level has a statistically significant effect on alcohol content.")
    print(f"We reject the null hypothesis that pH levels have equal mean alcohol.")
else:
    print(f"\nNOT SIGNIFICANT (p ≥ {alpha})")
    print(f"pH level does NOT have a statistically significant effect on alcohol.")
    print(f"We fail to reject the null hypothesis.")

# Model Summary Statistics
print("\nMODEL SUMMARY:")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"F-statistic: {model.fvalue:.4f}")
print(f"P-value (overall model): {model.f_pvalue:.6f}")

# Calculate effect sizes (Eta-squared)
print("\nEFFECT SIZES (η² - Eta-squared)")
ss_total = anova_table['sum_sq'].sum()
eta_sq_quality = anova_table.loc['C(Quality)', 'sum_sq'] / ss_total
eta_sq_ph = anova_table.loc['C(pH_Level)', 'sum_sq'] / ss_total

print(f"Quality effect size (η²): {eta_sq_quality:.4f}")
if eta_sq_quality < 0.01:
    print("Small effect")
elif eta_sq_quality < 0.06:
    print("Medium effect")
else:
    print("Large effect")

print(f"\npH Level effect size (η²): {eta_sq_ph:.4f}")
if eta_sq_ph < 0.01:
    print("Small effect")
elif eta_sq_ph < 0.06:
    print("Medium effect")
else:
    print("Large effect")

# ============================================================================
# STEP 7: COMPREHENSIVE INTERPRETATION
# ============================================================================
print("\nSTEP 7: COMPREHENSIVE INTERPRETATION")

print("""
SUMMARY OF FINDINGS:

1. DATA OVERVIEW:
   - Analyzed the relationship between wine quality ratings (5, 6, 7) and
     pH levels (Low, Medium, High) on alcohol content
   - Used a 2-way ANOVA without replication design
   - Total combinations analyzed: 9 (3 quality levels × 3 pH levels)

2. ASSUMPTION CHECKS:
   - Normality: Assessed using Shapiro-Wilk test and Q-Q plot
   - Homogeneity of Variances: Assessed using Levene's test
   - Note: 2-way ANOVA is relatively robust to moderate violations

3. STATISTICAL RESULTS:
""")

if quality_p < alpha:
    print(f"Quality Factor: SIGNIFICANT (p = {quality_p:.6f})")
    print(f" - Higher quality wines tend to have different alcohol levels")
    print(f" - Effect size (η²) = {eta_sq_quality:.4f}")
else:
    print(f"Quality Factor: NOT SIGNIFICANT (p = {quality_p:.6f})")

if ph_p < alpha:
    print(f"\npH Level Factor: SIGNIFICANT (p = {ph_p:.6f})")
    print(f" - Different pH levels are associated with varying alcohol content")
    print(f" - Effect size (η²) = {eta_sq_ph:.4f}")
else:
    print(f"\npH Level Factor: NOT SIGNIFICANT (p = {ph_p:.6f})")

print(f"""
4. PRACTICAL IMPLICATIONS:
   - The model explains {model.rsquared*100:.2f}% of variance in alcohol content
   - Both factors should be considered when analyzing wine characteristics
   - Results suggest that wine quality and pH chemistry relate to alcohol levels

5. LIMITATIONS:
   - This is an observational study; causation cannot be inferred
   - Sample represents averaged values (no replication within cells)
   - Results specific to red wine data; may not generalize to other wine types

6. RECOMMENDATIONS:
   - For wine production: Monitor both quality targets and pH levels
   - For further research: Consider interaction effects with replication
   - Validate findings with additional data sources
""")