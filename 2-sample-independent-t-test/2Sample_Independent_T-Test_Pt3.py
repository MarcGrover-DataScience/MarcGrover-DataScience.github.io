# Two-Sample Independent Samples t-test: Compares the means of two independent groups.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set Dataframe printing options
desired_width = 320                                                 # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps


#========================================
# STEP 0 - Data Load & Validation
#========================================

print("DATA LOAD\n")
df_ensata = pd.read_excel('Iris_ensata.xlsx')
print("RAW DATA")
print(df_ensata)
ensata_1 = df_ensata[df_ensata['Test set'] == 1]['Length']
ensata_2 = df_ensata[df_ensata['Test set'] == 2]['Length']
print("RAW DATA - SAMPLE GROUP 1")
print(ensata_1)

print("\nDATA VALIDATION\n")
print(f"Dataset shape: {df_ensata.shape}")
print(f"\nMissing values:\n{df_ensata.isnull().sum()}")
print(f"\nDuplicate rows: {df_ensata.duplicated().sum()}")
print(f"\nData types:\n{df_ensata.dtypes}")
print(f"\nGroup distribution:\n{df_ensata['Test set'].value_counts().sort_index()}")
print(f"\nDescriptive statistics:\n{df_ensata.describe()}")

# DESCRIPTIVE STATISTICS
print("\nDESCRIPTIVE STATISTICS")
print(f"Group 1 (Ensata Group 1):  n={len(ensata_1)}, Mean={ensata_1.mean():.3f}, SD={ensata_1.std():.3f}")
print(f"Group 2 (Ensata Group 2):  n={len(ensata_2)}, Mean={ensata_2.mean():.3f}, SD={ensata_2.std():.3f}")
print(f"Difference in Means:  {ensata_2.mean() - ensata_1.mean():.3f}")

#========================================
# STEP 1 - Test hypothesis that the mean petal length is the same in 2 independent samples
#========================================
# This tests the length of sepal petals for 2 batches of Iris Ensata grown in different conditions
#========================================

print("\nTWO-SAMPLE INDEPENDENT T-TEST (STUDENT'S)")

# CHECK ASSUMPTIONS
print("\nASSUMPTIONS CHECK")

# Test for equal variances (Levene's test)
levene_stat, levene_p = stats.levene(ensata_1, ensata_2)
print(f"Levene's Test for Equal Variances:")
print(f"F-statistic = {levene_stat:.4f}, p-value = {levene_p:.4f}")
if levene_p > 0.05:
    print("Variances are approximately equal (use Student's t-test)")
    equal_var = True
else:
    print("Variances differ significantly (consider Welch's t-test)")
    equal_var = False

# Test for normality (Shapiro-Wilk test)
_, p_ensata1 = stats.shapiro(ensata_1)
_, p_ensata2 = stats.shapiro(ensata_2)
print(f"\nShapiro-Wilk Normality Test:")
print(f"Ensata Group1: p={p_ensata1:.4f} {'(Normal)' if p_ensata1 > 0.05 else '(Non-normal)'}")
print(f"Ensata Group2: p={p_ensata2:.4f} {'(Normal)' if p_ensata2 > 0.05 else '(Non-normal)'}")
print(f"Note: With n=50 each, CLT ensures robustness to non-normality")

# PERFORM T-TEST
print("\nTWO-SAMPLE T-TEST")

# Student's t-test (equal variances)
t_stat_student, p_value_student = stats.ttest_ind(ensata_1, ensata_2, equal_var=True)

# Welch's t-test (unequal variances) - for comparison
t_stat_welch, p_value_welch = stats.ttest_ind(ensata_1, ensata_2, equal_var=False)

print(f"Student's t-test (equal variances assumed):")
print(f"t-statistic = {t_stat_student:.4f}")
print(f"p-value = {p_value_student:.4e}")
print(f"Degrees of freedom = {len(ensata_1) + len(ensata_2) - 2}")

print(f"\nWelch's t-test (unequal variances):")
print(f"t-statistic = {t_stat_welch:.4f}")
print(f"p-value = {p_value_welch:.4e}")

# EFFECT SIZE
print("\nEFFECT SIZE")

# Cohen's d (pooled standard deviation)
pooled_std = np.sqrt(((len(ensata_1)-1)*ensata_1.var() +
                      (len(ensata_2)-1)*ensata_2.var()) /
                     (len(ensata_1) + len(ensata_2) - 2))
cohens_d = (ensata_2.mean() - ensata_1.mean()) / pooled_std

print(f"Cohen's d = {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect = "Small"
elif abs(cohens_d) < 0.5:
    effect = "Medium"
elif abs(cohens_d) < 0.8:
    effect = "Large"
else:
    effect = "Very Large"
print(f"Effect Size: {effect}")

# CONFIDENCE INTERVAL
print("\nCONFIDENCE INTERVAL FOR DIFFERENCE IN MEANS")

# Calculate 95% CI for difference in means
mean_diff = ensata_2.mean() - ensata_1.mean()
se = pooled_std * np.sqrt(1/len(ensata_1) + 1/len(ensata_2))
df = len(ensata_1) + len(ensata_2) - 2                       # Degrees of freedom
ci_95 = stats.t.interval(0.95, df, loc=mean_diff, scale=se)

print(f"Mean Difference: {mean_diff:.4f}")
print(f"Standard Error: {se:.4f}")
print(f"95% Confidence Interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

#========================================
# STEP 2 - Produce Test Hypothesis Results
#========================================

# STATISTICAL DECISION
print("\nSTATISTICAL DECISION\n")
alpha = 0.05
if p_value_student < alpha:
    print(f"REJECT the null hypothesis (p={p_value_student:.4e} < alpha={alpha})")
    print(f"Conclusion: There IS a statistically significant difference")
    print(f"between the mean sepal lengths of Iris Ensata in Group 1 and Group 2.")
else:
    print(f"FAIL TO REJECT the null hypothesis (p={p_value_student:.4f} > alpha={alpha})")
    print(f"Conclusion: No significant difference detected.")

# GENERATE VISUALIZATIONS

# Plot 1: Overlapping Histograms
# Create a single figure
plt.figure(figsize=(10, 6))

# Plot histograms using seaborn
sns.histplot(ensata_1, bins=12, alpha=0.6, label='Group 1',
             color='#FF6B6B', edgecolor='black', kde=True)
sns.histplot(ensata_2, bins=12, alpha=0.6, label='Group 2',
             color='#45B7D1', edgecolor='black', kde=True)

# Add mean lines
plt.axvline(ensata_1.mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean Group 1: {ensata_1.mean():.2f}')
plt.axvline(ensata_2.mean(), color='blue', linestyle='--',
            linewidth=2, label=f'Mean Group 2: {ensata_2.mean():.2f}')

# Labels and styling
plt.xlabel('Sepal Length (cm)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('2s_ttest_hist.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 2: Box Plots
# Create DataFrame
data_box = pd.DataFrame({
    'Sepal Length': np.concatenate([ensata_1, ensata_2]),
    'Species': ['Group 1']*len(ensata_1) + ['Group 2']*len(ensata_2)
})

# Create a single figure
plt.figure(figsize=(10, 6))

# Create the boxplot
sns.boxplot(data=data_box, x='Species', y='Sepal Length', hue='Species',
            palette=['#FF6B6B', '#45B7D1'])

# Add title and grid
plt.title('Box Plot Comparison', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('2s_ttest_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 3: Violin Plots with Data Points
plt.figure(figsize=(10, 6))
sns.violinplot(data=data_box, x='Species', y='Sepal Length', hue='Species',
               palette=['#FF6B6B', '#45B7D1'])
sns.swarmplot(data=data_box, x='Species', y='Sepal Length',
              color='black', alpha=0.5, size=3)
plt.title('Violin Plot with Data Points', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('2s_ttest_violin.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot 4: Mean Comparison with 95% Confidence Interval Error Bars
plt.figure(figsize=(10, 6))

# Calculate CIs for each group individually
from scipy.stats import t as t_dist

def mean_ci(data, confidence=0.95):
    n = len(data)
    se = data.std() / np.sqrt(n)
    df_ci = n - 1
    h = t_dist.ppf((1 + confidence) / 2, df_ci) * se
    return data.mean(), h  # returns (mean, half-width)

mean1, ci1 = mean_ci(ensata_1)
mean2, ci2 = mean_ci(ensata_2)

groups = ['Group 1', 'Group 2']
means = [mean1, mean2]
errors = [ci1, ci2]
colors = ['#FF6B6B', '#45B7D1']

plt.errorbar(groups, means, yerr=errors, fmt='o', capsize=8, capthick=2,
             markersize=10, linewidth=2, color='black', zorder=5)

for i, (grp, mean, color) in enumerate(zip(groups, means, colors)):
    plt.scatter([grp], [mean], color=color, s=150, zorder=6)

# Add individual data points (strip plot)
import random
for i, (data, grp, color) in enumerate(zip([ensata_1, ensata_2], groups, colors)):
    x_jitter = [i + random.uniform(-0.08, 0.08) for _ in data]
    plt.scatter(x_jitter, data, color=color, alpha=0.3, s=20, zorder=3)

plt.ylabel('Sepal Length (cm)', fontsize=11)
plt.title('Mean Sepal Length with 95% Confidence Intervals', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('2s_ttest_mean_ci.png', dpi=150, bbox_inches='tight')

plt.show()

#========================================
# STEP 3 - Summarise Results
#========================================

# DETAILED SUMMARY TABLE
print("\nSUMMARY TABLE\n")

summary_df = pd.DataFrame({
    'Group': ['Ensata Group 1', 'Ensata Group 2', 'Difference'],
    'n': [len(ensata_1), len(ensata_2), ''],
    'Mean': [f'{ensata_1.mean():.3f}', f'{ensata_2.mean():.3f}',
             f'{mean_diff:.3f}'],
    'SD': [f'{ensata_1.std():.3f}', f'{ensata_2.std():.3f}', ''],
    'SE': ['', '', f'{se:.4f}']
})
print(summary_df.to_string(index=False))
print('Standard Error relates to approximated range of the mean of the difference')


print("\nREPORT SUMMARY\n")

print(f"Research Question: Does mean sepal length differ between two groups of Ensata Iris?")
print(f"Test Used: Two-sample independent t-test (Student's)")
print(f"Sample Sizes: n₁={len(ensata_1)}, n₂={len(ensata_2)}")
print(f"Test Statistic: t({df}) = {t_stat_student:.4f}")
print(f"P-value: {p_value_student:.4e}")
print(f"Effect Size: Cohen's d = {cohens_d:.4f} ({effect})")
print(f"95% CI for difference (i.e. the difference in means in this range): [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
print(f"\nConclusion: Ensata iris in group 2 have significantly longer sepals than in group 1")
print(f"(p < 0.001). The effect is small to medium and biologically meaningful.")



