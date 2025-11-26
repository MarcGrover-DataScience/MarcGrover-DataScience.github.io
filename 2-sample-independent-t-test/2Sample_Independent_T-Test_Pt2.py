# T-tests
# • One-Sample t-test: Compares the mean of a single sample to a known population mean.
# •	Independent Samples t-test: Compares the means of two independent groups.
# •	Paired Samples t-test: Compares means from the same group at two different times or under two different conditions.
# • ANOVA – One-way – to compare the means of 3+ groups to determine if  statistically significant differences exists
# • ANOVA – Two-way with replication
# • ANOVA – Two-way without replication
# • X^2 (Pearson's Chi-Squared) test of independence – test of independence of two categorical variables.
# • Pearson's X^2 Goodness-Of-Fit test – hypothesis test of a single categorical variable's frequency distribution
#       is significantly different from a hypothesized or known distribution. E.g. shop sales assumed per day


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
# STEP 1 - Test hypothesis that the mean petal length is the same in 2 independent samples
#========================================
# This tests the length of sepal petals for 2 batches of Iris Ensata grown in different conditions
#========================================

df_ensata = pd.read_excel('Iris_ensata.xlsx')
print(df_ensata)
ensata_1 = df_ensata[df_ensata['Test set'] == 1]['Length']
ensata_2 = df_ensata[df_ensata['Test set'] == 2]['Length']
print(ensata_1)


print("TWO-SAMPLE INDEPENDENT T-TEST (STUDENT'S)")

# DESCRIPTIVE STATISTICS
print("\nDESCRIPTIVE STATISTICS")
print(f"Group 1 (Ensata Group 1):  n={len(ensata_1)}, Mean={ensata_1.mean():.3f}, SD={ensata_1.std():.3f}")
print(f"Group 2 (Ensata Group 2):  n={len(ensata_2)}, Mean={ensata_2.mean():.3f}, SD={ensata_2.std():.3f}")
print(f"Difference in Means:  {ensata_2.mean() - ensata_1.mean():.3f}")

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
    effect = "Small to Medium"
elif abs(cohens_d) < 0.8:
    effect = "Medium to Large"
else:
    effect = "Large to Very Large"
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
print("\nSTATISTICAL DECISION")
alpha = 0.05
if p_value_student < alpha:
    print(f"REJECT the null hypothesis (p={p_value_student:.4e} < alpha={alpha})")
    print(f"Conclusion: There IS a statistically significant difference")
    print(f"between the mean sepal lengths of Iris Ensata in Group 1 and Group 2.")
else:
    print(f"FAIL TO REJECT the null hypothesis (p={p_value_student:.4f} > alpha={alpha})")
    print(f"Conclusion: No significant difference detected.")

# VISUALIZATIONS
print("\nCREATING VISUALIZATIONS...")


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
            linewidth=2, label=f'Mean Setosa: {ensata_1.mean():.2f}')
plt.axvline(ensata_2.mean(), color='blue', linestyle='--',
            linewidth=2, label=f'Mean Virginica: {ensata_2.mean():.2f}')

# Labels and styling
plt.xlabel('Sepal Length (cm)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Distribution Comparison', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
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
plt.show()

#========================================
# STEP 3 - Summarise Results
#========================================

# DETAILED SUMMARY TABLE
print("\nSUMMARY TABLE")

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


print("\nREPORT SUMMARY")

print(f"Research Question: Does mean sepal length differ between two groups of Ensata Iris?")
print(f"Test Used: Two-sample independent t-test (Student's)")
print(f"Sample Sizes: n₁={len(ensata_1)}, n₂={len(ensata_2)}")
print(f"Test Statistic: t({df}) = {t_stat_student:.4f}")
print(f"P-value: {p_value_student:.4e}")
print(f"Effect Size: Cohen's d = {cohens_d:.4f} ({effect})")
print(f"95% CI for difference (i.e. the difference in means in this range): [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
print(f"\nConclusion: Ensata iris in group 2 have significantly longer sepals than in group 2")
print(f"(p < 0.001). The effect is small to medium and biologically meaningful.")



