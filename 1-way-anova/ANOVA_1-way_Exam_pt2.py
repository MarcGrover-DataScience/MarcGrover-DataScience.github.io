# ANOVA – One-way – to compare the means of 3+ groups to determine if  statistically significant differences exists
# This tests the mean exam scores of three groups of students using different training courses


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg

# Set Dataframe printing options
desired_width=320                                                   # shows columns with X or fewer characters
pd.set_option("display.width", desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option("display.max_columns", 10)                            # shows Y columns in the display
pd.set_option("display.max_rows", 20)                               # shows Z rows in the display
pd.set_option("display.min_rows", 10)                               # defines the minimum number of rows to show
pd.set_option("display.precision", 3)                               # displays numbers to 3 dps

#========================================
# STEP 1 - Generate the data
#========================================

# Declare participant numbers
n_participants = 100
np.random.seed(42)

# Simulate data:
# Simulate course 1 exam results
group1 = np.random.normal(loc=75.5, scale=10, size=n_participants)
group1 = np.clip(group1, 25, 99)    # Contrains values to a realistic range
group1 = np.round(group1, 1)            # round to 1 decimal place
# print(group1)

# Simulate course 2 exam results
group2 = np.random.normal(loc=76, scale=11, size=n_participants)
group2 = np.clip(group2, 25, 99)  # Contrains values to a realistic range
group2 = np.round(group2, 1)            # round to 1 decimal place
# print(group2)

# Simulate course 3 exam results
group3 = np.random.normal(loc=72, scale=10.5, size=n_participants)
group3 = np.clip(group3, 25, 99)  # Contrains values to a realistic range
group3 = np.round(group3, 1)            # round to 1 decimal place
# print(group3)

# Create single dataframe of all results
all_samples = np.concatenate([group1, group2, group3])
groups = ['Group1']*100 + ['Group2']*100 + ['Group3']*100
df = pd.DataFrame({'Score': all_samples,
                   'Group': groups
                   })
print(df)

# 1: Boxplot of exam scores by group
plt.figure(figsize=(8, 6))
sns.boxplot(x='Group', y='Score', data=df, hue= 'Group', palette='Greens')
plt.title('Exam Results by Group')
plt.xlabel('Group')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.savefig("1way_boxplot.png", dpi=150)
plt.show()

# 2: Violin plot of exam scores by group
plt.figure(figsize=(8, 6))
sns.violinplot(x='Group', y='Score', data=df, hue= 'Group', palette='Greens')
plt.title('Exam Results Distribution by Group (Violin Plot)')
plt.xlabel('Group')
plt.ylabel('Exam Score')
plt.tight_layout()
plt.savefig("1way_violin.png", dpi=150)
plt.show()

# 3: Histograms of each group separately
group_list = ['Group1', 'Group2', 'Group3']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (group, color) in enumerate(zip(group_list, colors)):
    group_data = df[df['Group'] == group]['Score']
    sns.histplot(group_data, bins=10, kde=True, color=color,
                 edgecolor='black', alpha=0.6, ax=axes[idx])
    axes[idx].set_xlabel('Score', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'{group.capitalize()}', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Exam Score Distribution with KDE by Group', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("1way_histograms_group.png", dpi=150)
plt.show()

# 4: Descriptive stats by group
print("\nDESCRIPTIVE STATISTICS BY GROUP")
for group in group_list:
    data = df[df['Group'] == group]['Score']
    print(f"\n{group}:")
    print(f"  Mean:  {data.mean():.3f}")
    print(f"  SD:    {data.std():.3f}")
    print(f"  SE:    {stats.sem(data):.3f}")
    print(f"  Min:   {data.min():.1f}")
    print(f"  Max:   {data.max():.1f}")


# Test for normality (Shapiro-Wilk test)
_, p_group1 = stats.shapiro(group1)
_, p_group2 = stats.shapiro(group2)
_, p_group3 = stats.shapiro(group3)
print(f"\nShapiro-Wilk Normality Test:")
print(f"Group1: p={p_group1:.4f} {'(Normal)' if p_group1 > 0.05 else '(Non-normal)'}")
print(f"Group2: p={p_group2:.4f} {'(Normal)' if p_group2 > 0.05 else '(Non-normal)'}")
print(f"Group3: p={p_group3:.4f} {'(Normal)' if p_group3 > 0.05 else '(Non-normal)'}")
print(f"Note: With n=100 each, CLT ensures robustness to non-normality")

print("\nFrom the plots, there is a clear similarity between groups 1 and 2. "
      "\nThe group 3 results look lower than groups 1 and 2 in regards to the mean and the quartiles."
      "\n'This evidence suggests the ANOVA test may find a significant result, but it isn't clear.")

print("\nNull Hypothesis: The means of exam scores are equal for all three groups ")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (group, color) in enumerate(zip(group_list, colors)):
    group_data = df[df['Group'] == group]['Score']
    stats.probplot(group_data, dist="norm", plot=axes[idx])
    axes[idx].set_title(f'Q-Q Plot: {group.capitalize()}', fontsize=12, fontweight='bold')
    axes[idx].get_lines()[0].set_markerfacecolor(color)
    axes[idx].get_lines()[0].set_markeredgecolor('black')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Q-Q Plots — Normality Check by Group', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("1way_qq_plots.png", dpi=150)
plt.show()

# Test for equal variances (Levene's test)
# To check the ANOVA assumption of homogeneity of variance (also called homoscedasticity),
# which states that all groups should have a similar spread (variance).
# The null Hypothesis is that the variance of all three groups in equal

# Determine variances for each group
print("\nGroup Variances (for context)")
print(f"Group1 variance: {group1.var():.4f}")
print(f"Group2 variance: {group2.var():.4f}")
print(f"Group3 variance: {group3.var():.4f}\n")

# Perform Levene's Test
print("\nLevene's Test for Equal Variances")
levene_stat, levene_p = stats.levene(group1, group2, group3)

print(f"Levene's Statistic: {levene_stat:.4f}")
print(f"P-Value: {levene_p:.4f}\n")

# Interpretation of Levene's Test
alpha = 0.05
if levene_p < alpha:
    print(f"Result (p={levene_p:.4f}): We REJECT the null hypothesis.")
    print("Conclusion: The variances are NOT equal.")
    print("The ANOVA assumption of homogeneity of variance is VIOLATED.")
else:
    print(f"Result (p={levene_p:.4f}): We FAIL to reject the null hypothesis.")
    print("Conclusion: The variances ARE equal.")
    print("The ANOVA assumption of homogeneity of variance is MET.")

# Levene's Test: The test confirms this visual inspection. The P-Value is 0.181 (which is p > 0.05).
# This does not violate the assumption of homogeneity of variance.

#========================================
# STEP 2 - Test hypothesis that the mean exam scores of 3 groups are the same
#========================================

# Perform the one-way ANOVA
f_statistic, p_value = stats.f_oneway(group1, group2, group3)

print("\nOne-Way ANOVA Results")
print(f"F-Statistic: {f_statistic:.2f}")
print(f"P-Value: {p_value}")
print("\n")

# Interpretation of ANOVA Test
alpha = 0.05
if p_value < alpha:
    print(f"Result (p={p_value:.4f}): We REJECT the null hypothesis.")
    print("Conclusion: The means are NOT equal.")
else:
    print(f"Result (p={p_value:.4f}): We FAIL to reject the null hypothesis.")
    print("Conclusion: The means ARE equal.")

# Effect Size
# Eta-squared: SS_between / SS_total
grand_mean = df['Score'].mean()
ss_between = sum(
    len(df[df['Group'] == s]) * (df[df['Group'] == s]['Score'].mean() - grand_mean) ** 2
    for s in group_list
)
ss_total = sum((df['Score'] - grand_mean) ** 2)
eta_squared = ss_between / ss_total

print(f"\nEffect Size (Eta-squared η²): {eta_squared:.4f}")

# Interpret eta-squared
if eta_squared < 0.01:
    eta_interpretation = "negligible"
elif eta_squared < 0.06:
    eta_interpretation = "small"
elif eta_squared < 0.14:
    eta_interpretation = "medium"
else:
    eta_interpretation = "large"

print(f"Effect size interpretation: {eta_interpretation}")


# --- Post-Hoc Test: Tukey's HSD ---
# This test will compare all pairs of groups

# We use the raw data and the group labels
tukey_result = pairwise_tukeyhsd(endog=df['Score'],
                                 groups=df['Group'],
                                 alpha=0.05)

print("\nTukey's HSD Post-Hoc Test Results")
print(tukey_result)
print("\n")

#  Tukey HSD Visualisation
tukey_result.plot_simultaneous(figsize=(8, 5), ylabel='Group', xlabel='Score')
plt.title("Tukey's HSD — Simultaneous 95% Confidence Intervals", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("1way_tukey_plot.png", dpi=150)
plt.show()


# --- Calculate Means and 95% Confidence Intervals for each group ---

print("Mean Exam Score and 95% Confidence Intervals by Group:")

# Loop through each group to calculate and print its statistics
for group in group_list:
    # 1. Filter the DataFrame for the current group
    group_data = df[df['Group'] == group]['Score']

    # 2. Calculate the required statistics
    mean = np.mean(group_data)
    n = len(group_data)
    std_err = stats.sem(group_data)  # Standard Error of the Mean

    # 3. Define confidence level and degrees of freedom
    confidence_level = 0.95
    degrees_freedom = n - 1

    # 4. Calculate the confidence interval
    #    We provide the confidence level, degrees of freedom,
    #    mean (loc), and standard error (scale)
    ci = stats.t.interval(confidence_level, degrees_freedom, loc=mean, scale=std_err)

    # 5. Print the formatted results
    print(f"\nGroup: {group}")
    print(f"  Mean Exam Score: {mean:.4f}")
    print(f"  95% CI for Mean: ({ci[0]:.4f}, {ci[1]:.4f})")

# --- Create the Interval Plot (using pointplot) ---
# Set the figure size
plt.figure(figsize=(10, 7))

# Create the point plot (or interval plot)
# This plot shows the mean as a point (the "point" estimate)
# and the 95% confidence interval as the vertical line.
sns.pointplot(
    x='Group',
    y='Score',
    data=df,
    hue='Group',
    palette='Blues',
    capsize=0.2,  # Adds 'caps' to the confidence interval lines
    linestyle='none'
)
# Add title and labels
plt.title('Mean Scores with 95% Confidence Intervals (Interval Plot)', fontsize=16)
plt.xlabel('Group', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7) # Add a grid for easier reading
plt.tight_layout()
plt.savefig("1way_point_plot.png", dpi=150)
plt.show()

# Note that for as the sample sizes are greater than 30, the Central Limit Theorem applies.
# This theorem states that the sampling distribution of the means will be approximately normal,
# regardless of the original data's distribution. Since ANOVA compares means, the test will still be valid.

# In general ANOVA is more robust when groups have a similar size and in this case they are equal
# If samples are small and normality tests (especially visual plots) show that the data is very non-normal
# (e.g., heavily skewed), other tests may be more applicable, for example the Kruskal-Wallis H Test.
# This is essentially the non-parametric version of the one-way ANOVA.


# Perform the Welch's one-way ANOVA
print("\nWelch's ANOVA (for completeness)")

# Perform the test

welch_result = pg.welch_anova(data=df, dv='Score', between='Group')
print(welch_result)

welch_p = welch_result['p_unc'].values[0]
if welch_p < 0.05:
    print(f"\nWelch's ANOVA Result (p={welch_p:.4f}): Reject H₀ — consistent with standard ANOVA conclusion.")
else:
    print(f"\nWelch's ANOVA Result (p={welch_p:.4f}): Fail to reject H₀.")