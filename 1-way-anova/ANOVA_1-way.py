# ANOVA – One-way – to compare the means of 3+ groups to determine if  statistically significant differences exists
# This tests the mean petal lengths of three species of iris and follows on from the T-test.py tests

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
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
# STEP 1 - Assess the data
#========================================

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create a pandas DataFrame
df = pd.DataFrame(data=X, columns=feature_names)
df['species'] = y

# Map target names (0, 1, 2) to actual species names for clarity
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_map)

# Display the first few rows of the data
print("\nOriginal data:")
print(df)
print("\n")

# Prepare the data for the ANOVA test
# We need to pass the petal lengths for each species as a separate array
group1 = df[df['species'] == 'setosa']['petal length (cm)']
group2 = df[df['species'] == 'versicolor']['petal length (cm)']
group3 = df[df['species'] == 'virginica']['petal length (cm)']

# 1: Boxplot of the petal lengths
plt.figure(figsize=(8, 6))
sns.boxplot(x='species', y='petal length (cm)', data=df, hue= 'species', palette='Greens')
plt.title('Petal Length Distribution by Iris Species (Boxplot)')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig("1way_boxplot.png", dpi=150)
plt.show()

# 2: Violin Plot of the petal lengths
plt.figure(figsize=(8, 6))
sns.violinplot(x='species', y='petal length (cm)', data=df, hue= 'species', palette='Greens')
plt.title('Petal Length Distribution by Iris Species (Violin Plot)')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig("1way_violin.png", dpi=150)
plt.show()

# 3: Histograms of each species separately
species_list = ['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (species, color) in enumerate(zip(species_list, colors)):
    species_data = df[df['species'] == species]['petal length (cm)']
    sns.histplot(species_data, bins=10, kde=True, color=color,
                 edgecolor='black', alpha=0.6, ax=axes[idx])
    axes[idx].set_xlabel('Petal Length (cm)', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'{species.capitalize()}', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Petal Length Distribution with KDE by Species', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("1way_histograms_species.png", dpi=150)
plt.show()

# Test for normality (Shapiro-Wilk test)
_, p_group1 = stats.shapiro(group1)
_, p_group2 = stats.shapiro(group2)
_, p_group3 = stats.shapiro(group3)
print(f"\nShapiro-Wilk Normality Test:")
print(f"Setosa: p={p_group1:.4f} {'(Normal)' if p_group1 > 0.05 else '(Non-normal)'}")
print(f"Versicolor: p={p_group2:.4f} {'(Normal)' if p_group2 > 0.05 else '(Non-normal)'}")
print(f"Virginica: p={p_group3:.4f} {'(Normal)' if p_group3 > 0.05 else '(Non-normal)'}")
print(f"Note: With n=50 each, CLT ensures robustness to non-normality")


print("\nFrom both plots, there is a very clear and strong visual difference between the groups. "
      "\nThe 'setosa' species has a much smaller petal length, while 'virginica' has the largest. "
      "\n'Versicolor' is in the middle. This evidence strongly suggests our ANOVA test will find a significant result.")

print("\nNull Hypothesis: The means of petal length are equal for all three species ")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (species, color) in enumerate(zip(species_list, colors)):
    species_data = df[df['species'] == species]['petal length (cm)']
    stats.probplot(species_data, dist="norm", plot=axes[idx])
    axes[idx].set_title(f'Q-Q Plot: {species.capitalize()}', fontsize=12, fontweight='bold')
    axes[idx].get_lines()[0].set_markerfacecolor(color)
    axes[idx].get_lines()[0].set_markeredgecolor('black')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Q-Q Plots — Normality Check by Species', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("1way_qq_plots.png", dpi=150)
plt.show()

# Test for equal variances (Levene's test)
# To check the ANOVA assumption of homogeneity of variance (also called homoscedasticity),
# which states that all groups should have a similar spread (variance).
# The null Hypothesis is that the variance of all three groups in equal

# Determine variances for each group
print("\nGroup Variances (for context)")
print(f"Setosa variance: {group1.var():.4f}")
print(f"Versicolor variance: {group2.var():.4f}")
print(f"Virginica variance: {group3.var():.4f}\n")

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

# Levene's Test: The test confirms this visual inspection. The P-Value is 0.0000 (which is p < 0.05).
# This technically violates the assumption of homogeneity of variance.
# Does this invalidate our ANOVA? In this specific case, no.
# The result of our original ANOVA (F=1180, p-value approx 10^{-91}) was so overwhelmingly significant.
# As such this violation doesn't change the final conclusion.
# The difference between the means is so massive that it's still by far the most important factor.

# The formal solution? If the ANOVA result were less clear (e.g., p = 0.04), we would be more concerned.
# The standard solution for an ANOVA where the variances are unequal is to use a Welch's ANOVA.
# Alternatively, the non-parametric Kruskal-Wallis test also does not assume equal variances.

# ASSUMPTION: Independence of observations
# Each observation belongs to exactly one species group, and no participant appears in more than one group.
# This assumption is satisfied by the structure of the Iris dataset.


#========================================
# STEP 2 - Test hypothesis that the mean petal length of 3 species are the same
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
grand_mean = df['petal length (cm)'].mean()
ss_between = sum(
    len(df[df['species'] == s]) * (df[df['species'] == s]['petal length (cm)'].mean() - grand_mean) ** 2
    for s in species_list
)
ss_total = sum((df['petal length (cm)'] - grand_mean) ** 2)
eta_squared = ss_between / ss_total

print(f"Effect Size (Eta-squared η²): {eta_squared:.4f}")

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
tukey_result = pairwise_tukeyhsd(endog=df['petal length (cm)'],
                                 groups=df['species'],
                                 alpha=0.05)

print("\nTukey's HSD Post-Hoc Test Results")
print(tukey_result)
print("\n")

#  Tukey HSD Visualisation
tukey_result.plot_simultaneous(figsize=(8, 5), ylabel='Species', xlabel='Petal Length (cm)')
plt.title("Tukey's HSD — Simultaneous 95% Confidence Intervals", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("1way_tukey_plot.png", dpi=150)
plt.show()


# --- Calculate Means and 95% Confidence Intervals for each species ---

print("Mean Petal Length and 95% Confidence Intervals by Species:")

# Get the unique species names
species_list = df['species'].unique()

# Loop through each species to calculate and print its statistics
for species in species_list:
    # 1. Filter the DataFrame for the current species
    species_data = df[df['species'] == species]['petal length (cm)']

    # 2. Calculate the required statistics
    mean = np.mean(species_data)
    n = len(species_data)
    std_err = stats.sem(species_data)  # Standard Error of the Mean

    # 3. Define confidence level and degrees of freedom
    confidence_level = 0.95
    degrees_freedom = n - 1

    # 4. Calculate the confidence interval
    #    We provide the confidence level, degrees of freedom,
    #    mean (loc), and standard error (scale)
    ci = stats.t.interval(confidence_level, degrees_freedom, loc=mean, scale=std_err)

    # 5. Print the formatted results
    print(f"\nSpecies: {species}")
    print(f"  Mean Petal Length: {mean:.4f} cm")
    print(f"  95% CI for Mean: ({ci[0]:.4f} cm, {ci[1]:.4f} cm)")

# --- Create the Interval Plot (using pointplot) ---
# Set the figure size
plt.figure(figsize=(10, 7))

# Create the point plot (or interval plot)
# This plot shows the mean as a point (the "point" estimate)
# and the 95% confidence interval as the vertical line.
sns.pointplot(
    x='species',
    y='petal length (cm)',
    data=df,
    hue='species',
    palette='Greens',
    capsize=0.2,  # Adds 'caps' to the confidence interval lines
    linestyle='none'
    # join=False      # Set to False so it doesn't draw a line connecting the species
)

# Add title and labels
plt.title('Mean Petal Length with 95% Confidence Intervals (Interval Plot)', fontsize=16)
plt.xlabel('Species', fontsize=12)
plt.ylabel('Mean Petal Length (cm)', fontsize=12)
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
# Set equal_var=False to run Welch's ANOVA
# f_statistic_welch, p_value_welch = stats.f_oneway(group1, group2, group3, equal_var=False)
# print(f"Welch's F-Statistic: {f_statistic_welch:.4f}")
# print(f"P-Value: {p_value_welch:.4f}\n")
#
# if p_value_welch < 0.05:
#     print("Conclusion: Reject the null hypothesis. There is a significant difference between group means.")
# else:
#     print("Conclusion: Fail to reject the null hypothesis.")

welch_result = pg.welch_anova(data=df, dv='petal length (cm)', between='species')
print(welch_result)