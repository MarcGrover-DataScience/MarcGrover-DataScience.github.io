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

#========================================
# STEP 1 - Test hypothesis that the mean petal length is 6.0
#========================================
# One-Sample T-Test in Python
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

print("ONE-SAMPLE T-TEST")
# Load data (example: Iris dataset from Kaggle)
from sklearn.datasets import load_iris
iris = load_iris()
sepal_length = iris.data[:, 0]
print(f'Sample size: {sepal_length.size}')
print('Set of values of sepal_length: ')
print(sepal_length)

# Converting load_iris() into a dataframe
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df_iris)

# Hypothesized population mean
mu_0 = 6.0

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sepal_length, mu_0)

# Calculate descriptive statistics
sample_mean = np.mean(sepal_length)
sample_std = np.std(sepal_length, ddof=1)
sample_size = len(sepal_length)
standard_error = sample_std / np.sqrt(sample_size)

# Calculate confidence interval
confidence_level = 0.95
df = sample_size - 1                    # degrees of freedom
ci = stats.t.interval(confidence_level, df,
                      loc=sample_mean,
                      scale=standard_error)

# Print results
print(f"Sample Mean: {sample_mean:.3f}")
print(f"Standard Deviation: {sample_std:.3f}")
print(f"Standard Error: {standard_error:.3f}")
print(f"T-Statistic: {t_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print(f"Reject H₀: Mean significantly differs from {mu_0}")
else:
    print(f"Fail to reject H₀: No significant difference from {mu_0}")


# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))


sns.histplot(sepal_length, bins=10, kde=True, color='skyblue',
            edgecolor='black', alpha=0.6, ax=axes[0])
axes[0].set_xlabel('Sepal Length (cm)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Distribution of Sepal Length')
axes[0].grid(axis='y', alpha=0.3)


# Confidence interval plot
axes[1].errorbar([1], [sample_mean], yerr=[[sample_mean-ci[0]], [ci[1]-sample_mean]],
                 fmt='o', markersize=10, capsize=10, capthick=2, color='blue')
axes[1].axhline(mu_0, color='red', linestyle='--', linewidth=2, label=f'Hypothesized μ = {mu_0}')
axes[1].set_ylabel('Sepal Length (cm)')
axes[1].set_title('95% Confidence Interval')
axes[1].set_xlim(0.5, 1.5)
axes[1].set_xticks([])
axes[1].legend()

plt.tight_layout()
plt.show()

# Boxplot of sepal lengths - considering all species
df_iris_temp = df_iris
df_iris_temp['Temp'] = 'A'
plt.figure(figsize=(6, 6))
sns.boxplot(data=df_iris_temp, y='sepal length (cm)', hue='Temp', palette='Set2', legend=False)
plt.ylabel('Sepal Length (cm)', fontsize=12)
plt.xlabel('Species', fontsize=12)
plt.title('Distribution of Sepal Lengths by Iris Species', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#========================================
# STEP 2 - Investigate each species separately
#========================================

# Boxplot of sepal lengths - split by each species
plt.figure(figsize=(6, 6))
sns.boxplot(data=df_iris, x='species', y='sepal length (cm)', hue='species', palette='Set2')
plt.ylabel('Sepal Length (cm)', fontsize=12)
plt.xlabel('Species', fontsize=12)
plt.title('Distribution of Sepal Lengths by Iris Species', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary statistics for each species
print("\nSummary Statistics for Sepal Length by species:")
print(df_iris.groupby('species', observed=False)['sepal length (cm)'].describe())

print("\nMean sepal length by species:")
species_list = ['setosa', 'versicolor', 'virginica']
for species in species_list:
    mean_val = df_iris[df_iris['species'] == species]['sepal length (cm)'].mean()
    std_val = df_iris[df_iris['species'] == species]['sepal length (cm)'].std()
    print(f"{species.capitalize()}: {mean_val:.3f} ± {std_val:.3f} cm")

# Histograms of each species separately
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for idx, (species, color) in enumerate(zip(species_list, colors)):
    species_data = df_iris[df_iris['species'] == species]['sepal length (cm)']
    sns.histplot(species_data, bins=10, kde=True, color=color,
                 edgecolor='black', alpha=0.6, ax=axes[idx])
    axes[idx].set_xlabel('Sepal Length (cm)', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].set_title(f'{species.capitalize()}', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Sepal Length Distribution with KDE by Species', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()



#========================================
# STEP 3 - Test hypothesis that the mean petal length is 6.0 for Species = 'versicolor'
#========================================

print('\n\nEvidence exists that the sepal length for the versicolor species may be approximately 6.0. '
      '\nThis is to be tested to further understand this.')
df_versicolor = df_iris[df_iris['species'] == 'versicolor']
sepal_length_versicolor = df_versicolor['sepal length (cm)']
print(sepal_length_versicolor)

# Hypothesized population mean
mu_0 = 6.0

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(sepal_length_versicolor, mu_0)

# Calculate descriptive statistics
sample_mean = np.mean(sepal_length_versicolor)
sample_std = np.std(sepal_length_versicolor, ddof=1)
sample_size = len(sepal_length_versicolor)
standard_error = sample_std / np.sqrt(sample_size)

# Calculate confidence interval
confidence_level = 0.95
df = sample_size - 1                    # degrees of freedom
ci = stats.t.interval(confidence_level, df,
                      loc=sample_mean,
                      scale=standard_error)

# Print results
print(f"Sample Mean: {sample_mean:.3f}")
print(f"Standard Deviation: {sample_std:.3f}")
print(f"Standard Error: {standard_error:.3f}")
print(f"T-Statistic: {t_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")
print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print(f"Reject H₀: Mean significantly differs from {mu_0}")
else:
    print(f"Fail to reject H₀: No significant difference from {mu_0}")
