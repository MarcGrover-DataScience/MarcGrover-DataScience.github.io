# One-Sample t-test - Compares the mean of a single sample to a known population mean
# Independent Samples t-test - Compares the means of two independent groups
# Paired Samples t-test - Compares the means from the same group at two different times or conditions
# Chi-Squared (Pearsons Chi-Squared) test of independence – test of independence of two categorical variables.
#   The null hypothesis is that there is no association between the dimensions and the observations.
#   Example is association between gender and smoker/non-smoker, and counting instances
#   It is a test of independence of two categorical variables
# Pearson's chi-squared goodness-of-fit test is a statistical hypothesis test
#   determine if single categorical variable's freq dist is significantly different from hypothesized/ known dist
#   Essentially, it checks how well your observed data "fits" a predefined model or a set of expected proportions.
# A/B testing - e.g. assessing campaign performance
#   two versions of something are compared to determine which one performs better at achieving a specific goal.
#   Chi-Squared Test of Independence is a common technique for this

# Load packages
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
## EXAMPLE 1: Website Conversion Rate A/B Test
# ============================================================================
# Scenario: Testing if a new webpage design increases conversion rates
# Old webpage (control group) vs New webpage (treatment group)
print("CHI-SQUARED ANALYSIS FOR A/B TESTING")
print("EXAMPLE 1: Website Conversion Rate A/B Test")

data = {
    'Group': ['Control'] * 1000 + ['Treatment'] * 1000,
    'Converted': (
        ['No'] * 880 + ['Yes'] * 120 +  # Control: 12% conversion
        ['No'] * 820 + ['Yes'] * 180     # Treatment: 18% conversion
    )
}

df = pd.DataFrame(data)
print(df)
print(f"\nTotal observations: {len(df)}")

# Calculate conversion rates
print("\nConversion Rates by Group:")
conversion_rates = df.groupby('Group')['Converted'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
).reset_index()
conversion_rates.columns = ['Group', 'Conversion_Rate']
print(conversion_rates)

# Prepare data for statistical test
control_conversions = (df[df['Group'] == 'Control']['Converted'] == 'Yes').sum()
control_total = len(df[df['Group'] == 'Control'])
treatment_conversions = (df[df['Group'] == 'Treatment']['Converted'] == 'Yes').sum()
treatment_total = len(df[df['Group'] == 'Treatment'])

# Calculate additional metrics
lift = ((treatment_conversions / treatment_total) - (control_conversions / control_total)) / (control_conversions / control_total) * 100
absolute_diff = (treatment_conversions / treatment_total - control_conversions / control_total) * 100

print(f"\nPerformance Improvement:")
print(f"  Absolute Difference: {absolute_diff:.2f} percentage points")
print(f"  Relative Lift: {lift:.2f}%")

# Create contingency table
contingency_table = pd.crosstab(df['Group'], df['Converted'])
print("\nContingency Table:")
print(contingency_table)

# Perform chi-squared test
chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)

print("\nChi-Squared Test Results:")
print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")
print(f"\nExpected frequencies:")
print(pd.DataFrame(expected_freq,
                   index=contingency_table.index,
                   columns=contingency_table.columns))

# Interpretation
alpha = 0.05
print(f"\nNull hypothesis: There is no variance between the control and sample \nSignificance level (alpha): {alpha}")
if p_value < alpha:
    print(f"RESULT: Reject null hypothesis (p < {alpha})")
    print("There IS a statistically significant difference between groups.")
else:
    print(f"RESULT: Fail to reject null hypothesis (p ≥ {alpha})")
    print("There is NO statistically significant difference between groups.")

# Effect size (Cramér's V)
n = contingency_table.sum().sum()              # Total number of observations
k = min(contingency_table.shape)               # Minimum of rows or columns
cramers_v = (chi2_stat / (n * (k - 1))) ** 0.5
# print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"\nCramér's V: {cramers_v:.4f}")

print("Interpretation: ", end="")
if cramers_v < 0.1:
    print("Small effect")
elif cramers_v < 0.3:
    print("Medium effect")
else:
    print("Large effect")

# Calculate confidence intervals (95%)
def proportion_ci(successes, total, confidence=0.95):
    p = successes / total
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(p * (1 - p) / total)
    margin = z * se
    return (p - margin) * 100, (p + margin) * 100

control_ci = proportion_ci(control_conversions, control_total)
treatment_ci = proportion_ci(treatment_conversions, treatment_total)

print(f"\nControl Group:")
print(f"  Conversion Rate: {control_conversions/control_total*100:.2f}%")
print(f"  95% CI: [{control_ci[0]:.2f}%, {control_ci[1]:.2f}%]")
print(f"\nTreatment Group:")
print(f"  Conversion Rate: {treatment_conversions/treatment_total*100:.2f}%")
print(f"  95% CI: [{treatment_ci[0]:.2f}%, {treatment_ci[1]:.2f}%]")

# Visualization 1: Conversion Rate Comparison
fig, ax = plt.subplots(figsize=(8, 6))

# Bar chart of conversion rates
conversion_data = pd.DataFrame({
    'Group': ['Control', 'Treatment'],
    'Conversion_Rate': [
        (contingency_table.loc['Control', 'Yes'] / contingency_table.loc['Control'].sum()) * 100,
        (contingency_table.loc['Treatment', 'Yes'] / contingency_table.loc['Treatment'].sum()) * 100
    ]
})

colors = ['#3498db', '#e74c3c']
bars = ax.bar(conversion_data['Group'], conversion_data['Conversion_Rate'], color=colors, alpha=0.8)
ax.set_ylabel('Conversion Rate (%)', fontsize=11)
ax.set_title('Conversion Rate by Group', fontsize=12, fontweight='bold')
ax.set_ylim(0, max(conversion_data['Conversion_Rate']) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# Visualization 2: Confidence interval plot
plt.figure(figsize=(10, 6))
groups = ['Control', 'Treatment']
rates = [conversion_rates['Conversion_Rate'].iloc[0],
         conversion_rates['Conversion_Rate'].iloc[1]]
ci_lower = [control_ci[0], treatment_ci[0]]
ci_upper = [control_ci[1], treatment_ci[1]]

y_pos = np.arange(len(groups))
plt.barh(y_pos, rates, color=['#3498db', '#e74c3c'], alpha=0.8,
         edgecolor='black', linewidth=1)
plt.errorbar(rates, y_pos,
             xerr=[[rates[0] - ci_lower[0], rates[1] - ci_lower[1]],
                   [ci_upper[0] - rates[0], ci_upper[1] - rates[1]]],
             fmt='none', color='black', capsize=5, capthick=2, alpha=0.7)
plt.yticks(y_pos, groups)
plt.xlabel('Conversion Rate (%) with 95% CI', fontsize=12, fontweight='bold')
plt.title('Conversion Rates with Confidence Intervals', fontsize=14, fontweight='bold', pad=20)

# Add value labels
for i, (rate, lower, upper) in enumerate(zip(rates, ci_lower, ci_upper)):
    plt.text(rate + 0.5, i, f'{rate:.1f}%\n({lower:.1f}%-{upper:.1f}%)',
             va='center', fontsize=9)
plt.tight_layout()
plt.show()


# ============================================================================
# EXAMPLE 2: Email Campaign A/B Test (Three Variants)
# ============================================================================
print("\n \nEXAMPLE 2: Email Campaign A/B/C Test (Multiple Variants)")

# Testing three different email subject lines
email_data = {
    'Variant': ['A'] * 500 + ['B'] * 500 + ['C'] * 500,
    'Opened': (
            ['No'] * 425 + ['Yes'] * 75 +  # A: 15% open rate
            ['No'] * 400 + ['Yes'] * 100 +  # B: 20% open rate
            ['No'] * 375 + ['Yes'] * 125  # C: 25% open rate
    )
}

email_df = pd.DataFrame(email_data)
print(email_df)

print("\nEmail Campaign Results (%):")
open_rates = email_df.groupby('Variant')['Opened'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
)
print(open_rates)

# Contingency table
email_table = pd.crosstab(email_df['Variant'], email_df['Opened'])
print("\nContingency Table:")
print(email_table)

# Perform chi-squared test
chi2_stat, p_value, dof, expected = chi2_contingency(email_table)

print("Chi-Squared Test Results:")
print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Degrees of freedom: {dof}")

if p_value < 0.05:
    print("At least one variant performs differently from the others.")
    print("Recommendation: Conduct pairwise comparisons.")
else:
    print("No significant difference between variants.")


# ============================================================================
# EXAMPLE 3: Mobile App Feature Test
# ============================================================================
print("\n \nEXAMPLE 3: Mobile App Feature Test")

# Testing if a new onboarding flow increases user retention
# Measuring: Did user return after 7 days?

app_data = {
    'Version': ['Old'] * 800 + ['New'] * 800,
    'Retained': (
            ['No'] * 560 + ['Yes'] * 240 +  # Old: 30% retention
            ['No'] * 480 + ['Yes'] * 320  # New: 40% retention
    )
}

app_df = pd.DataFrame(app_data)
print(app_df)

print("\n7-Day Retention Rates:")
retention_rates = app_df.groupby('Version')['Retained'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100
)
print(retention_rates)

# Create contingency table
app_table = pd.crosstab(app_df['Version'], app_df['Retained'])
print("\napp_Table:")
print(app_table)

# Perform chi-squared test
chi2_stat, p_value, dof, expected = chi2_contingency(app_table)

print("\nChi-Squared Test Results:")
print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.6f}")

# Interpretation
alpha = 0.05
print(f"\nNull hypothesis: There is no variance between the control and sample \nSignificance level (alpha): {alpha}")
if p_value < alpha:
    print(f"RESULT: Reject null hypothesis (p < {alpha})")
    print("There IS a statistically significant difference between groups.")
else:
    print(f"RESULT: Fail to reject null hypothesis (p ≥ {alpha})")
    print("There is NO statistically significant difference between groups.")

# Calculate lift
old_rate = (app_table.loc['Old', 'Yes'] / app_table.loc['Old'].sum()) * 100
new_rate = (app_table.loc['New', 'Yes'] / app_table.loc['New'].sum()) * 100
lift = ((new_rate - old_rate) / old_rate) * 100

print(f"\nRelative Lift: {lift:.2f}%")
print(f"Absolute Difference: {new_rate - old_rate:.2f} percentage points")

# Effect size (Cramér's V)
n = app_table.sum().sum()              # Total number of observations
k = min(app_table.shape)               # Minimum of rows or columns
cramers_v = (chi2_stat / (n * (k - 1))) ** 0.5
print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"Cramér's V: {cramers_v:.4f}")

print("Interpretation: ", end="")
if cramers_v < 0.1:
    print("Small effect")
elif cramers_v < 0.3:
    print("Medium effect")
else:
    print("Large effect")



# ============================================================================
# EXAMPLE 4: Sample Size and Power Analysis
# ============================================================================
# Power refers to the probability that your test will detect an effect when there actually is one.
# Typically, this is 80% i.e. 80% chance of detecting a real effect should one exist.
# Power = Probability of rejecting the null hypothesis when it's actually false
# Note, alpha is 5% chance of false positive
print("\n \nEXAMPLE 4: Sample Size and Power Analysis")

def minimum_sample_size(p1, p2, alpha=0.05, power=0.80):
    """
    Estimate minimum sample size per group for A/B test

    Parameters:
    -----------
    p1 : float
        Expected conversion rate for control (e.g., 0.10 for 10%)
    p2 : float
        Expected conversion rate for treatment (e.g., 0.12 for 12%)
    alpha : float
        Significance level (default: 0.05)
    power : float
        Statistical power (default: 0.80)

    Returns:
    --------
    int : Minimum sample size per group
    """
    from scipy.stats import norm

    # Z-scores for alpha and beta
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Effect size
    effect_size = abs(p2 - p1)

    # Calculate sample size
    n = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / effect_size ** 2

    return int(np.ceil(n))


# Example calculation
baseline_rate = 0.11  # 11% conversion rate
target_rate = 0.16  # 16% conversion rate

min_n = minimum_sample_size(baseline_rate, target_rate)

print(f"\nScenario:")
print(f"  Baseline conversion rate: {baseline_rate * 100:.1f}%")
print(f"  Target conversion rate: {target_rate * 100:.1f}%")
print(f"  Relative lift: {((target_rate / baseline_rate - 1) * 100):.1f}%")
print(f"\nMinimum sample size per group: {min_n:,}")
print(f"Total participants needed: {min_n * 2:,}")

# ============================================================================
# SUMMARY AND BEST PRACTICES
# ============================================================================
print("\n\nBEST PRACTICES FOR CHI-SQUARED A/B TESTING")

best_practices = """
1. SAMPLE SIZE
   - Calculate required sample size before starting
   - Ensure minimum expected frequency ≥ 5 in each cell

2. RANDOMIZATION
   - Randomly assign users to control/treatment groups
   - Ensure groups are comparable at baseline

3. INTERPRETATION
   - Check p-value against significance level (typically α = 0.05)
   - Consider practical significance, not just statistical
   - Report effect size (Cramér's V) alongside p-value

4. AVOID COMMON PITFALLS
   - Don't peek at results early (increases false positives)
   - Don't run multiple tests without correction
   - Account for seasonal effects and external factors

5. REPORTING
   - Report both conversion rates and confidence intervals
   - Include sample sizes for transparency
   - Discuss practical implications of findings
---------------------------------------------------------------
"""

print(best_practices)