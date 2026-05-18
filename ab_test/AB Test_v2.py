# A/B testing - e.g. assessing campaign performance
#   two versions of something are compared to determine which one performs better at achieving a specific goal.
#   Chi-Squared Test of Independence is a common technique for this

# Load packages
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2
from scipy import stats
from scipy.stats import norm as sp_norm
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
print("---------------------------------------------------------------\n")


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

# Assumptions Checks
print("\nAssumption Checks — Chi-Squared Test of Independence:")

# Assumption 1: Independence of observations
# Each user is independently and randomly assigned to either Control or Treatment — no user appears in both groups by design.
print("\n1. Independence of Observations:")
print("SATISFIED — Each observation is a unique user, randomly assigned to one group only. No user appears in both Control and Treatment.")

# Assumption 2: Minimum expected cell frequency >= 5
# The chi-squared test relies on an approximation to the chi-squared distribution.
# This approximation is unreliable when any expected cell count is < 5. All cells must be checked, not just the observed counts.
print("\n2. Minimum Expected Cell Frequency (all cells must be >= 5):")
_, _, _, expected_assumption_check = chi2_contingency(contingency_table, correction=False)
expected_assumption_df = pd.DataFrame(
    expected_assumption_check,
    index=contingency_table.index,
    columns=contingency_table.columns
)
print(expected_assumption_df)
min_expected = expected_assumption_check.min()
if min_expected >= 5:
    print(f"SATISFIED — All expected frequencies >= 5 (minimum: {min_expected:.1f})")
    print("Chi-squared approximation is reliable.")
else:
    print(f"VIOLATED — Minimum expected frequency: {min_expected:.1f}")
    print("Consider Fisher's Exact Test for small-sample 2x2 tables.")

# Assumption 3: Yates' Continuity Correction
# scipy.stats.chi2_contingency applies Yates' correction by default for 2x2 tables (correction=True).
# This slightly reduces the chi-squared statistic, making the test more conservative and reducing the risk of a Type I error (false positive).
# Both values are reported here for transparency — in large samples the difference is typically negligible.
print("\n3. Yates' Continuity Correction (default for 2x2 tables in scipy):")
chi2_no_yates, p_no_yates, _, _ = chi2_contingency(contingency_table, correction=False)
chi2_with_yates, p_with_yates, _, _ = chi2_contingency(contingency_table, correction=True)
print(f"Without Yates' correction : chi2 = {chi2_no_yates:.4f},  p = {p_no_yates:.6f}")
print(f"With Yates' correction    : chi2 = {chi2_with_yates:.4f},  p = {p_with_yates:.6f}")
if p_no_yates < 0.05 and p_with_yates < 0.05:
    print("ROBUST — Both p-values < 0.05. Conclusion holds regardless of correction.")
elif p_no_yates < 0.05 and p_with_yates >= 0.05:
    print("SENSITIVE — Result changes with correction. Interpret with caution.")
else:
    print("Both p-values >= 0.05. Conclusion holds regardless of correction.")

# Assumption 4: Sample Size Adequacy
print("\n4. Sample Size Adequacy:")
print(f"   Control group:   n = {control_total:,}")
print(f"   Treatment group: n = {treatment_total:,}")
print(f"   Total sample:    n = {control_total + treatment_total:,}")
print("   SATISFIED — Large, balanced sample sizes. The chi-squared")
print("   approximation is fully reliable at this scale.")

# Perform chi-squared test
chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)

chi2_stat_ex1 = chi2_stat   # Preserve Example 1 statistic for the distribution chart below

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
print(f"\nNull hypothesis: There is no difference in conversion rates between the control and treatment groups \nSignificance level (alpha): {alpha}")
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

# colors = ['#3498db', '#e74c3c'] ['#3498db', '#e74c3c']
colors = ['seagreen', 'cornflowerblue']
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
plt.savefig('plot_01_ab_conversion.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Confidence interval plot
plt.figure(figsize=(10, 6))
groups = ['Control', 'Treatment']
rates = [conversion_rates['Conversion_Rate'].iloc[0],
         conversion_rates['Conversion_Rate'].iloc[1]]
ci_lower = [control_ci[0], treatment_ci[0]]
ci_upper = [control_ci[1], treatment_ci[1]]

y_pos = np.arange(len(groups))
plt.barh(y_pos, rates, color=colors, alpha=0.8,
         edgecolor='black', linewidth=0)
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
plt.savefig('plot_02_ab_conversion_ci.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 3: 100% Stacked Proportion Bar Chart
# Shows full group composition (converted + not converted) for each group, not just the conversion rate in isolation.
# This makes the relative scale of the effect immediately apparent and is standard in A/B reporting.

no_pcts = [
    contingency_table.loc['Control', 'No'] / contingency_table.loc['Control'].sum() * 100,
    contingency_table.loc['Treatment', 'No'] / contingency_table.loc['Treatment'].sum() * 100
]
yes_pcts = [
    contingency_table.loc['Control', 'Yes'] / contingency_table.loc['Control'].sum() * 100,
    contingency_table.loc['Treatment', 'Yes'] / contingency_table.loc['Treatment'].sum() * 100
]
group_labels = ['Control', 'Treatment']

fig, ax = plt.subplots(figsize=(8, 6))
bars_no = ax.bar(group_labels, no_pcts, color='darkgrey', alpha=1, label='Not Converted')
bars_yes = ax.bar(group_labels, yes_pcts, bottom=no_pcts, color='mediumseagreen', alpha=1, label='Converted')

for i, (no, yes) in enumerate(zip(no_pcts, yes_pcts)):
    ax.text(i, no / 2, f'{no:.1f}%', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(i, no + yes / 2, f'{yes:.1f}%', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

ax.set_ylabel('Percentage of Group (%)', fontsize=11)
ax.set_title('Conversion Proportions by Group (100% Stacked)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 110)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plot_03_conversion_stacked.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 4: Chi-Squared Distribution — Test Statistic vs Critical Value
# Contextualises the numerical test result by showing where the observed chi-squared statistic sits on the theoretical distribution under H0.
# The rejection region is shaded and the critical value is marked, making the margin of the result immediately visible.

x_range = np.linspace(0, 20, 1000)
y_dist = chi2.pdf(x_range, df=1)  # df=1 for a 2x2 contingency table
critical_val_95 = chi2.ppf(0.95, df=1)  # Critical value at alpha = 0.05, one-tailed

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_range, y_dist, color='steelblue', linewidth=2,
        label='Chi-squared distribution (df = 1)')
ax.fill_between(x_range, y_dist, where=(x_range >= critical_val_95),
                color='#e74c3c', alpha=0.35, label='Rejection region (α = 0.05)')
ax.axvline(x=chi2_stat_ex1, color='slateblue', linestyle='--', linewidth=2,
           label=f'Test statistic = {chi2_stat_ex1:.4f}')
ax.axvline(x=critical_val_95, color='#e74c3c', linestyle=':', linewidth=1.5,
           label=f'Critical value = {critical_val_95:.4f}')

# Annotate test statistic position
ax.annotate(f'  χ² = {chi2_stat_ex1:.2f}\n  p = {p_value:.5f}',
            xy=(chi2_stat_ex1, chi2.pdf(chi2_stat_ex1, df=1)),
            xytext=(chi2_stat_ex1 + 1.5, chi2.pdf(chi2_stat_ex1, df=1) + 0.05),
            fontsize=9, color='slateblue',
            arrowprops=dict(arrowstyle='->', color='slateblue', lw=1.2))

ax.set_xlabel('Chi-squared statistic', fontsize=11)
ax.set_ylabel('Probability Density', fontsize=11)
ax.set_title('Chi-Squared Distribution with Test Statistic\n'
             '(Example 1: Website Conversion Rate A/B Test)', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.set_xlim(0, 20)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plot_04_chi2_distribution.png', dpi=300, bbox_inches='tight')
plt.show()



# ============================================================================
# EXAMPLE 2: Email Campaign A/B Test (Three Variants)
# ============================================================================
print("\n \nEXAMPLE 2: Email Campaign A/B/C Test (Multiple Variants)")
print("---------------------------------------------------------------\n")


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

# Post-hoc Pairwise Comparisons — Example 2 (A/B/C Email Test)
# When a multi-variant chi-squared test is significant, the omnibus result only tells us that at least one variant differs
# — it does not tell us which specific pairs drive the difference.
# Pairwise chi-squared tests are required to isolate those differences.
#
# Multiple testing correction: running three pairwise tests at alpha=0.05 each inflates the family-wise Type I error rate.
# Bonferroni correction divides alpha by the number of comparisons to control for this:
#   Adjusted alpha = 0.05 / 3 pairs = 0.0167

bonferroni_alpha = 0.05 / 3
variant_pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
pairwise_p_values = {}

print("\nPOST-HOC PAIRWISE COMPARISONS (Bonferroni-corrected)")
print(f"Adjusted significance level: alpha / 3 = {bonferroni_alpha:.4f}")

for v1, v2 in variant_pairs:
    pair_subset = email_df[email_df['Variant'].isin([v1, v2])].copy()
    pair_table = pd.crosstab(pair_subset['Variant'], pair_subset['Opened'])
    chi2_pair, p_pair, _, _ = chi2_contingency(pair_table)
    pairwise_p_values[(v1, v2)] = p_pair

    rate_v1 = (pair_table.loc[v1, 'Yes'] / pair_table.loc[v1].sum()) * 100
    rate_v2 = (pair_table.loc[v2, 'Yes'] / pair_table.loc[v2].sum()) * 100
    sig_label = 'Significant *' if p_pair < bonferroni_alpha else 'Not significant'

    print(f"\nVariant {v1} ({rate_v1:.1f}%) vs Variant {v2} ({rate_v2:.1f}%)")
    print(f"  chi2 = {chi2_pair:.4f},  p = {p_pair:.6f}  →  {sig_label}")

print(f"\nSignificant at Bonferroni-corrected alpha = {bonferroni_alpha:.4f}")


# Visualization 5: Email Open Rates with Pairwise Significance Annotations
# Combines the overall open rate bar chart with significance brackets showing
# which pairwise comparisons reach the Bonferroni-corrected threshold.

email_open_rates_list = [
    (email_table.loc['A', 'Yes'] / email_table.loc['A'].sum()) * 100,
    (email_table.loc['B', 'Yes'] / email_table.loc['B'].sum()) * 100,
    (email_table.loc['C', 'Yes'] / email_table.loc['C'].sum()) * 100,
]
email_variant_labels = ['Variant A', 'Variant B', 'Variant C']
email_colors = ['seagreen', 'steelblue', 'slateblue']

fig, ax = plt.subplots(figsize=(9, 7))
bars_email = ax.bar(email_variant_labels, email_open_rates_list,
                    color=email_colors, alpha=0.8)

for bar in bars_email:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
            f'{height:.1f}%', ha='center', va='bottom',
            fontsize=10, fontweight='bold')


def significance_bracket(ax, x1, x2, y, h, label, fontsize=9):
    """Draw a significance bracket between two bar positions."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color='black', linewidth=1, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.1, label,
            ha='center', va='bottom', fontsize=fontsize)


ymax = max(email_open_rates_list)
h = 0.5  # bracket arm height

# A vs B  (x positions: 0 and 1)
p_AB = pairwise_p_values[('A', 'B')]
label_AB = f"p = {p_AB:.4f}{'*' if p_AB < bonferroni_alpha else ' (ns)'}"
significance_bracket(ax, 0, 1, ymax + 2, h, label_AB)

# B vs C  (x positions: 1 and 2)
p_BC = pairwise_p_values[('B', 'C')]
label_BC = f"p = {p_BC:.4f}{'*' if p_BC < bonferroni_alpha else ' (ns)'}"
significance_bracket(ax, 1, 2, ymax + 2, h, label_BC)

# A vs C  (x positions: 0 and 2 — placed highest)
p_AC = pairwise_p_values[('A', 'C')]
label_AC = f"p = {p_AC:.4f}{'*' if p_AC < bonferroni_alpha else ' (ns)'}"
significance_bracket(ax, 0, 2, ymax + 5, h, label_AC)

ax.set_ylabel('Email Open Rate (%)', fontsize=11)
ax.set_title('Email Open Rates by Variant\nwith Bonferroni-Corrected Pairwise Comparisons',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, ymax + 10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add significance key
ax.text(0.98, 0.04,
        f'* p < {bonferroni_alpha:.4f} (Bonferroni-corrected)\nns = not significant',
        transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('plot_05_email_open_rates_pairwise.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================================================
# EXAMPLE 3: Mobile App Feature Test
# ============================================================================
print("\n \nEXAMPLE 3: Mobile App Feature Test")
print("---------------------------------------------------------------\n")

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
print(f"\nNull hypothesis: There is no difference in 7-day retention rates between the Old and New app versions \nSignificance level (alpha): {alpha}")
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
print("---------------------------------------------------------------\n")

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

# Visualisation 6: Statistical Power Curve
# Shows how statistical power (probability of correctly detecting a true effect) increases as sample size grows,
# given the baseline and target conversion rates defined above.
# The minimum required sample size and the actual sample size used are both marked,
# providing immediate visual confirmation that the study was adequately powered.



def compute_power(p1, p2, n, alpha=0.05):
    """
    Compute statistical power for a two-proportion chi-squared test
    given sample size n per group.

    Parameters:
    -----------
    p1     : float  — baseline conversion rate (control)
    p2     : float  — expected conversion rate (treatment)
    n      : int    — sample size per group
    alpha  : float  — significance level (default 0.05, two-sided)

    Returns:
    --------
    float : statistical power (0 to 1)
    """
    p_pooled = (p1 + p2) / 2
    z_alpha = sp_norm.ppf(1 - alpha / 2)
    effect = abs(p2 - p1)
    se_null = np.sqrt(2 * p_pooled * (1 - p_pooled) / n)
    se_alt = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    z_power = (effect - z_alpha * se_null) / se_alt
    return sp_norm.cdf(z_power)


# Compute power across a range of sample sizes
sample_sizes = np.arange(50, 2001, 10)
power_values = [compute_power(baseline_rate, target_rate, n) * 100
                for n in sample_sizes]

# Power at the actual sample size used in Example 1 (n=1,000 per group)
actual_n = 1000
power_at_actual_n = compute_power(baseline_rate, target_rate, actual_n) * 100
power_at_min_n = compute_power(baseline_rate, target_rate, min_n) * 100

print(f"Power at minimum required sample size (n={min_n}):  {power_at_min_n:.1f}%")
print(f"Power at actual sample size (n={actual_n}):          {power_at_actual_n:.1f}%")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sample_sizes, power_values, color='steelblue', linewidth=2,
        label='Statistical Power')
ax.axhline(y=80, color='#e74c3c', linestyle='--', linewidth=1.5,
           label='80% power threshold (conventional minimum)')
ax.axvline(x=min_n, color='slateblue', linestyle=':', linewidth=1.8,
           label=f'Minimum required n = {min_n} per group')
ax.axvline(x=actual_n, color='seagreen', linestyle='-.', linewidth=1.8,
           label=f'Actual sample n = {actual_n:,} per group')

# Mark power at minimum required n
ax.scatter([min_n], [power_at_min_n], color='slateblue', zorder=5, s=60)
ax.text(min_n + 20, power_at_min_n - 4, f'{power_at_min_n:.1f}%',
        color='slateblue', fontsize=9, fontweight='bold')

# Mark power at actual n
ax.scatter([actual_n], [power_at_actual_n], color='seagreen', zorder=5, s=60)
ax.text(actual_n + 20, power_at_actual_n - 4, f'{power_at_actual_n:.1f}%',
        color='seagreen', fontsize=9, fontweight='bold')

ax.set_xlabel('Sample Size per Group (n)', fontsize=11)
ax.set_ylabel('Statistical Power (%)', fontsize=11)
ax.set_title(f'Statistical Power Curve\n'
             f'Detecting a lift from {baseline_rate * 100:.0f}% to {target_rate * 100:.0f}% '
             f'conversion rate  (α = 0.05, two-sided)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(0, 105)
ax.set_xlim(0, max(sample_sizes) + 50)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('plot_06_power_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# SUMMARY AND BEST PRACTICES
# ============================================================================
print("\n\nBEST PRACTICES FOR CHI-SQUARED A/B TESTING")
print("---------------------------------------------------------------\n")


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