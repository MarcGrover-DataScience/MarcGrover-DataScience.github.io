# Paired Samples t-test: Compares means from the same group at two different times or under two different conditions.
# Test hypothesis that sleep therapy intervention results in no difference to sleep hours before and after a sleep therapy intervention
# This tests the sleep hours before and after a sleep therapy intervention for 30 participants

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

# STEP 1 - GENERATE DATA

# Declare participant numbers
n_participants = 30

# Simulate data:
# Simulate sleep hours before treatment (typically lower)
np.random.seed(42)
sleep_before = np.random.normal(loc=5.5, scale=1.2, size=n_participants)
sleep_before = np.clip(sleep_before, 3, 8)  # Contrains values to a realistic range

# Simulate sleep after treatment (with some improvement)
# Adding individual improvement (not everyone responds the same)
improvement = np.random.normal(loc=0.5, scale=0.5, size=n_participants)
sleep_after = sleep_before + improvement
sleep_after = np.clip(sleep_after, 3, 10)

# Create DataFrame
df = pd.DataFrame({
    'Participant_ID': range(1, n_participants + 1),
    'Sleep_Before': sleep_before,
    'Sleep_After': sleep_after
})

# Calculate differences
df['Difference'] = df['Sleep_After'] - df['Sleep_Before']

print("\nPAIRED SAMPLE T-TEST: SLEEP THERAPY EFFECTIVENESS")
print("\nResearch Question:")
print("Does sleep therapy significantly improve sleep duration?")
print("\nNull Hypothesis (H0): Mean difference in sleep hours = 0")
print("Alternative Hypothesis (H1): Mean difference in sleep hours ≠ 0")

# Display sample data
print("\nSample Data:")
print(df)

# STEP 2 - EXPLORATORY DATA ANALYSIS

# Descriptive Statistics
print("\nDESCRIPTIVE STATISTICS\n")
print("NOTE - There is no need to test for normality of the Before and After data - the differences "
      "do need to be tested for normality (Shapiro-Wilk Test)")
print("NOTE - There is no need to test for equal variance between Before and After data, as the data "
      "relates to the the same subjects")

print(f"\nBefore Treatment:")
print(f"  Mean: {df['Sleep_Before'].mean():.3f} hours")
print(f"  SD:   {df['Sleep_Before'].std():.3f} hours")
print(f"  Min:  {df['Sleep_Before'].min():.3f} hours")
print(f"  Max:  {df['Sleep_Before'].max():.3f} hours")

print(f"\nAfter Treatment:")
print(f"  Mean: {df['Sleep_After'].mean():.3f} hours")
print(f"  SD:   {df['Sleep_After'].std():.3f} hours")
print(f"  Min:  {df['Sleep_After'].min():.3f} hours")
print(f"  Max:  {df['Sleep_After'].max():.3f} hours")

print(f"\nDifference (After - Before):")
print(f"  Mean: {df['Difference'].mean():.3f} hours")
print(f"  SD:   {df['Difference'].std():.3f} hours")
SE = stats.sem(df['Difference'])
print(f"  SE:  {SE:.3f} hours")
print("Standard Error measures the variability of the sample mean across different possible samples.")
# SE = SD / sqrt(n)
# SD = Variability of **individual differences** in your sample
# SE = Uncertainty about the **population mean** based on your sample

#STEP 3 - PAIRED SAMPLE T-TEST

# Perform Paired Sample T-Test
print("\n\nPAIRED SAMPLE T-TEST RESULTS")

t_statistic, p_value = stats.ttest_rel(df['Sleep_After'], df['Sleep_Before'])

print(f"\nT-statistic: {t_statistic:.4f}")
print(f"P-value:     {p_value:.6f}")
print(f"Degrees of freedom: {n_participants - 1}")

# Effect size (Cohen's d for paired samples)
mean_diff = df['Difference'].mean()
std_diff = df['Difference'].std()
cohens_d = mean_diff / std_diff

print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")

# Interpret effect size
if abs(cohens_d) < 0.2:
    effect_interpretation = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interpretation = "small"
elif abs(cohens_d) < 0.8:
    effect_interpretation = "medium"
else:
    effect_interpretation = "large"

print(f"Effect size interpretation: {effect_interpretation}")

# 95% Confidence Interval for the mean difference
confidence_level = 0.95
confidence_interval = stats.t.interval(
    confidence_level,
    df=n_participants - 1,
    loc=mean_diff,
    scale=stats.sem(df['Difference'])  # stats.sem() calculates standard error of the mean
)

print(f"\n95% Confidence Interval for mean difference:")
print("95% CI = Mean Difference ± (t-critical × SE)")
t_critical = stats.t.ppf(0.5+(confidence_level*0.5), df=29)
CI_range = t_critical * SE
# print(CI_range)
print(f"  {mean_diff:.3f} ± {CI_range:.3f}")
print(f"  [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}] hours")

# STEP 4 - STATISTICAL INTERPRETATION

print("\nINTERPRETATION\n")

alpha = 0.05
print(f"\nSignificance level (alpha): {alpha}")

if p_value < alpha:
    print(f"\nRESULT: STATISTICALLY SIGNIFICANT (p = {p_value:.6f} < {alpha})")
    print("\nConclusion:")
    print(f"  We reject the null hypothesis. There is sufficient evidence to")
    print(f"  conclude that sleep therapy significantly affects sleep duration.")
    print(f"  On average, participants slept {mean_diff:.3f} hours more after")
    print(f"  treatment (95% CI: [{confidence_interval[0]:.3f}, {confidence_interval[1]:.3f}]).")
    print(f"  The effect size is {effect_interpretation} (d = {cohens_d:.3f}).")
else:
    print(f"\nRESULT: NOT STATISTICALLY SIGNIFICANT (p = {p_value:.6f} ≥ {alpha})")
    print("\nConclusion:")
    print(f"  We fail to reject the null hypothesis. There is insufficient")
    print(f"  evidence to conclude that sleep therapy significantly affects")
    print(f"  sleep duration.")

# STEP 5 - CHECK ASSUMPTIONS

print("\nASSUMPTION CHECKS")

# 1. Normality of differences (Shapiro-Wilk test)
shapiro_stat, shapiro_p = stats.shapiro(df['Difference'])
print("\n1. Normality of Differences (Shapiro-Wilk Test):")
print(f"   W-statistic: {shapiro_stat:.4f}")        # Measures how well the data matches the normal curve
print(f"   P-value: {shapiro_p:.4f}")               # Probability of W-statistic if the data truly normally distributed

if shapiro_p > 0.05:
    print("   Differences are normally distributed (p > 0.05)")
else:
    print("   Differences may not be normally distributed (p ≤ 0.05)")
    print("   Note: With larger samples (n>30), t-test is robust to violations")

# Visualizations

# 1. Before vs After comparison
plt.figure(figsize=(8, 8))

# Plot lines connecting each participant's before and after
for i in range(len(df)):
    plt.plot([1, 2],
             [df.loc[i, 'Sleep_Before'], df.loc[i, 'Sleep_After']],
             'o-', alpha=0.4, color='gray', linewidth=1)

# Add mean lines
mean_before = df['Sleep_Before'].mean()
mean_after = df['Sleep_After'].mean()
plt.plot([1, 2], [mean_before, mean_after], 'o-',
         color='red', linewidth=3, markersize=10, label='Mean')

plt.xlim(0.5, 2.5)
plt.xticks([1, 2], ['Before', 'After'], fontsize=12)
plt.ylabel('Sleep Hours', fontsize=12)
plt.title('Sleep Hours: Before vs After Treatment', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("paired_before_after_comparison.png", dpi=150)
plt.show()


# 2. Box Plot Comparison with Individual Points Overlaid
plt.figure(figsize=(8, 6))

# Reshape data for Seaborn (long format)
df_long = pd.melt(df,
                  id_vars=['Participant_ID'],
                  value_vars=['Sleep_Before', 'Sleep_After'],
                  var_name='Condition',
                  value_name='Sleep_Hours')

# Rename conditions for better labels
df_long['Condition'] = df_long['Condition'].replace({
    'Sleep_Before': 'Before',
    'Sleep_After': 'After'
})

# Box plot
sns.boxplot(data=df_long,
            x='Condition',
            y='Sleep_Hours',
            hue='Condition',
            palette={'Before': 'seagreen', 'After': 'slateblue'},
            width=0.5,
            legend=False)

# Overlay individual points
sns.stripplot(data=df_long,
              x='Condition',
              y='Sleep_Hours',
              color='black',
              size=4,
              alpha=0.3)

plt.ylabel('Sleep Hours', fontsize=12)
plt.xlabel('')
plt.title('Distribution with Individual Values', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("paired_before_after_boxplot.png", dpi=150)
plt.show()


# 3. Difference distribution (histogram)
# Calculate mean difference
mean_diff = df['Difference'].mean()

# Create the histogram with KDE
plt.figure(figsize=(10, 6))
ax =sns.histplot(data=df,
             x='Difference',
             bins=8,
             kde=True,
             color='seagreen',
             alpha=0.7,
             edgecolor='grey',
             linewidth=0.5)
ax.lines[0].set_color('darkgreen')

# Add mean line
plt.axvline(mean_diff, color='red', linestyle='--', linewidth=2,
            label=f'Mean = {mean_diff:.3f}')

# Add zero reference line
plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5,
            label='No Change')

plt.xlabel('Difference (After - Before) Hours', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Differences', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("paired_difference_distribution.png", dpi=150)
plt.show()


# 4. Q-Q plot for normality check
# Create the Q-Q plot
plt.figure(figsize=(8, 6))
stats.probplot(df['Difference'], dist="norm", plot=plt)

plt.title('Q-Q Plot (Normality Check for Differences)', fontsize=14, fontweight='bold')
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Sample Quantiles', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("paired_q_q.png", dpi=150)
plt.show()


# 5. Before vs After comparison (Decreases highlighted)
plt.figure(figsize=(8, 8))

# Plot lines connecting each participant's before and after
for i in range(len(df)):
    before = df.loc[i, 'Sleep_Before']
    after = df.loc[i, 'Sleep_After']

    # Color orange if sleep decreased, gray if increased or stayed same
    if after < before:
        color = 'orange'
        alpha = 0.7
        linewidth = 1.5
    else:
        color = 'gray'
        alpha = 0.4
        linewidth = 1

    plt.plot([1, 2], [before, after],
             'o-', alpha=alpha, color=color, linewidth=linewidth)

# Add mean lines
mean_before = df['Sleep_Before'].mean()
mean_after = df['Sleep_After'].mean()
plt.plot([1, 2], [mean_before, mean_after], 'o-',
         color='red', linewidth=3, markersize=10, label='Mean', zorder=5)

plt.xlim(0.5, 2.5)
plt.xticks([1, 2], ['Before', 'After'], fontsize=12)
plt.ylabel('Sleep Hours', fontsize=12)
plt.title('Sleep Hours: Before vs After Treatment', fontsize=14, fontweight='bold')

# Add custom legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.6, label='Improved/Same'),
    Line2D([0], [0], color='orange', linewidth=1.5, alpha=0.7, label='Worsened'),
    Line2D([0], [0], color='red', linewidth=3, marker='o', label='Mean')
]
plt.legend(handles=legend_elements, loc='best')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("paired_difference_decreases.png", dpi=150)
plt.show()

# ADVANCED RESULTS

# 6. 95% CONFIDENCE INTERVAL FOREST PLOT
# A forest plot (also called a CI plot) is the standard visual in clinical and intervention research for communicating
# a point estimate alongside its uncertainty. The key features are:
#   - A horizontal line spanning the 95% CI
#   - A diamond (or square) at the mean difference
#   - A vertical zero line as the "no effect" reference
#
# If the CI does not cross zero, the result is statistically significant at the chosen alpha level — the visual makes this immediately legible.

fig, ax = plt.subplots(figsize=(9, 8))

# Shaded band across the CI
ax.barh(0,
        confidence_interval[1] - confidence_interval[0],
        left=confidence_interval[0],
        height=0.22,
        color='steelblue', alpha=0.20,
        label='95% Confidence Interval')

# Horizontal whisker line
ax.plot(confidence_interval, [0, 0], color='steelblue', linewidth=2.5)

# End caps on whiskers
for x in confidence_interval:
    ax.plot([x, x], [-0.07, 0.07], color='steelblue', linewidth=2.5)

# Mean difference — diamond shape
diamond_x = [mean_diff, mean_diff - 0.018, mean_diff, mean_diff + 0.018, mean_diff]
diamond_y = [0.10, 0, -0.10, 0, 0.10]
ax.fill(diamond_x, diamond_y, color='steelblue', zorder=5,
        label=f'Mean difference = {mean_diff:.3f} hrs')

# Zero reference line
ax.axvline(0, color='red', linestyle='--', linewidth=1.8,
           label='No effect (zero line)', zorder=3)

# Annotations: lower CI
ax.annotate(f'Lower 95% CI\n{confidence_interval[0]:.3f} hrs',
            xy=(confidence_interval[0], 0.01),
            xytext=(confidence_interval[0] - 0.07, 0.24),
            fontsize=9, ha='center', color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

# Annotations: upper CI
ax.annotate(f'Upper 95% CI\n{confidence_interval[1]:.3f} hrs',
            xy=(confidence_interval[1], 0.01),
            xytext=(confidence_interval[1] + 0.07, 0.24),
            fontsize=9, ha='center', color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

# Annotations: mean difference (below)
ax.annotate(f'Mean diff\n{mean_diff:.3f} hrs',
            xy=(mean_diff, -0.11),
            xytext=(mean_diff, -0.30),
            fontsize=9, ha='center', color='steelblue',
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

# Significance label
if confidence_interval[0] > 0 or confidence_interval[1] < 0:
    sig_text = "CI does not cross zero → Statistically significant (p < 0.05)"
    sig_color = 'darkgreen'
else:
    sig_text = "CI crosses zero → Not statistically significant (p ≥ 0.05)"
    sig_color = 'red'
ax.text(0.5, -0.42, sig_text, transform=ax.transAxes,
        ha='center', fontsize=10, color=sig_color, style='italic')

# Axis formatting
x_margin = 0.30
ax.set_xlim(confidence_interval[0] - x_margin, confidence_interval[1] + x_margin)
ax.set_ylim(-0.50, 0.55)
ax.set_yticks([])
ax.set_xlabel('Mean Difference in Sleep Hours (After − Before)', fontsize=11)
ax.set_title('95% Confidence Interval for Mean Difference\n(Paired Samples t-test)',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['left', 'right', 'top']].set_visible(False)

plt.tight_layout()
plt.savefig("paired_ci_forest_plot.png", dpi=150)
plt.show()


# POWER ANALYSIS

# Power analysis answers two questions:
#   1. Given the effect size we observed, how well-powered was this study?
#   2. How many participants would we have needed to reliably detect this effect?
#
# Statistical power = P(correctly rejecting H0 | H0 is false)
# Convention: 80% power is the widely accepted minimum threshold.
# A study with low power risks a Type II error (missing a real effect).

print("\nPOWER ANALYSIS")

def paired_ttest_power(effect_size, n, alpha=0.05):
    """
    Parameters:
        effect_size : Cohen's d
        n           : number of pairs (participants)
        alpha       : significance level (default 0.05, two-tailed)
    Returns:
        power (float between 0 and 1)
    """
    degrees_of_freedom = n - 1
    t_critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
    ncp = effect_size * np.sqrt(n)          # non-centrality parameter
    power = (1 - stats.t.cdf(t_critical, degrees_of_freedom, loc=ncp)
               + stats.t.cdf(-t_critical, degrees_of_freedom, loc=ncp))
    return power


def required_n_for_power(effect_size, target_power=0.80, alpha=0.05):
    """
    Find the minimum n needed to achieve a target power level.

    Parameters:
        effect_size  : Cohen's d
        target_power : desired power (default 0.80)
        alpha        : significance level (default 0.05)
    Returns:
        minimum n (int)
    """
    for n in range(2, 1000):
        if paired_ttest_power(effect_size, n, alpha) >= target_power:
            return n
    return None


# Achieved power for this study
achieved_power = paired_ttest_power(cohens_d, n_participants)

# Minimum n required for common power thresholds
n_for_80 = required_n_for_power(cohens_d, target_power=0.80)
n_for_90 = required_n_for_power(cohens_d, target_power=0.90)
n_for_95 = required_n_for_power(cohens_d, target_power=0.95)

alpha = 0.05   # reaffirm alpha (significance level) for all additions below

print(f"\nEffect size used (Cohen's d):     {cohens_d:.4f}  [{effect_interpretation}]")
print(f"Significance level (alpha):       {alpha}")
print(f"\nAchieved power (n={n_participants}):           {achieved_power:.4f}  ({achieved_power * 100:.1f}%)")

print(f"\nSample size requirements:")
print(f"  For 80% power:  n = {n_for_80}  participants")
print(f"  For 90% power:  n = {n_for_90}  participants")
print(f"  For 95% power:  n = {n_for_95}  participants")

print(f"\nPOWER INTERPRETATION")
if achieved_power >= 0.95:
    print(f"  This study is very well-powered ({achieved_power * 100:.1f}%). Given Cohen's d of {cohens_d:.3f},")
    print(f"  only n={n_for_80} participants were needed for 80% power. Our sample of n={n_participants}")
    print(f"  substantially exceeds this, meaning the study had excellent sensitivity")
    print(f"  to detect the observed effect and the risk of a Type II error was very low.")
elif achieved_power >= 0.80:
    print(f"  This study meets the conventional 80% power threshold ({achieved_power * 100:.1f}%).")
    print(f"  There is a reasonable probability of detecting a true effect of this magnitude.")
else:
    print(f"  WARNING: This study is underpowered ({achieved_power * 100:.1f}% < 80% threshold).")
    print(f"  A minimum of n={n_for_80} participants is recommended to achieve adequate power.")
    print(f"  The current sample risks a Type II error (failing to detect a real effect).")


# 7: Power Curve
n_range = np.arange(5, 80)
power_values = [paired_ttest_power(cohens_d, n) for n in n_range]

plt.figure(figsize=(10, 6))

plt.plot(n_range, power_values,
         color='steelblue', linewidth=2.5,
         label=f"Power curve (Cohen's d = {cohens_d:.3f})")

# Threshold lines
plt.axhline(0.80, color='green',  linestyle='--', linewidth=1.5, label='80% power threshold (conventional minimum)')
plt.axhline(0.90, color='orange', linestyle='--', linewidth=1.5, label='90% power threshold')
plt.axhline(0.95, color='red',    linestyle='--', linewidth=1.5, label='95% power threshold')

# Marker for this study
plt.axvline(n_participants, color='navy', linestyle=':', linewidth=2,
            label=f'This study (n={n_participants}, power={achieved_power * 100:.1f}%)')

# Annotate required n at each threshold
for target, n_req, color in [(0.80, n_for_80, 'green'),
                              (0.90, n_for_90, 'orange'),
                              (0.95, n_for_95, 'red')]:
    plt.annotate(f'n={n_req}',
                 xy=(n_req, target),
                 xytext=(n_req + 3, target - 0.05),
                 fontsize=9, color=color,
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.0))

plt.xlabel('Sample Size (n)', fontsize=12)
plt.ylabel('Statistical Power', fontsize=12)
plt.title('Power Analysis: Sample Size vs. Statistical Power', fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("paired_power_curve.png", dpi=150)
plt.show()

# WILCOXON SIGNED-RANK TEST (Non-parametric alternative)
# The paired t-test assumes the *differences* are approximately normally distributed (tested above with Shapiro-Wilk).
# If that assumption is violated — particularly in smaller samples — the Wilcoxon signed-rank test is the appropriate non-parametric alternative.
#
# Rather than comparing means, Wilcoxon ranks the absolute differences and tests whether the median difference is zero.
# It makes no distributional assumption about the differences.
#
# Running both tests is considered good practice: agreement between them strengthens confidence in the conclusion; divergence warrants investigation.

print("\nWILCOXON SIGNED-RANK TEST (Non-Parametric Alternative)")

print("\nNull Hypothesis (H0): Median difference in sleep hours = 0")
print("Alternative Hypothesis (H1): Median difference in sleep hours ≠ 0")

wilcoxon_stat, wilcoxon_p = stats.wilcoxon(df['Sleep_After'], df['Sleep_Before'])

print(f"\nW-statistic: {wilcoxon_stat:.4f}")
print(f"P-value:     {wilcoxon_p:.6f}")

median_diff = df['Difference'].median()
print(f"\nMedian difference (After - Before): {median_diff:.3f} hours")

if wilcoxon_p < alpha:
    wilcoxon_result = "STATISTICALLY SIGNIFICANT"
    wilcoxon_conclusion = "reject"
else:
    wilcoxon_result = "NOT STATISTICALLY SIGNIFICANT"
    wilcoxon_conclusion = "fail to reject"

print(f"\nRESULT: {wilcoxon_result} (p = {wilcoxon_p:.6f})")
print(f"  We {wilcoxon_conclusion} the null hypothesis.")

print(f"\nCOMPARISON WITH PAIRED T-TEST:")
print(f"  {'Test':<30} {'Test Statistic':<20} {'P-value':<15} {'Conclusion'}")
print(f"  {'-'*75}")
t_statistic, p_value = stats.ttest_rel(df['Sleep_After'], df['Sleep_Before'])
t_conclusion = "Reject H0" if p_value < alpha else "Fail to reject H0"
w_conclusion = "Reject H0" if wilcoxon_p < alpha else "Fail to reject H0"
print(f"  {'Paired t-test (parametric)':<30} {'t = ' + f'{t_statistic:.4f}':<20} {p_value:<15.6f} {t_conclusion}")
print(f"  {'Wilcoxon signed-rank (non-param)':<30} {'W = ' + f'{wilcoxon_stat:.1f}':<20} {wilcoxon_p:<15.6f} {w_conclusion}")

print(f"\nINTERPRETATION:")
if t_conclusion == w_conclusion:
    print(f"  Both tests reach the same conclusion, strengthening confidence in the result.")
    print(f"  The parametric and non-parametric approaches agree: the intervention")
    print(f"  {'had' if wilcoxon_p < alpha else 'did not have'} a statistically significant effect on sleep duration.")
else:
    print(f"  The two tests diverge in their conclusions. This warrants further")
    print(f"  investigation. A divergence often indicates that the normality assumption")
    print(f"  of the t-test may be influential — consider favouring the Wilcoxon result.")