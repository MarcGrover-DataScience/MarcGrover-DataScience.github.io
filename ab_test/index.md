---

layout: default

title: Web conversion rates  (A/B test)

permalink: /ab-test/

---

# This project is currently being extended

## Goals and objectives:

For this portfolio project, the business scenario concerns a website redesign: specifically, whether a newly developed webpage generates a higher rate of membership sign-ups than the existing design. 1,000 users were exposed to the original webpage (the control group) and 1,000 to the new design (the treatment group), with each user's outcome recorded as a binary result — converted or not converted. The objective is to determine whether the difference in conversion rates between the two groups is statistically significant, or whether it could plausibly be explained by chance alone.

The chi-squared test of independence is the appropriate technique here because the outcome variable is categorical — each observation falls into one of two classes — and the analysis compares the distribution of that outcome across two independent groups. A statistically significant result provides the business with evidence to support or reject the decision to migrate to the new webpage, and to model the expected impact on membership volumes and associated revenue.

Beyond the primary hypothesis test, the project extends into several areas of analytical depth that are important in any applied A/B testing context. Assumption checking is conducted formally before the test is applied, verifying the independence of observations, confirming that all expected cell frequencies meet the minimum threshold required for the chi-squared approximation to be reliable, and examining the effect of Yates' continuity correction on the result. Effect size is quantified using Cramér's V, which contextualises the statistical result in terms of practical magnitude — an important distinction in settings where even a small conversion rate improvement carries meaningful commercial value.

A second example extends the framework to a three-variant email campaign, where the objective is to identify which of three subject line variants drives the highest open rate. This introduces the multi-variant A/B testing problem and demonstrates why an omnibus chi-squared test alone is insufficient: when the overall test finds a significant result, pairwise post-hoc comparisons with Bonferroni correction are required to identify exactly which variant pairs are driving the difference, while controlling the family-wise Type I error rate that arises from running multiple simultaneous tests.

A third example applies the same chi-squared framework to a mobile app onboarding redesign, measuring 7-day user retention across old and new versions. This demonstrates the generalisability of the technique across different business metrics and sectors.

The final section addresses a prerequisite step that is often overlooked in practice: before an A/B test is run, it is essential to determine how large a sample is needed to reliably detect a meaningful difference should one exist. Statistical power — the probability of correctly rejecting the null hypothesis when it is false — is calculated as a function of sample size, and a power curve is produced to visually confirm that the actual sample size used in the study was sufficient to meet the conventional 80% power threshold.

Taken together, the project demonstrates not only the correct implementation of chi-squared A/B testing, but also the analytical judgement to validate assumptions, interpret results in both statistical and practical terms, handle the multiple comparisons problem in multi-variant designs, and approach the study design question of sample size adequacy with the same rigour applied to the analysis itself.

## Application:  

The appropriate statistical test for an A/B test depends entirely on the nature of the outcome being measured. When the outcome is **categorical** — that is, each observation falls into one of a discrete set of classes — the chi-squared test of independence is the correct choice. This project analyses a binary outcome (converted or not converted), making the chi-squared test the natural fit: it tests whether the distribution of a categorical variable differs significantly between two or more independent groups.

The alternative in A/B testing is the **independent two-sample t-test**, which is used when the outcome is **continuous** — for example, comparing the average time spent on a page, average order value, or average session duration between a control and treatment group. The t-test compares group means and assumes the underlying variable is measured on a numerical scale. Applying it to a binary outcome such as converted/not converted would be incorrect.

It is also worth noting that the chi-squared test extends naturally beyond the standard two-group (A/B) design to multi-variant tests involving three or more groups simultaneously — such as the three-variant email campaign tested in this project. In these cases, the omnibus chi-squared result indicates whether at least one group differs from the others, but does not identify which specific pairs are responsible. Post-hoc pairwise comparisons with a multiple testing correction — Bonferroni correction is applied here — are then required to answer that question precisely, while controlling the inflated false positive risk that arises from running several tests at once.

Chi-squared A/B testing is foundational to data-driven decision-making across many sectors, particularly wherever business decisions hinge on a binary or categorical outcome.

💻 In **technology**, A/B testing is ubiquitous in product development and digital marketing. Teams continuously test changes to user interfaces, page layouts, call-to-action wording, and onboarding flows to determine which version drives better engagement. Outcomes such as sign-up conversion, feature adoption, and click-through rate are all binary in nature, making chi-squared the standard analytical tool. The ability to attribute a change in conversion rate to a specific design decision — with statistical confidence — is what separates data-driven product development from intuition-led guesswork.

🛍️ In **retail**, both online and physical channels use this framework to assess the effectiveness of promotional campaigns, pricing experiments, and packaging changes. A retailer might test whether a redesigned product page increases add-to-basket rate, whether a targeted discount email outperforms a generic one, or whether a revised store layout changes the proportion of customers who reach a particular aisle. Each of these produces a binary outcome (acted or did not act), and chi-squared testing provides the rigour to distinguish genuine improvements from random variation in customer behaviour.

🏦 In **financial services**, chi-squared A/B testing is applied throughout the customer acquisition and onboarding journey. Lenders test whether different application form designs affect submission completion rates; wealth managers test whether varying the framing of investment risk disclosures influences the proportion of clients who proceed to the next stage; and insurance providers test whether personalised quote presentations improve policy uptake. In a sector where regulatory and compliance constraints limit how aggressively firms can iterate, statistical rigour in evaluating even small incremental changes carries significant commercial value.

🏭 In **manufacturing and quality control**, the chi-squared framework is applied offline to compare production conditions rather than digital experiences. A manufacturer might test whether a change to process temperature, raw material supplier, or equipment setting alters the proportion of units that pass or fail a quality inspection. Because the outcome is categorical (pass/fail, defective/non-defective), chi-squared is the appropriate test — and given the cost implications of both false positives (unnecessary process changes) and false negatives (undetected quality problems), rigorous assumption checking and adequate sample sizing are particularly important in this context.

## Methodology:  

The workflow was developed in Python using NumPy and Pandas for data construction and manipulation, SciPy for all statistical computation, and Matplotlib for visualisations. All datasets were synthetically generated within the script using a fixed random seed for reproducibility, with the conversion and retention rates deliberately set to produce analytically interesting findings across the four examples.

**Data preparation** was minimal by design: each dataset was constructed directly as a Pandas DataFrame and transformed into a contingency table using pd.crosstab(). A contingency table records the observed frequency of each combination of group membership and outcome, and is the required input structure for the chi-squared test of independence. No imputation, outlier removal, or scaling was necessary given the nature of the data.

**Assumption checking** was conducted formally before applying the chi-squared test in Example 1, covering four criteria. First, independence of observations: each record represents a unique user assigned to one group only, satisfying this requirement by design. Second, minimum expected cell frequency: the chi-squared test relies on an asymptotic approximation that becomes unreliable when any expected cell count falls below five; expected frequencies were computed programmatically and verified against this threshold, with Fisher's Exact Test noted as the appropriate fallback for small samples. Third, Yates' continuity correction: SciPy applies this correction by default to 2×2 tables, reducing the chi-squared statistic slightly to guard against Type I error inflation; both the corrected and uncorrected statistics and p-values are reported to confirm the result is robust to this choice. Fourth, sample size adequacy: the size and balance of each group is stated explicitly and assessed against the minimum required sample, which is calculated separately in Example 4.

**The chi-squared test of independence** was applied to each contingency table using scipy.stats.chi2_contingency(). The test evaluates the null hypothesis that the distribution of the outcome variable is independent of group membership — that is, there is no difference in conversion or retention rates between the groups. The test statistic, degrees of freedom, p-value, and table of expected frequencies (representing the counts that would be observed if the null hypothesis were true) are all reported. The significance level was set at α = 0.05 throughout.

**Effect size** was quantified using Cramér's V, calculated from the chi-squared statistic, total sample size, and the minimum dimension of the contingency table. Cramér's V returns a value between 0 and 1, where values below 0.1 indicate a small effect, 0.1–0.3 a medium effect, and above 0.3 a large effect. This is reported alongside the p-value in all examples, as statistical significance and practical magnitude are distinct: a result can be highly significant in large samples while the underlying effect remains small in absolute terms.

**Confidence intervals** for the true conversion rates were calculated using the normal approximation method (Wilson-type interval), with a 95% confidence level. The standard error of each proportion is computed from the observed rate and sample size, and the margin of error is derived using the corresponding z-score from the standard normal distribution. These intervals quantify the plausible range of the true underlying conversion rate in each group, supporting business planning that needs to account for uncertainty around the point estimate.

**Absolute difference and relative lift** were calculated for Examples 1 and 3. The absolute difference is the straight percentage point gap between the treatment and control conversion rates. The relative lift expresses that gap as a percentage of the control rate, which is the more commercially relevant figure when evaluating the proportional improvement a new design or feature delivers over the baseline.

**Multi-variant testing and post-hoc comparisons** are demonstrated in Example 2, where three email subject line variants (A, B, and C) are tested simultaneously. An omnibus chi-squared test across all three variants first determines whether any significant differences exist. Because this result does not identify which specific pairs differ, three pairwise chi-squared tests are then conducted between all variant combinations (A vs B, A vs C, B vs C). Bonferroni correction is applied to control the family-wise error rate: the significance threshold is divided by the number of comparisons (α / 3 = 0.0167), ensuring that the probability of at least one false positive across the three tests remains at or below 5%.

**Sample size and power analysis** are addressed in Example 4. A sample size function was implemented to calculate the minimum number of observations required per group to detect a specified difference in conversion rates at α = 0.05 with 80% statistical power — the conventional minimum. Power is the probability of correctly rejecting the null hypothesis when a true effect of the specified size exists; at 80% power, there is a 20% chance of a Type II error (failing to detect a real effect). A complementary power function was implemented to compute achieved power across a range of sample sizes, producing a power curve that visually maps the relationship between sample size and the probability of detection. Both the minimum required sample size and the actual sample size used in Example 1 are marked on this curve, confirming that the study was sufficiently powered.

**Six visualisations** were produced across the four examples: a grouped bar chart of conversion rates by group; a horizontal bar chart of conversion rates with 95% confidence interval error bars; a 100% stacked proportion bar chart showing the full group composition for both converted and not-converted outcomes; the chi-squared theoretical distribution with the test statistic, critical value, and rejection region annotated; a bar chart of email open rates by variant with Bonferroni-corrected pairwise significance brackets; and a statistical power curve with the minimum required and actual sample sizes marked.

## Results:

### Hypothesis Test:  

The data being used for the A/B test contains 1,000 observations for each group, where 18% of users converted in the treatment group and 12% of users converted in the control group.

![conversion](plot_01_ab_conversion.png)

This data was used to create the contingency table:

```
Converted   No  Yes
Group              
Control    880  120
Treatment  820  180
```

The chi-squared test was applied to the data, with the null hypothesis that there is no variance between the control and treatment groups, with the significance level (alpha) equal to 0.05.  

The result of the chi-squared test was a p_value of 0.00022, and as this is <0.05 we can reject the null hypothesis and provide evidence that there is a statistically significant difference in conversion rates between the 2 groups.

An output of the chi-squared test was the expected frequencies table, which represents the expected number of conversions should there be no difference in conversion rates between groups.

```
Converted     No    Yes
Group                  
Control    850.0  150.0
Treatment  850.0  150.0
```

Using Cramér's V which is a measure of association between two nominal variables, returning a number between 0 and 1 that indicates how strongly two categorical variables are associated.  The calculated Cramér's V was 0.0826, which is interpretted as being a 'small' effect, i.e. moving from the control to treatment group will return a statistically significant difference but the scale of that effect is small.  It should be noted that this is a subjective effect 'size', and may well produce a meaningful and positive business improvement, and as such the Cramér's V is to be interpreted within the business context.  As an example, increasing conversion rates by a few percent may have significant business benefit and meet the goals of the web-site development.

Given the data available, we want to determine the range of values that the true conversion rates are in, with 95% confidence.  From the data we cannot be sure that the true conversion rate via the new web page is exactly 18%.

It was determined that the 95% confidence intervals for true conversion rates are:  

Control Group: 95% Confidence Interval of Conversion Rate: (9.99%, 14.01%)

Treatment Group: 95% Confidence Interval of Conversion Rate: (15.62%, 20.38%)

Visualising these ranges on a chart to support interpretation, and further confirm that the conversion rates improve for the new web-site design:  

![conversion_ci](plot_02_ab_conversion_ci.png)

### Sample size and power analysis:  

When setting up A/B tests and recording observations, it is important to determine the sample size required to meaningfully determine if there is a difference between the groups.  

Power refers to the probability that your test will detect an effect when there actually is one.  Typically this is 80%, i.e. an 80% chance of detecting a real effect should one exist.  More formally:  

Power = Probability of rejecting the null hypothesis when it's actually false

Note that it is common to set the significance level (alpha) to 0.05, which is the chance of a false positive.

Taking the example above, the business previously had data to imply that the conversion rate on the old website was 11%, and was hoping for a conversion rate of up to 16%.  Using these values, it was determined that a sample size of at least 733 observations per group was required.  The data analysed has 1,000 observations per groups, and as such we can be confident that the sample size was sufficiently large to detect the approximated differences in conversion rates between websites.

![plot_03_conversion_stacked](plot_03_conversion_stacked.png)

![plot_04_chi2_distribution](plot_04_chi2_distribution.png)

![plot_05_email_open_rates_pairwise](plot_05_email_open_rates_pairwise.png)

![plot_06_power_curve](plot_06_power_curve.png)

## Conclusions:



## Next steps:
The primary recommendations would include:
* the new website should be deployed as there is evidence that it results in an increased volume of memberships being taken (higher conversion rate).
* constantly track the conversion rates of the new website to understand if the rate achieved in the test is reflected going forward, and understand any changes or trends over time
* use a range of analytical techniques, potentially including time-series analysis and comparative analysis methods on new observations recorded
* other website designs are tested to see if they produce even greater conversion rates

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/AB Test_v2.py)
