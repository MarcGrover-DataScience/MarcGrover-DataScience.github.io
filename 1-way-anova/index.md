---

layout: default

title: Teaching Methods Comparison (1-Way ANOVA)

permalink: /1-way-anova/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the simulated business scenario concerns a fictional educational study involving 300 students assigned across three groups, each subject to a different teaching method, with the goal of determining whether the choice of teaching method produces a statistically meaningful difference in exam performance. A simple observation that the group means differ would be insufficient for this purpose, where raw differences between sample means are always present to some degree, and without a formal statistical framework there is no basis for distinguishing a genuine effect of the teaching method from differences that could plausibly arise from natural variability between students.  

The objective is therefore to apply a One-Way Analysis of Variance (ANOVA), which tests whether the observed variation in exam scores between the three groups is greater than would be expected by chance alone. Critically, this project is designed to illustrate the value of statistical rigour in a scenario where the answer is not obvious from visual inspection alone. The three groups' score distributions overlap substantially, and no single chart makes the conclusion clear. It is only through the formal hypothesis test that a reliable conclusion can be reached. This reflects a realistic and important class of analytical problem: one where the signal exists in the data but requires the correct statistical machinery to extract it confidently.  

A key objective is to demonstrate correct application of the full ANOVA workflow, from assumption validation through to post-hoc testing. Where the ANOVA identifies that at least one group mean differs significantly from the others, it cannot on its own identify which specific pairs of groups are responsible for that difference. Tukey's Honest Significant Difference (HSD) post-hoc test is therefore applied to resolve this, controlling the error rate across all pairwise comparisons and identifying precisely where the significant differences lie. This two-stage structure is central to the project and reflects the precision required when comparing more than two groups.  

A secondary objective is to demonstrate the assumptions that underpin the One-Way ANOVA, specifically the requirements that observations are independent, that the residuals within each group are approximately normally distributed, and that the variances across groups are approximately equal (homoscedasticity).  Each assumption is tested; residue normality via the Shapiro-Wilk test and Q-Q plots, and homogeneity of variance via Levene's test.  Where the equal variance assumption is questioned, Welch's ANOVA is additionally presented as the appropriate robust alternative, with both results compared to confirm the stability of the conclusion.  

By grounding every analytical decision in a clear methodological rationale, this project demonstrates not only technical proficiency in Python-based statistical testing, but also the ability to select the appropriate analytical tool for the structure of the data, validate the conditions under which that tool is appropriate, and communicate findings in a way that is meaningful to both technical and non-technical audiences.

## Application:  

A One-Way ANOVA (Analysis of Variance) is a statistical test used to determine whether the means of three or more independent groups are significantly different from one another. It extends the logic of a two-sample t-test to situations involving multiple groups, while avoiding the inflation of Type I error that would result from running repeated pairwise t-tests across all group combinations.  

It answers the question: "Is there a statistically meaningful difference in outcomes between these groups, or could the observed variation be explained by chance alone?"  

The technique works by partitioning the total variance in the data into two components: the variance between groups (reflecting genuine differences in group means) and the variance within groups (reflecting natural variability among individuals within the same group). The F-statistic is the ratio of these two quantities — a large F indicates that the between-group differences are substantial relative to within-group noise, and therefore unlikely to have arisen by chance. Where the ANOVA confirms that at least one group mean differs significantly, a post-hoc test such as Tukey's HSD is applied to identify precisely which group pairs are responsible for the difference.  

This approach is applicable across many sectors and scenarios. Practical examples showing where a One-Way ANOVA provides clear business value include: 

🛍️ **Retail Sector**:

* **Promotional Strategy Testing** — Compare average basket size across three or more different promotional strategies (e.g. percentage discount, buy-one-get-one, and loyalty points) applied to separate customer groups, to determine whether promotion type has a statistically significant effect on spend per visit.
* **Store Format Performance** — Assess whether mean weekly revenue differs significantly across different store formats (e.g. high street, retail park, and outlet) to inform future investment decisions.
* **Customer Satisfaction by Region** — Test whether mean customer satisfaction scores differ significantly across geographic regions to identify where service improvements are most needed.

💻 **Technology Sector**:

* **Algorithm Benchmarking** — Compare the mean execution time or accuracy of three or more competing algorithms on the same task, to determine whether performance differences are statistically significant rather than artefacts of random variation in test conditions.
* **User Engagement by Onboarding Flow** — Assess whether mean session duration or feature adoption rates differ significantly across different onboarding experiences assigned to separate user cohorts.
* **Infrastructure Configuration Testing** — Compare mean system latency or error rates across multiple server configurations or cloud regions to determine whether the choice of configuration has a genuine effect on performance.

🔬 **Science & Research Sector**:

* **Drug Dosage Trials** — Compare mean physiological outcomes (e.g. blood pressure reduction, biomarker levels) across patients assigned to different dosage groups to assess whether dosage level has a statistically significant effect on treatment response.  
* **Material Testing** — Assess whether mean tensile strength or thermal resistance differs significantly across samples produced using different manufacturing processes or chemical formulations.
* **Educational Interventions** — Test whether mean assessment scores differ significantly across student cohorts assigned to different pedagogical approaches, as demonstrated in this project.

🏭 **Manufacturing Sector**:

* **Supplier Quality Comparison** — Compare mean defect rates across components sourced from three or more suppliers to determine whether supplier choice has a statistically significant impact on production quality.
* **Machine Calibration Settings** — Assess whether mean output precision differs significantly across multiple calibration configurations on the same production line, to identify the optimal setting.
* **Shift Performance Analysis** — Test whether mean units produced or mean defect rate differs significantly across morning, afternoon, and night shifts, to identify whether time of shift is a meaningful driver of production variability.

It is worth noting the relationship between One-Way ANOVA and other hypothesis tests in this portfolio. Where only two groups are being compared, an independent samples t-test is the appropriate tool. Where the same subjects are measured under multiple conditions, a repeated measures ANOVA is more appropriate than a one-way design. And where the outcome variable is categorical rather than continuous, a chi-squared test of independence becomes the relevant framework — all of which are covered elsewhere in this portfolio.

## Methodology:  

The methodology adopted for this project follows the end-to-end data science workflow, progressing from data generation and exploratory analysis through to hypothesis testing, post-hoc investigation, and the extraction of business insight. The project is implemented in Python, using pandas for data manipulation, scipy and pingouin for statistical testing, and seaborn and matplotlib for visualisation. Each stage of the pipeline is described below. 

**Data Generation**:

The dataset is synthetically generated as part of the Python script, simulating an educational study in which 300 students are assigned equally across three groups of 100, each taught using a different teaching method. Exam scores for each group are drawn from a normal distribution with group-specific means, and standard deviations reflecting realistic between-student variability. The group means are deliberately set close together, with substantial distributional overlap across groups, ensuring the dataset reflects the kind of scenario where a formal statistical test is necessary to reach a reliable conclusion. 

**Exploratory Data Analysis**: 

Descriptive statistics are calculated for each group individually, including the mean, standard deviation, standard error, minimum, and maximum. Three charts are produced to support visual inspection of the data:

* A **boxplot**, providing a side-by-side comparison of the central tendency, spread, and absence of extreme outliers across the three groups.
* A **violin plot**, combining the boxplot summary with a kernel density estimate to show the full shape of each group's distribution.
* **Histograms with KDE overlays**, allowing the distributional shape of each group to be assessed individually and the approximate normality of the data to be visually inspected.

**Testing Assumptions**:
The One-Way ANOVA has three core assumptions, each requiring a specific diagnostic check.

* **Independence of Observations** — Each student belongs to exactly one group and contributes exactly one score. No student's result influences any other. This assumption is satisfied by design in the data generation process and requires no further diagnostic testing.
* **Normality Within Each Group** — The ANOVA assumes that the scores within each group are approximately normally distributed. This is assessed using two complementary approaches: Q-Q plots for each group provide visual confirmation, and the Shapiro-Wilk test formalises the assessment with a p-value for each group individually. If this assumption were violated, the appropriate non-parametric alternative would be the Kruskal-Wallis H test.
* **Homogeneity of Variance (Homoscedasticity)** — The ANOVA assumes that the variance of scores is approximately equal across all groups. The group variances are inspected as context, and Levene's test is applied as the formal diagnostic test. If this assumption were violated, Welch's ANOVA, which relaxes the equal variance requirement, would be the appropriate alternative, and is additionally presented in this analysis as a robustness check regardless of the Levene's outcome.

**Statistical Testing and Effect Size**:

The One-Way ANOVA is performed using scipy.stats.f_oneway(), testing the null hypothesis that the mean exam score is equal across all three groups against the alternative that at least one group mean differs significantly. The result is evaluated against a significance threshold of α = 0.05.

In addition to the F-statistic and p-value, eta-squared (η²) is calculated as the effect size measure, defined as the proportion of total variance in exam scores attributable to the group factor. This quantifies the practical magnitude of any teaching method effect independently of sample size, and is interpreted using conventional benchmarks (negligible: <0.01, small: 0.01–0.06, medium: 0.06–0.14, large: >0.14). A 95% confidence interval for the mean is also calculated for each group and presented visually via an interval plot.

**Post-Hoc Analysis — Tukey's HSD**:

Where the ANOVA returns a significant result, Tukey's Honest Significant Difference (HSD) post-hoc test is applied using statsmodels.stats.multicomp.pairwise_tukeyhsd(). This test compares all three pairwise group combinations simultaneously, controlling the family-wise error rate at α = 0.05 to avoid the inflation of Type I error that would result from running multiple independent t-tests. The output identifies precisely which group pairs drive the significant ANOVA result, and a simultaneous confidence interval plot is produced to visualise the pairwise differences directly.

**Welch's ANOVA**:

Welch's ANOVA is performed using pingouin.welch_anova() as a robustness check, providing a cross-validation of the standard ANOVA result under a framework that does not assume equal variances. Both results are compared to assess the stability of the conclusion.

**Business Insight Extraction and Visualisation**:

The outputs of the hypothesis testing and post-hoc stages are synthesised into a structured interpretation that addresses the original business question directly: does teaching method have a statistically significant and practically meaningful effect on exam performance, and if so, which specific methods differ? Results are communicated using clear, non-technical language supported by the visualisations produced during the exploratory and assumption-checking phases, ensuring the findings are accessible to both technical and non-technical stakeholders.

## Results:

**Descriptive Statistics**
Descriptive statistics were calculated for each of the three groups across all 100 participants per group. The results are summarised below.
Group 1 (Teaching Method A):

* Mean: 74.463
* SD: 9.035
* SE: 0.908
* Min: 49.3    Max: 94.0

Group 2 (Teaching Method B):

* Mean: 76.123
* SD: 10.136
* SE: 1.019
* Min: 54.9    Max: 99.0

Group 3 (Teaching Method C):

* Mean: 72.549
* SD: 10.925
* SE: 1.098
* Min: 38.0    Max: 99.0

The group means are close together — spanning a range of just 3.6 points — and the distributions overlap substantially. Group 2 has the highest mean score and Group 3 the lowest, but neither the raw means nor the descriptive statistics alone are sufficient to determine whether these differences are statistically meaningful or simply the product of natural variability between students. This is precisely the scenario where a formal hypothesis test adds value over visual inspection alone.
The boxplot and violin plot below illustrate the distributions for each group. Both confirm the overlapping nature of the score distributions, with similar medians and interquartile ranges across all three groups. No extreme outliers are present in any group.

![1way_boxplot](/1way_boxplot.png)  

![1way_violin](/1way_violin.png)  

The histograms with KDE overlays provide a clearer view of the shape of each group's distribution individually.

![1way_histograms_group](/1way_histograms_group.png)  

**Testing Assumptions**  

Before proceeding with the One-Way ANOVA, the three core assumptions of the test were validated.  
**Independence of observations** is satisfied by design. Each student belongs to exactly one group, their exam score is recorded once, and no student's result influences that of any other. This assumption requires no further diagnostic testing.
**Normality within each group** was assessed using the Shapiro-Wilk test applied to each group's scores individually, supported by visual inspection of Q-Q plots for each group.  

* Group 1: W = 0.9899, p = 0.6526 — normally distributed (p > 0.05)  
* Group 2: W = 0.9772, p = 0.0800 — normally distributed (p > 0.05)  
* Group 3: W = 0.9912, p = 0.7568 — normally distributed (p > 0.05)  

All three groups pass the Shapiro-Wilk test comfortably. The Q-Q plots below provide consistent visual confirmation, with sample quantiles tracking closely along the theoretical normal line across the full range of each group's data.

![1way_qq_plots](/1way_qq_plots.png)  

**Homogeneity of variance** (homoscedasticity) was assessed using Levene's test, which tests the null hypothesis that the variances of all three groups are equal. The raw group variances provide useful context ahead of the formal test:

* Group 1 variance: 81.64
* Group 2 variance: 102.73
* Group 3 variance: 119.36

While the variances increase from Group 1 to Group 3, Levene's test assesses whether this spread is statistically significant. The test returned a Levene statistic of 1.7464 and a p-value of 0.1762. As p > 0.05, we fail to reject the null hypothesis — the variances are not significantly different, and the homoscedasticity assumption is satisfied. The standard One-Way ANOVA is therefore appropriate.  
All three assumptions are met and the analysis proceeds with the parametric One-Way ANOVA. Welch's ANOVA is additionally presented as a robustness check, given the visible trend in group variances.  

**One-Way ANOVA**
The One-Way ANOVA was performed using scipy.stats.f_oneway(), testing the null and alternative hypotheses:

* **H₀**: The mean exam scores are equal across all three groups (μ₁ = μ₂ = μ₃)
* **H₁**: At least one group mean differs significantly from the others

The test returned the following results:

* F-statistic: 3.1279
* P-value: 0.0453
* Eta-squared (η²): 0.0206

As p = 0.0453 < 0.05, the null hypothesis is rejected. There is statistically significant evidence that the teaching method has an effect on exam performance — at least one group mean is significantly different from the others.  

The effect size, measured by eta-squared, is 0.0206, indicates that approximately 2.1% of the total variance in exam scores is attributable to the group factor — i.e. to the choice of teaching method. By conventional benchmarks this is a small effect, which is consistent with the modest separation between group means observed in the descriptive statistics. The result is statistically significant, but the practical magnitude of the teaching method's influence on exam performance is limited in this dataset.

The interval plot below shows the mean and 95% confidence interval for each group, making the relative positions and uncertainty around each group mean clear at a glance.

![1way_point_plot](/1way_point_plot.png)  

The confidence intervals for all three groups overlap substantially, which is consistent with both the small effect size and the marginal p-value. Group 2 has the highest mean but its interval overlaps considerably with both Group 1 and Group 3.

**Post-Hoc Analysis — Tukey's HSD**  
The One-Way ANOVA confirms that at least one group mean differs significantly, but does not identify which specific pairs of groups are responsible. Tukey's Honest Significant Difference (HSD) post-hoc test was applied to resolve this, comparing all three pairwise combinations while controlling the family-wise error rate at α = 0.05.

```
        Comparison   Mean Difference  P-value (adjusted)            95% CI     Significant
Group 1 vs Group 2            +1.660              0.4777    (−1.709, 5.029)             No
Group 1 vs Group 3            −1.914              0.3751    (−5.283, 1.455)             No
Group 2 vs Group 3            −3.574              0.0346   (−6.943, −0.205)            Yes
```

The source of the significant ANOVA result is the difference between Group 2 and Group 3. With an adjusted p-value of 0.035, the mean score difference of 3.574 points between these two groups is statistically significant after controlling for multiple comparisons. The confidence interval for this difference lies entirely below zero, confirming that Group 2 outperformed Group 3 by a meaningful and statistically reliable margin. The comparisons between Group 1 and Group 2, and between Group 1 and Group 3, do not reach significance — Group 1 occupies an intermediate position that is statistically indistinguishable from either of the other two groups.
The Tukey simultaneous confidence interval plot below visualises these pairwise comparisons directly.

![1way_tukey_plot](/1way_tukey_plot.png)  

**Welch's ANOVA**
As a robustness check, Welch's ANOVA was additionally applied. This variant does not assume equal variances across groups and is therefore more conservative in the presence of heteroscedasticity. Given the visible upward trend in group variances noted during assumption testing, this provides a useful cross-validation of the standard ANOVA result.  
Welch's ANOVA returned an F-statistic of 2.837 and a p-value of 0.061. This narrowly exceeds the significance threshold of α = 0.05, meaning that under the more conservative framework that relaxes the equal variance assumption, the result does not reach conventional significance. This is an important nuance: while the standard ANOVA result is technically valid given that Levene's test was passed, the Welch's result signals that the conclusion sits close to the boundary and should be interpreted with appropriate caution. In a real-world context, this would warrant collecting additional data to determine whether the effect is robust, rather than treating the ANOVA result as a definitive finding.  

## Conclusions:

The One-Way ANOVA returns a statistically significant result (F = 3.128, p = 0.045), providing evidence that the choice of teaching method does have a measurable effect on exam performance. However, the result demands careful interpretation on two fronts.  

First, the effect size is small — eta-squared of 0.021 indicates that the teaching method accounts for only around 2% of the total variance in scores. The vast majority of the variability in student performance is driven by factors other than which group a student was assigned to, such as individual aptitude, prior knowledge, and study habits. Statistical significance confirms the effect is real; it does not imply that the teaching method is a dominant driver of outcomes.  

Second, the Tukey post-hoc analysis reveals that the significant ANOVA result is driven entirely by a single pairwise difference: Group 2 outperforming Group 3 by a mean of 3.57 points (adjusted p = 0.035). Group 1 is statistically indistinguishable from either Group 2 or Group 3. Any practical recommendation arising from this analysis would therefore focus on the relative underperformance of Group 3's teaching method compared to Group 2's, rather than on a broad conclusion that all three methods differ.  

Third, the Welch's ANOVA result of p = 0.061 — which narrowly misses significance under a framework that relaxes the equal variance assumption — adds an important caveat. The standard ANOVA conclusion is technically valid given that Levene's test was passed, but the proximity of the Welch's result to the significance threshold suggests the finding should be treated as indicative rather than conclusive. In a genuine business or research context, this would be a strong prompt to collect additional data before acting on the result.  

Taken together, the analysis demonstrates that the formal statistical framework of the One-Way ANOVA can surface real but subtle effects that would be invisible to visual inspection alone — and equally importantly, that interpreting the result responsibly requires looking beyond the p-value to the effect size, the post-hoc structure, and the robustness of the conclusion under alternative assumptions.  

## Next steps:  

The most immediately valuable extension would be to enrich the dataset with student-level covariates — such as prior academic performance, study hours, or attendance rate — and incorporate these into the analysis using a one-way ANCOVA (Analysis of Covariance). By controlling for pre-existing differences between students, ANCOVA would reduce within-group noise and increase the sensitivity of the test, potentially strengthening the marginal result observed here and providing a cleaner estimate of the teaching method's true effect. This would also begin to address the question of why Group 3 underperforms relative to Group 2 — whether it reflects a genuine weakness in the teaching method itself, or a confounding difference in the students assigned to that group.  

Another suggested extension would be to apply a Two-Way ANOVA, introducing a second categorical factor such as student cohort, school, or class size alongside teaching method. This would allow the analysis to test not only the main effect of each factor independently, but also whether an interaction effect exists — for example, whether a particular teaching method is more effective in smaller classes or with higher-attaining cohorts. Interaction effects of this kind can be the most actionable finding in educational research, and the Two-Way ANOVA framework for detecting such interaction effects is covered in a subsequent project in this portfolio.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/ANOVA_1-way_Exam_pt2.py)
