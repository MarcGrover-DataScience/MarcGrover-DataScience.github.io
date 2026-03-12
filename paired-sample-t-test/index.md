---

layout: default

title: Sleep Therapy Intervention (Paired Sample T-Test)

permalink: /paired-sample-t-test/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the simulated business scenario concerns a fictional clinical sleep study involving 30 adult participants, with the goal of determining whether a structured sleep therapy intervention produces a statistically meaningful improvement in sleep duration.  A simple observation that average sleep hours increased after treatment would be insufficient for this purpose, as it cannot distinguish a genuine therapeutic effect from natural night-to-night variability in sleep patterns, regression to the mean, or the placebo effect.

The objective is therefore to apply a rigorous statistical framework that isolates the intervention's contribution with quantified confidence, producing a robust and clinically meaningful assessment of whether the therapy generated a genuine improvement in sleep duration, and if so, by how much and with what degree of certainty.  

A key objective is to demonstrate that evaluating the effectiveness of an intervention requires more than observing that an outcome changed after it was applied.  The analysis aims to show that only by correctly identifying the paired structure of the data — where each participant serves as their own control — and testing the distribution of individual-level differences against a null hypothesis of no effect, can a meaningful statistical conclusion be reached.  This within-subject design, and its implications for assumption testing and effect size interpretation, is central to the project and reflects the kind of methodological precision that separates a rigorous intervention analysis from a superficial before-and-after comparison.  

A secondary objective is to demonstrate awareness of the assumptions that underpin the paired samples t-test, specifically the requirement that the differences between paired observations are approximately normally distributed, and to show how these assumptions are tested and validated in practice — both through the Shapiro-Wilk test and through visual inspection via Q-Q plot. Where assumptions hold, the analysis proceeds with the parametric test; the Wilcoxon signed-rank test is additionally presented as the appropriate non-parametric alternative, with both results compared to illustrate the robustness of the conclusion.  A statistical power analysis is also included to contextualise the result, confirming whether the sample of 30 participants provides adequate sensitivity to detect an effect of the observed magnitude.  

By grounding every analytical decision in a clear methodological rationale, this project aims to demonstrate not only technical proficiency in Python and SciPy-based statistical testing, but also the ability to design, validate, and communicate a complete intervention analysis in a way that is meaningful to both technical and non-technical audiences.  

Ultimately, the project reflects a core principle of applied statistics: that understanding whether a treatment caused a change — and quantifying the size of that change with appropriate uncertainty — is far more valuable than simply observing that a difference exists.

## Application:  

A Paired Sample t‑Test (also called a dependent t‑test) is used to determine whether the mean difference between two related sets of observations is statistically significant. 

These “paired” observations come from the same subjects measured twice (before and after), or two conditions applied to the same participant, item, or process.

It answers the question:  “Did something change in a meaningful way?”

The technique computes the differences in observations, and using the mean and standard deviation of the differences, determines if this difference is statistically significant, and as a consequnce if there is a meaningful difference in the two sets of observations.

This approach is applicable across many sectors and scenarios.  Practical examples showing where a paired t‑test provides clear business value include:

🛍️ **Retail Sector**:
* Measure Impact of Store Layout Changes - Before‑and‑after comparison of customer dwell time, footfall flows, or basket size when a new layout is introduced.  This helps determine whether the redesign increases sales.
* A/B Testing In‑Store Promotions - Compare sales per customer before and after applying a new discount strategy in the same store.  This helps retailers optimise promotional return on investment.
* Training Effectiveness for Store Staff - Assess whether customer satisfaction scores for the same team improved after a training programme.

💻 **Technology Sector**:
* Software Performance Benchmarking - Compare system performance metrics (e.g., latency, CPU load) before and after code optimisation.  this identifies whether the new version truly improves performance.
* User Experience (UX) Improvements - Measure user completion time for tasks before and after a UI redesign.  This validates design choices based on statistically significant improvements.
* Cybersecurity Patch Impact - Compare false‑positive detection rates or scan times before vs after new threat‑detection algorithms.

🔬 **Science & Research Sector**:
* Clinical Trials & Experiments - Measure physiological indicators (e.g., heart rate, blood pressure) pre‑ and post‑treatment on the same subjects.
* Environmental Measurements - Assess changes in pollutant concentration before and after the introduction of a filtering system.
* Psychology & Behavioural Experiments - Compare participant scores on a cognitive task before and after an intervention such as mindfulness training.

🏭 **Manufacturing Sector**:
* Process Improvement (Lean / Six Sigma) - Compare defect rates from the same production line before vs after a process optimisation.
* Equipment Calibration Impact - Assess whether recalibration improves precision on the same machine.
* Energy Efficiency Testing - Compare power consumption of machinery before and after implementing efficiency controls.

A/B Testing and Paired Sample t‑Tests are related but significantly different.  A paired sample t‑test is a specific statistical test, whereas a A/B testing is an experimental framework that may use a t‑test (paired or unpaired), but also uses many other statistical methods.  For example, within this portfolio, there is an A/B Test example using chi-squared test of independence.

## Methodology:  

The methodology adopted for this project follows the end-to-end data science workflow, progressing from raw data through to the extraction and communication of business insight. The project is implemented in Python, using pandas for data manipulation, scipy for statistical testing, and seaborn and matplotlib for visualisation. Each stage of the pipeline is described in detail below.  

**Data Generation**:  The dataset is synthetically generated as part of the Python script, simulating a clinical sleep study involving 30 adult participants. Sleep hours before treatment are drawn from a normal distribution with a mean of 5.5 hours and a standard deviation of 1.2 hours, clipped to the realistic range of 3–8 hours.  Individual improvement scores are then sampled from a separate normal distribution (mean 0.5 hours, SD 0.5 hours) and added to each participant's pre-treatment value to produce their post-treatment sleep duration, clipped to a maximum of 10 hours. This design introduces realistic between-participant variability in treatment response, ensuring the simulated data reflects the heterogeneity typical of real clinical populations while remaining fully reproducible via a fixed random seed.

**Exploratory Data Analysis**:  Exploratory analysis is performed on the Before treatment values, the After treatment values, and most critically the pairwise Differences (After − Before), which are the direct subject of the paired t-test.  Descriptive statistics are computed for all three series, including the mean, standard deviation, minimum, and maximum.  For the Difference scores specifically, the Standard Error of the Mean is also calculated, which quantifies the uncertainty around the observed mean difference as an estimate of the true population effect.  

Four charts are produced to support exploratory and diagnostic analysis, two of which further support the Testing Assumptions:

* A slope plot connecting each participant's Before and After values.
* A boxplot with overlaid strip plot, comparing the Before and After distributions side-by-side.
* A histogram with KDE of the Difference scores
* A Q-Q plot of the Difference scores

**Testing assumptions**:  A paired t-test has three core assumptions, each requiring a specific diagnostic check.  

* **Independence of Pairs (Between-Pair Independence)** — Each pair of observations must come from an independent subject; one participant's data must not influence another's. While each participant contributes two measurements, the pairs themselves must be independent of one another. This is satisfied by design in the data generation process, where each participant's values are drawn independently.
* **Normality of the Difference Scores** — The paired t-test requires that the pairwise differences (After − Before) are approximately normally distributed. Importantly, normality is not required of the Before or After values individually — only of their differences. This assumption is assessed using three complementary approaches: a histogram with KDE overlay, a Q-Q plot, and the Shapiro-Wilk test. If this assumption were violated, the appropriate non-parametric alternative would be the Wilcoxon Signed-Rank Test, which is also presented in this analysis for comparison.
* **No Extreme Outliers in the Difference Scores** — Outliers in the Difference scores can distort the mean difference and inflate the t-statistic, potentially producing a misleading result. This is assessed visually using a boxplot of the Difference scores, with any flagged values inspected individually.

**Statistical Testing and Effect Size**: The paired samples t-test is performed using scipy.stats.ttest_rel(), comparing After against Before values. The test computes the mean of the Difference scores, assesses how far this departs from zero relative to the variability in the differences, and returns a t-statistic and two-tailed p-value. The result is evaluated against a significance threshold of α = 0.05.  

In addition to the p-value, Cohen's d is calculated as the effect size measure, defined as the mean difference divided by the standard deviation of the differences. This quantifies the practical magnitude of the treatment effect independently of sample size, and is interpreted using conventional benchmarks (negligible: <0.2, small: 0.2–0.5, medium: 0.5–0.8, large: >0.8).  A 95% confidence interval for the mean difference is also constructed, providing a plausible range for the true population-level effect.  

**Statistical Power Analysis**: A power analysis is conducted to contextualise the result and assess whether the sample of 30 participants provides sufficient sensitivity to detect an effect of the observed magnitude. Power is calculated using the observed Cohen's d and a significance level of α = 0.05. A study is conventionally considered adequately powered when it achieves 80% power or above — meaning there is at least an 80% probability of correctly rejecting the null hypothesis when a true effect of the observed size exists.  

## Results:

Using the stated methodology the following results were generated.

**Exploratory Data Analysis**: The descriptive statistics for the Before, After and Difference statistics were:

Before Treatment:  
* Mean: 5.274 hours  
* SD:   1.080 hours  
* Min:  3.204 hours  
* Max:  7.395 hours  

After Treatment:  
* Mean: 5.714 hours  
* SD:   1.218 hours  
* Min:  3.191 hours  
* Max:  7.999 hours  

Difference (After - Before):  
* Mean: 0.439 hours  
* SD:   0.466 hours  
* SE:   0.085 hours  (the variability of the sample mean across different possible samples)  

The slope plot connects each participant's Before and After values.  The outcome-coded colouring, distinguishes participants whose sleep improved (grey) from those whose sleep worsened (orange), allowing the proportion and pattern of non-responders to be identified.  The group mean is highlighted red.  This shows that 24 of the 30 participants recorded improved sleep, and also provides a visual overview of the magnitude of the increases and decreases.  

![paired_difference_decreases](paired_difference_decreases.png)

The boxplot with overlaid strip plot, compares the Before and After distributions side-by-side, highlighting the shifts in central tendency and spread visible, which shows that mean is higher in the After measurements, and the spread of the quartiles also seems greater in the After measurements (which is backed up by the higher standard deviation identified previously).  

![paired_before_after_boxplot](paired_before_after_boxplot.png)

The histogram with KDE of the Difference scores allows visual confirmation that the approximate normality of the distribution.  It also shows the mean difference is greater than (0.439), and the distribution is predominantly greater than zero. and at zero (no change), used to visually assess the shape and approximate normality of the differences.  

![paired_difference_distribution](paired_difference_distribution.png)

The Q-Q plot of the Difference scores, visually supports that the distribution of the Differences is normal.  This is used in conjunction with the Differences histogram, and Shapiro-Wilk test to assess the normality assumption visually (see the Testing assumptions below).  

![paired_q_q](paired_q_q.png)

**Testing assumptions**: Before proceeding with the paired samples t-test, the three core assumptions of the test were validated.

* **Independence of Pairs (Between-Pair Independence)** is satisfied by design. Each participant's data was generated independently, and no participant's values influence those of any other. This assumption requires no further diagnostic testing.
* **Normality of the Difference scores** was assessed using three complementary approaches. Visually, the histogram of Difference scores (After − Before) shows a roughly bell-shaped distribution centred slightly above zero, with the KDE overlay confirming an approximately normal shape and no evidence of heavy tails or strong skew.  The histogram is shown above.

The Q-Q plot (aalso above) provides further visual confirmation as the sample quantiles track closely along the theoretical normal line across the full range of the data, with only minor deviations at the tails, consistent with normal sampling variation rather than a systematic departure from normality.

The Shapiro-Wilk test formalises this visual assessment. The test returned a W-statistic of 0.98 and a p-value of 0.91 - which is greater than 0.05.  This means we fail to reject the null hypothesis of normality. The Difference scores are consistent with a normally distributed population, and the parametric paired t-test is therefore appropriate.
* **No Extreme Outliers in the Difference Scores** were confirmed by visual inspection of the boxplot, shown above. The interquartile range is compact and symmetrically positioned, with no data points falling beyond the whiskers. The assumption is satisfied.

As such all three assumptions are met. Therefore the analysis proceeds with the parametric paired samples t-test.  For the sake of completeness, the Wilcoxon signed-rank test is additionally presented as a robustness check.

**Statistical Testing and Effect Size**: The paired samples t-test was performed using scipy.stats.ttest_rel(), comparing After against Before sleep hours across all 30 participants. The null hypothesis states that the mean difference in sleep duration is zero — i.e. that the therapy produced no systematic effect.  

The test returned a t-statistic of 5.17 and a p-value of < 0.001 (29 degrees of freedom). As this is well below the significance threshold of α = 0.05, the null hypothesis is rejected. There is strong statistical evidence that the sleep therapy intervention produced a genuine improvement in sleep duration.  

The mean difference in sleep hours was +0.44 hours (approximately 26 minutes), with a standard deviation of 0.47 hours and a standard error of 0.085 hours. The 95% confidence interval for the mean difference is [0.27, 0.61] hours, meaning we can be 95% confident that the true population-level effect of the therapy lies between approximately 16 and 36 minutes of additional sleep per night.  

The CI forest plot below places this finding in its clearest form: the mean difference and its confidence interval sit entirely to the right of the zero line, confirming that a null effect is not a plausible explanation for the observed data.

![paired_ci_forest_plot](paired_ci_forest_plot.png)

Cohen's d was calculated as 0.94, which by conventional benchmarks represents a large effect. This is a practically significant finding: the therapy not only cleared the bar of statistical significance but produced an effect of substantial real-world magnitude relative to the variability in individual responses.  

As a robustness check, the Wilcoxon signed-rank test was also applied. This non-parametric alternative makes no assumption about the distribution of differences and is appropriate as a cross-validation of the parametric result. The Wilcoxon test returned a p-value of < 0.001, consistent with the t-test conclusion, confirming that the result is not sensitive to the normality assumption.  

The slope plot shown above visualises the individual-level data underlying these statistics, showing all participants' trajectories with the group mean highlighted in red; the second colour-codes the minority of participants whose sleep worsened after treatment (orange), illustrating that while the intervention was not universally effective, improvement was the clear dominant pattern.

**Statistical Power Analysis**:

A post-hoc power analysis was conducted to assess whether the sample of 30 participants was adequate to reliably detect an effect of the observed magnitude.  Power is defined as the probability of correctly rejecting the null hypothesis when a true effect exists — in other words, the sensitivity of the study design.  

Using the observed Cohen's d of 0.94 and a significance level of α = 0.05 (two-tailed), the achieved statistical power for this study is > 0.99. This substantially exceeds the conventional threshold of 0.80, confirming that the sample was more than sufficiently powered to detect an effect of this size.  

The power curve below contextualises this result across a range of sample sizes. It shows the relationship between sample size and achieved power for the observed effect size, with reference lines marking the 80% power threshold and the actual sample of 30 participants. This illustrates that even a considerably smaller sample would have been sufficient to reliably detect an effect of this magnitude — a reflection of how large the true effect is.  A sample of 11 would have exceeded the 80% power threshold, and a sample of 17 would have exceeded the 95% power threshold.

![paired_power_curve](paired_power_curve.png)

It should be noted that because this was computed after observing the effect size, the power result should be interpreted as a characterisation of the study's sensitivity given the observed data, rather than as a prospective design justification.  In a real clinical context, a pre-specified power analysis would be conducted before data collection to determine the minimum required sample size.

## Conclusions:

The analysis provides strong and consistent evidence that the sleep therapy intervention produced a genuine improvement in sleep duration. All three test assumptions were satisfied, the paired t-test result was highly significant, and the Wilcoxon signed-rank test confirmed the conclusion under a more conservative non-parametric framework. The effect size was large by conventional standards, and the study was more than adequately powered to detect it.  

Two aspects of the result are worth highlighting beyond the headline significance finding. First, the 95% confidence interval sits entirely above zero, meaning a null effect is not a statistically plausible outcome — the therapy improved sleep, and the analysis quantifies the likely range of that improvement with precision. Second, the slope plots reveal that while improvement was the dominant pattern, a minority of participants did not respond positively to the intervention. In a real clinical context, understanding who responds and who does not would be as valuable as the group-level finding itself, and would represent a natural direction for further investigation.

## Next steps:  

The most immediately valuable extension would be to move beyond the group-level result and examine heterogeneity in treatment response. Segmenting participants by baseline sleep duration — for example, comparing those who were severely sleep-deprived before treatment against those with near-normal baseline values — could reveal whether the therapy is differentially effective depending on the severity of the presenting condition. This kind of subgroup analysis would transform the binary conclusion of the t-test into an actionable insight about which patients are most likely to benefit.  

A longer-term extension would be to apply a mixed-effects model to a richer version of the dataset, incorporating additional follow-up measurements taken weeks or months after the intervention. This would allow the durability of the treatment effect to be assessed over time, addressing the clinically important question of whether the improvement in sleep duration is sustained or whether participants revert toward their baseline once the structured therapy programme ends.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/Paired Sample T-Test.py)
