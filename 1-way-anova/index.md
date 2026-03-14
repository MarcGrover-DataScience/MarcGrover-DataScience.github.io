---

layout: default

title: Teaching Methods Comparison (1-Way ANOVA)

permalink: /1-way-anova/

---

# This project is in development

## Goals and objectives:

For this portfolio project, the simulated business scenario concerns a fictional educational study involving 300 students assigned across three groups, each subject to a different teaching method, with the goal of determining whether the choice of teaching method produces a statistically meaningful difference in exam performance. A simple observation that the group means differ would be insufficient for this purpose — raw differences between sample means are always present to some degree, and without a formal statistical framework there is no basis for distinguishing a genuine effect of the teaching method from differences that could plausibly arise from natural variability between students.  

The objective is therefore to apply a One-Way Analysis of Variance (ANOVA), which tests whether the observed variation in exam scores between the three groups is greater than would be expected by chance alone. Critically, this project is designed to illustrate the value of statistical rigour in a scenario where the answer is not obvious from visual inspection alone. The three groups' score distributions overlap substantially, and no single chart makes the conclusion clear — it is only through the formal hypothesis test that a reliable conclusion can be reached. This reflects a realistic and important class of analytical problem: one where the signal exists in the data but requires the correct statistical machinery to extract it confidently.  

A key objective is to demonstrate correct application of the full ANOVA workflow, from assumption validation through to post-hoc testing. Where the ANOVA identifies that at least one group mean differs significantly from the others, it cannot on its own identify which specific pairs of groups are responsible for that difference. Tukey's Honest Significant Difference (HSD) post-hoc test is therefore applied to resolve this, controlling the family-wise error rate across all pairwise comparisons and identifying precisely where the significant differences lie. This two-stage structure — omnibus test followed by targeted post-hoc analysis — is central to the project and reflects the methodological precision required when comparing more than two groups.  

A secondary objective is to demonstrate awareness of the assumptions that underpin the One-Way ANOVA, specifically the requirements that observations are independent, that the residuals within each group are approximately normally distributed, and that the variances across groups are approximately equal (homoscedasticity). Each assumption is tested in practice — normality via the Shapiro-Wilk test and Q-Q plots, and homogeneity of variance via Levene's test. Where the equal variance assumption is questioned, Welch's ANOVA is additionally presented as the appropriate robust alternative, with both results compared to confirm the stability of the conclusion.  

By grounding every analytical decision in a clear methodological rationale, this project aims to demonstrate not only technical proficiency in Python-based statistical testing, but also the ability to select the right tool for the structure of the data, validate the conditions under which that tool is appropriate, and communicate findings in a way that is meaningful to both technical and non-technical audiences.

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

Details of the methodology applied in the project.

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

![1way_boxplot](/1way_boxplot.png)  

![1way_violin](/1way_violin.png)  

![1way_histograms_group](/1way_histograms_group.png)  


![1way_qq_plots](/1way_qq_plots.png)  

![1way_tukey_plot](/1way_tukey_plot.png)  

![1way_point_plot](/1way_point_plot.png)  

## Conclusions:

Conclusions from the project findings and results.

## Next steps:  

With any analysis it is important to assess how the model and application of the analytical methods can be used and evolved to support the business goals and business decisions and yield tangible benefits.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/ANOVA_1-way_Exam_pt2.py)
