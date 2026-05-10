---

layout: default

title: Iris sepal petal lengths (Two-Sample Independent T-Test)

permalink: /2-sample-independent-t-test/

---

## Goals and objectives:

The business objective is to test the sepal petal lengths of iris plants (species = ensata) grown under 2 separate conditions (group 1 and group 2) to determine if the lengths have a statistically significant difference, and hence understand if the growing conditions is a factor in the length of the sepal petal length. The business wants to grow the biggest plants as they can be sold at a higher price.  This project demonstrates how a Two-Sample Independent T-Test can be used to assess this null hypothesis using the data available.  The sample dataset includes 50 sepal petal length measurements from each of the 2 groups of ensata iris plants.   

This test provided evidence that there is a statistically significant difference in the mean length of the sepal petals in the two groups, with evidence that those in group 2 are statistically longer. 

## Application:  

The Two-Sample Independent T-Test is a powerful statistical tool used to determine if there is a statistically significant difference between the means of two completely separate (independent) groups. In a business context, this is essential for A/B testing, competitive analysis, and comparing performance between different segments or strategies.  

* An asset management firm tests two different portfolio construction strategies (A and B). They use the t-test to determine if the mean annual return of a sample of portfolios managed under Strategy A is significantly different from the mean annual return of a sample managed under Strategy B.
* Comparing the average credit score of applicants who default on loans versus those who do not default to validate a credit scoring model's predictive power.
* A retailer launches two different website layouts or email promotions (Version A and Version B) to two random, independent groups of customers. They use the t-test to compare the mean conversion rate or mean revenue per user between the two groups.
* A software company tests two different application workflows (Workflow 1 vs. Workflow 2) and measures the mean task completion time for two independent groups of new users to see which design is more efficient.
* After implementing a change to a production line process, a manufacturer compares the mean defect rate of products produced before the change to the mean defect rate of products produced after the change to quantify the improvement.

## Methodology:  

The methodology adopted for this project follows the end-to-end data science workflow, progressing from data loading and validation through exploratory analysis, assumption testing, hypothesis testing, and the extraction of business insight. The project is implemented in Python, using pandas for data manipulation, scipy for statistical testing, and seaborn and matplotlib for visualisation. Each stage of the pipeline is described below.

### Data Loading and Validation:

The dataset is loaded from a locally stored Excel file (`Iris_ensata.xlsx`) using pandas. A structured validation audit is conducted prior to any analysis, checking for missing values across both columns (Length and Test set), identifying and removing any duplicate records, and confirming that both columns carry the expected data types — numeric for Length and integer for Test set. The distribution of observations across the two groups is also confirmed to be balanced at n=50 each, and descriptive statistics are printed for all variables to provide an initial picture of the data before any formal analysis begins.

### Hypothesis Statement:

The null and alternative hypotheses for this test are:

* **H₀**: The mean sepal petal length of Iris ensata in Group 1 = the mean sepal petal length of Iris ensata in Group 2 (i.e. there is no statistically significant difference)
* **H₁**: The mean sepal petal length of Iris ensata in Group 1 ≠ the mean sepal petal length of Iris ensata in Group 2 (i.e. a statistically significant difference exists)

The significance threshold is set at α = 0.05, representing a 95% confidence level. The assumption of independence of observations is accepted by design of the experiment — the two groups of plants were grown under separate conditions with no overlap between them.

### Exploratory Data Analysis:

Exploratory analysis is performed on the sepal length measurements for both groups before any formal testing is conducted. Three charts are produced to support visual inspection of the data:

* A **histogram with KDE overlay** for each group, used to assess the shape of each group's length distribution and to provide an initial visual check of the normality assumption. Departures from a bell-shaped, symmetric distribution would be visible here before the formal test is applied.
* A **boxplot** for each group, used to inspect central tendency, spread, and the presence or absence of extreme outliers. Outliers in a two-sample t-test can distort the t-statistic and inflate or deflate the p-value, so confirming their absence is a necessary pre-analysis step.
* A **violin plot with individual data points overlaid**, which combines a kernel density estimate of the distribution shape with the individual observation positions. This provides a richer view of the within-group distribution than the boxplot alone — particularly useful for identifying multi-modality or asymmetry that a boxplot would not reveal.

### Assumption Testing:

Before performing the Two-Sample T-Test, two formal assumption checks are conducted:

**Normality — Shapiro-Wilk Test**: The Shapiro-Wilk test is applied to each group separately. Its null hypothesis is that the sample is drawn from a normally distributed population. A p-value greater than 0.05 supports the normality assumption. It should be noted that with n=50 observations in each group, the Central Limit Theorem (CLT) provides robustness to moderate departures from normality in any case — the sampling distribution of the mean will be approximately normal at this sample size regardless of the underlying population distribution. The Shapiro-Wilk test therefore serves as a formal confirmation of what the histograms suggest visually.

**Homogeneity of Variances — Levene's Test**: Levene's test examines whether the variances of the two groups are equal. Its null hypothesis is that the two population variances are the same. The outcome of this test determines which variant of the t-test is applied: if the variances are equal (p > 0.05), Student's t-test is used, which assumes a common pooled variance. If the variances differ significantly (p ≤ 0.05), Welch's t-test is more appropriate, as it does not assume equal variances and adjusts the degrees of freedom accordingly. Both variants are computed for completeness.

### Hypothesis Testing:

The Two-Sample T-Test is applied using scipy's `ttest_ind()` function, with the `equal_var` parameter set based on the outcome of Levene's test. The test produces a t-statistic and a p-value, which are evaluated against the significance threshold of α = 0.05.

**Effect size** is quantified using Cohen's d, defined as the difference between the two group means divided by the pooled standard deviation. Cohen's d provides a scale-independent measure of the practical magnitude of the difference between the groups, complementing the p-value which reflects only the probability of the observed result under H₀. Standard thresholds are: negligible (< 0.2), small (< 0.5), medium (< 0.8), and large (≥ 0.8).

A **95% confidence interval for the difference in means** is also constructed, providing a plausible range for the true population difference. A CI that does not contain zero is consistent with a significant p-value, and the sign and width of the interval convey directional and precision information that the p-value alone does not.

A fourth chart is produced to directly visualise the core quantity of interest — a **mean comparison plot with 95% confidence interval error bars** for each group, with individual data points overlaid. This chart makes the relative position of the two group means and the uncertainty around each immediately apparent, and provides intuitive visual support for the statistical conclusion reached by the test.

## Results:

### Descriptive statistics:  

Initially a histogram and KDE of the iris sepal petal lengths for each group of observations was created to visually inspect the distribution.

![Histogram of petal length by group](/2s_ttest_hist.png)

The histogram and KDE curves show that both groups follow an approximately symmetric, bell-shaped distribution with no obvious skew. The distributions overlap substantially, but the centre of Group 2's distribution sits visibly to the right of Group 1's, indicating a higher mean length. Both KDE curves are smooth and unimodal, providing initial visual support for the normality assumption that will be tested formally below.

Boxplot and violin plots of the values for each group were also produced, to further understand the distributions.

![Boxplot of petal length by group](/2s_ttest_boxplot.png)

![Violin plot of petal length by group](/2s_ttest_violin.png)

The boxplot confirms that neither group contains extreme outliers. Both interquartile ranges are compact and the whiskers are of proportionate length, with no individual observations plotted beyond the whisker boundaries. This confirms that no individual measurements would materially distort the t-statistic, and the outlier assumption of the Two-Sample T-Test is satisfied. The median line for Group 2 sits noticeably higher than that of Group 1, consistent with what the histogram suggested.

The violin plot reinforces these observations: both distributions are approximately symmetric and unimodal, with the widest density mass concentrated around the respective group means. The overlaid individual data points confirm a dense, well-behaved spread in both groups with no isolated observations that would warrant investigation or removal. The combined evidence from the histogram, boxplot, and violin plot supports proceeding to formal assumption testing and hypothesis testing with confidence in the quality and distribution of the data.

Simple descriptive statistics for each group:

```
Ensata Group 1:  n=50, Mean=8.159cm, SD=0.381  
Ensata Group 2:  n=50, Mean=8.333cm, SD=0.401  
Difference in Means:  0.174cm
```
The mean sepal petal length of Group 2 is 0.174cm greater than that of Group 1. The standard deviations are similar in magnitude (0.381 and 0.401), suggesting comparable within-group 
variability. The chart below visualises the mean and 95% confidence interval for each group, with individual data points overlaid.

![Mean sepal length with 95% confidence intervals](/2s_ttest_mean_ci.png)

The mean comparison plot makes the direction and magnitude of the difference between the two groups immediately apparent. The 95% confidence interval for Group 1 spans approximately 8.05cm to 8.27cm, and for Group 2 approximately 8.22cm to 8.45cm. Crucially, the two confidence intervals show only minimal overlap at their extremities, which provides visual confirmation that the difference in means is statistically significant at the α = 0.05 level — a finding that is formally confirmed by the t-test below. The individual data points overlaid on the chart illustrate the within-group spread and confirm that the group means are representative summaries of their respective distributions, with no extreme values pulling either mean away from the centre of the data.

### Hypothesis Test:

First we test the data for normality, addressing both groups separately, using the Shapiro-Wilk Normality Test. It should be noted that as there are 50 observations for each group, the Central Limit Theorem (CLT) ensures robustness to non-normality anyway. The null hypothesis of the Shapiro-Wilk Normality Test is that the observations are normally distributed.

Shapiro-Wilk Normality Test results:

```
Ensata Group1: p=0.9173 (Normal)  
Ensata Group2: p=0.6300 (Normal)
```

As both p-values are considerably greater than 0.05, both samples are considered to be normally distributed. The evidence strongly supports the null hypothesis of the Shapiro-Wilk test, and the normality assumption of the Two-Sample T-Test is satisfied. This is consistent with the visual evidence from the histogram and violin plot.

Another assumption that we need to test is for homogeneity of variances, using Levene's Test for Equal Variances, where the null hypothesis is that the variances of the observations in Group 1 and Group 2 are equal. The results were:

```
F-statistic = 0.5494, p-value = 0.4604
```

As 0.4604 > 0.05, the evidence supports the null hypothesis of Levene's Test — the variances of the two groups are not significantly different. Student's t-test (which assumes equal, pooled variance) is therefore the appropriate variant to apply. For completeness, Welch's t-test (which does not assume equal variances) is also computed; in this case the two variants produce near-identical results, which is expected when variances are equal.

The Two-Sample T-Test was applied to the data for the two groups, where the significance level (alpha) was set to 0.05 — i.e. a 95% confidence level. The results were:

```
T-Statistic:  -2.2317  
P-Value:       0.0279
```

As 0.0279 < 0.05 we reject the null hypothesis (H₀). The evidence supports the alternative hypothesis that the means of the two groups are statistically significantly different — the sepal petals from Group 2 are statistically significantly longer than those from Group 1.

To further understand the magnitude and precision of the difference, two additional measures are reported:

**Effect size:** Cohen's d = 0.4463, classified as a **Small** effect size under the standard Cohen (1988) thresholds (small < 0.5, medium < 0.8, large ≥ 0.8). It should be noted that the value sits close to the medium threshold, indicating a difference of moderate practical consequence. The interpretation of effect size is always contextual — a discussion of what this means for the business is provided in the Conclusions section.

**95% Confidence Interval for the difference in means:** (0.019cm, 0.330cm), with a mean difference of 0.174cm. The interval is entirely positive, confirming the directional finding that Group 2 has a higher mean than Group 1. The interval does not contain zero, which is consistent with the significant p-value. The width of the interval (0.311cm) reflects the uncertainty associated with estimating the true population difference from samples of n=50 each.

## Conclusions:

### Statistical Interpretation

The Two-Sample Independent T-Test provides strong statistical evidence that the mean sepal petal length of Iris ensata differs between the two growing conditions. The three key statistical outputs — the p-value, effect size, and confidence interval — each contribute different and complementary information to this conclusion, and should be read together rather than in isolation.

The p-value of 0.0279 indicates that, under the assumption that H₀ is true (i.e. that the two groups have the same population mean), the probability of observing a difference in sample means as large as 0.174cm or larger is only 2.79%. This falls below the pre-specified significance threshold of α = 0.05, and so H₀ is rejected. The result is statistically significant.

Cohen's d of 0.4463 quantifies the magnitude of that difference in standardised terms. While the p-value tells us the result is unlikely to be due to chance, it says nothing about how large the difference is. Cohen's d — a scale-independent measure — indicates the two group means sit approximately 0.45 standard deviations apart. This is classified as a Small effect under Cohen's standard thresholds, sitting close to the medium boundary. The 95% confidence interval of [0.019cm, 0.330cm] provides a plausible range for the true population difference; while the lower bound approaches zero (suggesting the true difference could be modest), the upper bound of 0.330cm would represent a commercially meaningful size advantage. Taken together, all three measures point in the same direction: a genuine, positive difference in favour of Group 2, with moderate practical magnitude.

### Business Interpretation

An important distinction is that statistical significance and practical significance are not equivalent, and both must be considered when drawing business conclusions.

The test confirms that the difference in means is unlikely to be due to chance. However, the practical value of a 0.174cm difference in mean sepal length depends entirely on the commercial context. In this case, where the business objective is to maximise plant size in order to achieve higher sale prices, even a modest but consistent size advantage across a population of plants could accumulate into meaningful revenue — particularly at scale. A grower producing thousands of plants per season would see the 0.174cm mean advantage translate into a material shift in the proportion of plants meeting a premium size threshold. The growing conditions applied in Group 2 appear to produce longer sepal petals and should be considered as the preferred approach going forward, pending further investigation as described in the Next Steps below.

It should also be noted that the 95% confidence interval, spanning 0.019cm to 0.330cm, means the business cannot be certain the advantage is as large as 0.174cm — the true difference could be anywhere in that range. This uncertainty is a natural consequence of working with samples of n=50 per group, and should be acknowledged in any business decision based on this analysis.

### Limitations

Several limitations of this analysis should be considered when interpreting the findings:

**Sample size:** With n=50 observations per group, the analysis is adequately powered for detecting a medium-sized effect, but the relatively modest sample size contributes to a wide confidence interval (0.311cm). Larger samples would narrow this interval and provide a more precise estimate of the true population difference. The current results should be treated as indicative rather than definitive.

**Single measurement:** The analysis is based solely on sepal petal length. A comprehensive assessment of plant size and commercial value would also consider petal length, petal width, and sepal width. It is possible that the growing conditions benefit one measurement but not others, or that the overall morphological profile of the two groups differs in ways not captured by this single variable.

**Growing conditions not formally described:** The analysis treats the two groups as distinct experimental conditions, but the precise nature of those conditions is not specified in the dataset. Without a formal description of what differs between the conditions (e.g. soil composition, irrigation, light exposure, fertiliser type), it is not possible to attribute the observed difference to a specific causal factor. Additional experimental documentation would be required before any specific horticultural recommendation could be made with confidence.

**Independence assumption:** The assumption of independence of observations is accepted by design. However, if plants within a group were grown in proximity (e.g. in the same greenhouse bay or batch), within-group measurements may be correlated due to shared environmental micro-conditions. Clustered or nested designs of this kind would require mixed-effects models rather than a standard two-sample t-test to correctly account for that dependency structure.

## Next steps:

Having concluded that the mean iris sepal length for group 2 is longer, with the effect being considered 'small to medium', it is recommended to make additional measurements in the future to further test the findings, as well as trying to grow ensata iris plants other growing conditions to allow additional tests to determine if other conditions result in even longer sepal lengths.

Should additional growing conditions be tested, a natural methodological extension is the one-way ANOVA, which would allow simultaneous comparison of three or more group means without inflating the Type I error rate that multiple pairwise t-tests would introduce. This technique is covered elsewhere in this portfolio.

A further consideration is the analysis of additional morphological measurements — petal length, petal width, sepal width — to determine whether the growing condition effect is consistent across all measurements, or specific to sepal petal length. A multivariate approach such as MANOVA could test whether the overall measurement profile differs between groups.

The assumption of independence of observations was accepted by design. In a real experimental setting, confirming that plants in each group were genuinely independently assigned (e.g. via randomisation, not batch or greenhouse clustering) would be an important validation step. Clustered or nested designs would require mixed-effects models rather than a standard two-sample t-test.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/2Sample_Independent_T-Test_Pt3.py)

The original data used for the analysis is here:
[Access input data](/Iris_ensata.xlsx)
