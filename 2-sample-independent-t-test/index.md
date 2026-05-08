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

A workflow in Python was developed using libraries Scipy, Pandas and Numpy, utilising Matplotlib and Seaborn for visualisations.  The data came from a publicly available dataset of iris measurements from the library scikit-learn, which was then extended to support the generation of statistically interesting findings.  

### Data Loading and Validation:

The dataset is loaded from a locally stored Excel file (Iris_ensata.xlsx) using pandas. A structured validation audit is conducted prior to any analysis: missing values are checked across both columns (Length and Test set), duplicate records are identified, and the data types of each column are confirmed to be numeric and categorical as expected. Descriptive statistics are printed for all variables and the raw distribution of observations across the two test groups is confirmed to be balanced (n=50 each).

### Hypothesis Testing
The Two-Sample T-Test was used to test the null hypothesis that the mean sepal petal lengths for group 1 and group 2 are the same.  

The null and alternative hypotheses for the test are:

* H₀: The mean sepal petal length of Iris ensata in Group 1 = the mean sepal petal length in Group 2
* H₁: The mean sepal petal length of Iris ensata in Group 1 ≠ the mean sepal petal length in Group 2

The assumption of independence of observations is assumed due to design of the experiment.

Data preparation:  Minor transformation of data into a pandas dataframe for analytical purposes.

## Results:

### Descriptive statistics:  

Initially a histogram and KDE of the iris petal lengths for each group of observations was created to visually inspect the distribution.    

![Histogram of petal length by group](/2s_ttest_hist.png) 

Boxplot and violin plots of the values for each group were also produced, to further understand the distributions. 

![Boxplot of petal length by group](/2s_ttest_boxplot.png) 
![Violin plot of petal length by group](/2s_ttest_violin.png) 


The boxplot confirms that neither group contains extreme outliers. Both interquartile ranges are compact and the whiskers are of proportionate length, indicating no individual observations that
would materially distort the t-statistic. The outlier assumption of the Two-Sample T-Test is satisfied.  The violin plot confirms that both distributions are approximately symmetric and unimodal, providing initial visual support for the normality assumption ahead of formal testing.


Simple descriptive statistics for each group:  
Ensata Group 1:  n=50, Mean=8.159cm, SD=0.381  
Ensata Group 2:  n=50, Mean=8.333cm, SD=0.401  
Difference in Means:  0.174cm 

It is noted that the histograms / KDEs look normal for each group, but we shall test that also.

### Hypothesis Test:

First we test the data for normality, addressing both groups separately, using the Shapiro-Wilk Normality Test.  It should be noted that as there are 50 observations for each group, the Central Limit Theorem (CLT) ensures robustness to non-normality anyway.  The null hypothesis of the Shapiro-Wilk Normality Test is that the observations are normally distributed.

Shapiro-Wilk Normality Test results:  
Ensata Group1: p=0.9173 (Normal)  
Ensata Group2: p=0.6300 (Normal)  
As both p-values are greater than 0.05, then both samples are considered to be normally distributed, i.e. the evidence supports the null hypothesis of the Shapiro-Wilk Normality Test. 

Another assumption that we need to test is for homogeneity of variances, using Levene's Test for Equal Variances, where the null hypothesis is that the variances of the observations in group 1 and group 2 are equal. The results were:

F-statistic = 0.5494, p-value = 0.4604  
Conclusion:  As 0.4604 > 0.05, the evidence supports the null hypothesis of Levene's Test, that the variances are equal, therefore we can use Student's t-test to test the overall research null hypothesis.  Should variances not be equal, then an alternative test, such as Welch's t-test may be more applicable.

The Two-Sample T-Test was applied to the data for the two groups, where the significance level (alpha) was set to 0.05 - i.e. 95% confidence level. The results were: 

```
T-Statistic:  -2.2317  
P-Value:       0.0279
```

As 0.0279 < 0.05 we can reject the null hypothesis (H₀) and that the evidence supports the alternate hypothesis that the means of the two groups are statistically significantly different, i.e. the sepal petals from group 2 have a statistically significant longer length.

We wish to further understand the differences in mean lengths: 

Effect size:  The Cohen's D measure is used to quantify the difference between the two group means, the value is: Cohen's d = 0.4463.  Using the standard interpretation of Cohen's D this is considered a 'Medium' effect size, noting that the value sits near the 'High' effect size threshold.  It should be noted that the interpretation of the effect size is subjective, and the true business context should be considered, for example in this case the difference in mean lengths can provide significant business benefit, for example it makes the plants more valuable and desirable to buyers.

## Conclusions:

95% Confidence Interval (CI) of the difference in means is: (0.019cm, 0.330cm), with the mean difference 0.174cm - noting that these are all positive, i.e. confirming that the group 2 mean is greater than the group 1 mean.

An important distinction is that statistical significance and practical significance are not equivalent.  The test confirms that the difference in means is unlikely to be due to chance (p = 0.028), but Cohen's d of 0.446 — classified as Small to Medium — indicates the magnitude of the difference is moderate.  In this business context, where the commercial value of the iris plants depends on physical size, even a 0.174cm difference in mean sepal length may carry meaningful pricing implications. The practical significance of a result must always be evaluated in the context of the domain, not by the effect size label alone.

## Next steps:
Having concluded that the mean iris sepal length for group 2 is longer, with the effect being considered 'small to medium', it is recommended to make additional measurements in the future to further test the findings, as well as trying to grow ensata iris plants other growing conditions to allow additional tests to determine if other conditions result in even longer sepal lengths.

Should additional growing conditions be tested, a natural methodological extension is the one-way ANOVA, which would allow simultaneous comparison of three or more group means without inflating the Type I error rate that multiple pairwise t-tests would introduce. This technique is covered elsewhere in this portfolio.

A further consideration is the analysis of additional morphological measurements — petal length, petal width, sepal width — to determine whether the growing condition effect is consistent across all measurements, or specific to sepal petal length. A multivariate approach such as MANOVA could test whether the overall measurement profile differs between groups.


The assumption of independence of observations was accepted by design. In a real experimental setting, confirming that plants in each group were genuinely independently assigned (e.g. via randomisation, not batch or greenhouse clustering) would be an important validation step. Clustered or nested designs would require mixed-effects models rather than a standard two-sample t-test.

## Python code:
You can view the full Python script used for the analysis here: 
[View the Python Script](/2Sample_Independent_T-Test_Pt2.py)

The original data used for the analysis is here:
[Access input data](/Iris_ensata.xlsx)
